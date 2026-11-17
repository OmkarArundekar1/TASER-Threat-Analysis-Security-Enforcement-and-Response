"""Offline, read-only analysis of Wazuh alert files. Streams each file, plain or gzip, writes nothing beside them, and copies nothing. Records counts, shapes and ids,
never payloads. Manager names are recorded only as short hashes. Ids are event seconds and file offsets, so they're kept as examples.
usage: python -I -B analyze_alert_ids.py OUTPUT_JSON FILE..."""
import collections
import datetime as dt
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

STRICT_ID = re.compile(r"[0-9]+\.[0-9]+")
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{4}")
MAX_TOLERANT_ID = 64
UNSAFE_ID_CHARACTER = re.compile(r"[\x00-\x20\x7f|]")


def short_hash(text):
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:12]


def digest(data):
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def id_kind(value):
    """The JSON type and condition of an id, as the proposed policy would see it."""
    if value is None:
        return "null_or_missing"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if not isinstance(value, str):
        return type(value).__name__
    if value == "":
        return "empty_string"
    if STRICT_ID.fullmatch(value):
        return "string_seconds_dot_offset"
    return "string_other_form"


def tolerant_policy_accepts(value):
    """The proposed policy, for evaluation only: a bounded, non-empty string, kept exactly. A value that could forge a key separator or carries whitespace or control
    characters is refused rather than rewritten. The manager is percent-encoded, so the first / is unambiguous, and an id holding / is safe. | isn't: it separates the request key."""
    return type(value) is str and 0 < len(value) <= MAX_TOLERANT_ID and not UNSAFE_ID_CHARACTER.search(value)


def manager_kind(alert):
    block = alert.get("manager", None)
    if block is None:
        return "no_manager_block", None
    if not isinstance(block, dict):
        return "manager_not_an_object", None
    name = block.get("name")
    if name is None:
        return "no_name", None
    if not isinstance(name, str):
        return "name_not_a_string", None
    if name == "":
        return "name_empty", None
    return "name_string", name


def legacy_key(alert):
    """generate_alert_id before Phase 2.6C, to count how often it would have merged alerts in the real archive."""
    data = alert.get("data") if isinstance(alert.get("data"), dict) else {}
    rule = alert.get("rule") if isinstance(alert.get("rule"), dict) else {}
    agent = alert.get("agent") if isinstance(alert.get("agent"), dict) else {}
    return (alert.get("timestamp", ""), rule.get("id", ""), agent.get("id", ""), data.get("srcip") or alert.get("srcip", ""), data.get("dstip") or alert.get("dstip", ""), alert.get("location", ""))


class Scan:
    def __init__(self):
        self.seen = {}           # hash of manager and id: (hash of the whole line, file number)
        self.seen_bare = {}      # hash of id alone: hash of the manager
        self.files = []
        self.totals = collections.Counter()
        self.id_kinds = collections.Counter()
        self.id_lengths = collections.Counter()
        self.manager_kinds = collections.Counter()
        self.managers = collections.Counter()
        self.manager_first_last = {}
        self.manager_months = collections.defaultdict(set)
        self.decoders = collections.Counter()
        self.rule_ids = set()
        self.agents = set()
        self.locations = set()
        self.months = collections.Counter()
        self.timestamp_offsets = collections.Counter()
        self.seconds_difference = collections.Counter()
        self.other_form_examples = []
        self.examples = {}
        self.duplicates_identical = 0
        self.duplicates_different = []
        self.duplicates_identical_across_files = collections.Counter()
        self.bare_across_managers = 0
        self.previous_file_last = None

    def add_file(self, path, name):
        stats = collections.Counter()
        legacy, previous_offset, previous_seconds = collections.Counter(), None, None
        first_id, last_id = None, None
        number = len(self.files)
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rb") as handle:
            for raw in handle:
                stats["lines"] += 1
                if not raw.endswith(b"\n"):
                    stats["partial_final_line"] += 1
                    continue
                if not raw.strip():
                    stats["blank"] += 1
                    continue
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    stats["not_utf8"] += 1
                    continue
                try:
                    alert = json.loads(text)
                except ValueError:
                    stats["malformed_json"] += 1
                    continue
                if not isinstance(alert, dict):
                    stats["not_an_object"] += 1
                    continue
                stats["alerts"] += 1
                self.observe(alert, raw, number, stats, legacy)
                value = alert.get("id")
                if isinstance(value, str) and STRICT_ID.fullmatch(value):
                    seconds, offset = map(int, value.split("."))
                    if previous_offset is not None:
                        stats["offset_fell"] += offset < previous_offset
                        stats["offset_repeated"] += offset == previous_offset
                        stats["seconds_fell"] += seconds < previous_seconds
                    previous_offset, previous_seconds = offset, seconds
                    first_id = first_id or value
                    last_id = value
        stats["legacy_key_groups"] = sum(1 for v in legacy.values() if v > 1)
        stats["legacy_key_alerts_merged"] = sum(v - 1 for v in legacy.values() if v > 1)
        row = {"file": name, **stats, "first_id": first_id, "last_id": last_id}
        self.files.append(row)
        for key in ("alerts", "not_utf8", "malformed_json", "not_an_object", "partial_final_line", "offset_fell", "offset_repeated", "seconds_fell", "legacy_key_alerts_merged", "legacy_key_groups", "lines", "blank"):
            self.totals[key] += stats[key]
        return row

    def observe(self, alert, raw, number, stats, legacy):
        value = alert.get("id")
        kind = id_kind(value)
        self.id_kinds[kind] += 1
        if isinstance(value, str):
            self.id_lengths[len(value)] += 1
            self.totals["tolerant_policy_accepts"] += tolerant_policy_accepts(value)
            if kind == "string_other_form" and len(self.other_form_examples) < 10:
                self.other_form_examples.append(value[:40])
        else:
            self.totals["tolerant_policy_refuses_non_string"] += 1
        status, name = manager_kind(alert)
        self.manager_kinds[status] += 1
        stamp = alert.get("timestamp")
        month = stamp[:7] if isinstance(stamp, str) else "none"
        self.months[month] += 1
        if isinstance(stamp, str):
            self.totals["timestamp_expected_form"] += bool(TIMESTAMP.fullmatch(stamp))
            self.timestamp_offsets[stamp[-5:]] += 1
            if isinstance(value, str) and STRICT_ID.fullmatch(value):
                try:
                    when = dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f%z")
                    difference = int(value.split(".")[0]) - int(when.timestamp())
                    self.seconds_difference[max(-3, min(3, difference))] += 1
                except ValueError:
                    self.totals["timestamp_unparsed"] += 1
        decoder = alert.get("decoder") if isinstance(alert.get("decoder"), dict) else {}
        self.decoders[str(decoder.get("name"))] += 1
        rule = alert.get("rule") if isinstance(alert.get("rule"), dict) else {}
        self.rule_ids.add(str(rule.get("id")))
        agent = alert.get("agent") if isinstance(alert.get("agent"), dict) else {}
        self.agents.add(short_hash(str(agent.get("id"))))
        self.locations.add(short_hash(str(alert.get("location"))))
        legacy[legacy_key(alert)] += 1
        if "cluster" in alert:
            self.totals["has_cluster_block"] += 1
        if name is not None:
            hashed = short_hash(name)
            self.managers[hashed] += 1
            self.manager_months[hashed].add(month)
            self.totals["manager_names_needing_encoding"] += not re.fullmatch(r"[A-Za-z0-9._~-]+", name)
        if isinstance(value, str) and value:
            line = digest(raw)
            bare = digest(value.encode("utf-8", "surrogatepass"))
            manager_hash = short_hash(name) if name is not None else None
            if name is not None:
                key = digest(f"{name}\x00{value}".encode("utf-8", "surrogatepass"))
                earlier = self.seen.get(key)
                if earlier is None:
                    self.seen[key] = (line, number)
                elif earlier[0] == line:
                    self.duplicates_identical += 1
                    self.duplicates_identical_across_files[f"{self.file_name(earlier[1])} and {self.file_name(number)}"] += 1
                elif len(self.duplicates_different) < 200:
                    self.duplicates_different.append({"id": value, "first_file": self.file_name(earlier[1]), "second_file": self.file_name(number)})
                    self.totals["duplicates_different_content"] += 1
                else:
                    self.totals["duplicates_different_content"] += 1
            if bare in self.seen_bare and self.seen_bare[bare] != manager_hash:
                self.bare_across_managers += 1
            self.seen_bare.setdefault(bare, manager_hash)
            if month not in self.examples and kind == "string_seconds_dot_offset":
                self.examples[month] = value

    def file_name(self, number):
        return self.files[number]["file"] if number < len(self.files) else self.current

    def report(self):
        return {"totals": dict(self.totals), "id_kinds": dict(self.id_kinds), "id_lengths": dict(sorted(self.id_lengths.items())), "manager_kinds": dict(self.manager_kinds),
                "managers_by_short_hash": {k: {"alerts": v, "months": sorted(self.manager_months[k])} for k, v in self.managers.items()},
                "unique_manager_and_id": len(self.seen), "repeats_identical_line": self.duplicates_identical, "repeats_identical_line_where": dict(self.duplicates_identical_across_files.most_common(20)),
                "repeats_different_content_examples": self.duplicates_different[:20], "same_id_under_different_manager": self.bare_across_managers,
                "months": dict(sorted(self.months.items())), "id_seconds_minus_timestamp_clamped": dict(sorted(self.seconds_difference.items())), "timestamp_offsets": dict(self.timestamp_offsets),
                "decoders_top": dict(self.decoders.most_common(20)), "decoder_count": len(self.decoders), "distinct_rules": len(self.rule_ids), "distinct_agents": len(self.agents), "distinct_locations": len(self.locations),
                "other_form_id_examples": self.other_form_examples, "example_id_per_month": self.examples, "files": self.files}


def main():
    output = Path(sys.argv[1])
    scan = Scan()
    for index, path in enumerate(sys.argv[2:]):
        name = "/".join(Path(path).parts[-3:])
        scan.current = name
        row = scan.add_file(path, name)
        print(f"{index + 1}/{len(sys.argv) - 2} {name}: {row.get('alerts', 0)} alerts", flush=True)
    output.write_text(json.dumps(scan.report(), indent=1), encoding="utf-8")
    print(json.dumps({k: scan.report()[k] for k in ("totals", "id_kinds", "manager_kinds", "unique_manager_and_id", "repeats_identical_line", "same_id_under_different_manager")})[:1500])


if __name__ == "__main__":
    main()

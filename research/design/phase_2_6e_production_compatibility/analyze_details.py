"""Offline, read-only details the first analysis raised: how the old key's merges break down, how far an id's seconds fell, peak rates, and the shape of bad lines.
Streams existing files, and prints counts, decoder names and ids, never payloads. usage: python -I -B analyze_details.py OUTPUT_JSON FILE..."""
import collections
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

from analyze_alert_ids import legacy_key

STRICT = re.compile(r"[0-9]+\.[0-9]+")


def content_digest(alert):
    """Everything except what must differ between two occurrences: the id, Wazuh's firing counter, and its previous-output field."""
    body = {k: v for k, v in alert.items() if k not in ("id", "previous_output", "previous_log")}
    if isinstance(body.get("rule"), dict):
        body["rule"] = {k: v for k, v in body["rule"].items() if k != "firedtimes"}
    return hashlib.blake2b(json.dumps(body, sort_keys=True, default=str).encode("utf-8", "surrogatepass"), digest_size=8).digest()


def analyze_file(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    row = collections.Counter()
    per_second = collections.Counter()
    drops = []
    bad_lines = []
    decoders = collections.Counter()
    first_by_key = {}
    previous = None
    with opener(path, "rb") as handle:
        for raw in handle:
            try:
                alert = json.loads(raw.decode("utf-8"))
            except ValueError:
                stripped = raw.rstrip(b"\r\n")
                bad_lines.append({"length": len(raw), "ends_in_newline": raw.endswith(b"\n"), "ends_in_brace": stripped.endswith(b"}"), "starts_with_brace": stripped.startswith(b"{")})
                continue
            if not isinstance(alert, dict):
                continue
            value = alert.get("id")
            if isinstance(value, str) and STRICT.fullmatch(value):
                seconds = int(value.split(".")[0])
                per_second[seconds] += 1
                if previous is not None and seconds < previous:
                    drops.append(previous - seconds)
                previous = seconds
            key = legacy_key(alert)
            digest = content_digest(alert)
            if key in first_by_key:
                row["merged"] += 1
                row["merged_identical_content_apart_from_id"] += first_by_key[key] == digest
                decoder = alert.get("decoder") if isinstance(alert.get("decoder"), dict) else {}
                decoders[str(decoder.get("name"))] += 1
                data = alert.get("data") if isinstance(alert.get("data"), dict) else {}
                row["merged_uses_src_ip_spelling"] += bool(data.get("src_ip"))
                row["merged_has_no_network_field"] += not any(data.get(k) for k in ("srcip", "src_ip", "dstip", "dest_ip"))
            else:
                first_by_key[key] = digest
            row["alerts"] += 1
    return {"alerts": row["alerts"], "merged": row["merged"], "merged_identical_content_apart_from_id": row["merged_identical_content_apart_from_id"], "merged_uses_src_ip_spelling": row["merged_uses_src_ip_spelling"],
            "merged_has_no_network_field": row["merged_has_no_network_field"], "merged_decoders": dict(decoders.most_common(5)), "busiest_second": max(per_second.values(), default=0), "seconds_with_100_or_more": sum(v >= 100 for v in per_second.values()),
            "id_seconds_drops": sorted(drops, reverse=True)[:10], "bad_lines": bad_lines}


def main():
    output = Path(sys.argv[1])
    files = {}
    for path in sys.argv[2:]:
        files["/".join(Path(path).parts[-3:])] = analyze_file(path)
    totals = collections.Counter()
    for row in files.values():
        for key in ("alerts", "merged", "merged_identical_content_apart_from_id", "merged_uses_src_ip_spelling", "merged_has_no_network_field"):
            totals[key] += row[key]
    summary = {"totals": dict(totals), "busiest_second_anywhere": max(r["busiest_second"] for r in files.values()), "seconds_with_100_or_more": sum(r["seconds_with_100_or_more"] for r in files.values()),
               "largest_id_seconds_drop": max((max(r["id_seconds_drops"]) for r in files.values() if r["id_seconds_drops"]), default=0),
               "bad_lines": {k: r["bad_lines"] for k, r in files.items() if r["bad_lines"]}, "id_seconds_drops": {k: r["id_seconds_drops"] for k, r in files.items() if r["id_seconds_drops"]},
               "top_merging_days": sorted(({"file": k, "alerts": r["alerts"], "merged": r["merged"], "identical_apart_from_id": r["merged_identical_content_apart_from_id"], "decoders": r["merged_decoders"], "no_network_field": r["merged_has_no_network_field"]} for k, r in files.items()), key=lambda r: -r["merged"])[:6]}
    output.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary)[:3500])


if __name__ == "__main__":
    main()

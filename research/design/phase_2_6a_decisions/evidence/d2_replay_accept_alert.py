"""Read only and offline. Replays archived alerts through the listener's REAL accept_alert and generate_alert_id,
then models what process_alert's later dedupe would have done. Starts nothing: the functions are lifted from the
source by AST, so importing the listener (signal handlers, log files) never happens. No database, no network.
Prints counts and field names, never alert content.
usage (from isfcr/CYUKTI/backend): python -I d2_replay_accept_alert.py ALERT_FILE"""
import ast
import collections
import datetime as dt
import hashlib
import json
import os
import sys
from threading import Lock

BACKEND = os.getcwd()
path = sys.argv[1]
assert not os.path.abspath(path).startswith("/var/ossec")


def lift(rel, *names, extra=None):
    """Executes only the named top-level definitions from a source file, in a fresh namespace."""
    source = open(os.path.join(BACKEND, rel), encoding="utf-8").read()
    tree = ast.parse(source)
    picked = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            picked.append(node)
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in names:
            picked.append(node)
    found = {getattr(n, "name", None) or n.targets[0].id for n in picked}
    assert found == set(names), (rel, set(names) - found)
    namespace = dict(extra or {})
    exec(compile(ast.Module(picked, []), rel, "exec"), namespace)
    return namespace


def sha_of(rel):
    return hashlib.sha256(open(os.path.join(BACKEND, rel), "rb").read()).hexdigest()


print(f"input {path} sha256 {hashlib.sha256(open(path, 'rb').read()).hexdigest()}")
for rel in ("listener/wazuh_listener.py", "listener/utils.py", "realtime_socgraph.py", "dedup_engine.py", "config.py", "mitre_rule_registry.py"):
    print(f"source {rel} sha256 {sha_of(rel)}")

alerts = [json.loads(line) for line in open(path, encoding="utf-8", errors="replace") if line.strip()]
generate_alert_id = lift("listener/utils.py", "generate_alert_id", extra={"hashlib": hashlib})["generate_alert_id"]
limits = lift("listener/wazuh_listener.py", "MAX_ALERT_CACHE", "ALERT_CACHE_TTL")
print(f"MAX_ALERT_CACHE {limits['MAX_ALERT_CACHE']} ALERT_CACHE_TTL {limits['ALERT_CACHE_TTL']}")


class _Queue:
    """Never full: this measures the identity check alone. The worker is assumed to keep up."""

    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)


class _Clock:
    """accept_alert uses wall time. Pacing it from each alert's timestamp models a listener reading live."""
    now = 0.0

    def time(self):
        return self.now


def run(paced):
    clock, queue = _Clock(), _Queue()
    namespace = lift("listener/wazuh_listener.py", "accept_alert", extra={
        "time": clock, "alert_queue": queue, "seen_alerts": collections.OrderedDict(), "seen_alerts_lock": Lock(),
        "ALERT_CACHE_TTL": limits["ALERT_CACHE_TTL"], "MAX_ALERT_CACHE": limits["MAX_ALERT_CACHE"]})
    dropped = []
    for alert in alerts:
        if paced:
            clock.now = dt.datetime.strptime(alert["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
        else:
            clock.now += 0.001
        item = dict(alert)
        item["alert_id"] = generate_alert_id(item)
        if not namespace["accept_alert"](item):
            dropped.append(item)
    return queue.items, dropped


for label, paced in (("paced by alert timestamps", True), ("read as one fast backlog", False)):
    kept, dropped = run(paced)
    print(f"\naccept_alert, {label}: accepted {len(kept)}, refused as repeats {len(dropped)}")

kept, dropped = run(True)
first = {}
for item in kept:
    first.setdefault(item["alert_id"], item)


def paths(value, prefix=""):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from paths(inner, f"{prefix}{key}.")
    else:
        yield prefix.rstrip("."), json.dumps(value, sort_keys=True, default=str)


print("\n--- how each refused alert differs from the accepted one with its alert_id")
differing = collections.Counter()
only_id = 0
for item in dropped:
    a, b = dict(paths({k: v for k, v in item.items() if k != "alert_id"})), dict(paths({k: v for k, v in first[item["alert_id"]].items() if k != "alert_id"}))
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    if changed <= {"id"}:
        only_id += 1
    for key in changed:
        differing[key] += 1
print(f"differ only in Wazuh id: {only_id}; differ in more: {len(dropped) - only_id}; distinct Wazuh ids among refused: {len({i.get('id') for i in dropped})}")
print("fields differing, with counts:", dict(differing.most_common(25)))
print("share a timestamp with the accepted one:", sum(1 for i in dropped if i.get("timestamp") == first[i["alert_id"]].get("timestamp")))
print("refused alerts by rule level:", dict(sorted(collections.Counter(i.get("rule", {}).get("level") for i in dropped).items())))
print("refused alerts by whether srcip is present:", dict(collections.Counter(bool((i.get("data") or {}).get("srcip")) for i in dropped)))

# ------------------------------------------------------------ what the later steps would have done
pipeline = lift("realtime_socgraph.py", "extract_iocs", "extract_audit_command", "should_ignore_audit_command", "IGNORED_AUDIT_COMMANDS",
                extra={"os": os, "logger": type("L", (), {"debug": staticmethod(lambda *a: None)})})
registry_source = open(os.path.join(BACKEND, "mitre_rule_registry.py"), encoding="utf-8").read()
registry = {}
exec(compile(registry_source, "mitre_rule_registry.py", "exec"), registry)
reviewed = registry.get("REVIEWED_RULE_MAPPINGS", {})
print(f"\nreviewed rule mappings: {len(reviewed)}. Treated as valid here, because checking them needs the graph. An assumption")
window = lift("config.py", "DEDUP_WINDOW")["DEDUP_WINDOW"]


def classify(stream):
    """A model of process_alert's routing: the same fingerprint and window as dedup_engine, without Neo4j."""
    counts = collections.Counter()
    cache = {}
    for alert in stream:
        if not alert.get("rule"):
            counts["no rule"] += 1
            continue
        if pipeline["should_ignore_audit_command"](alert):
            counts["ignored audit helper"] += 1
            continue
        rule = alert["rule"]
        native = (rule.get("mitre") or {}).get("id") or []
        entry = reviewed.get(str(rule.get("id")))
        if native:
            technique, attributed = native[0], True
        elif entry and entry.get("enabled", True) and entry.get("technique_ids"):
            technique, attributed = entry["technique_ids"][0], True
        else:
            technique, attributed = None, False
        attacker, victim = pipeline["extract_iocs"](alert)
        agent = alert.get("agent", {}).get("id", "UNKNOWN")
        fingerprint = f"{attacker}|{victim}|{technique}|{rule.get('id')}|{agent}"
        now = dt.datetime.strptime(alert["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
        last = cache.get(fingerprint)
        kind = "attributed" if attributed else "unattributed"
        if last is not None and now - last <= window:
            counts[f"{kind} duplicate"] += 1
            cache[fingerprint] = max(last, now)
        else:
            counts[f"{kind} new event"] += 1
            cache[fingerprint] = now
    return counts


everything, as_listener = classify(alerts), classify(kept)
print(f"DEDUP_WINDOW {window}. Model of routing, for all alerts:", dict(sorted(everything.items())))
print("The same model, for only what accept_alert lets through:", dict(sorted(as_listener.items())))
print("Lost to accept_alert, by category:", {k: everything[k] - as_listener[k] for k in sorted(everything) if everything[k] != as_listener[k]})
print("refused alerts that would have been a new event rather than a repeat:", sum(v for k, v in (everything - as_listener).items() if "new event" in k))
attacker_differs = sum(1 for i in dropped if pipeline["extract_iocs"](i) != pipeline["extract_iocs"](first[i["alert_id"]]))
print("refused alerts whose attacker or victim differs from the accepted one:", attacker_differs)
print("consecutive-position check: refused alerts adjacent in the file to their accepted twin:", end=" ")
position = {id(a): n for n, a in enumerate(alerts)}
ids = {}
for n, a in enumerate(alerts):
    ids.setdefault(generate_alert_id(a), []).append(n)
print(sum(1 for v in ids.values() if len(v) > 1 and v[-1] - v[0] == len(v) - 1), "of", sum(1 for v in ids.values() if len(v) > 1), "groups are contiguous")
print("largest group of alerts sharing one alert_id:", max(len(v) for v in ids.values()))

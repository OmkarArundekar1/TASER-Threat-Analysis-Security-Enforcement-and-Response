"""Read only and offline. For the D13 candidate: builds the parameters update_duplicate_event would receive for each
unattributed duplicate in the sample, and passes them through the REAL journal serializer, ledger fingerprint and key
check, twice, from two separate parses. Constructs no journal, no ledger and no driver, so no file or connection opens.
The event_id is a fixed stand-in, because the real one needs the graph. Prints counts, never alert content.
usage (from isfcr/CYUKTI/backend): python -I -B d3_d13_payload_analysis.py ALERT_FILE"""
import ast
import collections
import hashlib
import json
import os
import re
import sys
import urllib.parse

sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())
path = sys.argv[1]
assert not os.path.abspath(path).startswith("/var/ossec")

import event_time as et  # noqa: E402  stdlib only, and has no side effects at import
import neo4j_idempotency_ledger as ledger  # noqa: E402
import neo4j_write_journal as journal  # noqa: E402


def constant_in(rel, function):
    tree = ast.parse(open(rel, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return max((n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)), key=len)
    raise LookupError(function)


def lift(rel, *names):
    tree = ast.parse(open(rel, encoding="utf-8").read())
    picked = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names) or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in names)]
    namespace = {"os": os, "logger": type("L", (), {"debug": staticmethod(lambda *a: None)})}
    exec(compile(ast.Module(picked, []), rel, "exec"), namespace)
    return namespace


QUERY = constant_in("neo4j_client.py", "update_duplicate_event")
print("input sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())
print("statement sha256", hashlib.sha256(QUERY.encode()).hexdigest(), "| carries MATCHES:", ":MATCHES" in QUERY)
pipeline = lift("realtime_socgraph.py", "extract_iocs", "extract_audit_command", "should_ignore_audit_command", "IGNORED_AUDIT_COMMANDS")
WINDOW = 90
PLACEHOLDER = "00000000-0000-4000-8000-000000000000"


def parse():
    return [json.loads(line) for line in open(path, encoding="utf-8", errors="replace") if line.strip()]


def key_for(alert):
    manager = urllib.parse.quote(alert["manager"]["name"], safe="-._~")
    return f"cyukti.v1|wazuh:{manager}/{alert['id']}|duplicate_event.update"


def request(alert):
    rule = alert["rule"]
    compact = json.dumps({
        "rule_id": rule.get("id"), "rule_description": rule.get("description"), "rule_level": rule.get("level"),
        "rule_groups": rule.get("groups", []), "rule_firedtimes": rule.get("firedtimes"), "mitre": rule.get("mitre"),
        "network": alert.get("data", {}).get("network", {}), "http": alert.get("data", {}).get("http", {}),
        "suricata": alert.get("data", {}).get("suricata", {}), "agent": alert.get("agent", {}),
        "location": alert.get("location"), "timestamp": alert.get("timestamp")})
    parameters = {"event_id": PLACEHOLDER, "tps": 0, "investigation_payload": compact,
                  "event_time": et.resolve_event_time(alert, et.POLICY_STRICT).value}
    return [(QUERY, parameters)], len(compact)


def walk(alerts):
    cache, events = {}, collections.Counter()
    seen_ids = collections.Counter()
    for alert in alerts:
        rule = alert.get("rule")
        if not rule or pipeline["should_ignore_audit_command"](alert):
            continue
        native = (rule.get("mitre") or {}).get("id") or []
        attacker, victim = pipeline["extract_iocs"](alert)
        fingerprint = f"{attacker}|{victim}|{native[0] if native else None}|{rule.get('id')}|{alert.get('agent', {}).get('id', 'UNKNOWN')}"
        now = et.resolve_event_time(alert, et.POLICY_STRICT).value.timestamp()
        last = cache.get(fingerprint)
        duplicate = last is not None and now - last <= WINDOW
        cache[fingerprint] = max(last, now) if duplicate else now
        if not native:
            if duplicate:
                yield alert
            else:
                events[fingerprint] += 1
    walk.events = events


first, second = parse(), parse()
pairs = list(zip(walk(first), walk(second)))
print(f"unattributed duplicates in the model: {len(pairs)}")
print(f"fingerprints with more than one modelled unattributed event: {sum(1 for v in walk.events.values() if v > 1)} of {len(walk.events)}")

same_serial, same_fp, refused, sizes, key_sizes, key_bad, per_second = 0, 0, collections.Counter(), [], [], 0, collections.Counter()
tz_types, fingerprints = collections.Counter(), set()
for a, b in pairs:
    statements_a, size = request(a)
    statements_b, _ = request(b)
    sizes.append(size)
    tz_types[type(statements_a[0][1]["event_time"].tzinfo).__name__] += 1
    try:
        text_a, _ = journal._serialize_intent(statements_a, False)
        text_b, _ = journal._serialize_intent(statements_b, False)
        same_serial += text_a == text_b
    except Exception as error:
        refused[type(error).__name__] += 1
        continue
    fp_a, fp_b = ledger.fingerprint_payload("duplicate_event.update", statements_a), ledger.fingerprint_payload("duplicate_event.update", statements_b)
    same_fp += fp_a == fp_b
    fingerprints.add(json.dumps(str(fp_a)))
    key = key_for(a)
    key_sizes.append(len(key))
    try:
        ledger._checked_key(key)
    except (TypeError, ValueError):
        key_bad += 1
    per_second[a["id"].split(".")[0]] += 1
print(f"journal serializer: identical across parses {same_serial}; refused {dict(refused)}")
print(f"ledger fingerprint: identical across parses {same_fp}; distinct {len(fingerprints)}")
print(f"tzinfo types on event_time: {dict(tz_types)}")
sizes.sort()
print(f"investigation_payload bytes: median {sizes[len(sizes)//2]}, p99 {sizes[int(len(sizes)*.99)]}, max {sizes[-1]}, total {sum(sizes)}")
key_sizes.sort()
print(f"proposed key characters: min {key_sizes[0]}, max {key_sizes[-1]}; refused by the ledger's key check: {key_bad}")
span = max(int(k) for k in per_second) - min(int(k) for k in per_second)
print(f"W07U requests: busiest second {max(per_second.values())}; seconds with 100 or more {sum(1 for v in per_second.values() if v >= 100)}; rate over the span {len(pairs)/max(span,1)*86400:.0f} per day")

alerts = parse()
print("--- other facts")
print("rule.firedtimes present:", sum(1 for a in alerts if "firedtimes" in (a.get("rule") or {})), "of", len(alerts))
print("native technique id spelled 'None':", sum(1 for a in alerts if "None" in ((a.get("rule") or {}).get("mitre") or {}).get("id", [])))
print("timestamps the strict policy refuses:", end=" ")
bad = 0
for a in alerts:
    try:
        et.resolve_event_time(a, et.POLICY_STRICT)
    except et.EventTimeError:
        bad += 1
print(bad)
print("manager names needing percent-encoding under the proposed rule:", sum(1 for a in alerts if urllib.parse.quote(a["manager"]["name"], safe="-._~") != a["manager"]["name"]))
print("ids matching the proposed pattern:", sum(1 for a in alerts if re.fullmatch(r"[0-9]+\.[0-9]+", a["id"])))

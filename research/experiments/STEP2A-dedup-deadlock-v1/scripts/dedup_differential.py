"""Before/after dedup comparison. No Neo4j, no replay of the pipeline, no writes.
Feeds identical inputs to the pre-fix and fixed DeduplicationEngine, with the DB
fallback stubbed to "nothing recent" (the only DB answer the pre-fix engine can
survive), and compares every decision and the final cache. Then checks the real
archive's decisions against Step 2 run A's recorded per-alert paths.
usage: dedup_differential.py <backend> <prefix_module.py> <step2_view.json> <alerts.json>"""
import importlib.util, json, os, random, sys
from datetime import datetime, timedelta, timezone

backend, prefix_path, view_path, alerts_path = sys.argv[1:5]
os.environ["NEO4J_URI"] = "bolt://127.0.0.1:9"
sys.path.insert(0, backend)
import dedup_engine as fixed  # noqa
from event_time import parse_timestamp, raw_timestamp  # noqa
spec = importlib.util.spec_from_file_location("dedup_prefix", prefix_path)
prefix = importlib.util.module_from_spec(spec); spec.loader.exec_module(prefix)
for m in (fixed, prefix):
    m.find_recent_duplicate = lambda fingerprint, reference_time=None: None


def drive(module, ops):
    e = module.DeduplicationEngine(); out = []
    for args, t in ops:
        dup, fp = e.is_duplicate(*args, event_time=t)
        out.append((dup, fp))
        if not dup:
            e.register_event(fp, campaign_id="C" + fp[:6], event_id="E" + fp[:6], event_time=t)
    return out, e.cache


# 1. randomized sequences: window edges, late alerts, ties, cleanup
rng = random.Random(20261104)
T0 = datetime(2026, 3, 11, tzinfo=timezone.utc)
total = mismatches = 0
for trial in range(200):
    t = T0; ops = []
    for _ in range(400):
        t += timedelta(seconds=rng.choice([0, 0.5, 1, 30, 89, 90, 90.5, 91, 200, 1000]))
        when = t - timedelta(seconds=rng.choice([0, 0, 0, 5, 60, 500]))     # some late alerts
        ops.append(((rng.choice("ab"), "v", rng.choice(["T1", "T2", None]), str(rng.randint(1, 4)), "001"), when))
    a, ca = drive(prefix, ops); b, cb = drive(fixed, ops)
    total += len(ops); mismatches += (a != b) + (ca != cb)
print(f"randomized: {total} decisions over 200 sequences, sequences differing: {mismatches}")

# 2. real archive, same order as the harness
sys.path.insert(0, backend)
from replay.harness import load_alerts  # noqa
loaded = load_alerts(alerts_path)


def ioc(alert):  # identical to realtime_socgraph.extract_iocs
    data = alert.get("data", {}); agent = alert.get("agent", {})
    attacker = data.get("src_ip") or data.get("srcip")
    victim = data.get("dest_ip") or data.get("dstip")
    if not victim:
        victim = data.get("dest_ip") or data.get("dstip") or agent.get("name") or agent.get("id") or data.get("hostname") or alert.get("hostname") or alert.get("location")
    return attacker or victim, victim


ops = []
for _, event_time, alert in loaded.ordered:
    rule = alert["rule"]; ids = (rule.get("mitre") or {}).get("id") or []
    a, v = ioc(alert)
    ops.append(((a, v, ids[0] if ids else None, rule.get("id"), alert.get("agent", {}).get("id", "UNKNOWN")), event_time))
pa, _ = drive(prefix, ops); fa, _ = drive(fixed, ops)
view = json.load(open(view_path))["assignments"]
recorded = [x["path"].startswith("DUPLICATE") for x in view]
print(f"archive: {len(ops)} alerts | prefix==fixed: {pa == fa} | duplicates fixed: {sum(d for d, _ in fa)} | "
      f"alerts where fixed differs from Step 2 run A record: {sum(1 for (d, _), r in zip(fa, recorded) if d != r)}")

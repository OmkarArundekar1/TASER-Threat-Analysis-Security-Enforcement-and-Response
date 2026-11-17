"""Feed the archive through the Step-1 (pre-event-time) code in the SAME
event-time order the harness uses, recording each alert's campaign.
usage: feed_before.py <step1_backend> <current_backend> <alerts> <out_dir> [speed]"""
import json, os, sys, time

step1, current, alerts_path, out_dir = sys.argv[1:5]
speed = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "none" else None
assert "1769" in os.environ["NEO4J_URI"], "disposable instance only"
os.environ["GNN_ENABLED"] = "false"
os.environ["MISP_URL"] = "https://127.0.0.1:9"
sys.path.insert(0, step1)          # Step-1 pipeline modules
sys.path.append(current)           # only event_time.py + replay/ resolve from here

from replay.harness import load_alerts, snapshot_graph, comparable_view, view_hash, _require_fresh_database  # noqa
import realtime_socgraph as rsg  # noqa  (Step-1 version)
from neo4j_client import driver  # noqa
assert not hasattr(rsg, "event_clock"), "expected the pre-Step-2 pipeline"

_require_fresh_database(driver, False)
rsg.sync.publish_campaign = lambda incident: {"success": False, "action": "suppressed"}

class _NoNet:
    def post(self, *a, **k): raise RuntimeError("blocked")
rsg.requests = _NoNet()

current_trace = {}
_orig_cae, _orig_cuae, _orig_dup = rsg.create_attack_event, rsg.create_unattributed_attack_event, rsg.dedup_engine.is_duplicate
def cae(**kw):
    current_trace.update(path="RESOLVED", campaign_id=kw["campaign_id"]); return _orig_cae(**kw)
def cuae(**kw):
    current_trace.update(path="UNATTRIBUTED", campaign_id=kw["campaign_id"]); return _orig_cuae(**kw)
def dup(**kw):
    d, fp = _orig_dup(**kw)
    if d:
        current_trace.update(path="DUPLICATE", campaign_id=(rsg.dedup_engine.get_metadata(fp) or {}).get("campaign_id"))
    return d, fp
rsg.create_attack_event, rsg.create_unattributed_attack_event, rsg.dedup_engine.is_duplicate = cae, cuae, dup

loaded = load_alerts(alerts_path)
traces, prev = [], None
started = time.monotonic()
for index, (position, event_time, alert) in enumerate(loaded.ordered):
    if speed and prev is not None:
        gap = (event_time - prev).total_seconds()
        if gap > 0:
            time.sleep(gap / speed)
    prev = event_time
    current_trace = {"index": index, "position": position, "event_time": event_time.isoformat()}
    ok = rsg.process_alert(alert)
    current_trace["ok"] = bool(ok)
    traces.append(current_trace)
wall = round(time.monotonic() - started, 3)

snap = snapshot_graph(driver)
view = comparable_view(traces, [], snap)
os.makedirs(out_dir, exist_ok=True)
manifest = {"code": "step1 (pre-event-time)", "speed": speed, "wall_seconds": wall,
            "input_sha256": loaded.sha256, "accepted": len(loaded.ordered), "rejected": len(loaded.rejected),
            "view_sha256": view_hash(view),
            "counts": {"campaigns": len(view["campaigns"]), "operations": len(view["operations"]),
                       "transitions": len(view["transitions"])}}
json.dump(manifest, open(f"{out_dir}/manifest.json", "w"), indent=1)
json.dump(view, open(f"{out_dir}/view.json", "w"), indent=1, sort_keys=True, default=str)
json.dump(snap, open(f"{out_dir}/graph_snapshot.json", "w"), indent=1, default=str)
with open(f"{out_dir}/traces.jsonl", "w") as f:
    for t in traces: f.write(json.dumps(t) + "\n")
print(json.dumps(manifest, indent=1))

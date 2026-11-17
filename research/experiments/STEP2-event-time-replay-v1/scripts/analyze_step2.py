import collections, json, os, sys
from datetime import datetime, timedelta

R = sys.argv[1]
def load(n, f): return json.load(open(os.path.join(R, n, f)))
def jl(n, f): return [json.loads(l) for l in open(os.path.join(R, n, f))]
dt = lambda s: datetime.fromisoformat(s) if s else None
out = {}

# ---- 4. ordering / ties / late (event-time runs)
tr = jl("A_after_unpaced", "traces.jsonl")
times = [t["event_time"] for t in tr if t.get("event_time")]
tie_groups = collections.Counter(times)
out["ordering"] = {
    "alerts": len(tr), "with_event_time": len(times),
    "monotone_non_decreasing": all(dt(a) <= dt(b) for a, b in zip(times, times[1:])),
    "late_flagged": sum(1 for t in tr if t.get("late")),
    "fallback_event_time": sum(1 for t in tr if t.get("event_time_source") != "ALERT_TIMESTAMP"),
    "distinct_timestamps": len(tie_groups),
    "alerts_sharing_a_timestamp": sum(c for c in tie_groups.values() if c > 1),
    "largest_tie_group": max(tie_groups.values()),
    "tie_order_is_file_order": all(
        a["position"] < b["position"] for a, b in zip(tr, tr[1:]) if a.get("event_time") == b.get("event_time")),
    "rejected": len(load("A_after_unpaced", "rejected.json")),
    "first_event": times[0], "last_event": times[-1],
}

# ---- 5. lifecycle consistency (A vs E)
def lifecycle(n, lo, hi):
    snap = load(n, "graph_snapshot.json")
    decisions = jl(n, "decisions.jsonl") if os.path.exists(os.path.join(R, n, "decisions.jsonl")) else []
    expired_ids = {d["campaign_id"] for d in decisions if d["decision"] in ("CLOSE_EXPIRED", "MAINTENANCE_EXPIRE")}
    score_closed = {d["campaign_id"]: d["event_time"] for d in decisions if d["decision"] == "CLOSE_BELOW_THRESHOLD"}
    rows = snap["campaigns"]
    res = {"campaigns": len(rows),
           "first_le_last": sum(1 for c in rows if dt(c["first_seen"]) <= dt(c["last_seen"])),
           "within_archive_window": sum(1 for c in rows if lo <= dt(c["first_seen"]) and dt(c["last_seen"]) <= hi),
           "status": dict(collections.Counter(c["status"] for c in rows)),
           "closed": sum(1 for c in rows if c["closed_at"])}
    if decisions:
        ok_exp = [c for c in rows if c["campaign_id"] in expired_ids]
        res["expired_closed_at_eq_last_seen_plus_120"] = sum(
            1 for c in ok_exp if dt(c["closed_at"]) == dt(c["last_seen"]) + timedelta(seconds=120))
        res["expired_total"] = len(ok_exp)
        # in-memory last_seen can be older than Neo4j last_seen (UNKNOWN/duplicate refresh, audit B3)
        res["expired_closed_at_lt_neo4j_last_seen_plus_120"] = sum(
            1 for c in ok_exp if dt(c["closed_at"]) < dt(c["last_seen"]) + timedelta(seconds=120))
        res["closed_below_threshold"] = len(score_closed)
        res["decision_counts"] = dict(collections.Counter(d["decision"] for d in decisions))
    return res
lo, hi = dt(times[0]), dt(times[-1])
out["lifecycle"] = {"A_after_unpaced": lifecycle("A_after_unpaced", lo, hi),
                    "E_before_unpaced": lifecycle("E_before_unpaced", lo, hi)}
eb = load("E_before_unpaced", "graph_snapshot.json")["campaigns"]
out["lifecycle"]["E_before_example_first_seen"] = sorted(c["first_seen"] for c in eb)[:2]

# ---- 6. operations
for n in ("A_after_unpaced", "D_after_x200_maint05", "E_before_unpaced", "F_before_x200"):
    v = load(n, "view.json")
    out.setdefault("operations", {})[n] = {
        "operations": len(v["operations"]),
        "status": dict(collections.Counter(o["status"] for o in v["operations"])),
        "closed_at_values": sorted({o["closed_at"] for o in v["operations"] if o["closed_at"]})[:3],
        "campaigns_without_operation": len({c["campaign"] for c in v["campaigns"]} -
                                           {x for o in v["operations"] for x in o["campaigns"]}),
    }
out["operations"]["A_vs_D_identical"] = load("A_after_unpaced", "view.json")["operations"] == load("D_after_x200_maint05", "view.json")["operations"]

# ---- 7. dedup
for n in ("A_after_unpaced", "C_after_x2000_maint005", "D_after_x200_maint05", "E_before_unpaced", "F_before_x200"):
    v = load(n, "view.json")
    p = collections.Counter(a["path"] for a in v["assignments"])
    out.setdefault("dedup", {})[n] = {"duplicates": p["DUPLICATE"] + p["DUPLICATE_UNATTRIBUTED"],
                                      "new_events": p["RESOLVED"] + p["UNATTRIBUTED"]}
ea, ee, ef = (load(n, "view.json")["assignments"] for n in ("A_after_unpaced", "E_before_unpaced", "F_before_x200"))
out["dedup"]["E_vs_F_alerts_with_different_dup_status"] = sum(
    1 for x, y in zip(ee, ef) if (x["path"] == "DUPLICATE") != (y["path"] == "DUPLICATE"))

# ---- 9. concrete before/after examples on the 192.168.56.106 -> pes1ug23cs411-VirtualBox pair
def pair_campaigns(n):
    v = load(n, "view.json")
    return [(c["campaign"], c["first_seen"], c["last_seen"], c["sequence"][:8], len(c["sequence"]))
            for c in v["campaigns"] if c["attacker_ip"] == "192.168.56.106" and c["victim_ip"] == "pes1ug23cs411-VirtualBox"]
out["example_pair"] = {n: pair_campaigns(n) for n in ("A_after_unpaced", "E_before_unpaced", "F_before_x200")}

# gaps between consecutive non-duplicate events for that pair in event time (why after splits)
gaps = []
prev = None
for a in ea:
    if a["path"] in ("RESOLVED", "UNATTRIBUTED") and a["campaign"]:
        pass
snapA = load("A_after_unpaced", "graph_snapshot.json")
out["after_campaign_span_seconds"] = sorted(
    (dt(c["last_seen"]) - dt(c["first_seen"])).total_seconds() for c in snapA["campaigns"])[-5:]
json.dump(out, open(os.path.join(R, "analysis.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))

"""Read-only comparison of the six B3 v2 runs. Anomalies are reported as found;
nothing is corrected, normalised or dropped. No ground truth exists for this
archive, so nothing here measures detection accuracy.

usage (cwd = b3_runs_v2): python comparison/compare_six.py
Writes comparison/summary.json, comparison/runs.csv, comparison/campaign_anomalies.csv.
"""
import collections
import csv
import hashlib
import json
import math
import os
import re
import statistics
from datetime import datetime as D, timedelta as TD

RUNS = ["CUR_norestart", "CUR_restart_r2", "A_norestart", "A_restart", "B_norestart", "B_restart"]
RESTART_PAIRS = [("CUR_norestart", "CUR_restart_r2"), ("A_norestart", "A_restart"), ("B_norestart", "B_restart")]
VARIANT_PAIRS = [("CUR_norestart", "A_norestart"), ("CUR_norestart", "B_norestart"), ("A_norestart", "B_norestart"),
                 ("CUR_restart_r2", "A_restart"), ("CUR_restart_r2", "B_restart"), ("A_restart", "B_restart")]
TIMEOUT = TD(seconds=120)
RESTART_INDEX = 4500
dt = lambda s: D.fromisoformat(s) if s else None  # noqa: E731


def load(name):
    d = f"runs/{name}"
    run = {
        "name": name,
        "status": open(f"{d}/status").read().strip(), "exit": open(f"{d}/exit_status").read().strip(),
        "status_json": json.load(open(f"{d}/status.json")), "manifest": json.load(open(f"{d}/manifest.json")),
        "view": json.load(open(f"{d}/view.json")), "snapshot": json.load(open(f"{d}/graph_snapshot.json")),
        "traces": [json.loads(l) for l in open(f"{d}/traces.jsonl")],
        "decisions": [json.loads(l) for l in open(f"{d}/decisions.jsonl")],
        "resources": open(f"{d}/neo4j_resources.txt").read(), "diagnostics": open(f"{d}/diagnostics.log").read(),
    }
    # Stable cross-run identity for a campaign: file position of its first alert.
    first = {}
    for a in run["view"]["assignments"]:
        if a["campaign"] and a["campaign"] not in first:
            first[a["campaign"]] = a["position"]
    run["first_alert"] = first
    run["assign"] = {a["position"]: a["campaign"] for a in run["view"]["assignments"]}
    run["campaign_by_key"] = {first[c["campaign"]]: c for c in run["view"]["campaigns"] if c["campaign"] in first}
    return run


def partition(assign, lo=None, hi=None):
    groups = collections.defaultdict(set)
    for position, label in assign.items():
        if label and (lo is None or lo <= position) and (hi is None or position < hi):
            groups[label].add(position)
    return {frozenset(g) for g in groups.values()}


def adjusted_rand(a, b):
    keys = [k for k in a if a[k] and b.get(k)]
    n = len(keys)
    c2 = lambda x: x * (x - 1) / 2  # noqa: E731
    joint = collections.Counter((a[k], b[k]) for k in keys)
    ca, cb = collections.Counter(a[k] for k in keys), collections.Counter(b[k] for k in keys)
    index = sum(c2(v) for v in joint.values())
    sa, sb = sum(c2(v) for v in ca.values()), sum(c2(v) for v in cb.values())
    expected = sa * sb / c2(n) if n > 1 else 0
    maximum = (sa + sb) / 2
    return 1.0 if maximum == expected else (index - expected) / (maximum - expected)


def memory(text):
    get = lambda p: (re.search(p, text) or [None, None])[1]  # noqa: E731
    to_mb = lambda v: round(int(v) / 1e6, 1) if v else None  # noqa: E731
    return {"peak_mb": to_mb(get(r"memory_peak=(\d+)")), "anon_mb": to_mb(get(r"anon (\d+)")),
            "file_mb": to_mb(get(r"file (\d+)")), "cap_mb": to_mb(get(r"cgroup_memory_max=(\d+)"))}


def run_metrics(r):
    v, s, m = r["view"], r["snapshot"], r["manifest"]
    campaigns = v["campaigns"]
    spans = [(dt(c["last_seen"]) - dt(c["first_seen"])).total_seconds() for c in campaigns]
    in_ops = {x for o in v["operations"] for x in o["campaigns"]}
    op_counts = collections.Counter(x for o in v["operations"] for x in o["campaigns"])
    paths = collections.Counter(a["path"] for a in v["assignments"])
    scores = [d["score"] for d in r["decisions"] if "score" in d]
    expired_ids = {d["campaign_id"] for d in r["decisions"] if d["decision"] in ("CLOSE_EXPIRED", "MAINTENANCE_EXPIRE")}
    raw = {c["campaign_id"]: c for c in s["campaigns"]}
    expired = [raw[i] for i in expired_ids if i in raw]
    st = r["status_json"]
    return {
        "status": r["status"], "exit_code": int(r["exit"]), "stage": st["stage_at_end"], "watchdog_fired": st["watchdog_fired"],
        "tracebacks": st["checks"]["tracebacks"], "process_alert_errors": st["checks"]["process_alert_errors"],
        "artifact_problems": st["artifact_problems"], "view_sha256": m["view_sha256"],
        "view_hash_recomputed_ok": hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest() == m["view_sha256"],
        "alerts": len(v["assignments"]), "alerts_ok": st["checks"]["alerts_ok"], "late": sum(1 for t in r["traces"] if t.get("late")),
        "paths": dict(sorted(paths.items())), "duplicates": paths["DUPLICATE"] + paths["DUPLICATE_UNATTRIBUTED"],
        "campaigns": len(campaigns), "campaign_status": dict(collections.Counter(c["status"] for c in campaigns)),
        "unknown_only": sum(all(x == "UNKNOWN" for x in c["sequence"]) for c in campaigns),
        "singletons": sum(len(c["sequence"]) == 1 for c in campaigns),
        "self_loops": sum(c["attacker_ip"] == c["victim_ip"] for c in campaigns),
        "per_pair": dict(collections.Counter(f'{c["attacker_ip"]}->{c["victim_ip"]}' for c in campaigns).most_common()),
        "span_seconds": {"median": statistics.median(spans), "p90": sorted(spans)[int(len(spans) * 0.9)], "max": max(spans)},
        "reopened_total": sum(c["reopened_count"] or 0 for c in campaigns),
        "operations": len(v["operations"]), "operation_status": dict(collections.Counter(o["status"] for o in v["operations"])),
        "campaigns_without_operation": sum(c["campaign"] not in in_ops for c in campaigns),
        "campaigns_in_multiple_operations": sum(n > 1 for n in op_counts.values()),
        "transitions_distinct": len(v["transitions"]), "transitions_total": sum(t["count"] for t in v["transitions"]),
        "transitions": {f'{t["src"]}->{t["dst"]}': t["count"] for t in v["transitions"]},
        "decisions": dict(sorted(collections.Counter(d["decision"] for d in r["decisions"]).items())),
        "scores": {"n": len(scores), "min": min(scores), "median": statistics.median(scores), "max": max(scores),
                   "below_0.35": sum(x < 0.35 for x in scores), "at_or_above_0.70": sum(x >= 0.70 for x in scores)},
        "closed_before_neo4j_last_seen_plus_timeout": sum(1 for c in expired if dt(c["closed_at"]) and dt(c["closed_at"]) < dt(c["last_seen"]) + TIMEOUT),
        "expired_total": len(expired),
        "misp_would_publish": m["misp_would_publish"], "misp_suppressed": m["misp_publications_suppressed"], "http_blocked": m["outbound_http_blocked"],
        "harness_wall_seconds": m["wall_seconds"], "memory": memory(r["resources"]),
        "restart_markers": st["checks"].get("restart_markers"),
        "final_flush_records": (re.search(r"FINAL_FLUSH_DONE records=(\d+)", r["diagnostics"]) or [None, None])[1],
        "restart_flush_records": (re.search(r"RESTART_FLUSH_DONE records=(\d+)", r["diagnostics"]) or [None, None])[1],
        "container_status": st["container_status"], "cleanup_status": st["cleanup_status"],
    }


def compare(a, b):
    va, vb = a["view"], b["view"]
    paths = sum(x["path"] != y["path"] for x, y in zip(va["assignments"], vb["assignments"]))
    pa, pb = partition(a["assign"]), partition(b["assign"])
    ca, cb = a["campaign_by_key"], b["campaign_by_key"]
    shared = set(ca) & set(cb)
    fields = collections.Counter()
    timestamp_examples = []
    for key in sorted(shared):
        for f in ("last_seen", "closed_at", "status", "reopened_count", "sequence", "first_seen"):
            if ca[key].get(f) != cb[key].get(f):
                fields[f] += 1
                if f in ("last_seen", "closed_at") and len(timestamp_examples) < 12:
                    timestamp_examples.append({"first_alert": key, "pair": f'{ca[key]["attacker_ip"]}->{ca[key]["victim_ip"]}',
                                               "field": f, a["name"]: ca[key].get(f), b["name"]: cb[key].get(f)})
    transitions_a = {(t["src"], t["dst"]): t["count"] for t in va["transitions"]}
    transitions_b = {(t["src"], t["dst"]): t["count"] for t in vb["transitions"]}
    # lifecycle: compare alert-driven decisions as ordered tuples without campaign labels
    la = [(d["decision"], d.get("technique"), d["event_time"], d.get("score")) for d in va["lifecycle_decisions"]]
    lb = [(d["decision"], d.get("technique"), d["event_time"], d.get("score")) for d in vb["lifecycle_decisions"]]
    sa, sb = collections.Counter(la), collections.Counter(lb)
    na = collections.Counter(x[:3] for x in la)
    nb = collections.Counter(x[:3] for x in lb)
    score_only = sum(((na & nb) - (sa & sb).__class__(x[:3] for x in (sa & sb).elements())).values())
    return {
        "hash_equal": a["manifest"]["view_sha256"] == b["manifest"]["view_sha256"],
        "sections_equal": {k: va[k] == vb[k] for k in va},
        "alerts_with_different_path": paths,
        "alerts_with_different_label": sum(a["assign"].get(k) != b["assign"].get(k) for k in a["assign"]),
        "campaigns": [len(pa), len(pb)], "identical_campaign_memberships": len(pa & pb),
        "adjusted_rand_index_between_runs": round(adjusted_rand(a["assign"], b["assign"]), 6),
        "campaigns_sharing_first_alert": len(shared), "campaign_field_differences_on_shared": dict(fields),
        "timestamp_examples": timestamp_examples,
        "transitions_only_a": {f"{s}->{d}": n for (s, d), n in transitions_a.items() if (s, d) not in transitions_b},
        "transitions_only_b": {f"{s}->{d}": n for (s, d), n in transitions_b.items() if (s, d) not in transitions_a},
        "transitions_count_changed": {f"{k[0]}->{k[1]}": [transitions_a[k], transitions_b[k]] for k in transitions_a
                                      if k in transitions_b and transitions_a[k] != transitions_b[k]},
        "lifecycle_lengths": [len(la), len(lb)],
        "lifecycle_only_a": sum((sa - sb).values()), "lifecycle_only_b": sum((sb - sa).values()),
        "lifecycle_same_decision_different_score": score_only,
    }


def anomalies(r):
    """closed_at < last_seen, as stored. Reported, never corrected."""
    out = []
    for key, c in sorted(r["campaign_by_key"].items()):
        if c["closed_at"] and dt(c["closed_at"]) < dt(c["last_seen"]):
            out.append({"run": r["name"], "campaign": c["campaign"], "first_alert_position": key,
                        "attacker": c["attacker_ip"], "victim": c["victim_ip"], "first_seen": c["first_seen"],
                        "last_seen": c["last_seen"], "closed_at": c["closed_at"],
                        "gap_seconds": (dt(c["last_seen"]) - dt(c["closed_at"])).total_seconds(),
                        "status": c["status"], "events": len(c["sequence"])})
    return out


runs = {name: load(name) for name in RUNS}
metrics = {name: run_metrics(r) for name, r in runs.items()}
anomaly_rows = [row for r in runs.values() for row in anomalies(r)]

# Tracked by first alert: the two known campaigns (v1/CUR_norestart labels C0002, C0045).
base = runs["CUR_norestart"]
known = {label: pos for label, pos in base["first_alert"].items() if label in ("C0002", "C0045", "C0034", "C0028")}
tracked = {}
for label, pos in known.items():
    tracked[label] = {"first_alert_position": pos}
    for name, r in runs.items():
        owner = r["assign"].get(pos)
        c = next((x for x in r["view"]["campaigns"] if x["campaign"] == owner), None)
        tracked[label][name] = None if c is None else {
            "label": owner, "starts_here": r["first_alert"].get(owner) == pos, "first_seen": c["first_seen"],
            "last_seen": c["last_seen"], "closed_at": c["closed_at"], "status": c["status"], "events": len(c["sequence"]),
            "closed_before_last_seen": bool(c["closed_at"] and dt(c["closed_at"]) < dt(c["last_seen"])),
            "reopen_scores": [d.get("score") for d in r["view"]["lifecycle_decisions"] if d["campaign_id"] == owner and d["decision"] == "REOPEN"],
        }

summary = {
    "note": "No ground truth for this archive: these are behavioural differences between runs, not detection accuracy.",
    "runs": metrics,
    "restart_effects": {f"{a} vs {b}": compare(runs[a], runs[b]) for a, b in RESTART_PAIRS},
    "variant_effects": {f"{a} vs {b}": compare(runs[a], runs[b]) for a, b in VARIANT_PAIRS},
    "closed_at_before_last_seen": anomaly_rows, "tracked_campaigns": tracked,
}
os.makedirs("comparison", exist_ok=True)
json.dump(summary, open("comparison/summary.json", "w"), indent=1, default=str)

columns = ["status", "exit_code", "alerts", "alerts_ok", "duplicates", "campaigns", "unknown_only", "singletons", "self_loops",
           "reopened_total", "operations", "campaigns_without_operation", "campaigns_in_multiple_operations", "transitions_distinct",
           "transitions_total", "closed_before_neo4j_last_seen_plus_timeout", "expired_total", "misp_would_publish",
           "harness_wall_seconds", "view_sha256"]
with open("comparison/runs.csv", "w", newline="") as handle:
    w = csv.writer(handle)
    w.writerow(["run"] + columns + ["closed_at_before_last_seen", "memory_peak_mb", "memory_anon_mb", "memory_file_mb", "span_median_s", "span_max_s"])
    for name, m in metrics.items():
        w.writerow([name] + [m[c] for c in columns] + [sum(a["run"] == name for a in anomaly_rows), m["memory"]["peak_mb"],
                    m["memory"]["anon_mb"], m["memory"]["file_mb"], m["span_seconds"]["median"], m["span_seconds"]["max"]])
with open("comparison/campaign_anomalies.csv", "w", newline="") as handle:
    fields = ["run", "campaign", "first_alert_position", "attacker", "victim", "first_seen", "last_seen", "closed_at", "gap_seconds", "status", "events"]
    w = csv.DictWriter(handle, fieldnames=fields)
    w.writeheader()
    w.writerows(anomaly_rows)
print("written")

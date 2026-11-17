"""Summarise and compare the Step 2 runs. usage: compare_step2.py <runs_dir> <out_json>"""
import collections, json, os, sys

runs_dir, out_path = sys.argv[1:3]
sys.path.insert(0, os.path.expanduser("~/capstone/isfcr/CYUKTI/backend"))
from replay.harness import compare_runs, view_hash  # noqa

names = sorted(d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d)))
views, manifests = {}, {}
for n in names:
    views[n] = json.load(open(os.path.join(runs_dir, n, "view.json")))
    manifests[n] = json.load(open(os.path.join(runs_dir, n, "manifest.json")))

def summary(n):
    v = views[n]
    per_pair = collections.Counter((c["attacker_ip"], c["victim_ip"]) for c in v["campaigns"])
    paths = collections.Counter(a["path"] for a in v["assignments"])
    sizes = collections.Counter(a["campaign"] for a in v["assignments"] if a["campaign"])
    return {
        "view_sha256": view_hash(v), "wall_seconds": manifests[n].get("wall_seconds"),
        "speed": manifests[n].get("speed"), "maintenance_interval": manifests[n].get("maintenance_interval"),
        "campaigns": len(v["campaigns"]), "operations": len(v["operations"]),
        "transitions": len(v["transitions"]), "paths": dict(paths),
        "alerts_reaching_campaign_logic": paths.get("RESOLVED", 0) + paths.get("UNATTRIBUTED", 0),
        "campaigns_per_pair": {f"{a}->{b}": k for (a, b), k in per_pair.most_common()},
        "singleton_campaigns": sum(1 for c in v["campaigns"] if len(c["sequence"]) == 1),
        "largest_campaign_alerts": max(sizes.values()) if sizes else 0,
    }

result = {"runs": {n: summary(n) for n in names}, "pairwise": {}}
after = [n for n in names if "_after_" in n]
before = [n for n in names if "_before_" in n]
for group in (after, before):
    for a in group:
        for b in group:
            if a < b:
                result["pairwise"][f"{a} vs {b}"] = compare_runs(os.path.join(runs_dir, a), os.path.join(runs_dir, b))

# before vs after: assignment-level agreement (same alerts, same order)
def partition(v):
    groups = collections.defaultdict(set)
    for a in v["assignments"]:
        if a["campaign"]:
            groups[a["campaign"]].add(a["position"])
    return {frozenset(s) for s in groups.values()}

if after and before:
    a0, b0 = after[0], before[0]
    pa, pb = partition(views[a0]), partition(views[b0])
    same_path = sum(1 for x, y in zip(views[a0]["assignments"], views[b0]["assignments"]) if x["path"] == y["path"])
    result["before_vs_after"] = {
        "after_run": a0, "before_run": b0,
        "identical_campaign_partitions": len(pa & pb),
        "campaigns_after": len(pa), "campaigns_before": len(pb),
        "alerts_with_same_path": same_path, "alerts": len(views[a0]["assignments"]),
    }
json.dump(result, open(out_path, "w"), indent=1, default=str)
print(json.dumps({"runs": result["runs"],
                  "pairwise": {k: {"identical": v["identical"],
                                   "differing_sections": [s for s, e in v["sections"].items() if not e["identical"]]}
                               for k, v in result["pairwise"].items()},
                  "before_vs_after": result.get("before_vs_after")}, indent=1, default=str))

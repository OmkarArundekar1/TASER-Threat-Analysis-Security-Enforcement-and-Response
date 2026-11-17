"""Step 1 live verification: feed alerts through the REAL process_alert()
into a throwaway Neo4j, then read back NEXT_TECHNIQUE edges and compare
them with the transitions implied by each campaign's own stored event
sequence.

usage: python verify_chain_live.py <backend_dir> <mode: synthetic|archive> <archive_path> <out_json>
Requires NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD pointing at a DISPOSABLE
instance, MISP_URL pointing at a closed port.
"""
import collections
import json
import os
import sys

backend_dir, mode, archive_path, out_path = sys.argv[1:5]
assert "17687" in os.environ["NEO4J_URI"], "refusing to run against anything but the disposable instance"
sys.path.insert(0, backend_dir)

import realtime_socgraph as rsg  # noqa: E402
from neo4j_client import driver  # noqa: E402


def synthetic():
    def a(t, rid):
        return {"rule": {"id": rid, "level": 10, "description": f"synthetic {t}", "groups": [],
                         "mitre": {"id": [t], "technique": [t], "tactic": []}},
                "agent": {"id": "001", "name": "victim"},
                "data": {"src_ip": "10.99.0.5", "dest_ip": "10.99.0.9"},
                "timestamp": "2026-11-03T12:00:00.000+0000"}
    return [a("T1595", "900001"), a("T1110", "900002"), a("T1078", "900003")]


def archive():
    """Native-mapped alerts from the real 2026-03-11 archive for the
    attacker->victim pair with the most distinct techniques, in file order."""
    by_pair = collections.defaultdict(list)
    for line in open(archive_path, encoding="utf-8", errors="replace"):
        al = json.loads(line)
        ids = ((al.get("rule") or {}).get("mitre") or {}).get("id") or []
        if not ids:
            continue
        pair = rsg.extract_iocs(al)
        by_pair[pair].append(al)
    pair, alerts = max(by_pair.items(), key=lambda kv: len({x["rule"]["mitre"]["id"][0] for x in kv[1]}))
    return alerts


alerts = synthetic() if mode == "synthetic" else archive()
results = [rsg.process_alert(al) for al in alerts]

with driver.session() as s:
    edges = s.run("MATCH (a:Technique)-[r:NEXT_TECHNIQUE]->(b:Technique) "
                  "RETURN a.attack_id AS src, b.attack_id AS dst, r.count AS count ORDER BY src, dst").data()
    campaigns = s.run(
        "MATCH (c:Campaign)-[:HAS_EVENT]->(e:AttackEvent) WHERE e.attack_id IS NOT NULL "
        "WITH c, e ORDER BY e.first_seen "
        "RETURN c.campaign_id AS cid, c.reopened_count AS reopened, collect(e.attack_id) AS seq").data()

# Expected under the (unchanged) learning policy: within each campaign, every
# consecutive pair of resolved events with different techniques, excluding the
# segment-start event. Valid only for campaigns never reopened (a reopen starts
# a new segment mid-sequence); reopened campaigns are reported separately.
expected = collections.Counter()
reopened = []
for c in campaigns:
    if c["reopened"]:
        reopened.append(c["cid"])
        continue
    seq = c["seq"]
    for prev, cur in zip(seq, seq[1:]):
        if prev != cur:
            expected[(prev, cur)] += 1

actual = collections.Counter({(e["src"], e["dst"]): e["count"] for e in edges})
out = {
    "mode": mode, "backend_dir": backend_dir, "alerts_fed": len(alerts),
    "process_alert_true": sum(1 for r in results if r),
    "campaigns": len(campaigns), "reopened_campaigns": reopened,
    "edges_actual": {f"{k[0]}->{k[1]}": v for k, v in sorted(actual.items())},
    "edges_expected_from_event_sequences": {f"{k[0]}->{k[1]}": v for k, v in sorted(expected.items())},
    "match": (actual == expected) if not reopened else "n/a (reopened campaigns present)",
    "campaign_sequences": [{"cid": c["cid"], "reopened": c["reopened"], "seq": c["seq"]} for c in campaigns],
}
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "campaign_sequences"}, indent=1))

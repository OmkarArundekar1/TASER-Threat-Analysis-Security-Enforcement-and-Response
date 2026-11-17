"""
READ-ONLY validation of research/BASELINE_AUDIT.md findings against the live
CYUKTI Neo4j graph (Decision 5).

Every query runs inside session.execute_read(); Neo4j rejects any write
attempted in a read transaction, so this script cannot modify the graph even
by mistake. It contains no CREATE/MERGE/SET/DELETE/REMOVE clause (asserted
below before anything is sent to the server).

usage (from isfcr/CYUKTI/backend, with the normal .env):
    python ../../../research/queries/readonly_live_graph_validation.py <out.json>
"""
import json
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")
from neo4j import READ_ACCESS  # noqa: E402
from neo4j_client import driver  # noqa: E402

QUERIES = {
    # A2c -- provenance of the recorded transitions
    "next_technique_edges": """
        MATCH (a:Technique)-[r:NEXT_TECHNIQUE]->(b:Technique)
        RETURN a.attack_id AS src, b.attack_id AS dst, r.count AS count,
               r.confidence AS confidence, toString(r.created_at) AS created_at,
               toString(r.last_seen) AS last_seen, keys(r) AS properties
        ORDER BY created_at""",
    # For each edge: campaigns whose ordered resolved events contain src
    # immediately followed (ignoring UNKNOWNs) by dst, with timestamps.
    "edge_supporting_campaigns": """
        MATCH (a:Technique)-[:NEXT_TECHNIQUE]->(b:Technique)
        MATCH (c:Campaign)-[:HAS_EVENT]->(e:AttackEvent)
        WHERE e.attack_id IS NOT NULL
        WITH a, b, c, e ORDER BY e.first_seen
        WITH a, b, c, collect({t: e.attack_id, ts: toString(e.first_seen)}) AS ev
        UNWIND range(0, size(ev) - 2) AS i
        WITH a, b, c, ev[i] AS x, ev[i + 1] AS y
        WHERE x.t = a.attack_id AND y.t = b.attack_id
        RETURN a.attack_id AS src, b.attack_id AS dst, c.campaign_id AS campaign,
               x.ts AS src_event_time, y.ts AS dst_event_time
        ORDER BY src_event_time""",
    "event_time_range": """
        MATCH (e:AttackEvent)
        RETURN toString(min(e.first_seen)) AS first, toString(max(e.first_seen)) AS last,
               count(e) AS events""",
    # A6 / C2 -- campaign and operation state
    "campaign_status": """
        MATCH (c:Campaign) RETURN c.status AS status, count(*) AS n ORDER BY n DESC""",
    "operations": """
        MATCH (o:Operation) RETURN count(o) AS operations""",
    "campaigns_without_operation": """
        MATCH (c:Campaign) WHERE NOT EXISTS { MATCH (:Operation)-[:HAS_CAMPAIGN]->(c) }
        OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:AttackEvent)
        WITH c, count(e) AS events, count(e.attack_id) AS resolved_events
        RETURN count(c) AS n,
               sum(CASE WHEN resolved_events = 0 THEN 1 ELSE 0 END) AS of_which_unknown_only,
               sum(CASE WHEN events = 0 THEN 1 ELSE 0 END) AS of_which_no_events""",
    "campaigns_in_multiple_operations": """
        MATCH (o:Operation)-[:HAS_CAMPAIGN]->(c:Campaign)
        WITH c, count(DISTINCT o) AS ops WHERE ops > 1
        RETURN count(c) AS n, max(ops) AS max_ops""",
    # B1 -- labelling starvation as stored
    "event_mitre_status": """
        MATCH (e:AttackEvent)
        RETURN coalesce(e.mitre_status, 'MISSING') AS mitre_status,
               coalesce(e.mitre_provenance, 'MISSING') AS provenance, count(*) AS n
        ORDER BY n DESC""",
    # C3 -- attacker := victim fallback
    "self_loop_campaigns": """
        MATCH (c:Campaign) WHERE c.attacker_ip = c.victim_ip
        RETURN count(c) AS n, collect(DISTINCT c.attacker_ip)[..10] AS identities""",
    "attacker_victim_pairs": """
        MATCH (c:Campaign)
        RETURN c.attacker_ip AS attacker, c.victim_ip AS victim, count(*) AS campaigns
        ORDER BY campaigns DESC LIMIT 25""",
}

_WRITE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|LOAD\s+CSV|CALL\s+apoc\.(create|merge|refactor))\b", re.I)
for name, q in QUERIES.items():
    assert not _WRITE.search(q), f"write clause in {name}"


def _run(tx, q):
    return [dict(r) for r in tx.run(q)]


def main(out_path):
    results = {"generated_at": datetime.now(timezone.utc).isoformat(), "read_only": True, "results": {}}
    with driver.session(default_access_mode=READ_ACCESS) as session:
        for name, q in QUERIES.items():
            try:
                results["results"][name] = session.execute_read(_run, q)
            except Exception as e:  # report, never retry as write
                results["results"][name] = {"error": repr(e)}
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "readonly_live_graph_validation.json")

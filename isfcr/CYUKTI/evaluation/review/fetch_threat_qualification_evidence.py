"""
review/fetch_threat_qualification_evidence.py
=================================================
Independent-adjudication evidence gatherer (review/evaluation_threat_qualification_adjudication.md).
Reads the real campaign IDs directly from the existing
threat_qualification_v0 ground-truth dataset (never a hardcoded
duplicate list, to avoid drift) and pulls, for each, the raw
Campaign/AttackEvent/Operation fields an independent reviewer needs:
attacker/victim, timestamps, technique set with per-event occurrence
(dedup) counts, MITRE provenance/confidence, risk_score, cti_score,
reopened_count, and operation correlation. Read-only -- no write query.

Usage: cd evaluation && python review/fetch_threat_qualification_evidence.py
Output: results/threat_qualification_adjudication_evidence.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from ground_truth import store
from neo4j_client import driver


def fetch():
    records, _ = store.load_best_available("threat_qualification_v0")
    campaign_ids = [r.raw_event_id for r in records]

    with driver.session() as s:
        results = {}
        for cid in campaign_ids:
            row = s.run(
                "MATCH (c:Campaign {campaign_id:$id}) "
                "OPTIONAL MATCH (op:Operation)-[:HAS_CAMPAIGN]->(c) "
                "RETURN c.attacker_ip AS attacker_ip, c.victim_ip AS victim_ip, "
                "c.first_seen AS first_seen, c.last_seen AS last_seen, "
                "c.last_technique AS last_technique, c.risk_score AS risk_score, "
                "c.cti_score AS cti_score, c.status AS status, "
                "c.reopened_count AS reopened_count, op.operation_id AS operation_id LIMIT 1",
                id=cid,
            ).single()

            events = list(s.run(
                "MATCH (c:Campaign {campaign_id:$id})-[:HAS_EVENT]->(e:AttackEvent) "
                "OPTIONAL MATCH (e)-[:MATCHES]->(t:Technique) "
                "RETURN e.attack_id AS attack_id, e.first_seen AS first_seen, e.occurrences AS occurrences, "
                "e.mitre_provenance AS provenance, e.mitre_confidence AS confidence, t.attack_id AS technique "
                "ORDER BY e.first_seen",
                id=cid,
            ))

            results[cid] = {"campaign": dict(row) if row else None, "events": [dict(e) for e in events]}

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "results", "threat_qualification_adjudication_evidence.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote {len(results)} campaign evidence records -> {out_path}")


if __name__ == "__main__":
    fetch()

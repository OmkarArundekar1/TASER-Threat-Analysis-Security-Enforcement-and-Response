import os
import sys
from collections import defaultdict

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from neo4j_client import driver  # noqa: E402

with driver.session() as s:
    print("=== PHASE 1: independent reproduction ===")
    total_events = s.run("MATCH (e:AttackEvent) RETURN count(e) AS n").single()["n"]
    linked_events = s.run(
        "MATCH (:Campaign)-[:HAS_EVENT]->(e:AttackEvent) RETURN count(DISTINCT e) AS n"
    ).single()["n"]
    orphan_events = s.run(
        "MATCH (e:AttackEvent) WHERE NOT EXISTS { MATCH (:Campaign)-[:HAS_EVENT]->(e) } RETURN count(e) AS n"
    ).single()["n"]
    print(f"total AttackEvents: {total_events}")
    print(f"linked (via HAS_EVENT): {linked_events}")
    print(f"orphaned (no HAS_EVENT): {orphan_events}")
    print(f"sanity check linked+orphan == total: {linked_events + orphan_events == total_events}")

    print("\n=== PHASE 2: full profile of every orphaned event ===")
    rows = s.run(
        """
        MATCH (e:AttackEvent)
        WHERE NOT EXISTS { MATCH (:Campaign)-[:HAS_EVENT]->(e) }
        OPTIONAL MATCH (e)-[:MATCHES]->(t:Technique)
        RETURN e.event_id AS event_id, e.campaign_id AS campaign_id,
               e.first_seen AS first_seen, e.last_seen AS last_seen,
               e.attack_id AS attack_id, e.technique AS technique, e.stage AS stage,
               e.attacker_ip AS attacker_ip, e.victim_ip AS victim_ip,
               e.fingerprint AS fingerprint, e.rule_id AS rule_id, e.agent_id AS agent_id,
               e.occurrences AS occurrences, e.tps AS tps, e.rule_level AS rule_level,
               t.attack_id AS matched_technique
        ORDER BY e.campaign_id, e.first_seen
        """
    ).data()
    for r in rows:
        print(r)

    print(f"\ntotal orphan rows printed: {len(rows)}")

    print("\n=== PHASE 3: grouped by orphan campaign_id ===")
    groups = defaultdict(list)
    for r in rows:
        groups[r["campaign_id"]].append(r)

    print(f"distinct orphan campaign_ids: {len(groups)}")
    for cid, events in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        attackers = {e["attacker_ip"] for e in events}
        victims = {e["victim_ip"] for e in events}
        techniques = {e["attack_id"] for e in events}
        first = min(e["first_seen"] for e in events)
        last = max(e["last_seen"] for e in events)
        print(f"{cid:16s} n={len(events):2d} techniques={sorted(techniques)} "
              f"attackers={attackers} victims={victims} first={first} last={last}")

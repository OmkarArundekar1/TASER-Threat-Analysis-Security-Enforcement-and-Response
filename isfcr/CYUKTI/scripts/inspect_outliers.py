import os
import sys

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from neo4j_client import driver  # noqa: E402

for cid in ["CAMP_427A075C", "CAMP_75FBEEBC"]:
    print(f"=== {cid} ===")
    with driver.session() as s:
        r = s.run(
            """
            MATCH (c:Campaign {campaign_id: $cid})-[:HAS_EVENT]->(e:AttackEvent)
            RETURN e.event_id AS id, e.technique_id AS tech, e.occurrences AS occ,
                   e.tps AS tps, e.rule_level AS lvl, e.first_seen AS first_seen
            ORDER BY e.first_seen
            """,
            cid=cid,
        ).data()
        total_tps = 0
        for row in r:
            print(row)
            total_tps += (row["tps"] or 0) * (row["occ"] or 1)
        c = s.run("MATCH (c:Campaign {campaign_id: $cid}) RETURN c.risk_score AS rs, c.total_tps AS ttps, "
                   "c.occurrences AS occ", cid=cid).single()
        print("Campaign risk_score/total_tps/occurrences:", dict(c))
    print()

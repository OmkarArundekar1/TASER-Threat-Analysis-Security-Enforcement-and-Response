import os
import sys

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from neo4j_client import driver  # noqa: E402
from config import TPS_MAP  # noqa: E402

print("=== TPS_MAP (config.py) ===")
for stage, tps in TPS_MAP.items():
    print(f"  {stage}: {tps}")

print("\n=== Events for the two highest risk_score campaigns ===")
with driver.session() as s:
    for cid in ["CAMP_427A075C", "CAMP_75FBEEBC"]:
        print(f"\n{cid}:")
        rows = s.run(
            "MATCH (c:Campaign {campaign_id: $cid})-[:HAS_EVENT]->(e:AttackEvent) "
            "RETURN e.stage AS stage, e.technique AS tech, e.attack_id AS attack_id, "
            "e.occurrences AS occ, e.tps AS tps ORDER BY e.first_seen",
            cid=cid,
        ).data()
        for row in rows:
            print(f"    {row}  (tps/occ = {row['tps'] / row['occ'] if row['occ'] else None})")

    print("\n=== Full risk_score distribution across all 33 campaigns ===")
    scores = sorted(r["rs"] for r in s.run("MATCH (c:Campaign) RETURN c.risk_score AS rs"))
    print("sorted:", scores)

    import statistics
    print("\nmin:", min(scores), "max:", max(scores), "median:", statistics.median(scores))
    print("mean:", round(statistics.mean(scores), 1))
    n = len(scores)
    for p in [50, 75, 90, 95, 99]:
        k = (p / 100) * (n - 1)
        f = int(k)
        c = min(f + 1, n - 1)
        val = scores[f] + (scores[c] - scores[f]) * (k - f)
        print(f"p{p}: {round(val, 1)}")

    print("\n=== Stage distribution across ALL AttackEvents in the dataset ===")
    stage_rows = s.run(
        "MATCH (e:AttackEvent) RETURN e.stage AS stage, count(*) AS n, sum(e.occurrences) AS total_occ, "
        "sum(e.tps) AS total_tps ORDER BY total_tps DESC"
    ).data()
    for row in stage_rows:
        print(f"  {row}")

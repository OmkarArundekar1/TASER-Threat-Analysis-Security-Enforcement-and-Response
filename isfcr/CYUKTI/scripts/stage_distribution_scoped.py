import os
import sys

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from neo4j_client import driver  # noqa: E402

with driver.session() as s:
    print("=== Stage distribution, scoped to the 33 real Campaign-linked events only ===")
    rows = s.run(
        """
        MATCH (c:Campaign)-[:HAS_EVENT]->(e:AttackEvent)
        RETURN e.stage AS stage, count(*) AS n, sum(e.occurrences) AS total_occ, sum(e.tps) AS total_tps
        ORDER BY total_tps DESC
        """
    ).data()
    for row in rows:
        print(row)

    print("\n=== Corrected risk_score percentiles (Campaign-linked only) ===")
    scores = sorted(r["rs"] for r in s.run("MATCH (c:Campaign) RETURN c.risk_score AS rs"))
    import statistics
    print("sorted:", scores)
    print("min:", min(scores), "max:", max(scores), "median:", statistics.median(scores))
    print("mean:", round(statistics.mean(scores), 1))
    n = len(scores)
    for p in [25, 50, 75, 90, 95, 99]:
        k = (p / 100) * (n - 1)
        f = int(k)
        c = min(f + 1, n - 1)
        val = scores[f] + (scores[c] - scores[f]) * (k - f)
        print(f"p{p}: {round(val, 1)}")

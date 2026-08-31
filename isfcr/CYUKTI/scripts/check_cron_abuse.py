import os
import sys

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from neo4j_client import driver  # noqa: E402

with driver.session() as s:
    rows = s.run(
        "MATCH (e:AttackEvent {stage: 'Cron abuse'}) "
        "RETURN e.event_id AS id, e.attack_id AS attack_id, e.campaign_id AS campaign_id, "
        "e.technique AS technique, e.occurrences AS occ, e.tps AS tps"
    ).data()
    for r in rows:
        print(r)

    print("\n=== Unknown-stage events: attack_id breakdown ===")
    rows2 = s.run(
        "MATCH (e:AttackEvent {stage: 'Unknown'}) "
        "RETURN e.attack_id AS attack_id, count(*) AS n, sum(e.occurrences) AS total_occ"
    ).data()
    for r in rows2:
        print(r)

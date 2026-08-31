import os
import sys

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from neo4j_client import driver  # noqa: E402

with driver.session() as s:
    total_campaigns = s.run("MATCH (c:Campaign) RETURN count(c) AS n").single()["n"]
    print("Total Campaign nodes:", total_campaigns)

    for cid in ["CAMP_EA630D49", "CAMP_F2565DF7", "CAMP_599ECA0F", "CAMP_24C62977"]:
        exists = s.run("MATCH (c:Campaign {campaign_id: $cid}) RETURN c", cid=cid).single()
        print(cid, "-> Campaign node exists:", exists is not None)

    orphan_count = s.run(
        "MATCH (e:AttackEvent) WHERE NOT EXISTS { MATCH (:Campaign)-[:HAS_EVENT]->(e) } RETURN count(e) AS n"
    ).single()["n"]
    print("Orphaned AttackEvents (no incoming HAS_EVENT from any Campaign):", orphan_count)

    total_events = s.run("MATCH (e:AttackEvent) RETURN count(e) AS n").single()["n"]
    print("Total AttackEvent nodes:", total_events)

    print("\nDistinct campaign_id values referenced by AttackEvent.campaign_id property "
          "that have no matching Campaign node:")
    rows = s.run(
        """
        MATCH (e:AttackEvent)
        WHERE NOT EXISTS { MATCH (c:Campaign {campaign_id: e.campaign_id}) }
        RETURN DISTINCT e.campaign_id AS cid, count(*) AS n
        """
    ).data()
    for r in rows:
        print(r)

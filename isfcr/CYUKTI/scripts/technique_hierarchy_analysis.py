import os
import sys

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from neo4j_client import driver  # noqa: E402

with driver.session() as s:
    print("=== SUBTECHNIQUE_OF relationships for the 15 techniques observed in real campaigns ===")
    rows = s.run(
        """
        MATCH (c:Campaign)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t:Technique)
        WITH DISTINCT t
        OPTIONAL MATCH (t)-[:SUBTECHNIQUE_OF]->(parent:Technique)
        RETURN t.attack_id AS technique, t.name AS name, t.is_subtechnique AS is_sub,
               parent.attack_id AS parent_id, parent.name AS parent_name
        ORDER BY technique
        """
    ).data()
    for row in rows:
        print(row)

    print("\n=== Do parent techniques of observed subtechniques ALSO appear as directly-observed techniques? ===")
    rows2 = s.run(
        """
        MATCH (c:Campaign)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t:Technique)
        WITH collect(DISTINCT t.attack_id) AS observed
        MATCH (t2:Technique)-[:SUBTECHNIQUE_OF]->(parent:Technique)
        WHERE t2.attack_id IN observed
        RETURN t2.attack_id AS subtechnique, parent.attack_id AS parent,
               parent.attack_id IN observed AS parent_also_observed
        """
    ).data()
    for row in rows2:
        print(row)

    print("\n=== LIKELY_NEXT predicted techniques: are they ever subtechniques, or always parents? ===")
    rows3 = s.run(
        """
        MATCH (c:Campaign)-[:LIKELY_NEXT]->(pt:Technique)
        RETURN DISTINCT pt.attack_id AS predicted, pt.is_subtechnique AS is_sub
        """
    ).data()
    for row in rows3:
        print(row)

    print("\n=== NEXT_TECHNIQUE relationship: how is it populated? sample + count ===")
    count = s.run("MATCH ()-[r:NEXT_TECHNIQUE]->() RETURN count(r) AS n").single()["n"]
    print("total NEXT_TECHNIQUE edges:", count)
    rows4 = s.run(
        "MATCH (a:Technique)-[r:NEXT_TECHNIQUE]->(b:Technique) "
        "RETURN a.attack_id AS from_tech, a.is_subtechnique AS from_is_sub, "
        "b.attack_id AS to_tech, b.is_subtechnique AS to_is_sub, r.count AS count"
    ).data()
    for row in rows4:
        print(row)

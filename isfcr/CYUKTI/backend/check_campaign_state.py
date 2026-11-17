#!/usr/bin/env python3
"""Check current campaign and operation state in Neo4j"""
from neo4j_client import driver

with driver.session() as session:
    # Count campaigns
    result = session.run('MATCH (c:Campaign) RETURN count(c) as count, c.status as status')
    print("Campaign Status:")
    for record in result:
        print(f"  {record['status']}: {record['count']}")

    # Count by host pairs
    result = session.run('''
        MATCH (c:Campaign)
        RETURN c.attacker_ip as attacker, c.victim_ip as victim,
               c.status as status, count(*) as count
        ORDER BY count DESC
        LIMIT 10
    ''')
    print("\nTop Campaign Host Pairs:")
    for record in result:
        print(f"  {record['attacker']} → {record['victim']}: {record['count']} ({record['status']})")

    # Count AttackEvents
    result = session.run('MATCH (ae:AttackEvent) RETURN count(ae) as count').single()
    print(f"\nTotal AttackEvents: {result['count']}")

    # Count Operations
    result = session.run('MATCH (o:Operation) RETURN count(o) as count, o.status as status')
    print("\nOperation Status:")
    for record in result:
        print(f"  {record['status']}: {record['count']}")

    # Check transition edges
    result = session.run('''
        MATCH ()-[r:NEXT_TECHNIQUE]->()
        RETURN count(r) as total,
               sum(r.count) as observations
    ''').single()
    print(f"\nTransition Edges: {result['total']} (total observations: {result['observations']})")

    # Sample transitions
    result = session.run('''
        MATCH (a:Technique)-[r:NEXT_TECHNIQUE]->(b:Technique)
        RETURN a.attack_id as src, b.attack_id as dst,
               r.count as count, r.confidence as confidence
        ORDER BY r.count DESC
        LIMIT 10
    ''')
    print("\nTop Transitions:")
    for record in result:
        print(f"  {record['src']} → {record['dst']}: count={record['count']}, confidence={record['confidence']:.2f}")

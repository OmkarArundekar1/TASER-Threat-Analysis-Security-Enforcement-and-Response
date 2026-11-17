#!/usr/bin/env python3
"""Check what alert data exists in Neo4j"""
from neo4j_client import driver

with driver.session() as session:
    result = session.run('MATCH (ae:AttackEvent) RETURN count(ae) as count')
    count = result.single()['count']
    print(f'Total AttackEvents in Neo4j: {count}')

    if count > 0:
        result = session.run('''
            MATCH (ae:AttackEvent)
            RETURN ae.rule_id as rule_id, ae.mitre_technique as tech, ae.provenance as prov
            LIMIT 20
        ''')
        print('\nSample AttackEvents:')
        for record in result:
            print(f"  Rule {record['rule_id']}: {record['tech']} ({record['prov']})")

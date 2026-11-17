#!/usr/bin/env python3
"""Check AttackEvent schema and relationships"""
from neo4j_client import driver

with driver.session() as session:
    # Get one sample AttackEvent with all properties
    result = session.run('''
        MATCH (ae:AttackEvent)
        RETURN ae LIMIT 1
    ''')
    record = result.single()
    if record:
        ae = record['ae']
        print('AttackEvent properties:')
        for key in sorted(ae.keys()):
            print(f'  {key}: {ae[key]}')

    # Check relationships
    result = session.run('''
        MATCH (ae:AttackEvent)-[r]->(t)
        RETURN type(r) as rel_type, labels(t) as target_labels, count(*) as count
    ''')
    print('\nAttackEvent relationships:')
    for record in result:
        print(f'  -{record["rel_type"]}-> {record["target_labels"]}: {record["count"]}')

    # Check distribution of rule_ids
    result = session.run('''
        MATCH (ae:AttackEvent)
        RETURN ae.rule_id as rule_id, count(*) as count
        ORDER BY count DESC
        LIMIT 15
    ''')
    print('\nTop rule_ids by frequency:')
    for record in result:
        print(f'  {record["rule_id"]}: {record["count"]}')

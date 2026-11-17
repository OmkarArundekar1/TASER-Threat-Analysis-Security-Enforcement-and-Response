#!/usr/bin/env python3
from neo4j_client import driver

with driver.session() as session:
    # Check AttackEvent relationships
    result = session.run('''
        MATCH (ae:AttackEvent)-[r]->()
        RETURN type(r) as rel_type, count(*) as count
    ''')

    print("AttackEvent Outgoing Relationships:")
    for record in result:
        print(f"  {record['rel_type']}: {record['count']}")

    # Check if campaign_id property exists
    result = session.run('''
        MATCH (ae:AttackEvent)
        WHERE ae.campaign_id IS NOT NULL
        RETURN count(ae) as count
    ''').single()

    print(f"\nAttackEvents with campaign_id property: {result['count']}")

    # Sample campaign_id values
    result = session.run('''
        MATCH (ae:AttackEvent)
        WHERE ae.campaign_id IS NOT NULL
        RETURN ae.campaign_id as campaign_id
        LIMIT 5
    ''')

    print("\nSample campaign_id values:")
    for record in result:
        print(f"  {record['campaign_id']}")

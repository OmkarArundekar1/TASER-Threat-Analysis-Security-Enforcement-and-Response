#!/usr/bin/env python3
"""
R2.1: Check what relationships actually exist between Campaign and Operation nodes.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'isfcr', 'CYUKTI', 'backend'))

from neo4j_client import driver

print("=" * 80)
print("R2.1 RELATIONSHIP INVESTIGATION")
print("=" * 80)

with driver.session() as session:
    # Check all relationships from Campaign to Operation
    result = session.run("""
        MATCH (c:Campaign)-[r]->(o:Operation)
        RETURN type(r) as rel_type, count(*) as count
    """)

    print("\nCAMPAIGN -> OPERATION Relationships:")
    print("-" * 40)
    records = list(result)
    if records:
        for record in records:
            print(f"{record['rel_type']}: {record['count']}")
    else:
        print("No relationships found.")

    # Check all relationships from Operation to Campaign
    result = session.run("""
        MATCH (o:Operation)-[r]->(c:Campaign)
        RETURN type(r) as rel_type, count(*) as count
    """)

    print("\nOPERATION -> CAMPAIGN Relationships:")
    print("-" * 40)
    records = list(result)
    if records:
        for record in records:
            print(f"{record['rel_type']}: {record['count']}")
    else:
        print("No relationships found.")

    # Check if there are ANY relationships involving Operation nodes
    result = session.run("""
        MATCH (o:Operation)-[r]-()
        RETURN type(r) as rel_type, count(*) as count
        ORDER BY count DESC
    """)

    print("\nAll Operation relationships (either direction):")
    print("-" * 40)
    records = list(result)
    if records:
        for record in records:
            print(f"{record['rel_type']}: {record['count']}")
    else:
        print("No relationships found.")

    # Get sample Operation IDs to inspect
    result = session.run("""
        MATCH (o:Operation)
        RETURN o.operation_id as operation_id, o.status as status,
               o.campaign_count as campaign_count
        LIMIT 5
    """)

    print("\nSample Operations:")
    print("-" * 40)
    for record in result:
        print(f"ID: {record['operation_id']}")
        print(f"  Status: {record['status']}")
        print(f"  Campaign Count: {record['campaign_count']}")

    # Check if campaigns have any operation_id property
    result = session.run("""
        MATCH (c:Campaign)
        WHERE c.operation_id IS NOT NULL
        RETURN count(*) as count
    """)

    count = list(result)[0]['count']
    print(f"\nCampaigns with operation_id property: {count}")

    if count > 0:
        result = session.run("""
            MATCH (c:Campaign)
            WHERE c.operation_id IS NOT NULL
            RETURN c.campaign_id as campaign_id, c.operation_id as operation_id
            LIMIT 5
        """)
        print("\nSample campaigns with operation_id:")
        print("-" * 40)
        for record in result:
            print(f"Campaign: {record['campaign_id']}")
            print(f"  Operation ID: {record['operation_id']}")

print("\n" + "=" * 80)

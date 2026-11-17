#!/usr/bin/env python3
"""
R2.1: Operation Grouping Analysis
==================================
Analyzes how Campaigns are grouped into Operations via HAS_CAMPAIGN relationships.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'isfcr', 'CYUKTI', 'backend'))

from neo4j_client import driver
import json

print("=" * 80)
print("R2.1 OPERATION GROUPING ANALYSIS")
print("=" * 80)

with driver.session() as session:
    # Count totals
    result = session.run("""
        MATCH (c:Campaign)
        RETURN count(c) as total_campaigns
    """)
    total_campaigns = list(result)[0]['total_campaigns']

    result = session.run("""
        MATCH (o:Operation)
        RETURN count(o) as total_operations
    """)
    total_operations = list(result)[0]['total_operations']

    result = session.run("""
        MATCH (o:Operation)-[:HAS_CAMPAIGN]->(c:Campaign)
        RETURN count(c) as attached_campaigns, count(DISTINCT c) as distinct_attached
    """)
    rec = list(result)[0]
    attached_campaigns = rec['attached_campaigns']
    distinct_attached = rec['distinct_attached']

    unattached = total_campaigns - distinct_attached

    print(f"\nDATASET INVENTORY:")
    print("-" * 40)
    print(f"Total Campaigns:           {total_campaigns}")
    print(f"Total Operations:          {total_operations}")
    print(f"Attached Campaigns:        {distinct_attached} ({distinct_attached/total_campaigns*100:.1f}%)")
    print(f"Unattached Campaigns:      {unattached} ({unattached/total_campaigns*100:.1f}%)")
    print(f"HAS_CAMPAIGN edges:        {attached_campaigns}")

    # Operations distribution
    result = session.run("""
        MATCH (o:Operation)
        OPTIONAL MATCH (o)-[:HAS_CAMPAIGN]->(c:Campaign)
        WITH o, count(c) as campaign_count
        RETURN campaign_count, count(o) as operation_count
        ORDER BY campaign_count
    """)

    print(f"\nOPERATION SIZE DISTRIBUTION:")
    print("-" * 40)
    print(f"{'Size':<10} {'Operations':<12} {'Campaigns':<12}")
    print("-" * 40)

    distribution = {}
    for record in result:
        size = record['campaign_count']
        count = record['operation_count']
        distribution[size] = count
        campaigns_in_group = size * count
        print(f"{size:<10} {count:<12} {campaigns_in_group:<12}")

    # Calculate effective grouping
    operations_with_multiple = sum(count for size, count in distribution.items() if size > 1)
    campaigns_in_groups = sum(size * count for size, count in distribution.items() if size > 1)

    print(f"\nGROUPING EFFECTIVENESS:")
    print("-" * 40)
    print(f"Operations with 0 campaigns:    {distribution.get(0, 0)}")
    print(f"Operations with 1 campaign:     {distribution.get(1, 0)} (singletons)")
    print(f"Operations with >1 campaign:    {operations_with_multiple}")
    print(f"Campaigns grouped (size >1):    {campaigns_in_groups} ({campaigns_in_groups/total_campaigns*100:.1f}%)")

    # Top grouped Operations
    result = session.run("""
        MATCH (o:Operation)-[:HAS_CAMPAIGN]->(c:Campaign)
        WITH o, count(c) as campaign_count,
             collect(c.campaign_id)[0..5] as sample_campaigns
        WHERE campaign_count > 1
        RETURN o.operation_id as operation_id,
               o.status as status,
               campaign_count,
               sample_campaigns
        ORDER BY campaign_count DESC
        LIMIT 10
    """)

    print(f"\nTOP GROUPED OPERATIONS:")
    print("-" * 40)
    for record in result:
        print(f"\n{record['operation_id']} ({record['status']}):")
        print(f"  Campaigns: {record['campaign_count']}")
        print(f"  Sample: {', '.join(record['sample_campaigns'][:3])}")

    # Check for unattached campaigns characteristics
    result = session.run("""
        MATCH (c:Campaign)
        WHERE NOT exists((c)<-[:HAS_CAMPAIGN]-(:Operation))
        RETURN count(c) as count
    """)
    unattached_count = list(result)[0]['count']

    if unattached_count > 0:
        print(f"\nUNATTACHED CAMPAIGNS SAMPLE:")
        print("-" * 40)
        result = session.run("""
            MATCH (c:Campaign)
            WHERE NOT exists((c)<-[:HAS_CAMPAIGN]-(:Operation))
            MATCH (ae:AttackEvent {campaign_id: c.campaign_id})
            WITH c, count(ae) as event_count
            RETURN c.campaign_id as campaign_id,
                   c.status as status,
                   c.attacker_ip as attacker,
                   c.victim_ip as victim,
                   event_count
            ORDER BY event_count DESC
            LIMIT 10
        """)

        for record in result:
            print(f"\n{record['campaign_id']} ({record['status']}):")
            print(f"  Events: {record['event_count']}")
            print(f"  {record['attacker']} -> {record['victim']}")

    # Compare to campaign fragmentation
    result = session.run("""
        MATCH (ae:AttackEvent)
        WITH ae.campaign_id as campaign_id, count(*) as event_count
        WHERE event_count = 1
        RETURN count(*) as singleton_campaigns
    """)
    singleton_campaigns = list(result)[0]['singleton_campaigns']

    print(f"\nFRAGMENTATION COMPARISON:")
    print("-" * 40)
    print(f"Singleton Campaigns (1 event):  {singleton_campaigns} ({singleton_campaigns/total_campaigns*100:.1f}%)")
    print(f"Singleton Operations (1 camp):  {distribution.get(1, 0)} ({distribution.get(1, 0)/total_operations*100:.1f}%)")

    # Export data
    output = {
        "total_campaigns": total_campaigns,
        "total_operations": total_operations,
        "attached_campaigns": distinct_attached,
        "unattached_campaigns": unattached,
        "has_campaign_edges": attached_campaigns,
        "operation_distribution": distribution,
        "operations_with_multiple": operations_with_multiple,
        "campaigns_in_groups": campaigns_in_groups,
        "singleton_campaigns": singleton_campaigns,
        "singleton_operations": distribution.get(1, 0),
        "empty_operations": distribution.get(0, 0)
    }

    with open('research/experiments/r2_1/operation_grouping.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Data exported to: operation_grouping.json")
    print("=" * 80)

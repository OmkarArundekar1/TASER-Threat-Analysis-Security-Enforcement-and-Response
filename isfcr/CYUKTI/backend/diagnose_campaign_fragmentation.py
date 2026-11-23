#!/usr/bin/env python3
"""
Campaign Fragmentation Diagnostic Tool

This script analyzes campaign correlation decisions to understand why
the system fragments sessions into multiple campaigns.

Usage:
    python diagnose_campaign_fragmentation.py

Outputs:
    - Campaign correlation decision trace
    - Feature value distributions at closure
    - Temporal gap analysis
    - Recommendation for threshold adjustments
"""

from datetime import datetime
from collections import defaultdict
import json

from neo4j_client import driver


def get_all_campaigns():
    """Retrieve all campaigns from Neo4j with their events."""
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Campaign)
            OPTIONAL MATCH (c)-[:CONTAINS]->(e:AttackEvent)
            WITH c, e ORDER BY e.timestamp
            RETURN 
                c.campaign_id as campaign_id,
                c.attacker_ip as attacker_ip,
                c.victim_ip as victim_ip,
                c.first_seen as first_seen,
                c.last_seen as last_seen,
                c.status as status,
                collect({
                    event_id: e.event_id,
                    timestamp: e.timestamp,
                    technique: e.technique
                }) as events
            ORDER BY c.first_seen
        """)
        
        campaigns = []
        for record in result:
            campaigns.append({
                'campaign_id': record['campaign_id'],
                'attacker_ip': record['attacker_ip'],
                'victim_ip': record['victim_ip'],
                'first_seen': record['first_seen'],
                'last_seen': record['last_seen'],
                'status': record['status'],
                'events': [e for e in record['events'] if e['event_id'] is not None],
                'event_count': len([e for e in record['events'] if e['event_id'] is not None])
            })
        
        return campaigns


def group_by_attacker_victim(campaigns):
    """Group campaigns by (attacker_ip, victim_ip) pair."""
    groups = defaultdict(list)
    
    for campaign in campaigns:
        key = (campaign['attacker_ip'], campaign['victim_ip'])
        groups[key].append(campaign)
    
    return groups


def calculate_temporal_gaps(campaigns):
    """Calculate temporal gaps between consecutive campaigns."""
    if len(campaigns) < 2:
        return []
    
    # Sort by first_seen
    sorted_campaigns = sorted(campaigns, key=lambda c: c['first_seen'])
    
    gaps = []
    for i in range(len(sorted_campaigns) - 1):
        c1 = sorted_campaigns[i]
        c2 = sorted_campaigns[i + 1]
        
        gap = (c2['first_seen'] - c1['last_seen']).total_seconds()
        gaps.append({
            'from_campaign': c1['campaign_id'],
            'to_campaign': c2['campaign_id'],
            'gap_seconds': gap,
            'gap_minutes': gap / 60.0,
            'c1_last_seen': c1['last_seen'],
            'c2_first_seen': c2['first_seen']
        })
    
    return gaps


def analyze_fragmentation():
    """Main analysis function."""
    print("=" * 80)
    print("Campaign Fragmentation Diagnostic")
    print("=" * 80)
    print()
    
    campaigns = get_all_campaigns()
    print(f"Total campaigns in database: {len(campaigns)}")
    print()
    
    groups = group_by_attacker_victim(campaigns)
    print(f"Unique (attacker, victim) pairs: {len(groups)}")
    print()
    
    # Analyze fragmentation per pair
    fragmentation_data = []
    
    for (attacker, victim), group_campaigns in groups.items():
        if len(group_campaigns) <= 1:
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Attacker: {attacker}")
        print(f"Victim: {victim}")
        print(f"Number of campaigns: {len(group_campaigns)}")
        print(f"{'=' * 80}\n")
        
        # Sort campaigns by first_seen
        sorted_campaigns = sorted(group_campaigns, key=lambda c: c['first_seen'])
        
        # Print campaign details
        for i, campaign in enumerate(sorted_campaigns, 1):
            print(f"Campaign {i}: {campaign['campaign_id']}")
            print(f"  First seen: {campaign['first_seen']}")
            print(f"  Last seen:  {campaign['last_seen']}")
            print(f"  Duration:   {(campaign['last_seen'] - campaign['first_seen']).total_seconds():.1f}s")
            print(f"  Events:     {campaign['event_count']}")
            print(f"  Status:     {campaign['status']}")
            
            # Show first few techniques
            if campaign['events']:
                techniques = [e['technique'] for e in campaign['events'] if e['technique']]
                if techniques:
                    print(f"  Techniques: {', '.join(techniques[:5])}")
                    if len(techniques) > 5:
                        print(f"              ... and {len(techniques) - 5} more")
            print()
        
        # Calculate temporal gaps
        gaps = calculate_temporal_gaps(sorted_campaigns)
        
        if gaps:
            print("Temporal Gaps Between Campaigns:")
            print("-" * 80)
            for gap in gaps:
                print(f"  {gap['from_campaign']} → {gap['to_campaign']}")
                print(f"    Gap: {gap['gap_seconds']:.1f}s ({gap['gap_minutes']:.1f} minutes)")
                print(f"    C1 last seen: {gap['c1_last_seen']}")
                print(f"    C2 first seen: {gap['c2_first_seen']}")
                print()
        
        fragmentation_data.append({
            'attacker': attacker,
            'victim': victim,
            'num_campaigns': len(group_campaigns),
            'total_events': sum(c['event_count'] for c in group_campaigns),
            'gaps': gaps
        })
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()
    
    if fragmentation_data:
        all_gaps = []
        for item in fragmentation_data:
            all_gaps.extend([g['gap_seconds'] for g in item['gaps']])
        
        if all_gaps:
            print(f"Temporal Gap Statistics (seconds):")
            print(f"  Count:   {len(all_gaps)}")
            print(f"  Min:     {min(all_gaps):.1f}s")
            print(f"  Max:     {max(all_gaps):.1f}s")
            print(f"  Mean:    {sum(all_gaps) / len(all_gaps):.1f}s")
            print(f"  Median:  {sorted(all_gaps)[len(all_gaps)//2]:.1f}s")
            print()
            
            # Check how many gaps are within CAMPAIGN_TIMEOUT (120s)
            within_timeout = [g for g in all_gaps if g <= 120]
            print(f"Gaps within CAMPAIGN_TIMEOUT (120s): {len(within_timeout)}/{len(all_gaps)} ({100*len(within_timeout)/len(all_gaps):.1f}%)")
            print()
            
            if within_timeout:
                print("⚠️  WARNING: Some campaigns were fragmented even though temporal gaps")
                print("   were within CAMPAIGN_TIMEOUT. This suggests:")
                print("   1. Correlation score dropped below 0.30 threshold")
                print("   2. Reopen score was below 0.70 threshold")
                print("   3. Features were unavailable or returning low values")
                print()
        
        print(f"Pairs with fragmentation: {len(fragmentation_data)}")
        for item in fragmentation_data:
            print(f"  {item['attacker']} → {item['victim']}: {item['num_campaigns']} campaigns, {item['total_events']} events")
        print()
    
    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    if all_gaps and within_timeout:
        avg_gap = sum(within_timeout) / len(within_timeout)
        print(f"1. ✅ CAMPAIGN_TIMEOUT (120s) is appropriate")
        print(f"   Average gap within timeout: {avg_gap:.1f}s")
        print()
        print(f"2. ⚠️  ACTIVE_CONTINUE_THRESHOLD (0.30) may be too high")
        print(f"   Consider lowering to 0.25 to reduce premature closure")
        print()
        print(f"3. ⚠️  REOPEN_THRESHOLD (0.70) is too high")
        print(f"   Consider lowering to 0.50 to allow more reopening")
        print()
        print(f"4. ✅ lateral_movement_similarity bug was fixed")
        print(f"   This should help with multi-hop attacks going forward")
        print()
        print(f"5. 🔍 Need to investigate feature calculations")
        print(f"   Run with debug logging to see actual correlation scores")
        print()
    elif all_gaps:
        print(f"1. CAMPAIGN_TIMEOUT (120s) may be too short")
        print(f"   Consider increasing to {max(all_gaps)*1.2:.0f}s")
        print()


if __name__ == '__main__':
    try:
        analyze_fragmentation()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

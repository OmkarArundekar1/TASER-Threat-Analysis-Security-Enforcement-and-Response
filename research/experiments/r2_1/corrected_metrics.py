#!/usr/bin/env python3
"""
R2.1 Corrected Metrics
=====================
Produces internally consistent, verified campaign fragmentation metrics.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'isfcr', 'CYUKTI', 'backend'))

import json
from collections import Counter, defaultdict
from neo4j_client import driver

def extract_canonical_dataset():
    """Extract canonical campaign assignment from Neo4j using campaign_id property."""
    with driver.session() as session:
        # Get all AttackEvents with their campaign assignments
        result = session.run('''
            MATCH (ae:AttackEvent)
            RETURN ae.event_id as event_id,
                   ae.campaign_id as campaign_id,
                   ae.rule_id as rule_id,
                   ae.attack_id as attack_id,
                   ae.attacker_ip as attacker_ip,
                   ae.victim_ip as victim_ip,
                   ae.first_seen as first_seen
            ORDER BY ae.first_seen
        ''')

        events = []
        campaign_events = defaultdict(list)

        for record in result:
            event = {
                'event_id': record['event_id'],
                'campaign_id': record['campaign_id'],
                'rule_id': record['rule_id'],
                'attack_id': record['attack_id'],
                'attacker_ip': record['attacker_ip'],
                'victim_ip': record['victim_ip'],
                'first_seen': record['first_seen']
            }
            events.append(event)

            if record['campaign_id']:
                campaign_events[record['campaign_id']].append(event)

        return events, campaign_events

def calculate_corrected_metrics():
    """Calculate all metrics using consistent definitions."""
    events, campaign_events = extract_canonical_dataset()

    with driver.session() as session:
        # Get Campaign node metadata
        result = session.run('''
            MATCH (c:Campaign)
            RETURN c.campaign_id as campaign_id,
                   c.status as status,
                   c.attacker_ip as attacker_ip,
                   c.victim_ip as victim_ip,
                   c.first_seen as first_seen,
                   c.last_seen as last_seen,
                   c.reopened_count as reopened_count
        ''')

        campaigns = {}
        for record in result:
            campaigns[record['campaign_id']] = {
                'campaign_id': record['campaign_id'],
                'status': record['status'],
                'attacker_ip': record['attacker_ip'],
                'victim_ip': record['victim_ip'],
                'first_seen': record['first_seen'],
                'last_seen': record['last_seen'],
                'reopened_count': record['reopened_count'] or 0
            }

    # Calculate metrics
    total_events = len(events)
    total_campaigns = len(campaigns)

    # Events with valid campaign_id
    events_assigned = sum(1 for e in events if e['campaign_id'])
    events_unassigned = total_events - events_assigned

    # Distinct campaigns from events
    distinct_campaigns_in_events = len(set(e['campaign_id'] for e in events if e['campaign_id']))

    # Campaign sizes
    campaign_sizes = []
    singleton_count = 0

    for campaign_id, campaign_meta in campaigns.items():
        size = len(campaign_events.get(campaign_id, []))
        campaign_sizes.append(size)
        if size == 1:
            singleton_count += 1

    # Host pair fragmentation
    host_pair_campaigns = defaultdict(set)
    for campaign_id, campaign_meta in campaigns.items():
        host_pair = (campaign_meta['attacker_ip'], campaign_meta['victim_ip'])
        host_pair_campaigns[host_pair].add(campaign_id)

    fragmented_host_pairs = {
        f"{attacker}->{victim}": len(campaign_set)
        for (attacker, victim), campaign_set in host_pair_campaigns.items()
        if len(campaign_set) > 1
    }

    # Technique diversity
    technique_counts_per_campaign = []
    for campaign_id in campaigns.keys():
        techniques = set(e['attack_id'] for e in campaign_events.get(campaign_id, []) if e['attack_id'])
        technique_counts_per_campaign.append(len(techniques))

    # Reopened campaigns
    reopened_campaigns = sum(1 for c in campaigns.values() if c['reopened_count'] > 0)

    # Calculate durations
    campaign_durations = []
    zero_duration_count = 0

    for campaign_id, campaign_meta in campaigns.items():
        first_seen = campaign_meta['first_seen']
        last_seen = campaign_meta['last_seen']

        if first_seen and last_seen:
            # Convert to native datetime
            if hasattr(first_seen, 'to_native'):
                first_seen = first_seen.to_native()
            if hasattr(last_seen, 'to_native'):
                last_seen = last_seen.to_native()

            duration = last_seen - first_seen

            if hasattr(duration, 'total_seconds'):
                hours = duration.total_seconds() / 3600
            else:
                hours = 0

            campaign_durations.append(hours)
            if hours == 0:
                zero_duration_count += 1

    return {
        'dataset': {
            'total_events': total_events,
            'events_assigned': events_assigned,
            'events_unassigned': events_unassigned,
            'assignment_rate': events_assigned / total_events if total_events > 0 else 0
        },
        'campaigns': {
            'total_campaign_nodes': total_campaigns,
            'distinct_campaigns_in_events': distinct_campaigns_in_events,
            'campaigns_with_events': sum(1 for s in campaign_sizes if s > 0),
            'campaigns_without_events': sum(1 for s in campaign_sizes if s == 0),
            'singleton_campaigns': singleton_count,
            'singleton_ratio': singleton_count / total_campaigns if total_campaigns > 0 else 0,
            'reopened_campaigns': reopened_campaigns,
            'size_distribution': dict(Counter(campaign_sizes)),
            'sizes': {
                'min': min(campaign_sizes) if campaign_sizes else 0,
                'max': max(campaign_sizes) if campaign_sizes else 0,
                'mean': sum(campaign_sizes) / len(campaign_sizes) if campaign_sizes else 0,
                'median': sorted(campaign_sizes)[len(campaign_sizes)//2] if campaign_sizes else 0
            }
        },
        'technique_diversity': {
            'mean': sum(technique_counts_per_campaign) / len(technique_counts_per_campaign) if technique_counts_per_campaign else 0,
            'max': max(technique_counts_per_campaign) if technique_counts_per_campaign else 0
        },
        'temporal': {
            'campaigns_with_timestamps': len(campaign_durations),
            'zero_duration_campaigns': zero_duration_count,
            'non_zero_duration_campaigns': len(campaign_durations) - zero_duration_count,
            'duration_hours': {
                'min': min(campaign_durations) if campaign_durations else 0,
                'max': max(campaign_durations) if campaign_durations else 0,
                'mean': sum(campaign_durations) / len(campaign_durations) if campaign_durations else 0,
                'median': sorted(campaign_durations)[len(campaign_durations)//2] if campaign_durations else 0
            }
        },
        'host_pair_fragmentation': {
            'total_host_pairs': len(host_pair_campaigns),
            'fragmented_pairs': len(fragmented_host_pairs),
            'top_fragmented': dict(sorted(fragmented_host_pairs.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    }

def print_corrected_metrics(metrics):
    """Print corrected metrics report."""
    print("=" * 80)
    print("R2.1 CORRECTED METRICS")
    print("=" * 80)
    print()

    dataset = metrics['dataset']
    print("DATASET")
    print("-" * 80)
    print(f"Total Events:              {dataset['total_events']}")
    print(f"Events Assigned:           {dataset['events_assigned']}")
    print(f"Events Unassigned:         {dataset['events_unassigned']}")
    print(f"Assignment Rate:           {dataset['assignment_rate']*100:.1f}%")
    print()

    campaigns = metrics['campaigns']
    print("CAMPAIGNS")
    print("-" * 80)
    print(f"Total Campaign Nodes:      {campaigns['total_campaign_nodes']}")
    print(f"Distinct in Events:        {campaigns['distinct_campaigns_in_events']}")
    print(f"With Events:               {campaigns['campaigns_with_events']}")
    print(f"Without Events:            {campaigns['campaigns_without_events']}")
    print(f"Singleton Campaigns:       {campaigns['singleton_campaigns']}")
    print(f"Singleton Ratio:           {campaigns['singleton_ratio']*100:.1f}%")
    print(f"Reopened Campaigns:        {campaigns['reopened_campaigns']}")
    print()

    sizes = campaigns['sizes']
    print("CAMPAIGN SIZE")
    print("-" * 80)
    print(f"Min:                       {sizes['min']}")
    print(f"Max:                       {sizes['max']}")
    print(f"Mean:                      {sizes['mean']:.2f}")
    print(f"Median:                    {sizes['median']}")
    print(f"Distribution:              {campaigns['size_distribution']}")
    print()

    tech = metrics['technique_diversity']
    print("TECHNIQUE DIVERSITY")
    print("-" * 80)
    print(f"Mean Techniques/Campaign:  {tech['mean']:.2f}")
    print(f"Max Techniques/Campaign:   {tech['max']}")
    print()

    temp = metrics['temporal']
    print("TEMPORAL")
    print("-" * 80)
    print(f"Campaigns with Timestamps: {temp['campaigns_with_timestamps']}")
    print(f"Zero Duration:             {temp['zero_duration_campaigns']}")
    print(f"Non-Zero Duration:         {temp['non_zero_duration_campaigns']}")

    dur = temp['duration_hours']
    print(f"Duration (hours):")
    print(f"  Min:                     {dur['min']:.2f}")
    print(f"  Max:                     {dur['max']:.2f}")
    print(f"  Mean:                    {dur['mean']:.2f}")
    print(f"  Median:                  {dur['median']:.2f}")
    print()

    frag = metrics['host_pair_fragmentation']
    print("HOST PAIR FRAGMENTATION")
    print("-" * 80)
    print(f"Total Host Pairs:          {frag['total_host_pairs']}")
    print(f"Fragmented Pairs (>1):     {frag['fragmented_pairs']}")
    print(f"Top Fragmented:")
    for pair, count in list(frag['top_fragmented'].items())[:5]:
        print(f"  {pair}: {count} campaigns")
    print()
    print("=" * 80)

if __name__ == '__main__':
    metrics = calculate_corrected_metrics()
    print_corrected_metrics(metrics)

    output_file = 'corrected_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Corrected metrics saved to: {output_file}")

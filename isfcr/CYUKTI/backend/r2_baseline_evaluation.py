#!/usr/bin/env python3
"""
R2 Baseline Campaign Reconstruction Evaluation
==============================================
Measures current campaign reconstruction behavior WITHOUT modifications.

This establishes the baseline for comparison against:
- Approach B: Transition smoothing analysis
- Approach C: Operation grouping analysis
"""
import json
from collections import Counter, defaultdict
from datetime import timedelta
from neo4j_client import driver

def calculate_campaign_metrics():
    """Calculate comprehensive campaign reconstruction metrics."""
    with driver.session() as session:
        # Get all campaigns with their events (using campaign_id property)
        result = session.run('''
            MATCH (c:Campaign)
            OPTIONAL MATCH (ae:AttackEvent)
            WHERE ae.campaign_id = c.campaign_id
            WITH c, collect(ae) as events
            RETURN c.campaign_id as campaign_id,
                   c.status as status,
                   c.attacker_ip as attacker_ip,
                   c.victim_ip as victim_ip,
                   c.first_seen as first_seen,
                   c.last_seen as last_seen,
                   c.reopened_count as reopened_count,
                   size(events) as event_count,
                   events
            ORDER BY c.first_seen
        ''')

        campaigns = []
        campaign_sizes = []
        singleton_count = 0
        total_events_assigned = 0
        reopened_campaigns = 0

        host_pair_campaigns = defaultdict(list)
        technique_diversity = []
        temporal_spans = []

        for record in result:
            campaign_id = record['campaign_id']
            event_count = record['event_count']
            events = record['events']

            campaign_sizes.append(event_count)
            if event_count == 1:
                singleton_count += 1
            if event_count > 0:
                total_events_assigned += event_count

            if record['reopened_count'] and record['reopened_count'] > 0:
                reopened_campaigns += 1

            # Technique diversity per campaign
            techniques = set()
            for event in events:
                if event.get('attack_id'):
                    techniques.add(event['attack_id'])
            technique_diversity.append(len(techniques))

            # Temporal span
            if record['first_seen'] and record['last_seen']:
                # Handle Neo4j Duration objects
                first = record['first_seen']
                last = record['last_seen']
                if hasattr(first, 'iso_format'):
                    first = first.iso_format()
                if hasattr(last, 'iso_format'):
                    last = last.iso_format()

                # Calculate duration if both are datetime-like
                try:
                    # Try to calculate difference
                    if hasattr(last, '__sub__') and hasattr(first, '__sub__'):
                        span = last - first
                        if hasattr(span, 'seconds'):
                            temporal_spans.append(span.seconds / 3600)  # hours
                        elif hasattr(span, 'hours'):
                            temporal_spans.append(span.hours)
                except:
                    pass

            host_pair_campaigns[(record['attacker_ip'], record['victim_ip'])].append(campaign_id)

            campaigns.append({
                'campaign_id': campaign_id,
                'status': record['status'],
                'event_count': event_count,
                'technique_count': len(techniques),
                'reopened': record['reopened_count'] or 0
            })

        # Calculate fragmentation by host pair
        fragmentation_by_host = {}
        for (attacker, victim), camp_list in host_pair_campaigns.items():
            if len(camp_list) > 1:
                fragmentation_by_host[f"{attacker}->{victim}"] = len(camp_list)

        # Unassigned events (events with no campaign_id or campaign not in Campaign nodes)
        result = session.run('''
            MATCH (ae:AttackEvent)
            WHERE ae.campaign_id IS NULL
            RETURN count(ae) as unassigned_count
        ''').single()
        unassigned_count = result['unassigned_count']

        # Operation metrics
        result = session.run('''
            MATCH (o:Operation)
            OPTIONAL MATCH (c:Campaign)-[r]->(o)
            WITH o, count(c) as campaign_count
            RETURN o.operation_id as operation_id,
                   o.status as status,
                   campaign_count
            ORDER BY campaign_count DESC
        ''')

        operations = []
        operation_sizes = []
        for record in result:
            size = record['campaign_count']
            operations.append({
                'operation_id': record['operation_id'],
                'status': record['status'],
                'campaign_count': size
            })
            operation_sizes.append(size)

        return {
            'total_campaigns': len(campaigns),
            'total_events': 308,  # Known from inventory
            'events_assigned': total_events_assigned,
            'unassigned_events': unassigned_count,
            'singleton_campaigns': singleton_count,
            'singleton_ratio': singleton_count / len(campaigns) if campaigns else 0,
            'reopened_campaigns': reopened_campaigns,
            'campaign_sizes': {
                'min': min(campaign_sizes) if campaign_sizes else 0,
                'max': max(campaign_sizes) if campaign_sizes else 0,
                'mean': sum(campaign_sizes) / len(campaign_sizes) if campaign_sizes else 0,
                'median': sorted(campaign_sizes)[len(campaign_sizes)//2] if campaign_sizes else 0,
                'distribution': dict(Counter(campaign_sizes))
            },
            'technique_diversity': {
                'mean': sum(technique_diversity) / len(technique_diversity) if technique_diversity else 0,
                'max': max(technique_diversity) if technique_diversity else 0
            },
            'temporal_continuity': {
                'mean_span_hours': sum(temporal_spans) / len(temporal_spans) if temporal_spans else 0,
                'max_span_hours': max(temporal_spans) if temporal_spans else 0
            },
            'host_pair_fragmentation': fragmentation_by_host,
            'operations': {
                'total': len(operations),
                'mean_campaigns_per_operation': sum(operation_sizes) / len(operation_sizes) if operation_sizes else 0,
                'max_campaigns_per_operation': max(operation_sizes) if operation_sizes else 0
            }
        }

def print_baseline_report(metrics):
    """Print comprehensive baseline report."""
    print("=" * 80)
    print("R2 BASELINE EVALUATION - APPROACH A: EXISTING RECONSTRUCTION")
    print("=" * 80)
    print()

    print("CAMPAIGN ASSIGNMENT")
    print("-" * 80)
    print(f"Total Campaigns:           {metrics['total_campaigns']}")
    print(f"Total Events:              {metrics['total_events']}")
    print(f"Events Assigned:           {metrics['events_assigned']}")
    print(f"Unassigned Events:         {metrics['unassigned_events']}")
    print(f"Assignment Rate:           {metrics['events_assigned']/metrics['total_events']*100:.1f}%")
    print()

    print("CAMPAIGN FRAGMENTATION")
    print("-" * 80)
    print(f"Singleton Campaigns:       {metrics['singleton_campaigns']}")
    print(f"Singleton Ratio:           {metrics['singleton_ratio']*100:.1f}%")
    print(f"Reopened Campaigns:        {metrics['reopened_campaigns']}")
    print()

    sizes = metrics['campaign_sizes']
    print("CAMPAIGN SIZE DISTRIBUTION")
    print("-" * 80)
    print(f"Min:                       {sizes['min']}")
    print(f"Max:                       {sizes['max']}")
    print(f"Mean:                      {sizes['mean']:.2f}")
    print(f"Median:                    {sizes['median']}")
    print(f"Distribution:              {sizes['distribution']}")
    print()

    tech = metrics['technique_diversity']
    print("TECHNIQUE DIVERSITY")
    print("-" * 80)
    print(f"Mean Techniques/Campaign:  {tech['mean']:.2f}")
    print(f"Max Techniques/Campaign:   {tech['max']}")
    print()

    temp = metrics['temporal_continuity']
    print("TEMPORAL CONTINUITY")
    print("-" * 80)
    print(f"Mean Campaign Span:        {temp['mean_span_hours']:.2f} hours")
    print(f"Max Campaign Span:         {temp['max_span_hours']:.2f} hours")
    print()

    if metrics['host_pair_fragmentation']:
        print("HOST PAIR FRAGMENTATION (campaigns > 1)")
        print("-" * 80)
        for host_pair, count in sorted(metrics['host_pair_fragmentation'].items(),
                                       key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {host_pair}: {count} campaigns")
        print()

    ops = metrics['operations']
    print("OPERATION-LEVEL GROUPING")
    print("-" * 80)
    print(f"Total Operations:          {ops['total']}")
    print(f"Mean Campaigns/Operation:  {ops['mean_campaigns_per_operation']:.2f}")
    print(f"Max Campaigns/Operation:   {ops['max_campaigns_per_operation']}")
    print()

    print("=" * 80)

if __name__ == '__main__':
    metrics = calculate_campaign_metrics()
    print_baseline_report(metrics)

    # Save baseline
    output_file = 'r2_baseline_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Baseline metrics saved to: {output_file}")

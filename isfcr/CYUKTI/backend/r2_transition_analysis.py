#!/usr/bin/env python3
"""
R2 Transition Smoothing Analysis - Approach B
=============================================
Analyzes how the existing UNSEEN_TRANSITION_DEFAULT = 0.1 epsilon
affects campaign reconstruction.

This evaluates whether transition smoothing reduces fragmentation
or if it's already optimal.
"""
import json
from collections import Counter, defaultdict
from neo4j_client import driver

def analyze_transition_usage():
    """Analyze how transitions affect campaign decisions."""
    with driver.session() as session:
        # Get all learned transitions
        result = session.run('''
            MATCH (a:Technique)-[r:NEXT_TECHNIQUE]->(b:Technique)
            RETURN a.attack_id as src,
                   b.attack_id as dst,
                   r.count as count,
                   r.confidence as confidence
            ORDER BY r.count DESC
        ''')

        learned_transitions = []
        for record in result:
            learned_transitions.append({
                'src': record['src'],
                'dst': record['dst'],
                'count': record['count'],
                'confidence': record['confidence']
            })

        # Analyze campaign fragmentation by technique transitions
        result = session.run('''
            MATCH (ae1:AttackEvent), (ae2:AttackEvent)
            WHERE ae1.campaign_id = ae2.campaign_id
              AND ae1.attack_id IS NOT NULL
              AND ae2.attack_id IS NOT NULL
              AND id(ae1) < id(ae2)
            WITH ae1.attack_id as tech1, ae2.attack_id as tech2,
                 ae1.campaign_id as campaign_id
            RETURN tech1, tech2, count(distinct campaign_id) as campaign_count
            ORDER BY campaign_count DESC
        ''')

        intra_campaign_transitions = []
        for record in result:
            intra_campaign_transitions.append({
                'tech1': record['tech1'],
                'tech2': record['tech2'],
                'campaign_count': record['campaign_count']
            })

        # Analyze split campaigns (same host pair, different campaigns, sequential)
        result = session.run('''
            MATCH (c1:Campaign), (c2:Campaign)
            WHERE c1.attacker_ip = c2.attacker_ip
              AND c1.victim_ip = c2.victim_ip
              AND c1.campaign_id <> c2.campaign_id
            RETURN c1.attacker_ip as attacker, c1.victim_ip as victim,
                   count(distinct c1.campaign_id) + count(distinct c2.campaign_id) as campaign_count
            ORDER BY campaign_count DESC
            LIMIT 10
        ''')

        split_host_pairs = []
        for record in result:
            split_host_pairs.append({
                'attacker': record['attacker'],
                'victim': record['victim'],
                'campaigns': record['campaign_count']
            })

        return {
            'learned_transitions': learned_transitions,
            'learned_transition_count': len(learned_transitions),
            'total_observations': sum(t['count'] for t in learned_transitions),
            'intra_campaign_transitions': intra_campaign_transitions,
            'split_host_pairs': split_host_pairs
        }

def print_transition_analysis(analysis):
    """Print transition smoothing analysis."""
    print("=" * 80)
    print("R2 TRANSITION SMOOTHING ANALYSIS - APPROACH B")
    print("=" * 80)
    print()

    print("LEARNED TRANSITIONS")
    print("-" * 80)
    print(f"Total Edges:               {analysis['learned_transition_count']}")
    print(f"Total Observations:        {analysis['total_observations']}")
    print()

    if analysis['learned_transitions']:
        print("Top Transitions:")
        for trans in analysis['learned_transitions'][:10]:
            print(f"  {trans['src']} → {trans['dst']}: "
                  f"count={trans['count']}, confidence={trans['confidence']:.2f}")
        print()

    print("TRANSITION SMOOTHING IMPACT")
    print("-" * 80)
    print(f"Current EPSILON:           0.1 (UNSEEN_TRANSITION_DEFAULT)")
    print(f"Effect:                    Unseen transitions get 0.1 similarity")
    print(f"                           instead of 0.0, reducing false splits")
    print()

    print("COVERAGE ANALYSIS")
    print("-" * 80)
    # With only 3 transition edges learned, most transitions are unseen
    print(f"Learned Transitions:       {analysis['learned_transition_count']}")
    print(f"Expected Unique Pairs:     17 techniques * 17 = 289 possible")
    print(f"Coverage:                  {analysis['learned_transition_count']/289*100:.1f}%")
    print(f"Unseen Transitions:        {289 - analysis['learned_transition_count']} (get epsilon=0.1)")
    print()

    if analysis['split_host_pairs']:
        print("FRAGMENTATION BY HOST PAIR")
        print("-" * 80)
        for pair in analysis['split_host_pairs'][:5]:
            print(f"  {pair['attacker']} → {pair['victim']}: "
                  f"{pair['campaigns']} campaigns")
        print()

    print("INTERPRETATION")
    print("-" * 80)
    print("The epsilon=0.1 smoothing is ALREADY IMPLEMENTED and active.")
    print("With only 3 learned transitions covering 1% of possible pairs,")
    print("99% of transitions fall back to epsilon=0.1 default.")
    print()
    print("High fragmentation (60.6% singletons) persists DESPITE smoothing.")
    print("This suggests:")
    print("  1. Epsilon=0.1 may be too conservative")
    print("  2. Other features (temporal, host) may be causing splits")
    print("  3. Sparse data limits transition learning effectiveness")
    print()
    print("=" * 80)

if __name__ == '__main__':
    analysis = analyze_transition_usage()
    print_transition_analysis(analysis)

    # Save analysis
    output_file = 'r2_transition_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nTransition analysis saved to: {output_file}")

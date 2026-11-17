#!/usr/bin/env python3
"""
Baseline ATT&CK Resolution Coverage Evaluation
===============================================
Evaluates the existing mitre_resolver against AttackEvents currently in Neo4j.

This provides a reproducible baseline before any Reviewed Registry modifications.
"""
import json
from collections import Counter, defaultdict
from neo4j_client import driver
from mitre_resolver import resolve_mitre, PROVENANCE_UNKNOWN, PROVENANCE_AMBIGUOUS

def extract_alerts_from_neo4j():
    """Extract all AttackEvents and reconstruct their original alerts."""
    alerts = []
    with driver.session() as session:
        result = session.run('''
            MATCH (ae:AttackEvent)
            RETURN ae.event_id as event_id,
                   ae.rule_id as rule_id,
                   ae.attack_id as attack_id,
                   ae.investigation_payload as payload
            ORDER BY ae.first_seen
        ''')
        for record in result:
            payload = record['payload']
            if payload:
                try:
                    raw_alert = json.loads(payload)

                    # Transform flattened investigation_payload to Wazuh alert structure
                    # The payload has rule_id, rule_description, etc. at root level
                    # Need to nest them under 'rule' key for resolver
                    alert = {
                        'rule': {
                            'id': raw_alert.get('rule_id'),
                            'description': raw_alert.get('rule_description'),
                            'level': raw_alert.get('rule_level'),
                            'groups': raw_alert.get('rule_groups', []),
                            'firedtimes': raw_alert.get('rule_firedtimes')
                        },
                        'agent': raw_alert.get('agent', {}),
                        'timestamp': raw_alert.get('timestamp'),
                        'location': raw_alert.get('location')
                    }

                    # Only include mitre block if present (for native mappings)
                    if 'mitre' in raw_alert and raw_alert['mitre']:
                        alert['rule']['mitre'] = raw_alert['mitre']

                    alerts.append({
                        'event_id': record['event_id'],
                        'rule_id': record['rule_id'],
                        'stored_attack_id': record['attack_id'],
                        'alert': alert,
                        'raw_payload': raw_alert
                    })
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Warning: Failed to parse payload for event {record.get('event_id')}: {e}")
    return alerts

def evaluate_resolver(alerts):
    """Run resolve_mitre on all alerts and collect statistics."""
    stats = {
        'total': len(alerts),
        'by_provenance': Counter(),
        'by_confidence': Counter(),
        'unknown_rules': Counter(),
        'unknown_rule_descriptions': Counter(),
        'resolved_techniques': Counter(),
        'unique_rules': set(),
        'alerts_by_rule': defaultdict(list)
    }

    for item in alerts:
        alert = item['alert']
        rule = alert.get('rule', {})
        rule_id = rule.get('id', 'UNKNOWN')
        rule_desc = rule.get('description', 'No description')

        stats['unique_rules'].add(rule_id)
        stats['alerts_by_rule'][rule_id].append(item)

        resolution = resolve_mitre(alert)
        stats['by_provenance'][resolution.provenance] += 1
        stats['by_confidence'][resolution.confidence] += 1

        if resolution.provenance in (PROVENANCE_UNKNOWN, PROVENANCE_AMBIGUOUS):
            stats['unknown_rules'][rule_id] += 1
            stats['unknown_rule_descriptions'][(rule_id, rule_desc)] += 1
        else:
            for tech_id in resolution.technique_ids:
                stats['resolved_techniques'][tech_id] += 1

    return stats

def print_baseline_report(stats):
    """Print comprehensive baseline coverage report."""
    total = stats['total']

    print("=" * 80)
    print("ATT&CK RESOLUTION BASELINE EVALUATION")
    print("=" * 80)
    print(f"Dataset: Neo4j AttackEvents")
    print(f"Total Alerts: {total}")
    print(f"Unique Wazuh Rules: {len(stats['unique_rules'])}")
    print()

    print("RESOLUTION TIER DISTRIBUTION")
    print("-" * 80)
    print(f"{'Tier':<30} {'Count':>8}  {'Percentage':>10}")
    print("-" * 80)
    for prov, count in stats['by_provenance'].most_common():
        pct = (count / total * 100) if total else 0
        print(f"{prov:<30} {count:>8}  {pct:>9.1f}%")
    print("-" * 80)
    print()

    print("CONFIDENCE DISTRIBUTION")
    print("-" * 80)
    for conf, count in stats['by_confidence'].most_common():
        print(f"  {conf:<20} {count:>6}")
    print()

    print("TOP UNMAPPED RULES (UNKNOWN/AMBIGUOUS)")
    print("-" * 80)
    print(f"{'Rule ID':>8}  {'Count':>6}  Description")
    print("-" * 80)
    for (rule_id, desc), count in stats['unknown_rule_descriptions'].most_common(20):
        # Handle None descriptions
        desc = desc or "No description"
        # Truncate long descriptions
        desc_short = (desc[:60] + '...') if len(desc) > 60 else desc
        print(f"{rule_id:>8}  {count:>6}  {desc_short}")
    print()

    print("RULE-LEVEL MAPPING COVERAGE")
    print("-" * 80)
    print(f"{'Rule ID':>8}  {'Total':>6}  {'Mapped':>6}  {'Unmapped':>8}  {'%Mapped':>8}")
    print("-" * 80)

    rule_coverage = {}
    for rule_id in stats['unique_rules']:
        rule_alerts = stats['alerts_by_rule'][rule_id]
        mapped = sum(1 for item in rule_alerts
                    if resolve_mitre(item['alert']).provenance not in (PROVENANCE_UNKNOWN, PROVENANCE_AMBIGUOUS))
        unmapped = len(rule_alerts) - mapped
        pct_mapped = (mapped / len(rule_alerts) * 100) if rule_alerts else 0
        rule_coverage[rule_id] = {
            'total': len(rule_alerts),
            'mapped': mapped,
            'unmapped': unmapped,
            'pct': pct_mapped
        }

    # Sort by total count descending
    for rule_id in sorted(rule_coverage.keys(), key=lambda r: rule_coverage[r]['total'], reverse=True):
        cov = rule_coverage[rule_id]
        print(f"{rule_id:>8}  {cov['total']:>6}  {cov['mapped']:>6}  {cov['unmapped']:>8}  {cov['pct']:>7.1f}%")
    print()

    if stats['resolved_techniques']:
        print("RESOLVED TECHNIQUE FREQUENCY")
        print("-" * 80)
        for tech_id, count in stats['resolved_techniques'].most_common(20):
            print(f"  {tech_id:<15} {count:>5}")
        print()

    # Calculate summary metrics
    native = stats['by_provenance'].get('NATIVE_WAZUH', 0)
    reviewed = stats['by_provenance'].get('REVIEWED_RULE_MAPPING', 0)
    curated = stats['by_provenance'].get('CURATED_WAZUH_MAPPING', 0)
    inferred = stats['by_provenance'].get('DETERMINISTIC_INFERENCE', 0)
    unknown = stats['by_provenance'].get('UNKNOWN', 0)
    ambiguous = stats['by_provenance'].get('AMBIGUOUS', 0)

    print("BASELINE SUMMARY")
    print("=" * 80)
    print(f"Native Wazuh Mappings:        {native:>6}  ({native/total*100:>5.1f}%)")
    print(f"Reviewed Registry Mappings:   {reviewed:>6}  ({reviewed/total*100:>5.1f}%)")
    print(f"Curated Wazuh Mappings:       {curated:>6}  ({curated/total*100:>5.1f}%)")
    print(f"Deterministic Inference:      {inferred:>6}  ({inferred/total*100:>5.1f}%)")
    print(f"UNKNOWN:                      {unknown:>6}  ({unknown/total*100:>5.1f}%)")
    print(f"AMBIGUOUS:                    {ambiguous:>6}  ({ambiguous/total*100:>5.1f}%)")
    print(f"Total:                        {total:>6}  (100.0%)")
    print("=" * 80)
    print()

    # Return structured results for programmatic access
    return {
        'total': total,
        'native': native,
        'reviewed': reviewed,
        'curated': curated,
        'inferred': inferred,
        'unknown': unknown,
        'ambiguous': ambiguous,
        'unique_rules': len(stats['unique_rules']),
        'top_unmapped_rules': stats['unknown_rule_descriptions'].most_common(10)
    }

def main():
    print("Extracting alerts from Neo4j...")
    alerts = extract_alerts_from_neo4j()

    if not alerts:
        print("No alerts found in Neo4j. Cannot establish baseline.")
        return

    print(f"Extracted {len(alerts)} alerts.")
    print("Evaluating resolver...")
    print()

    stats = evaluate_resolver(alerts)
    summary = print_baseline_report(stats)

    # Save baseline snapshot for comparison
    baseline_file = 'mitre_baseline_snapshot.json'
    with open(baseline_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Baseline snapshot saved to: {baseline_file}")

if __name__ == '__main__':
    main()

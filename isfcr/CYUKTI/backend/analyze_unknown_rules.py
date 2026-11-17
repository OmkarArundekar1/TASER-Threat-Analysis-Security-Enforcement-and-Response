#!/usr/bin/env python3
"""Detailed analysis of UNKNOWN rules for mapping candidates"""
import json
from collections import defaultdict
from neo4j_client import driver
from mitre_resolver import resolve_mitre, PROVENANCE_UNKNOWN

def analyze_unknown_rules():
    """Extract detailed information about UNKNOWN rules."""
    with driver.session() as session:
        result = session.run('''
            MATCH (ae:AttackEvent)
            RETURN ae.rule_id as rule_id,
                   ae.investigation_payload as payload
        ''')

        rule_details = defaultdict(lambda: {
            'count': 0,
            'descriptions': set(),
            'groups': set(),
            'levels': set(),
            'sample_payloads': []
        })

        for record in result:
            try:
                raw_alert = json.loads(record['payload'])
                alert = {
                    'rule': {
                        'id': raw_alert.get('rule_id'),
                        'description': raw_alert.get('rule_description'),
                        'level': raw_alert.get('rule_level'),
                        'groups': raw_alert.get('rule_groups', []),
                    },
                    'agent': raw_alert.get('agent', {}),
                }

                if 'mitre' in raw_alert:
                    alert['rule']['mitre'] = raw_alert['mitre']

                resolution = resolve_mitre(alert)
                if resolution.provenance == PROVENANCE_UNKNOWN:
                    rule_id = alert['rule']['id']
                    details = rule_details[rule_id]
                    details['count'] += 1

                    if alert['rule']['description']:
                        details['descriptions'].add(alert['rule']['description'])
                    if alert['rule']['level']:
                        details['levels'].add(alert['rule']['level'])
                    if alert['rule']['groups']:
                        details['groups'].update(alert['rule']['groups'])

                    if len(details['sample_payloads']) < 2:
                        details['sample_payloads'].append(raw_alert)

            except Exception as e:
                pass

        # Sort by count descending
        sorted_rules = sorted(rule_details.items(), key=lambda x: x[1]['count'], reverse=True)

        print("=" * 100)
        print("DETAILED ANALYSIS OF UNKNOWN RULES")
        print("=" * 100)
        print()

        for rule_id, details in sorted_rules[:15]:
            print(f"Rule ID: {rule_id}")
            print(f"Count: {details['count']} alerts")
            print(f"Descriptions: {', '.join(sorted(details['descriptions']))}")
            print(f"Groups: {', '.join(sorted(details['groups']))}")
            print(f"Levels: {', '.join(map(str, sorted(details['levels'])))}")

            if details['sample_payloads']:
                print(f"\nSample alert:")
                sample = details['sample_payloads'][0]
                print(f"  rule_description: {sample.get('rule_description', 'N/A')}")
                print(f"  rule_groups: {sample.get('rule_groups', [])}")
                print(f"  rule_level: {sample.get('rule_level', 'N/A')}")

                # Show relevant context fields
                if 'suricata' in sample and sample['suricata']:
                    print(f"  suricata data: {sample['suricata']}")
                if 'data' in sample and sample['data']:
                    print(f"  data: {sample['data']}")

            print()
            print("-" * 100)
            print()

        return rule_details

if __name__ == '__main__':
    analyze_unknown_rules()

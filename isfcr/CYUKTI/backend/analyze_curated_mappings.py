#!/usr/bin/env python3
"""Analyze which alerts are being mapped by CURATED_WAZUH_MAPPING tier"""
import json
from collections import Counter
from neo4j_client import driver
from mitre_resolver import resolve_mitre, PROVENANCE_CURATED_WAZUH_MAPPING

def analyze_curated_mappings():
    with driver.session() as session:
        result = session.run('''
            MATCH (ae:AttackEvent)
            RETURN ae.rule_id as rule_id,
                   ae.investigation_payload as payload
        ''')

        curated_rules = Counter()
        curated_techniques = Counter()

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
                if resolution.provenance == PROVENANCE_CURATED_WAZUH_MAPPING:
                    rule_id = alert['rule']['id']
                    curated_rules[rule_id] += 1
                    for tech_id in resolution.technique_ids:
                        curated_techniques[tech_id] += 1

            except Exception as e:
                pass

        print("CURATED_WAZUH_MAPPING Analysis")
        print("=" * 80)
        print("\nRules mapped by curated tier:")
        for rule_id, count in curated_rules.most_common():
            print(f"  Rule {rule_id}: {count} alerts")

        print("\nTechniques assigned by curated tier:")
        for tech_id, count in curated_techniques.most_common():
            print(f"  {tech_id}: {count} alerts")

        return curated_rules

if __name__ == '__main__':
    analyze_curated_mappings()

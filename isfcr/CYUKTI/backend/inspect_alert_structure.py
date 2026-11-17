#!/usr/bin/env python3
"""Inspect actual alert structure in investigation_payload"""
import json
from neo4j_client import driver

with driver.session() as session:
    result = session.run('''
        MATCH (ae:AttackEvent)
        RETURN ae.event_id as event_id,
               ae.rule_id as rule_id,
               ae.investigation_payload as payload
        LIMIT 5
    ''')

    for i, record in enumerate(result, 1):
        print(f"\n{'='*80}")
        print(f"Sample Alert {i}")
        print(f"{'='*80}")
        print(f"Event ID: {record['event_id']}")
        print(f"Stored rule_id: {record['rule_id']}")

        payload = record['payload']
        if payload:
            try:
                alert = json.loads(payload)
                print(f"\nAlert JSON structure:")
                print(f"  Keys at root: {list(alert.keys())}")

                if 'rule' in alert:
                    print(f"  rule.keys(): {list(alert['rule'].keys())}")
                    print(f"  rule.id: {alert['rule'].get('id', 'MISSING')}")
                    print(f"  rule.description: {alert['rule'].get('description', 'MISSING')[:100]}")
                    if 'mitre' in alert['rule']:
                        print(f"  rule.mitre: {alert['rule']['mitre']}")

                # Print full alert for first one
                if i == 1:
                    print(f"\nFull alert JSON:")
                    print(json.dumps(alert, indent=2)[:1000])
            except Exception as e:
                print(f"Error parsing payload: {e}")

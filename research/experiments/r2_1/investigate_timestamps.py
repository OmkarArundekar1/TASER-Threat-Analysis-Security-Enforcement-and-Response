#!/usr/bin/env python3
"""
R2.1 Timestamp Investigation
============================
Investigates timestamp types, values, and calculation issues from R2.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'isfcr', 'CYUKTI', 'backend'))

import json
from datetime import datetime, timezone
from neo4j_client import driver

def inspect_timestamp_types():
    """Inspect actual timestamp types returned by Neo4j."""
    with driver.session() as session:
        result = session.run('''
            MATCH (ae:AttackEvent)
            RETURN ae.event_id as event_id,
                   ae.first_seen as first_seen,
                   ae.campaign_id as campaign_id
            ORDER BY ae.first_seen
            LIMIT 5
        ''')

        print("=" * 80)
        print("TIMESTAMP TYPE INSPECTION")
        print("=" * 80)

        for record in result:
            first_seen = record['first_seen']
            print(f"\nEvent: {record['event_id']}")
            print(f"  Type: {type(first_seen)}")
            print(f"  Value: {first_seen}")
            print(f"  Has iso_format: {hasattr(first_seen, 'iso_format')}")
            print(f"  Has isoformat: {hasattr(first_seen, 'isoformat')}")
            print(f"  Has __sub__: {hasattr(first_seen, '__sub__')}")

            if hasattr(first_seen, 'iso_format'):
                print(f"  iso_format(): {first_seen.iso_format()}")
            elif hasattr(first_seen, 'isoformat'):
                print(f"  isoformat(): {first_seen.isoformat()}")

def calculate_dataset_duration():
    """Calculate actual dataset duration with proper timestamp handling."""
    with driver.session() as session:
        # Get all timestamps
        result = session.run('''
            MATCH (ae:AttackEvent)
            WHERE ae.first_seen IS NOT NULL
            RETURN ae.first_seen as timestamp
            ORDER BY ae.first_seen
        ''')

        timestamps = []
        for record in result:
            ts = record['timestamp']
            # Convert Neo4j DateTime to Python datetime
            if hasattr(ts, 'to_native'):
                ts = ts.to_native()
            timestamps.append(ts)

        if not timestamps:
            print("\nNo valid timestamps found")
            return None

        start_ts = timestamps[0]
        end_ts = timestamps[-1]

        print("\n" + "=" * 80)
        print("DATASET DURATION CALCULATION")
        print("=" * 80)
        print(f"Start timestamp: {start_ts}")
        print(f"  Type: {type(start_ts)}")
        print(f"End timestamp: {end_ts}")
        print(f"  Type: {type(end_ts)}")

        # Calculate duration
        try:
            duration = end_ts - start_ts
            print(f"\nDuration object: {duration}")
            print(f"  Type: {type(duration)}")

            # Handle different duration types
            if hasattr(duration, 'total_seconds'):
                total_seconds = duration.total_seconds()
            elif hasattr(duration, 'seconds'):
                # Neo4j Duration object
                total_seconds = duration.seconds
                if hasattr(duration, 'days'):
                    total_seconds += duration.days * 86400
            else:
                print(f"  Unknown duration type, cannot calculate seconds")
                total_seconds = None

            if total_seconds is not None:
                hours = total_seconds / 3600
                days = total_seconds / 86400

                print(f"\nTotal seconds: {total_seconds}")
                print(f"Total hours: {hours:.2f}")
                print(f"Total days: {days:.2f}")

                return {
                    'start': str(start_ts),
                    'end': str(end_ts),
                    'duration_seconds': total_seconds,
                    'duration_hours': hours,
                    'duration_days': days
                }
        except Exception as e:
            print(f"\nError calculating duration: {e}")
            import traceback
            traceback.print_exc()

        return None

def calculate_campaign_durations():
    """Calculate duration for each campaign with proper timestamp handling."""
    with driver.session() as session:
        result = session.run('''
            MATCH (c:Campaign)
            WHERE c.first_seen IS NOT NULL AND c.last_seen IS NOT NULL
            RETURN c.campaign_id as campaign_id,
                   c.first_seen as first_seen,
                   c.last_seen as last_seen,
                   c.status as status
            ORDER BY c.first_seen
        ''')

        durations = []
        campaigns_with_duration = 0
        campaigns_zero_duration = 0
        campaigns_invalid = 0

        print("\n" + "=" * 80)
        print("CAMPAIGN DURATION CALCULATION")
        print("=" * 80)

        for record in result:
            campaign_id = record['campaign_id']
            first_seen = record['first_seen']
            last_seen = record['last_seen']

            # Convert Neo4j DateTime to Python datetime
            if hasattr(first_seen, 'to_native'):
                first_seen = first_seen.to_native()
            if hasattr(last_seen, 'to_native'):
                last_seen = last_seen.to_native()

            try:
                duration = last_seen - first_seen

                # Extract seconds
                if hasattr(duration, 'total_seconds'):
                    total_seconds = duration.total_seconds()
                elif hasattr(duration, 'seconds'):
                    total_seconds = duration.seconds
                    if hasattr(duration, 'days'):
                        total_seconds += duration.days * 86400
                else:
                    campaigns_invalid += 1
                    continue

                duration_hours = total_seconds / 3600
                durations.append(duration_hours)

                if duration_hours == 0:
                    campaigns_zero_duration += 1
                else:
                    campaigns_with_duration += 1

            except Exception as e:
                campaigns_invalid += 1
                print(f"  Campaign {campaign_id}: Error calculating duration: {e}")

        if durations:
            durations_sorted = sorted(durations)
            mean_duration = sum(durations) / len(durations)
            median_duration = durations_sorted[len(durations) // 2]

            print(f"\nCampaigns with valid timestamps: {len(durations)}")
            print(f"Campaigns with non-zero duration: {campaigns_with_duration}")
            print(f"Campaigns with zero duration: {campaigns_zero_duration}")
            print(f"Campaigns with invalid timestamps: {campaigns_invalid}")
            print(f"\nDuration Statistics:")
            print(f"  Min: {min(durations):.2f} hours")
            print(f"  Max: {max(durations):.2f} hours")
            print(f"  Mean: {mean_duration:.2f} hours")
            print(f"  Median: {median_duration:.2f} hours")

            return {
                'total_campaigns': len(durations),
                'non_zero_duration': campaigns_with_duration,
                'zero_duration': campaigns_zero_duration,
                'invalid': campaigns_invalid,
                'min_hours': min(durations),
                'max_hours': max(durations),
                'mean_hours': mean_duration,
                'median_hours': median_duration
            }
        else:
            print("\nNo valid campaign durations found")
            return None

if __name__ == '__main__':
    inspect_timestamp_types()
    dataset_duration = calculate_dataset_duration()
    campaign_durations = calculate_campaign_durations()

    # Save results
    results = {
        'dataset_duration': dataset_duration,
        'campaign_durations': campaign_durations
    }

    output_file = 'timestamp_investigation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

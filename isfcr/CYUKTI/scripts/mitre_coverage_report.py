"""
mitre_coverage_report.py
============================
Phase 20 diagnostic: runs mitre_resolver.resolve_mitre() against every
alert currently in Wazuh's alerts.json and reports the resulting
provenance/confidence distribution. Purely read-only -- does not write
to Neo4j, does not modify alerts.json, does not touch the CYUKTI
listener's offset.dat.

Answers:
  - how many alerts have native/reviewed/inferred/unknown/ambiguous MITRE resolution
  - which rules generate the most UNKNOWN alerts
  - which techniques are actually being resolved
  - which resolved techniques have no MITRE_TO_STAGE entry (a separate,
    pre-existing concern this script surfaces but does not fix)
"""

import json
import os
import sys
from collections import Counter, defaultdict

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from mitre_resolver import resolve_mitre  # noqa: E402
from mitre_mapper import MITRE_TO_STAGE  # noqa: E402

ALERTS_FILE = os.environ.get("WAZUH_ALERT_FILE", "/var/ossec/logs/alerts/alerts.json")


def main():
    provenance_counts = Counter()
    confidence_counts = Counter()
    unknown_rules = Counter()
    resolved_techniques = Counter()
    rule_seen = Counter()
    resolved_techniques_missing_stage = set()
    total = 0
    parse_errors = 0

    with open(ALERTS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                alert = json.loads(line)
            except Exception:
                parse_errors += 1
                continue

            total += 1
            rule = alert.get("rule") or {}
            rule_id = rule.get("id")
            rule_desc = rule.get("description")
            rule_seen[(rule_id, rule_desc)] += 1

            resolution = resolve_mitre(alert)
            provenance_counts[resolution.provenance] += 1
            confidence_counts[resolution.confidence] += 1

            if resolution.provenance in ("UNKNOWN", "AMBIGUOUS"):
                unknown_rules[(rule_id, rule_desc)] += 1
            else:
                for technique_id in resolution.technique_ids:
                    resolved_techniques[technique_id] += 1
                    if technique_id not in MITRE_TO_STAGE:
                        resolved_techniques_missing_stage.add(technique_id)

    print("=" * 100)
    print(f"MITRE resolution coverage report -- {ALERTS_FILE}")
    print("=" * 100)
    print(f"Total alerts examined: {total}  (JSON parse errors skipped: {parse_errors})")
    print()

    print("Provenance distribution:")
    for provenance, count in provenance_counts.most_common():
        pct = 100.0 * count / total if total else 0.0
        print(f"  {provenance:<24} {count:>5}  ({pct:5.1f}%)")
    print()

    print("Confidence distribution:")
    for confidence, count in confidence_counts.most_common():
        print(f"  {confidence:<12} {count}")
    print()

    print("Rule-level mapping coverage (all rules seen):")
    for (rule_id, desc), count in rule_seen.most_common():
        status = "UNKNOWN/AMBIGUOUS" if (rule_id, desc) in unknown_rules else "resolved"
        print(f"  {rule_id:>8}  x{count:<5}  [{status}]  {desc}")
    print()

    print("Top rules generating UNKNOWN/AMBIGUOUS alerts:")
    for (rule_id, desc), count in unknown_rules.most_common(20):
        print(f"  {rule_id:>8}  x{count:<5}  {desc}")
    print()

    print("Resolved technique frequency:")
    for technique_id, count in resolved_techniques.most_common():
        stage_note = "" if technique_id in MITRE_TO_STAGE else "  <-- NOT in MITRE_TO_STAGE (tps=0)"
        print(f"  {technique_id:<12} x{count:<5}{stage_note}")
    print()

    if resolved_techniques_missing_stage:
        print("Resolved techniques with NO MITRE_TO_STAGE entry (documented, not fixed by this script):")
        for technique_id in sorted(resolved_techniques_missing_stage):
            print(f"  {technique_id}")
    else:
        print("All resolved techniques have a MITRE_TO_STAGE entry.")


if __name__ == "__main__":
    main()

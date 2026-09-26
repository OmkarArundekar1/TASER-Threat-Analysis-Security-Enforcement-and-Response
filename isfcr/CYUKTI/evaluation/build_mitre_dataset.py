"""
build_mitre_dataset.py
=========================
Scans the REAL Wazuh alert archive (current alerts.json + the rotated
2026/<Month>/ossec-alerts-*.json archive) for alerts whose rule.id
matches one of the pre-declared AttackScenario rule families, and
builds an AUTO_PROPOSED ground-truth dataset from them -- independent
of mitre_resolver.py (this script never imports it; only run_all.py /
mitre_eval.py call the real resolver, at evaluation time, on the frozen
raw alert this script saves).

Usage:
    cd evaluation && python build_mitre_dataset.py [--limit-per-scenario N]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ground_truth import raw_alerts, store
from ground_truth.builder import build_from_raw_alert
from scenarios.registry import load_registry

ALERTS_ROOT = "/var/ossec/logs/alerts"

# rule_id -> scenario_id, built directly from MITRE_MAPPING.md's documented
# rule-ID-to-technique-family fixes -- independent of mitre_resolver.py's
# own internal registry (this dict is hand-transcribed from the markdown
# doc, not imported from mitre_rule_registry.py).
RULE_ID_TO_SCENARIO = {
    "100500": "SCN-NMAP-001",
    "100510": "SCN-SSHBRUTE-001",
    "210010": "SCN-SSHBRUTE-001",
    "210011": "SCN-SSHBRUTE-001",
    "100100": "SCN-SSHBRUTE-001",
    "100200": "SCN-SSHBRUTE-001",
    "5503": "SCN-SSHBRUTE-001",
    "5760": "SCN-SSHBRUTE-001",
    "100513": "SCN-DOS-001",
    "210013": "SCN-DOS-001",
}


def _iter_alert_files():
    current = os.path.join(ALERTS_ROOT, "alerts.json")
    if os.path.exists(current):
        yield current
    archive_root = os.path.join(ALERTS_ROOT, "2026")
    if os.path.isdir(archive_root):
        for month_dir, _, files in os.walk(archive_root):
            for fname in sorted(files):
                if fname.endswith(".json") or fname.endswith(".json.gz"):
                    yield os.path.join(month_dir, fname)


def _open_maybe_gz(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def scan(limit_per_scenario: int = 25) -> dict:
    """Returns {scenario_id: [alert_dict, ...]}, capped per scenario."""
    found: dict[str, list] = {sid: [] for sid in set(RULE_ID_TO_SCENARIO.values())}
    files_scanned = 0
    lines_scanned = 0
    parse_errors = 0

    for path in _iter_alert_files():
        files_scanned += 1
        try:
            with _open_maybe_gz(path) as f:
                for line in f:
                    lines_scanned += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        alert = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        parse_errors += 1
                        continue
                    rule_id = str((alert.get("rule") or {}).get("id", ""))
                    scenario_id = RULE_ID_TO_SCENARIO.get(rule_id)
                    if scenario_id and len(found[scenario_id]) < limit_per_scenario:
                        found[scenario_id].append(alert)
        except (OSError, PermissionError) as e:
            print(f"  [skip] could not read {path}: {e}")
            continue

    print(f"Scanned {files_scanned} files, {lines_scanned} lines, {parse_errors} parse errors.")
    for sid, alerts in found.items():
        print(f"  {sid}: {len(alerts)} matching alerts found")
    return found


def build(limit_per_scenario: int = 25, dataset_name: str = "mitre_mapping_v0"):
    registry_by_id = {s.scenario_id: s for s in load_registry()}
    found = scan(limit_per_scenario)

    records = []
    raw_by_sample_id = {}
    for scenario_id, alerts in found.items():
        scenario = registry_by_id.get(scenario_id)
        if scenario is None:
            continue
        for i, alert in enumerate(alerts):
            sample_id = f"{scenario_id}-{i:03d}"
            record = build_from_raw_alert(scenario, alert, sample_id=sample_id)
            records.append(record)
            raw_by_sample_id[sample_id] = alert

    if not records:
        print("No matching alerts found in the archive for any registered scenario. "
              "Nothing built -- reporting UNMEASURABLE is the honest outcome here, not fabricating a dataset.")
        return

    store.save_records(dataset_name, records)
    raw_alerts.save_raw_alerts(dataset_name, raw_by_sample_id)
    print(f"Built {len(records)} AUTO_PROPOSED ground-truth records -> ground_truth/provisional/{dataset_name}.jsonl")
    print(f"Raw alerts saved -> ground_truth/raw/{dataset_name}_alerts.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-scenario", type=int, default=25)
    args = parser.parse_args()
    build(limit_per_scenario=args.limit_per_scenario)

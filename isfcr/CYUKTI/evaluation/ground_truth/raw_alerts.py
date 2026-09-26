"""
ground_truth/raw_alerts.py
=============================
A tiny companion store: sample_id -> the frozen raw Wazuh alert dict a
GroundTruthRecord was built from. Kept separate from the
GroundTruthRecord itself (which holds only the label + provenance) so
the raw evidence is trivially inspectable/auditable without parsing
JSONL rows meant for the evaluator.
"""

from __future__ import annotations

import json
import os

_RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def save_raw_alerts(dataset_name: str, alerts_by_sample_id: dict) -> str:
    os.makedirs(_RAW_DIR, exist_ok=True)
    path = os.path.join(_RAW_DIR, f"{dataset_name}_alerts.json")
    with open(path, "w") as f:
        json.dump(alerts_by_sample_id, f, indent=2)
    return path


def load_raw_alerts(dataset_name: str) -> dict:
    path = os.path.join(_RAW_DIR, f"{dataset_name}_alerts.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

"""
review/compute_threat_qualification_adjudication_metrics.py
===============================================================
Recomputes the threat-qualification evaluation using the independent
adjudication labels documented in
review/evaluation_threat_qualification_adjudication.md, against the
same live CYUKTI cti_score-derived classification the original
preliminary evaluation used. This is the exact, reproducible source of
every number in that report's Sections 6-7 -- run it to verify them,
not just read them.

IMPORTANT: `ADJUDICATED` below is MEASURED_ASSISTANT_ADJUDICATED, not
MEASURED_HUMAN_REVIEWED. It was produced by this AI assistant reading
real repository/Neo4j evidence and real MITRE ATT&CK / Atomic Red Team
reference material (see the full report for citations and per-record
rationale) -- it is NOT independent human ground truth and must not be
presented as such.

Usage: cd evaluation && python review/compute_threat_qualification_adjudication_metrics.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from evaluation_metrics import classification_report, confusion_matrix, false_positive_negative_rates

# The 24 real campaigns, grouped exactly as they appear in
# threat_qualification_v0.jsonl (scenario_id -> raw_event_id).
RECORDS = [
    "CAMP_1627943B", "CAMP_BB9A0F28", "CAMP_0E9C0284", "CAMP_C6E494CF", "CAMP_75FBEEBC", "CAMP_DDBFC6D9",
    "CAMP_10407C1A", "CAMP_D8605E81", "CAMP_DAE35509", "CAMP_E449E81D", "CAMP_BB680773", "CAMP_E8E9F042",
    "CAMP_23FB320A", "CAMP_1429ADB4", "CAMP_F91A7652", "CAMP_066240B7",
    "CAMP_947A7084", "CAMP_67FCB361", "CAMP_427A075C", "CAMP_99EE7A64", "CAMP_86D5837D", "CAMP_15E98B82",
    "CAMP_B177634A", "CAMP_1B8E9033",
]

# The original AI-proposed, per-scenario provisional label (unchanged --
# reused for the "before" comparison, never altered by this script).
PROVISIONAL = {
    **{cid: "SUSPICIOUS" for cid in RECORDS[:12]},
    "CAMP_23FB320A": "QUALIFIED_THREAT", "CAMP_1429ADB4": "QUALIFIED_THREAT",
    "CAMP_F91A7652": "QUALIFIED_THREAT", "CAMP_066240B7": "QUALIFIED_THREAT",
    **{cid: "SUSPICIOUS" for cid in RECORDS[16:]},
}

# CYUKTI's real, live classification -- re-derived from cti_score via
# cti_confidence_engine.py's own published thresholds (see
# threat_qualification_eval.py). Frozen here as the values observed on
# 2026-09-26 (evaluation/results/threat_qualification_adjudication_evidence.json);
# re-run threat_qualification_eval.py against live Neo4j to reproduce.
CYUKTI = {
    "CAMP_1627943B": "QUALIFIED_THREAT", "CAMP_BB9A0F28": "QUALIFIED_THREAT", "CAMP_0E9C0284": "QUALIFIED_THREAT",
    "CAMP_C6E494CF": "QUALIFIED_THREAT", "CAMP_75FBEEBC": "QUALIFIED_THREAT", "CAMP_DDBFC6D9": "QUALIFIED_THREAT",
    "CAMP_10407C1A": "QUALIFIED_THREAT", "CAMP_D8605E81": "QUALIFIED_THREAT", "CAMP_DAE35509": "QUALIFIED_THREAT",
    "CAMP_E449E81D": "QUALIFIED_THREAT", "CAMP_BB680773": "QUALIFIED_THREAT", "CAMP_E8E9F042": "QUALIFIED_THREAT",
    "CAMP_23FB320A": "SUSPICIOUS", "CAMP_1429ADB4": "QUALIFIED_THREAT", "CAMP_F91A7652": "QUALIFIED_THREAT",
    "CAMP_066240B7": "SUSPICIOUS",
    "CAMP_947A7084": "QUALIFIED_THREAT", "CAMP_67FCB361": "QUALIFIED_THREAT", "CAMP_427A075C": "QUALIFIED_THREAT",
    "CAMP_99EE7A64": "QUALIFIED_THREAT", "CAMP_86D5837D": "QUALIFIED_THREAT", "CAMP_15E98B82": "QUALIFIED_THREAT",
    "CAMP_B177634A": "QUALIFIED_THREAT", "CAMP_1B8E9033": "QUALIFIED_THREAT",
}

# MEASURED_ASSISTANT_ADJUDICATED -- see the full report for the
# per-record evidence and rationale behind each of these.
ADJUDICATED = {
    "CAMP_1627943B": "QUALIFIED_THREAT",   # brute force + T1078 (SSH auth success) same attacker, same session
    "CAMP_BB9A0F28": "SUSPICIOUS",          # brute-force attempts only, no success indicator
    "CAMP_0E9C0284": "QUALIFIED_THREAT",    # brute force + T1078 success indicator
    "CAMP_C6E494CF": "SUSPICIOUS",          # brute-force attempts only
    "CAMP_75FBEEBC": "QUALIFIED_THREAT",    # sustained brute force (59-occurrence burst) + T1078 x2, reopened 5x
    "CAMP_DDBFC6D9": "QUALIFIED_THREAT",    # brute force + T1078 success indicator
    "CAMP_10407C1A": "QUALIFIED_THREAT",    # NOT actually brute force -- real exploit chain (T1595.002/T1210/T1055/T1190); mislabeled by session-boundary grouping
    "CAMP_D8605E81": "SUSPICIOUS",          # brute-force attempts only (13+1 occurrences), no success indicator
    "CAMP_DAE35509": "SUSPICIOUS",          # minimal brute-force attempts, no success indicator
    "CAMP_E449E81D": "SUSPICIOUS",          # single brute-force attempt event
    "CAMP_BB680773": "QUALIFIED_THREAT",    # brute force + T1078, NATIVE_WAZUH/CONFIRMED provenance (post rule-fix)
    "CAMP_E8E9F042": "SUSPICIOUS",          # brute-force attempts only
    "CAMP_23FB320A": "INSUFFICIENT_EVIDENCE",  # single T1114 event, risk_score=0, no corroborating context
    "CAMP_1429ADB4": "QUALIFIED_THREAT",    # rich, corroborated multi-stage exploitation chain, risk_score=34820
    "CAMP_F91A7652": "SUSPICIOUS",          # single, weak T1110.001 event (occurrence=1)
    "CAMP_066240B7": "INSUFFICIENT_EVIDENCE",  # single T1078 event, no preceding brute-force/scan context -- MITRE itself says valid-account use alone is ambiguous
    "CAMP_947A7084": "SUSPICIOUS",          # reconnaissance only (T1595 is Reconnaissance tactic, not compromise)
    "CAMP_67FCB361": "SUSPICIOUS",
    "CAMP_427A075C": "SUSPICIOUS",          # sustained/repeated scanning, still reconnaissance tactic only
    "CAMP_99EE7A64": "SUSPICIOUS",
    "CAMP_86D5837D": "SUSPICIOUS",
    "CAMP_15E98B82": "SUSPICIOUS",
    "CAMP_B177634A": "SUSPICIOUS",
    "CAMP_1B8E9033": "SUSPICIOUS",
}


def _print_report(name, y_true, y_pred, labels):
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"--- {name} ---")
    print("accuracy:", report["accuracy"], "macro_f1:", report["macro_f1"], "weighted_f1:", report["weighted_f1"])
    print("per_label:", json.dumps(report["per_label"], indent=2))
    print("confusion matrix:", json.dumps(matrix, indent=2))
    print()


if __name__ == "__main__":
    agree_orig = sum(1 for r in RECORDS if PROVISIONAL[r] == CYUKTI[r])
    print(f"Sanity check -- original provisional-vs-CYUKTI agreement: {agree_orig}/{len(RECORDS)} = {agree_orig/len(RECORDS):.1%}")

    agree_adj = sum(1 for r in RECORDS if ADJUDICATED[r] == CYUKTI[r])
    print(f"Adjudicated-vs-CYUKTI agreement: {agree_adj}/{len(RECORDS)} = {agree_adj/len(RECORDS):.1%}")
    print()

    _print_report(
        "Evaluation A: 3-class (adjudicated = reference)",
        [ADJUDICATED[r] for r in RECORDS], [CYUKTI[r] for r in RECORDS],
        ["NOT_THREAT", "SUSPICIOUS", "QUALIFIED_THREAT", "INSUFFICIENT_EVIDENCE"],
    )

    binary_records = [r for r in RECORDS if ADJUDICATED[r] != "INSUFFICIENT_EVIDENCE"]
    to_binary = lambda label: "THREAT" if label == "QUALIFIED_THREAT" else "NON_THREAT"
    y_true_bin = [to_binary(ADJUDICATED[r]) for r in binary_records]
    y_pred_bin = [to_binary(CYUKTI[r]) for r in binary_records]
    print(f"Evaluation B: binary (n={len(binary_records)}, {len(RECORDS) - len(binary_records)} INSUFFICIENT_EVIDENCE excluded)")
    _print_report("Evaluation B: binary", y_true_bin, y_pred_bin, ["NON_THREAT", "THREAT"])
    print("false_positive_negative_rates (THREAT positive class):",
          false_positive_negative_rates(y_true_bin, y_pred_bin, "THREAT"))

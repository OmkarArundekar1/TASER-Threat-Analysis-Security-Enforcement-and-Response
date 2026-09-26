"""
review/export_review_queue.py
================================
Exports human-fillable review queues (Phase 11) -- the mechanism that
actually unblocks the NOT_MEASURED items in results/summary.md. Never
pre-fills a verdict/relevance judgment itself; only assembles the real
evidence a reviewer needs to make one, in a structured, low-friction
format (CSV).

Usage:
    cd evaluation && python review/export_review_queue.py --which investigation
    cd evaluation && python review/export_review_queue.py --which rag --source mitre_semantic
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

_OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# The 3 real Phase 21 investigations -- see review/phase21_real_investigation_validation.md.
# Frozen, historical, quoted verbatim (not re-derived) so exporting this queue never touches
# that file or re-runs the investigation.
_PHASE21_INVESTIGATIONS = [
    {"investigation_id": "CAMP_427A075C", "attacker": "192.168.56.106", "victim": "192.168.56.105",
     "final_investigation_confidence": 0.1835, "top_hypothesis_technique": "T1595",
     "evidence_summary": "Reconnaissance-only campaign (T1595), risk_score 8330. See "
                          "review/phase21_real_investigation_validation.md Section 7 for the full evidence trail."},
    {"investigation_id": "CAMP_1429ADB4", "attacker": "192.168.56.105", "victim": "pes1ug23cs411-VirtualBox",
     "final_investigation_confidence": 0.0687, "top_hypothesis_technique": "T1210",
     "evidence_summary": "Mixed exploitation campaign (T1055/T1059/T1059.007/T1190/T1210/T1595.002), "
                          "risk_score 34820. See review/phase21_real_investigation_validation.md Section 7."},
    {"investigation_id": "CAMP_D8605E81", "attacker": "192.168.56.106", "victim": "pes1ug23cs411-VirtualBox",
     "final_investigation_confidence": 0.0963, "top_hypothesis_technique": "T1110",
     "evidence_summary": "Brute-force campaign (T1110/T1110.001), risk_score 630. "
                          "See review/phase21_real_investigation_validation.md Section 7."},
]

# Real, observed-in-this-project candidate query texts, grounded in the actual
# scenarios this lab has run (evaluation/scenarios/scenarios.json) -- NOT
# pre-judged for relevance; a reviewer fills in relevant_ids after actually
# running each query against the retriever and inspecting its results.
_CANDIDATE_QUERIES = {
    "mitre_semantic": [
        "network reconnaissance active scanning technique",
        "SSH brute force repeated failed authentication",
        "service exhaustion denial of service flood",
        "process injection privilege escalation",
        "command and scripting interpreter JavaScript execution",
    ],
    "campaign_narrative": [
        "campaign targeting pes1ug23cs411-VirtualBox from 192.168.56.106",
        "campaign with mixed exploitation techniques and high risk score",
        "repeated brute-force campaign against the same victim host",
    ],
    "gnn_topology": [
        "campaign topologically similar to CAMP_D8605E81 (brute force)",
        "campaign topologically similar to CAMP_1429ADB4 (mixed exploitation)",
    ],
}


def export_investigation_queue() -> str:
    path = os.path.join(_OUT_DIR, "investigation_verdict_queue.csv")
    fields = ["investigation_id", "attacker", "victim", "final_investigation_confidence",
              "top_hypothesis_technique", "evidence_summary",
              "verdict (fill in: CORRECT|INCORRECT|PARTIALLY_CORRECT|INSUFFICIENT_EVIDENCE)", "reviewer"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in _PHASE21_INVESTIGATIONS:
            writer.writerow({
                **row,
                "verdict (fill in: CORRECT|INCORRECT|PARTIALLY_CORRECT|INSUFFICIENT_EVIDENCE)": "",
                "reviewer": "",
            })
    print(f"Wrote {len(_PHASE21_INVESTIGATIONS)} rows -> {path}")
    print("Fill in the verdict + reviewer columns, then run "
          "review/import_reviewed.py --which investigation to promote this into "
          "evaluation/labels/investigation_verdicts.json.")
    return path


def export_rag_queue(source: str) -> str:
    if source not in _CANDIDATE_QUERIES:
        raise ValueError(f"Unknown RAG source {source!r}; choose from {list(_CANDIDATE_QUERIES)}")
    path = os.path.join(_OUT_DIR, f"rag_query_queue_{source}.csv")
    fields = ["query_id", "query_text", "relevant_ids (fill in, comma-separated doc/campaign ids actually relevant)",
              "reviewer"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, text in enumerate(_CANDIDATE_QUERIES[source]):
            writer.writerow({
                "query_id": f"{source.upper()}-Q{i+1:02d}",
                "query_text": text,
                "relevant_ids (fill in, comma-separated doc/campaign ids actually relevant)": "",
                "reviewer": "",
            })
    print(f"Wrote {len(_CANDIDATE_QUERIES[source])} candidate queries -> {path}")
    print(f"For each query: run it against the real {source} retriever, inspect the results, fill in which "
          f"returned items are ACTUALLY relevant, then run review/import_reviewed.py --which rag --source {source}.")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["investigation", "rag"], required=True)
    parser.add_argument("--source", choices=list(_CANDIDATE_QUERIES), default=None)
    args = parser.parse_args()

    if args.which == "investigation":
        export_investigation_queue()
    else:
        if not args.source:
            for source in _CANDIDATE_QUERIES:
                export_rag_queue(source)
        else:
            export_rag_queue(args.source)

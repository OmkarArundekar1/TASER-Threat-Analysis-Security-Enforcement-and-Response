"""
review/import_reviewed.py
============================
Reads a filled-in review queue CSV (see export_review_queue.py) back
in and writes the HUMAN_REVIEWED artifact each evaluator expects.
Refuses rows missing a reviewer name or a verdict/relevance value --
promoting a row with a blank verdict would silently manufacture a
label, which this module will not do.

Usage:
    cd evaluation && python review/import_reviewed.py --which investigation
    cd evaluation && python review/import_reviewed.py --which rag --source mitre_semantic
"""

from __future__ import annotations

import argparse
import csv
import json
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_LABELS_DIR = os.path.join(os.path.dirname(_DIR), "labels")
_QUERIES_DIR = os.path.join(os.path.dirname(_DIR), "queries")


def import_investigation_verdicts() -> int:
    path = os.path.join(_DIR, "investigation_verdict_queue.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist -- run export_review_queue.py --which investigation first.")

    verdict_col = "verdict (fill in: CORRECT|INCORRECT|PARTIALLY_CORRECT|INSUFFICIENT_EVIDENCE)"
    accepted = []
    skipped = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            verdict = (row.get(verdict_col) or "").strip()
            reviewer = (row.get("reviewer") or "").strip()
            if not verdict or not reviewer:
                skipped += 1
                continue
            if verdict not in {"CORRECT", "INCORRECT", "PARTIALLY_CORRECT", "INSUFFICIENT_EVIDENCE"}:
                raise ValueError(f"Row {row['investigation_id']}: unrecognized verdict {verdict!r}")
            accepted.append({
                "investigation_id": row["investigation_id"],
                "verdict": verdict,
                "reviewer": reviewer,
                "investigation_confidence": float(row["final_investigation_confidence"]),
            })

    os.makedirs(_LABELS_DIR, exist_ok=True)
    out_path = os.path.join(_LABELS_DIR, "investigation_verdicts.json")
    with open(out_path, "w") as f:
        json.dump(accepted, f, indent=2)
    print(f"Imported {len(accepted)} reviewed verdicts ({skipped} rows skipped: no verdict/reviewer) -> {out_path}")
    return len(accepted)


def import_rag_queries(source: str) -> int:
    path = os.path.join(_DIR, f"rag_query_queue_{source}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist -- run export_review_queue.py --which rag --source {source} first.")

    relevant_col = "relevant_ids (fill in, comma-separated doc/campaign ids actually relevant)"
    accepted = []
    skipped = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            relevant_raw = (row.get(relevant_col) or "").strip()
            reviewer = (row.get("reviewer") or "").strip()
            if not relevant_raw or not reviewer:
                skipped += 1
                continue
            accepted.append({
                "query_id": row["query_id"],
                "query_text": row["query_text"],
                "relevant_ids": [x.strip() for x in relevant_raw.split(",") if x.strip()],
                "reviewer": reviewer,
                "review_status": "HUMAN_REVIEWED",
            })

    os.makedirs(_QUERIES_DIR, exist_ok=True)
    out_path = os.path.join(_QUERIES_DIR, f"rag_queries_{source}.json")
    with open(out_path, "w") as f:
        json.dump(accepted, f, indent=2)
    print(f"Imported {len(accepted)} reviewed queries ({skipped} rows skipped: no relevance/reviewer) -> {out_path}")
    return len(accepted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["investigation", "rag"], required=True)
    parser.add_argument("--source", choices=["mitre_semantic", "campaign_narrative", "gnn_topology"], default=None)
    args = parser.parse_args()

    if args.which == "investigation":
        import_investigation_verdicts()
    else:
        if not args.source:
            raise SystemExit("--source is required with --which rag")
        import_rag_queries(args.source)

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
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

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


def _promote_ground_truth_dataset(dataset_name: str, decision_col: str, accept_values: set,
                                    build_updates) -> tuple[int, int]:
    """Shared machinery for MITRE/attribution/threat-qualification:
    reads the exported review CSV, and for every row with BOTH a
    reviewer name AND a real decision, re-loads the matching
    AUTO_PROPOSED GroundTruthRecord, applies `build_updates(row)` (a
    dict of field overrides), sets review_status=HUMAN_REVIEWED with
    that reviewer, and re-saves. Refuses (skips, does not guess) any
    row missing a reviewer or a decision -- this is the one place a
    provisional label can become HUMAN_REVIEWED, and it never happens
    silently."""
    import ground_truth.store as store
    from ground_truth.schema import ReviewStatus

    csv_path_candidates = [
        os.path.join(_DIR, f"mitre_review_queue_{dataset_name}.csv"),
        os.path.join(_DIR, f"attribution_review_queue_{dataset_name}.csv"),
        os.path.join(_DIR, f"threat_qualification_review_queue_{dataset_name}.csv"),
    ]
    csv_path = next((p for p in csv_path_candidates if os.path.exists(p)), None)
    if csv_path is None:
        raise FileNotFoundError(f"No review queue CSV found for {dataset_name} in {_DIR}")

    records, _ = store.load_best_available(dataset_name)
    by_sample_id = {r.sample_id: r for r in records}

    promoted, skipped = 0, 0
    reviewed = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            reviewer = (row.get("reviewer") or "").strip()
            decision = (row.get(decision_col) or "").strip()
            if not reviewer or decision not in accept_values:
                skipped += 1
                continue
            record = by_sample_id.get(row["sample_id"])
            if record is None:
                skipped += 1
                continue
            for field, value in build_updates(row).items():
                setattr(record, field, value)
            record.reviewer = reviewer
            record.review_status = ReviewStatus.HUMAN_REVIEWED
            reviewed.append(record)
            promoted += 1

    if reviewed:
        store.save_records(dataset_name, reviewed)
    print(f"Promoted {promoted} record(s) to HUMAN_REVIEWED, skipped {skipped} "
          f"(no reviewer/decision, or ambiguous UNCERTAIN/REJECT rows -- those correctly stay AUTO_PROPOSED).")
    return promoted, skipped


def import_mitre_reviews(dataset_name: str = "mitre_mapping_v0") -> tuple[int, int]:
    decision_col = "decision (fill in: ACCEPT|MODIFY|REJECT|UNCERTAIN)"
    corrected_col = "corrected_techniques (fill in only if MODIFY, comma-separated ATT&CK IDs)"

    def build_updates(row):
        if row[decision_col] == "MODIFY":
            corrected = [x.strip() for x in (row.get(corrected_col) or "").split(",") if x.strip()]
            return {"expected_mitre_techniques": corrected}
        return {}  # ACCEPT: keep the AI-proposed techniques as-is, now human-confirmed

    return _promote_ground_truth_dataset(dataset_name, decision_col, {"ACCEPT", "MODIFY"}, build_updates)


def import_attribution_reviews(dataset_name: str = "attribution_v0") -> tuple[int, int]:
    decision_col = "attacker_identity (fill in: confirm the IP, correct it, or write UNKNOWN/UNDETERMINED)"

    def build_updates(row):
        value = row[decision_col].strip()
        return {"expected_attribution": None if value.upper() in ("UNKNOWN", "UNDETERMINED") else value}

    # any non-empty value (including UNKNOWN/UNDETERMINED, a valid honest answer) counts as a real decision
    with open([p for p in (
        os.path.join(_DIR, f"attribution_review_queue_{dataset_name}.csv"),
    ) if os.path.exists(p)][0], newline="") as f:
        accept_values = {row[decision_col].strip() for row in csv.DictReader(f) if row[decision_col].strip()}
    return _promote_ground_truth_dataset(dataset_name, decision_col, accept_values, build_updates)


def import_threat_qualification_reviews(dataset_name: str = "threat_qualification_v0") -> tuple[int, int]:
    decision_col = "ground_truth_threat_status (fill in: NOT_THREAT|SUSPICIOUS|QUALIFIED_THREAT)"

    def build_updates(row):
        return {"expected_threat_status": row[decision_col].strip()}

    return _promote_ground_truth_dataset(
        dataset_name, decision_col, {"NOT_THREAT", "SUSPICIOUS", "QUALIFIED_THREAT"}, build_updates
    )


def import_campaign_correlation_reviews(dataset_name: str = "campaign_correlation_v0") -> int:
    """Pairwise same/different labels don't fit the per-sample
    GroundTruthRecord shape -- imported as a separate labeled-pairs
    JSON file instead, consumed directly by a pairwise evaluator run,
    not promoted into ground_truth/reviewed/."""
    csv_path = os.path.join(_DIR, f"campaign_correlation_review_queue_{dataset_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist -- run export_review_queue.py --which campaign_correlation first.")

    decision_col = "same_campaign_ground_truth (fill in: SAME|DIFFERENT|UNCERTAIN)"
    accepted = []
    skipped = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            reviewer = (row.get("reviewer") or "").strip()
            decision = (row.get(decision_col) or "").strip()
            if not reviewer or decision not in {"SAME", "DIFFERENT"}:  # UNCERTAIN pairs stay unresolved, not guessed
                skipped += 1
                continue
            accepted.append({
                "campaign_a": row["campaign_a"], "campaign_b": row["campaign_b"],
                "same_campaign": decision == "SAME", "reviewer": reviewer,
            })

    os.makedirs(_LABELS_DIR, exist_ok=True)
    out_path = os.path.join(_LABELS_DIR, f"{dataset_name}_pairwise_reviewed.json")
    with open(out_path, "w") as f:
        json.dump(accepted, f, indent=2)
    print(f"Imported {len(accepted)} reviewed pairs ({skipped} skipped: no reviewer/decision or UNCERTAIN) -> {out_path}")
    return len(accepted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["investigation", "rag", "mitre", "attribution",
                                             "threat_qualification", "campaign_correlation"], required=True)
    parser.add_argument("--source", choices=["mitre_semantic", "campaign_narrative", "gnn_topology"], default=None)
    args = parser.parse_args()

    if args.which == "investigation":
        import_investigation_verdicts()
    elif args.which == "mitre":
        import_mitre_reviews()
    elif args.which == "attribution":
        import_attribution_reviews()
    elif args.which == "threat_qualification":
        import_threat_qualification_reviews()
    elif args.which == "campaign_correlation":
        import_campaign_correlation_reviews()
    else:
        if not args.source:
            raise SystemExit("--source is required with --which rag")
        import_rag_queries(args.source)

"""
ground_truth/store.py
========================
Load/save GroundTruthRecords as JSONL files under
evaluation/ground_truth/{provisional,reviewed,locked}/. One file per
dataset (e.g. "mitre_mapping_v0.jsonl"). Locking additionally writes a
manifest with a dataset-level hash so a locked dataset can be verified
byte-for-byte unchanged later.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from ground_truth.schema import GroundTruthRecord, ReviewStatus, record_hash

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # evaluation/
_DIRS = {
    ReviewStatus.AUTO_PROPOSED: os.path.join(_BASE, "ground_truth", "provisional"),
    ReviewStatus.HUMAN_REVIEWED: os.path.join(_BASE, "ground_truth", "reviewed"),
    ReviewStatus.LOCKED: os.path.join(_BASE, "ground_truth", "locked"),
}


def save_records(dataset_name: str, records: list[GroundTruthRecord]) -> str:
    """Writes all records to the JSONL file matching the records' OWN
    review_status (all records in one save call must share a status --
    mixed-status saves are rejected, since a dataset's directory is the
    source of truth for what state it's in)."""
    statuses = {r.review_status for r in records}
    if len(statuses) > 1:
        raise ValueError(f"save_records({dataset_name}): mixed review statuses {statuses} in one call")
    status = statuses.pop() if statuses else ReviewStatus.AUTO_PROPOSED
    for r in records:
        r.validate()

    directory = _DIRS[status]
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{dataset_name}.jsonl")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    return path


def load_records(dataset_name: str, status: ReviewStatus) -> list[GroundTruthRecord]:
    path = os.path.join(_DIRS[status], f"{dataset_name}.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(GroundTruthRecord.from_dict(json.loads(line)))
    return records


def load_best_available(dataset_name: str) -> tuple[list[GroundTruthRecord], ReviewStatus]:
    """Returns the highest-quality available copy of a dataset:
    LOCKED > HUMAN_REVIEWED > AUTO_PROPOSED. Callers (evaluators) must
    check the returned status before treating results as final."""
    for status in (ReviewStatus.LOCKED, ReviewStatus.HUMAN_REVIEWED, ReviewStatus.AUTO_PROPOSED):
        records = load_records(dataset_name, status)
        if records:
            return records, status
    return [], ReviewStatus.AUTO_PROPOSED


@dataclass
class LockManifest:
    dataset_name: str
    record_count: int
    dataset_hash: str
    reviewer: str
    locked_at: str
    source_commit: str


def lock_dataset(dataset_name: str, reviewer: str, source_commit: str) -> LockManifest:
    """Promotes a HUMAN_REVIEWED dataset to LOCKED. Refuses if any
    record is still AUTO_PROPOSED, or if the reviewed set is empty."""
    reviewed = load_records(dataset_name, ReviewStatus.HUMAN_REVIEWED)
    if not reviewed:
        raise ValueError(
            f"lock_dataset({dataset_name}): no HUMAN_REVIEWED records found -- "
            f"nothing to lock. AUTO_PROPOSED records cannot be locked directly."
        )
    for r in reviewed:
        if r.review_status != ReviewStatus.HUMAN_REVIEWED:
            raise ValueError(f"lock_dataset({dataset_name}): record {r.sample_id} is not HUMAN_REVIEWED")
        r.review_status = ReviewStatus.LOCKED
        r.dataset_version = f"{dataset_name}-locked"

    save_records(dataset_name, reviewed)

    combined_hash = hashlib.sha256(
        "".join(sorted(record_hash(r) for r in reviewed)).encode("utf-8")
    ).hexdigest()

    from datetime import datetime, timezone
    manifest = LockManifest(
        dataset_name=dataset_name,
        record_count=len(reviewed),
        dataset_hash=combined_hash,
        reviewer=reviewer,
        locked_at=datetime.now(timezone.utc).isoformat(),
        source_commit=source_commit,
    )
    manifest_path = os.path.join(_DIRS[ReviewStatus.LOCKED], f"{dataset_name}.manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest.__dict__, f, indent=2)
    return manifest

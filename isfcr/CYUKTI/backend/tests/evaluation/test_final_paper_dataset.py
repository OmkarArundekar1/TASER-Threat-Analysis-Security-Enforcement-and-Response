import json
import os

_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "evaluation", "results", "final_paper_dataset.json"
)


def _load():
    with open(_PATH) as f:
        return json.load(f)


def test_final_paper_dataset_is_valid_json_with_required_top_level_keys():
    data = _load()
    for key in ("artifact_version", "generated_at", "human_review_status", "metrics", "limitations"):
        assert key in data


def test_human_review_status_honestly_reports_zero_reviewed_rows():
    data = _load()
    assert data["human_review_status"]["reviewed_row_count"] == 0
    assert data["human_review_status"]["reviewers"] == []


def test_no_metric_claims_a_status_of_measured_or_locked():
    """The whole point of this artifact: nothing here may claim final
    MEASURED/LOCKED status while zero rows have been human-reviewed."""
    data = _load()
    for entry in data["metrics"]:
        assert entry["status"] not in ("MEASURED", "LOCKED", "HUMAN_REVIEWED"), entry["metric"]


def test_every_metric_has_a_real_evidence_source_pointer():
    data = _load()
    for entry in data["metrics"]:
        assert entry.get("evidence_source"), entry["metric"]


def test_prediction_entry_does_not_restate_the_phase18_number_as_this_phases_result():
    data = _load()
    prediction = next(m for m in data["metrics"] if m["metric"].startswith("NEXT_TECHNIQUE"))
    assert prediction["value"] is None
    assert prediction["status"] == "UNMEASURABLE"

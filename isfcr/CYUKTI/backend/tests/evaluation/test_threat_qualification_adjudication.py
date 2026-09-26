"""
Reproducibility guard: the exact numbers cited in
review/evaluation_threat_qualification_adjudication.md must always be
reproducible by running review/compute_threat_qualification_adjudication_metrics.py
-- if someone edits the ADJUDICATED dict without updating the report,
this test catches the drift.
"""

import importlib.util
import os

_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "evaluation", "review", "compute_threat_qualification_adjudication_metrics.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("adjudication_metrics", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_24_records_present_in_every_dict():
    m = _load_module()
    assert len(m.RECORDS) == 24
    for d in (m.PROVISIONAL, m.CYUKTI, m.ADJUDICATED):
        assert set(d) == set(m.RECORDS)


def test_original_provisional_agreement_reproduces_8_point_3_percent():
    m = _load_module()
    agree = sum(1 for r in m.RECORDS if m.PROVISIONAL[r] == m.CYUKTI[r])
    assert agree == 2
    assert round(agree / len(m.RECORDS) * 100, 1) == 8.3


def test_adjudicated_agreement_reproduces_29_point_2_percent():
    m = _load_module()
    agree = sum(1 for r in m.RECORDS if m.ADJUDICATED[r] == m.CYUKTI[r])
    assert agree == 7
    assert round(agree / len(m.RECORDS) * 100, 1) == 29.2


def test_adjudicated_label_distribution_matches_the_report():
    m = _load_module()
    from collections import Counter
    counts = Counter(m.ADJUDICATED.values())
    assert counts["QUALIFIED_THREAT"] == 7
    assert counts["SUSPICIOUS"] == 15
    assert counts["INSUFFICIENT_EVIDENCE"] == 2


def test_binary_recall_is_perfect_and_precision_is_zero_for_threat_class():
    """The report's central finding: 100% recall, 0% precision on
    QUALIFIED_THREAT/THREAT -- verified directly, not just asserted in prose."""
    m = _load_module()
    binary_records = [r for r in m.RECORDS if m.ADJUDICATED[r] != "INSUFFICIENT_EVIDENCE"]
    to_binary = lambda label: "THREAT" if label == "QUALIFIED_THREAT" else "NON_THREAT"
    y_true = [to_binary(m.ADJUDICATED[r]) for r in binary_records]
    y_pred = [to_binary(m.CYUKTI[r]) for r in binary_records]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "THREAT" and p == "THREAT")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "THREAT" and p == "NON_THREAT")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "NON_THREAT" and p == "THREAT")

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    assert recall == 1.0
    assert round(precision, 3) == 0.318


def test_no_expected_field_in_the_adjudication_module_is_derived_from_cyukti():
    """Static check: ADJUDICATED must not be defined as a copy of or
    transform of CYUKTI anywhere in the module source."""
    with open(_MODULE_PATH) as f:
        source = f.read()
    # crude but real: the ADJUDICATED dict literal must not reference the
    # CYUKTI dict as its value source anywhere in its own definition block
    adjudicated_block = source.split("ADJUDICATED = {")[1].split("\n}")[0]
    assert "CYUKTI[" not in adjudicated_block
    assert "CYUKTI.get" not in adjudicated_block

"""
Tests for backend/investigation after the confidence-architecture fix:
evidence reliability, evidence coverage, model confidence/uncertainty,
and investigation confidence are now distinct, separately-computed
values (previously a single weighted-mean scalar that saturated after
one relevant fact — found by running the loop against real campaigns).
"""

import pytest

from evidence.schema import Evidence, EvidenceSource, EvidenceType
from evidence.store import EvidenceStore
from investigation.actions import ACTION_METADATA, InvestigationAction
from investigation.confidence import estimate_confidence, evidence_coverage, predictive_entropy
from investigation.loop import run_investigation
from investigation.next_best_evidence import score_action, select_next_best_evidence
from investigation.stopping import check_stopping


def _evidence(source, confidence, relevance=1.0, relationships=None, type_=EvidenceType.THREAT_INTEL):
    return Evidence(
        source=source, source_id="x", timestamp="t", type=type_,
        content={}, confidence=confidence, relevance=relevance, relationships=relationships or [],
    )


# ---------------------------------------------------------------------------
# predictive_entropy / evidence_coverage
# ---------------------------------------------------------------------------

def test_predictive_entropy_certain_distribution_is_zero():
    assert predictive_entropy({"a": 1.0, "b": 0.0, "c": 0.0}) == 0.0


def test_predictive_entropy_uniform_distribution_is_one():
    assert predictive_entropy({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == pytest.approx(1.0)


def test_two_distributions_same_top_prob_different_entropy():
    concentrated = predictive_entropy({"a": 0.5, "b": 0.4, "c": 0.1})
    spread = predictive_entropy({"a": 0.5, "b": 0.25, "c": 0.25})
    assert spread > concentrated


def test_evidence_coverage_one_source_is_partial_not_full():
    store = EvidenceStore()
    store.add(_evidence(EvidenceSource.MITRE, confidence=1.0, relevance=1.0))
    # 1 of 6 possible EvidenceSource categories consulted
    coverage = evidence_coverage(store)
    assert 0.0 < coverage < 1.0
    assert coverage == round(1.0 / len(EvidenceSource), 4) or coverage == pytest.approx(1.0 / len(EvidenceSource))


def test_evidence_coverage_grows_with_distinct_sources():
    store = EvidenceStore()
    store.add(_evidence(EvidenceSource.MITRE, confidence=1.0, relevance=1.0))
    one_source = evidence_coverage(store)
    store.add(_evidence(EvidenceSource.GRAPH, confidence=1.0, relevance=1.0))
    two_sources = evidence_coverage(store)
    assert two_sources > one_source


# ---------------------------------------------------------------------------
# Phase 11, Case 1: highly reliable but irrelevant evidence
# ---------------------------------------------------------------------------

def test_reliable_irrelevant_evidence_does_not_yield_high_confidence():
    store = EvidenceStore()
    # MITRE fact: confidence=1.0 (authoritative), relevance=0.0 (nothing
    # has scored it against this investigation's actual question yet)
    store.add(_evidence(EvidenceSource.MITRE, confidence=1.0, relevance=0.0))
    est = estimate_confidence(store)
    assert est.evidence_reliability == 0.0  # relevance=0 excluded from the weighted mean entirely
    assert est.investigation_confidence < 0.3


# ---------------------------------------------------------------------------
# Phase 11, Case 2: relevant evidence with moderate reliability
# ---------------------------------------------------------------------------

def test_relevant_moderate_reliability_evidence_contributes_appropriately():
    store = EvidenceStore()
    store.add(_evidence(EvidenceSource.CTI, confidence=0.6, relevance=0.8))
    est = estimate_confidence(store)
    assert est.evidence_reliability == 0.6
    assert 0.0 < est.investigation_confidence < 0.6  # contributes, but coverage (1/6 sources) still gates it


# ---------------------------------------------------------------------------
# Phase 11, Case 3: conflicting evidence
# ---------------------------------------------------------------------------

def test_conflicting_evidence_reduces_confidence():
    store = EvidenceStore()
    store.add(_evidence(EvidenceSource.CTI, confidence=0.9, relevance=0.9, relationships=["ip1"]))
    store.add(_evidence(EvidenceSource.SIEM, confidence=0.1, relevance=0.9, relationships=["ip1"]))
    est = estimate_confidence(store)
    assert est.conflict_count == 1

    # same evidence, no conflict (different relationship key) for comparison
    store2 = EvidenceStore()
    store2.add(_evidence(EvidenceSource.CTI, confidence=0.9, relevance=0.9, relationships=["ipA"]))
    store2.add(_evidence(EvidenceSource.SIEM, confidence=0.1, relevance=0.9, relationships=["ipB"]))
    est2 = estimate_confidence(store2)

    assert est.investigation_confidence < est2.investigation_confidence


# ---------------------------------------------------------------------------
# Phase 11, Case 4: model probabilities remain uncertain
# ---------------------------------------------------------------------------

def test_uncertain_model_probabilities_do_not_falsely_report_near_certainty():
    store = EvidenceStore()
    for source in [EvidenceSource.MITRE, EvidenceSource.GRAPH, EvidenceSource.CTI]:
        store.add(_evidence(source, confidence=1.0, relevance=1.0))

    uncertain_probs = {"Critical": 0.40, "Medium": 0.35, "Low": 0.25}
    est = estimate_confidence(store, uncertain_probs)

    assert est.model_confidence == 0.40
    assert est.model_uncertainty > 0.9  # near-uniform 3-way split -> high entropy
    assert est.investigation_confidence < 0.3  # must not read as resolved


# ---------------------------------------------------------------------------
# Phase 11, Case 5: strong model confidence + strong relevant evidence
# ---------------------------------------------------------------------------

def test_strong_model_and_strong_evidence_allows_high_confidence():
    store = EvidenceStore()
    for source in [EvidenceSource.MITRE, EvidenceSource.GRAPH, EvidenceSource.CTI,
                    EvidenceSource.ATTRIBUTION, EvidenceSource.CAMPAIGN_HISTORY, EvidenceSource.SIEM]:
        store.add(_evidence(source, confidence=0.95, relevance=0.95))

    certain_probs = {"Critical": 0.97, "Medium": 0.02, "Low": 0.01}
    est = estimate_confidence(store, certain_probs)

    assert est.model_uncertainty < 0.15
    assert est.investigation_confidence >= 0.75


# ---------------------------------------------------------------------------
# Phase 11, Case 6: no model available
# ---------------------------------------------------------------------------

def test_no_model_available_falls_back_to_evidence_only():
    store = EvidenceStore()
    store.add(_evidence(EvidenceSource.MITRE, confidence=1.0, relevance=1.0))
    est = estimate_confidence(store, model_probabilities=None)
    assert est.model_confidence is None
    assert est.model_uncertainty is None
    assert est.model_probabilities is None
    # falls back to reliability * coverage, not an exception or a fabricated model value
    assert est.investigation_confidence == round(est.evidence_reliability * est.evidence_coverage, 4)


# ---------------------------------------------------------------------------
# next_best_evidence: uncertainty-awareness
# ---------------------------------------------------------------------------

def test_xgboost_prediction_gains_value_when_uncertainty_high():
    store = EvidenceStore()
    high_unc = score_action(InvestigationAction.XGBOOST_PREDICTION, store, current_uncertainty=1.0)
    low_unc = score_action(InvestigationAction.XGBOOST_PREDICTION, store, current_uncertainty=0.0)
    assert high_unc.value > low_unc.value
    assert high_unc.uncertainty_reduction == 1.0
    assert low_unc.uncertainty_reduction == 0.0


def test_non_conclusion_actions_unaffected_by_uncertainty():
    store = EvidenceStore()
    high = score_action(InvestigationAction.MITRE_KNOWLEDGE, store, current_uncertainty=1.0)
    low = score_action(InvestigationAction.MITRE_KNOWLEDGE, store, current_uncertainty=0.0)
    assert high.value == low.value
    assert high.uncertainty_reduction == 0.0


def test_select_next_best_evidence_prefers_higher_value():
    store = EvidenceStore()
    chosen = select_next_best_evidence(store, list(ACTION_METADATA.keys()), current_uncertainty=1.0)
    all_scores = [score_action(a, store, 1.0) for a in ACTION_METADATA]
    assert chosen.value == max(s.value for s in all_scores)


def test_select_next_best_evidence_empty_actions_returns_none():
    store = EvidenceStore()
    assert select_next_best_evidence(store, []) is None


# ---------------------------------------------------------------------------
# stopping.py
# ---------------------------------------------------------------------------

def _est(investigation_confidence, model_uncertainty=None, conflict_count=0):
    from investigation.confidence import ConfidenceEstimate
    return ConfidenceEstimate(
        evidence_reliability=investigation_confidence, evidence_coverage=1.0, conflict_count=conflict_count,
        model_uncertainty=model_uncertainty, investigation_confidence=investigation_confidence,
        uncertainty=1.0 - investigation_confidence,
    )


def test_stopping_on_confidence_and_low_uncertainty():
    est = _est(0.9, model_uncertainty=0.1)
    decision = check_stopping(est, steps_taken=1, best_remaining_action_value=0.5)
    assert decision.should_stop


def test_no_stop_on_high_confidence_but_high_model_uncertainty():
    # high investigation_confidence alone must not be sufficient if the
    # model's own distribution is still spread out
    est = _est(0.9, model_uncertainty=0.9)
    decision = check_stopping(est, steps_taken=1, best_remaining_action_value=0.5)
    assert not decision.should_stop


def test_no_stop_on_high_confidence_with_conflicts():
    est = _est(0.9, model_uncertainty=0.1, conflict_count=1)
    decision = check_stopping(est, steps_taken=1, best_remaining_action_value=0.5)
    assert not decision.should_stop


def test_stopping_on_max_steps():
    est = _est(0.2)
    decision = check_stopping(est, steps_taken=8, best_remaining_action_value=0.5, max_steps=8)
    assert decision.should_stop
    assert "maximum investigation depth" in decision.reason


def test_stopping_when_no_action_has_positive_value():
    est = _est(0.2)
    decision = check_stopping(est, steps_taken=1, best_remaining_action_value=-0.1)
    assert decision.should_stop
    assert "positive expected value" in decision.reason


def test_no_stop_when_confidence_low_and_actions_remain():
    est = _est(0.2)
    decision = check_stopping(est, steps_taken=1, best_remaining_action_value=0.5)
    assert not decision.should_stop


# ---------------------------------------------------------------------------
# loop.py (fake executor — no live infra)
# ---------------------------------------------------------------------------

def _fake_executor(evidence_by_action):
    def execute(action):
        return evidence_by_action.get(action, [])
    return execute


def test_loop_never_repeats_an_action():
    calls = []

    def executor(action):
        calls.append(action)
        return [_evidence(ACTION_METADATA[action].source, confidence=0.9, relevance=0.9)]

    run_investigation(executor, max_steps=8)
    non_model_calls = [c for c in calls if c != InvestigationAction.XGBOOST_PREDICTION]
    assert len(non_model_calls) == len(set(non_model_calls))


def test_loop_with_no_model_predictor_never_calls_model():
    def executor(action):
        return [_evidence(ACTION_METADATA[action].source, confidence=0.9, relevance=0.9)]

    record = run_investigation(executor, model_predictor=None, max_steps=8)
    assert record.final_confidence.model_probabilities is None


def test_loop_consults_model_when_predictor_provided():
    def executor(action):
        return [_evidence(ACTION_METADATA[action].source, confidence=0.9, relevance=0.9)]

    def model_predictor():
        return {"Critical": 0.9, "Low": 0.1}

    record = run_investigation(executor, model_predictor=model_predictor, max_steps=8)
    model_was_consulted = any(
        s.action_taken == InvestigationAction.XGBOOST_PREDICTION for s in record.steps
    )
    assert model_was_consulted
    assert record.final_confidence.model_probabilities is not None


def test_loop_record_to_dict_is_json_serializable():
    import json

    def executor(action):
        return [_evidence(ACTION_METADATA[action].source, confidence=0.6, relevance=0.6)]

    record = run_investigation(executor, max_steps=2, confidence_threshold=0.99)
    payload = record.to_dict()
    json.dumps(payload)
    assert payload["stopping_reason"]

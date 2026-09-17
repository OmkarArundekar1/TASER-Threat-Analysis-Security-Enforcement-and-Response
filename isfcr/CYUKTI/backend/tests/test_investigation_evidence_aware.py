"""
Tests for the evidence-dependency / redundancy model, multi-hypothesis
state, conflict-severity scaling, and the UNKNOWN safety invariant as it
applies specifically to the investigation module (this project's
"evidence-aware adaptive investigator" capability, built on top of the
existing evidence-reliability/coverage/model-confidence separation).
"""

import inspect

import pytest

from evidence.schema import Evidence, EvidenceSource, EvidenceType
from evidence.store import EvidenceStore
from investigation.actions import ACTION_METADATA, InvestigationAction
from investigation.confidence import estimate_confidence
from investigation.loop import (
    _candidate_hypotheses,
    default_action_executor,
    default_model_predictor,
    run_investigation,
)
from investigation.next_best_evidence import score_action


def _evidence(source, confidence=0.9, relevance=0.9, type_=EvidenceType.THREAT_INTEL, source_id="x"):
    return Evidence(
        source=source, source_id=source_id, timestamp="t", type=type_,
        content={}, confidence=confidence, relevance=relevance,
    )


# ---------------------------------------------------------------- redundancy penalty (Phase 6)

def test_dependent_action_is_penalized_once_its_dependency_is_taken():
    store = EvidenceStore()
    before = score_action(InvestigationAction.ATTRIBUTION_MATCH, store, taken_actions=frozenset())
    after = score_action(
        InvestigationAction.ATTRIBUTION_MATCH, store,
        taken_actions=frozenset({InvestigationAction.CAMPAIGN_HISTORY}),
    )
    assert before.redundancy_penalty == 0.0
    assert after.redundancy_penalty > 0.0
    assert after.value < before.value


def test_redundancy_penalty_only_applies_to_declared_dependencies():
    store = EvidenceStore()
    # CTI_LOOKUP has no declared dependency on anything -- taking an
    # unrelated action first must not penalize it.
    scored = score_action(
        InvestigationAction.CTI_LOOKUP, store,
        taken_actions=frozenset({InvestigationAction.CAMPAIGN_HISTORY, InvestigationAction.MITRE_KNOWLEDGE}),
    )
    assert scored.redundancy_penalty == 0.0


def test_redundant_evidence_does_not_receive_unlimited_weight():
    # ATTRIBUTION_MATCH declares exactly one dependency (CAMPAIGN_HISTORY).
    # Taking other, unrelated actions afterward must not keep compounding
    # its penalty -- it's bounded by the number of DECLARED dependencies
    # actually satisfied, not by how many total actions have been taken.
    store = EvidenceStore()
    penalty_one_dep_satisfied = score_action(
        InvestigationAction.ATTRIBUTION_MATCH, store,
        taken_actions=frozenset({InvestigationAction.CAMPAIGN_HISTORY}),
    ).redundancy_penalty
    penalty_many_unrelated_taken = score_action(
        InvestigationAction.ATTRIBUTION_MATCH, store,
        taken_actions=frozenset({
            InvestigationAction.CAMPAIGN_HISTORY, InvestigationAction.MITRE_KNOWLEDGE,
            InvestigationAction.CTI_LOOKUP, InvestigationAction.DETECTION_CHECK,
            InvestigationAction.GRAPH_STRUCTURE,
        }),
    ).redundancy_penalty
    assert penalty_one_dep_satisfied == penalty_many_unrelated_taken


def test_ranking_changes_as_investigation_state_changes():
    """The core adaptivity requirement: the same action's relative rank
    shifts once its dependency has been satisfied, proving selection is
    state-dependent rather than a fixed priority list."""
    store = EvidenceStore()
    scores_before = {
        a: score_action(a, store, taken_actions=frozenset())
        for a in (InvestigationAction.ATTRIBUTION_MATCH, InvestigationAction.CAMPAIGN_HISTORY)
    }
    scores_after = {
        a: score_action(a, store, taken_actions=frozenset({InvestigationAction.CAMPAIGN_HISTORY}))
        for a in (InvestigationAction.ATTRIBUTION_MATCH,)
    }
    assert scores_after[InvestigationAction.ATTRIBUTION_MATCH].value < scores_before[InvestigationAction.ATTRIBUTION_MATCH].value


def test_derived_from_populated_when_dependent_evidence_is_collected():
    calls = {}

    def executor(action):
        calls[action] = calls.get(action, 0) + 1
        if action == InvestigationAction.CAMPAIGN_HISTORY:
            return [_evidence(EvidenceSource.CAMPAIGN_HISTORY, source_id="hist-1")]
        if action == InvestigationAction.ATTRIBUTION_MATCH:
            return [_evidence(EvidenceSource.ATTRIBUTION, source_id="attr-1")]
        return [_evidence(ACTION_METADATA[action].source, source_id=action.value)]

    record = run_investigation(executor, max_steps=8)

    attribution_evidence = [
        e for e in record.evidence_store.all() if e.source == EvidenceSource.ATTRIBUTION
    ]
    campaign_history_evidence = [
        e for e in record.evidence_store.all() if e.source == EvidenceSource.CAMPAIGN_HISTORY
    ]
    assert attribution_evidence and campaign_history_evidence
    assert campaign_history_evidence[0].evidence_id in attribution_evidence[0].derived_from


# ---------------------------------------------------------------- conflict severity scaling

def test_conflict_penalty_compounds_with_more_conflicts():
    store_one_conflict = EvidenceStore()
    store_one_conflict.add(Evidence(
        source=EvidenceSource.CTI, source_id="x", timestamp="t", type=EvidenceType.THREAT_INTEL,
        content={}, confidence=0.9, relevance=0.9, relationships=["ip1"],
    ))
    store_one_conflict.add(Evidence(
        source=EvidenceSource.SIEM, source_id="y", timestamp="t", type=EvidenceType.DETECTION,
        content={}, confidence=0.1, relevance=0.9, relationships=["ip1"],
    ))

    store_two_conflicts = EvidenceStore()
    for rel in ("ip1", "ip2"):
        store_two_conflicts.add(Evidence(
            source=EvidenceSource.CTI, source_id=f"cti-{rel}", timestamp="t", type=EvidenceType.THREAT_INTEL,
            content={}, confidence=0.9, relevance=0.9, relationships=[rel],
        ))
        store_two_conflicts.add(Evidence(
            source=EvidenceSource.SIEM, source_id=f"siem-{rel}", timestamp="t", type=EvidenceType.DETECTION,
            content={}, confidence=0.1, relevance=0.9, relationships=[rel],
        ))

    est_one = estimate_confidence(store_one_conflict)
    est_two = estimate_confidence(store_two_conflicts)
    assert est_one.conflict_count == 1
    assert est_two.conflict_count == 2
    # Same evidence_reliability/coverage inputs by construction (symmetric
    # stores) -- the only difference is conflict count, so a lower
    # investigation_confidence here isolates the compounding penalty.
    assert est_two.investigation_confidence < est_one.investigation_confidence


# ---------------------------------------------------------------- multi-hypothesis state

def test_candidate_hypotheses_ranks_all_nonzero_probabilities_highest_first():
    hyps = _candidate_hypotheses({"Critical": 0.5, "Medium": 0.3, "Low": 0.2})
    assert hyps == [("Critical", 0.5), ("Medium", 0.3), ("Low", 0.2)]


def test_candidate_hypotheses_excludes_zero_probability_classes():
    hyps = _candidate_hypotheses({"Critical": 1.0, "Medium": 0.0, "Low": 0.0})
    assert hyps == [("Critical", 1.0)]


def test_candidate_hypotheses_empty_when_no_model_probabilities():
    assert _candidate_hypotheses(None) == []


def test_investigation_state_distinguishes_single_conclusion_from_candidate_hypotheses():
    def executor(action):
        return [_evidence(ACTION_METADATA[action].source)]

    def model_predictor():
        return {"Critical": 0.4, "Medium": 0.35, "Low": 0.25}

    record = run_investigation(executor, model_predictor=model_predictor, max_steps=8)
    steps_with_model = [s for s in record.steps if s.state_after.model_probabilities]
    assert steps_with_model
    last = steps_with_model[-1]
    assert last.state_after.conclusion == "Critical"  # single top hypothesis
    assert len(last.state_after.candidate_hypotheses) == 3  # full competing set preserved


# ---------------------------------------------------------------- trace auditability (Phase 10)

def test_trace_records_candidate_scores_and_why_selected():
    def executor(action):
        return [_evidence(ACTION_METADATA[action].source)]

    record = run_investigation(executor, max_steps=2, confidence_threshold=0.99)
    assert record.steps
    first_step = record.steps[0]
    assert len(first_step.candidate_scores) == len(ACTION_METADATA)  # every action was considered
    assert first_step.why_selected  # non-empty explanation string
    payload = record.to_dict()
    assert payload["steps"][0]["action_scores"]
    assert payload["steps"][0]["why_selected"]
    assert "previous_confidence" in payload["steps"][0]


# ---------------------------------------------------------------- UNKNOWN safety invariant (Phase 9)

NEO4J_WRITE_FUNCTION_NAMES = (
    "create_attack_event",
    "create_unattributed_attack_event",
    "append_technique",
    "update_attack_chain",
    "attach_campaign_to_operation",
    "create_operation_db",
    "update_operation_activity",
    "store_dynamic_risk",
    "store_cti_confidence",
    "reopen_campaign_db",
    "expire_stale_campaigns_db",
)


def test_default_action_executor_never_references_a_neo4j_write_function():
    """Static/structural proof (same pattern already used to prove the
    UNKNOWN ingestion branch can't reach chain/prediction/MISP calls):
    the investigation loop's real engine wiring is read-only. It gathers
    evidence and forms hypotheses; it never has the capability to write
    attack_id, a Technique relationship, or a NEXT_TECHNIQUE edge, because
    the functions that do those writes are never referenced in its source
    at all."""
    source = inspect.getsource(default_action_executor)
    for name in NEO4J_WRITE_FUNCTION_NAMES:
        assert name not in source, f"default_action_executor must never reference {name}"


def test_run_investigation_loop_body_never_references_a_neo4j_write_function():
    import investigation.loop as loop_module
    source = inspect.getsource(loop_module)
    for name in NEO4J_WRITE_FUNCTION_NAMES:
        assert name not in source, f"investigation/loop.py must never reference {name}"


def test_investigation_never_imports_technique_merge_capability():
    """The investigation module must not import any function whose name
    suggests it can create/merge a Technique node or NEXT_TECHNIQUE edge
    -- hypotheses stay hypotheses; only the existing, separately-gated
    resolved-alert ingestion path (mitre_resolver.py / realtime_socgraph.py)
    is ever allowed to write those."""
    import investigation.loop as loop_module
    source = inspect.getsource(loop_module)
    assert "MERGE (t:Technique" not in source
    assert "NEXT_TECHNIQUE" not in source
    assert "MATCHES]->" not in source

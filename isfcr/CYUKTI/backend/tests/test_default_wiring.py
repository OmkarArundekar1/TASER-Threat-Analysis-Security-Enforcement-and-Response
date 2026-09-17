"""
Behavioral tests for the real-engine wiring in investigation/loop.py:
default_action_executor and default_model_predictor.

Both were already imported by tests/test_investigation_evidence_aware.py
but never actually invoked there -- only inspected structurally (the
no-Neo4j-write-function proof). This file exercises the call path
itself: each InvestigationAction reaches the correct real engine
singleton and the correct real collector function, and
default_model_predictor performs a real feature-extraction -> XGBoost
inference round trip end to end. Everything that would otherwise need
live Neo4j/MISP is monkeypatched at the singleton boundary
(`module.engine`, `module.context`, ...) -- the collector functions
themselves (evidence/collectors/*.py) are the REAL functions, already
unit-tested in test_evidence.py; only their upstream engine calls are
faked here.

The final test in this file (`test_full_closed_loop_...`) is CYUKTI's
end-to-end sanity check per the BUILD->INTEGRATE mandate: incident
state -> feature extraction -> ML prediction -> candidate generation ->
NBE scoring -> action selection -> evidence acquisition -> context
update -> next decision, run for real (real run_investigation, real
default_action_executor, real default_model_predictor, real trained
XGBoost model, real NBE scoring) with only the external engine/DB calls
stubbed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from evidence.schema import Evidence, EvidenceSource, EvidenceType
from investigation.actions import InvestigationAction
from investigation.loop import default_action_executor, default_model_predictor, run_investigation
from ml.data_prep.generate_synthetic_dataset import generate_synthetic_records
from ml.train_xgboost import XGBoostCampaignClassifier


class _FakeCampaignContext:
    def __init__(self, campaign_id="camp-1", attacker_ip="10.0.0.1", victim_ip="10.0.0.2", techniques=None, attack_chain=None):
        self.campaign_id = campaign_id
        self.attacker_ip = attacker_ip
        self.victim_ip = victim_ip
        self.techniques = techniques or {"T1110"}
        self.attack_chain = attack_chain if attack_chain is not None else list(self.techniques)
        self.risk_score = 42.0


def _fake_evidence(source: EvidenceSource, source_id: str = "x") -> list[Evidence]:
    return [Evidence(
        source=source, source_id=source_id, timestamp="t",
        type=EvidenceType.TECHNIQUE_KNOWLEDGE, content={}, confidence=0.9, relevance=0.9,
    )]


# ---------------------------------------------------------------- default_action_executor wiring

def test_mitre_knowledge_calls_real_engine_and_real_collector(monkeypatch):
    import mitre_feature_engine
    import evidence.collectors.mitre_collector as mitre_collector_module

    sentinel_features = object()
    monkeypatch.setattr(mitre_feature_engine.engine, "extract_features", MagicMock(return_value=sentinel_features))
    captured = {}

    def fake_collect(features):
        captured["features"] = features
        return _fake_evidence(EvidenceSource.MITRE)

    monkeypatch.setattr(mitre_collector_module, "collect_mitre_evidence", fake_collect)

    executor = default_action_executor(_FakeCampaignContext(), "T1110", "evt-1")
    result = executor(InvestigationAction.MITRE_KNOWLEDGE)

    mitre_feature_engine.engine.extract_features.assert_called_once_with("T1110")
    assert captured["features"] is sentinel_features
    assert len(result) == 1 and result[0].source == EvidenceSource.MITRE


def test_mitre_semantic_search_queries_real_rag_retriever(monkeypatch):
    from rag.mitre_retriever import mitre_retriever

    monkeypatch.setattr(mitre_retriever, "query", MagicMock(return_value=_fake_evidence(EvidenceSource.MITRE)))

    context = _FakeCampaignContext(techniques={"T1110", "T1078"})
    executor = default_action_executor(context, "T1110", "evt-1")
    result = executor(InvestigationAction.MITRE_SEMANTIC_SEARCH)

    mitre_retriever.query.assert_called_once()
    query_text = mitre_retriever.query.call_args[0][0]
    assert "T1110" in query_text and "T1078" in query_text
    assert len(result) == 1


def test_campaign_narrative_search_queries_real_rag_retriever(monkeypatch):
    from rag.campaign_retriever import CampaignNarrativeRetriever

    fake_query = MagicMock(return_value=_fake_evidence(EvidenceSource.CAMPAIGN_HISTORY))
    monkeypatch.setattr(CampaignNarrativeRetriever, "query", fake_query)

    context = _FakeCampaignContext(techniques={"T1110", "T1078"}, attack_chain=["T1110", "T1078"])
    executor = default_action_executor(context, "T1110", "evt-1")
    result = executor(InvestigationAction.CAMPAIGN_NARRATIVE_SEARCH)

    fake_query.assert_called_once()
    # MagicMock isn't a descriptor, so patching the class attribute
    # directly means `self` is NOT auto-bound the way a real method
    # would be -- the mock is called with exactly the arguments the
    # production code passes, query_text alone.
    query_text = fake_query.call_args[0][0]
    assert "T1110" in query_text and "T1078" in query_text
    assert len(result) == 1


def test_campaign_narrative_search_degrades_gracefully_when_no_history_exists(monkeypatch):
    """rag/campaign_retriever.py raises RuntimeError when no historical
    campaign has a resolved technique yet -- default_action_executor
    must turn that into an empty evidence list, not a crash."""
    import attribution_context as attribution_context_module

    monkeypatch.setattr(attribution_context_module.context, "load_historical_campaigns", lambda: [])

    context = _FakeCampaignContext(techniques={"T1110"}, attack_chain=["T1110"])
    executor = default_action_executor(context, "T1110", "evt-1")
    result = executor(InvestigationAction.CAMPAIGN_NARRATIVE_SEARCH)

    assert result == []


def test_cti_lookup_calls_real_engine_with_attacker_ip(monkeypatch):
    import threat_intelligence_engine
    import evidence.collectors.cti_collector as cti_collector_module

    sentinel_result = object()
    monkeypatch.setattr(threat_intelligence_engine.engine, "calculate", MagicMock(return_value=sentinel_result))
    captured = {}

    def fake_collect(ip, result):
        captured["ip"] = ip
        captured["result"] = result
        return _fake_evidence(EvidenceSource.CTI)

    monkeypatch.setattr(cti_collector_module, "collect_cti_evidence", fake_collect)

    context = _FakeCampaignContext(attacker_ip="203.0.113.7")
    executor = default_action_executor(context, "T1110", "evt-1")
    result = executor(InvestigationAction.CTI_LOOKUP)

    threat_intelligence_engine.engine.calculate.assert_called_once_with("203.0.113.7")
    assert captured["ip"] == "203.0.113.7"
    assert captured["result"] is sentinel_result
    assert len(result) == 1


def test_detection_check_returns_empty_without_event_id():
    executor = default_action_executor(_FakeCampaignContext(), "T1110", None)
    assert executor(InvestigationAction.DETECTION_CHECK) == []


def test_detection_check_calls_real_engine_when_event_id_present(monkeypatch):
    import detection_confidence_engine
    import evidence.collectors.detection_collector as detection_collector_module

    sentinel_result = object()
    monkeypatch.setattr(detection_confidence_engine.engine, "calculate", MagicMock(return_value=sentinel_result))
    monkeypatch.setattr(
        detection_collector_module, "collect_detection_evidence",
        lambda event_id, result: _fake_evidence(EvidenceSource.SIEM, source_id=event_id),
    )

    executor = default_action_executor(_FakeCampaignContext(), "T1110", "evt-42")
    result = executor(InvestigationAction.DETECTION_CHECK)

    detection_confidence_engine.engine.calculate.assert_called_once_with("evt-42")
    assert len(result) == 1 and result[0].source_id == "evt-42"


def test_attribution_match_calls_real_engine_with_campaign_context(monkeypatch):
    import threat_attribution_engine
    import evidence.collectors.attribution_collector as attribution_collector_module

    context = _FakeCampaignContext()
    sentinel_result = object()
    monkeypatch.setattr(threat_attribution_engine.engine, "attribute", MagicMock(return_value=sentinel_result))
    monkeypatch.setattr(
        attribution_collector_module, "collect_attribution_evidence",
        lambda result: _fake_evidence(EvidenceSource.ATTRIBUTION),
    )

    executor = default_action_executor(context, "T1110", "evt-1")
    result = executor(InvestigationAction.ATTRIBUTION_MATCH)

    threat_attribution_engine.engine.attribute.assert_called_once_with(context)
    assert len(result) == 1 and result[0].source == EvidenceSource.ATTRIBUTION


def test_campaign_history_calls_real_context_loader(monkeypatch):
    import attribution_context as attribution_context_module
    import evidence.collectors.campaign_history_collector as history_collector_module

    sentinel_campaigns = [object()]
    monkeypatch.setattr(
        attribution_context_module.context, "load_historical_campaigns",
        MagicMock(return_value=sentinel_campaigns),
    )
    captured = {}

    def fake_collect(techniques, campaigns):
        captured["techniques"] = techniques
        captured["campaigns"] = campaigns
        return _fake_evidence(EvidenceSource.CAMPAIGN_HISTORY)

    monkeypatch.setattr(history_collector_module, "collect_campaign_history_evidence", fake_collect)

    context = _FakeCampaignContext(techniques={"T1110"})
    executor = default_action_executor(context, "T1110", "evt-1")
    result = executor(InvestigationAction.CAMPAIGN_HISTORY)

    attribution_context_module.context.load_historical_campaigns.assert_called_once_with()
    assert captured["campaigns"] is sentinel_campaigns
    assert len(result) == 1


def test_graph_structure_calls_real_engine_with_force_reload(monkeypatch):
    import graph_feature_engine
    import evidence.collectors.graph_collector as graph_collector_module

    sentinel_features = object()
    monkeypatch.setattr(graph_feature_engine.graph_analytics, "extract_features", MagicMock(return_value=sentinel_features))
    monkeypatch.setattr(
        graph_collector_module, "collect_graph_evidence",
        lambda campaign_id, features: _fake_evidence(EvidenceSource.GRAPH, source_id=campaign_id),
    )

    context = _FakeCampaignContext(campaign_id="camp-99")
    executor = default_action_executor(context, "T1110", "evt-1")
    result = executor(InvestigationAction.GRAPH_STRUCTURE)

    graph_feature_engine.graph_analytics.extract_features.assert_called_once_with("camp-99", force_reload=True)
    assert len(result) == 1 and result[0].source_id == "camp-99"


def test_xgboost_prediction_action_never_reaches_fact_gathering_executor():
    executor = default_action_executor(_FakeCampaignContext(), "T1110", "evt-1")
    assert executor(InvestigationAction.XGBOOST_PREDICTION) == []


# ---------------------------------------------------------------- default_model_predictor wiring

@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory):
    """A real (small, synthetic-data) trained XGBoost artifact -- same
    mechanism as tests/test_ml_pipeline.py, reused here so
    default_model_predictor can be exercised against a real saved
    model.json + .meta.joblib pair without depending on the committed
    production artifact (which may not exist in every environment)."""
    records = generate_synthetic_records(300, seed=11)
    df = pd.DataFrame([r.__dict__ for r in records])
    save_dir = tmp_path_factory.mktemp("model")
    clf = XGBoostCampaignClassifier()
    clf.train(df, target_column="severity", save_dir=str(save_dir))
    return str(save_dir / "xgb_severity.json")


def test_default_model_predictor_returns_none_when_model_missing(tmp_path):
    predictor = default_model_predictor(
        _FakeCampaignContext(), "T1110", "evt-1", model_path=str(tmp_path / "does_not_exist.json"),
    )
    assert predictor is None


def test_default_model_predictor_runs_real_feature_extraction_to_prediction(monkeypatch, trained_model_path):
    import ml.runtime_predictor  # noqa: F401  (triggers ml/__init__.py's sys.path bootstrap)
    import feature_extractors

    fixture_record = generate_synthetic_records(1, seed=11)[0]
    captured_args = {}

    def fake_extract(campaign_context, current_attack_id, attacker_ip, event_id, **labels):
        captured_args["campaign_context"] = campaign_context
        captured_args["current_attack_id"] = current_attack_id
        captured_args["attacker_ip"] = attacker_ip
        captured_args["event_id"] = event_id
        return fixture_record

    monkeypatch.setattr(feature_extractors.extractor, "extract", fake_extract)

    context = _FakeCampaignContext(campaign_id="camp-7", attacker_ip="198.51.100.2")
    predictor = default_model_predictor(context, "T1110", "evt-1", model_path=trained_model_path)
    assert predictor is not None

    probabilities = predictor()

    assert captured_args["campaign_context"] is context
    assert captured_args["current_attack_id"] == "T1110"
    assert captured_args["attacker_ip"] == "198.51.100.2"
    assert probabilities is not None
    assert set(probabilities.keys()) <= {"Low", "Medium", "High", "Critical"}
    assert abs(sum(probabilities.values()) - 1.0) < 1e-3


def test_default_model_predictor_returns_none_on_extraction_failure(monkeypatch, trained_model_path):
    import ml.runtime_predictor  # noqa: F401
    import feature_extractors

    def raising_extract(*args, **kwargs):
        raise RuntimeError("Neo4j unavailable")

    monkeypatch.setattr(feature_extractors.extractor, "extract", raising_extract)

    predictor = default_model_predictor(_FakeCampaignContext(), "T1110", "evt-1", model_path=trained_model_path)
    assert predictor is not None
    assert predictor() is None  # controlled fallback, not a raised exception


# ---------------------------------------------------------------- full closed-loop sanity check

def test_full_closed_loop_incident_context_to_next_decision(monkeypatch, trained_model_path):
    """The mission-level sanity check: a real run_investigation loop,
    real default_action_executor, real default_model_predictor, and a
    real trained XGBoost model, driven end to end with only the
    external engine/DB singletons stubbed. Verifies every stage of the
    closed loop actually executes and hands off to the next:
    IncidentContext -> feature extraction -> ML prediction -> candidate
    generation -> NBE scoring -> action selection -> evidence
    acquisition -> context (store) update -> next decision -> repeat
    until a real stopping condition fires.
    """
    import mitre_feature_engine
    import threat_intelligence_engine
    import detection_confidence_engine
    import threat_attribution_engine
    import attribution_context as attribution_context_module
    import graph_feature_engine
    import evidence.collectors.mitre_collector as mitre_collector_module
    import evidence.collectors.cti_collector as cti_collector_module
    import evidence.collectors.detection_collector as detection_collector_module
    import evidence.collectors.attribution_collector as attribution_collector_module
    import evidence.collectors.campaign_history_collector as history_collector_module
    import evidence.collectors.graph_collector as graph_collector_module
    from rag.mitre_retriever import mitre_retriever
    import ml.runtime_predictor  # noqa: F401
    import feature_extractors

    monkeypatch.setattr(mitre_feature_engine.engine, "extract_features", MagicMock(return_value=object()))
    monkeypatch.setattr(mitre_collector_module, "collect_mitre_evidence", lambda f: _fake_evidence(EvidenceSource.MITRE))
    monkeypatch.setattr(mitre_retriever, "query", MagicMock(return_value=_fake_evidence(EvidenceSource.MITRE, "rag")))
    monkeypatch.setattr(threat_intelligence_engine.engine, "calculate", MagicMock(return_value=object()))
    monkeypatch.setattr(cti_collector_module, "collect_cti_evidence", lambda ip, r: _fake_evidence(EvidenceSource.CTI))
    monkeypatch.setattr(detection_confidence_engine.engine, "calculate", MagicMock(return_value=object()))
    monkeypatch.setattr(detection_collector_module, "collect_detection_evidence", lambda e, r: _fake_evidence(EvidenceSource.SIEM))
    monkeypatch.setattr(threat_attribution_engine.engine, "attribute", MagicMock(return_value=object()))
    monkeypatch.setattr(attribution_collector_module, "collect_attribution_evidence", lambda r: _fake_evidence(EvidenceSource.ATTRIBUTION))
    monkeypatch.setattr(attribution_context_module.context, "load_historical_campaigns", MagicMock(return_value=[]))
    monkeypatch.setattr(history_collector_module, "collect_campaign_history_evidence", lambda t, c: _fake_evidence(EvidenceSource.CAMPAIGN_HISTORY))
    monkeypatch.setattr(graph_feature_engine.graph_analytics, "extract_features", MagicMock(return_value=object()))
    monkeypatch.setattr(graph_collector_module, "collect_graph_evidence", lambda cid, f: _fake_evidence(EvidenceSource.GRAPH))

    fixture_record = generate_synthetic_records(1, seed=11)[0]
    monkeypatch.setattr(
        feature_extractors.extractor, "extract",
        lambda campaign_context, current_attack_id, attacker_ip, event_id, **labels: fixture_record,
    )

    context = _FakeCampaignContext(campaign_id="camp-e2e", techniques={"T1110", "T1078"})
    executor = default_action_executor(context, "T1110", "evt-1")
    model_predictor = default_model_predictor(context, "T1110", "evt-1", model_path=trained_model_path)
    assert model_predictor is not None

    record = run_investigation(executor, model_predictor=model_predictor, max_steps=8)

    assert record.steps, "closed loop must take at least one real decision"
    assert record.stopping_reason
    actions_taken = [s.action_taken for s in record.steps]
    assert InvestigationAction.XGBOOST_PREDICTION in actions_taken, (
        "ML prediction must be reachable as a real candidate action, not bypassed"
    )
    xgboost_step = next(s for s in record.steps if s.action_taken == InvestigationAction.XGBOOST_PREDICTION)
    assert xgboost_step.state_after.model_probabilities
    assert xgboost_step.state_after.conclusion in xgboost_step.state_after.model_probabilities
    assert len(xgboost_step.state_after.candidate_hypotheses) >= 1
    # NBE actually drove the decision, not a fixed order: the model action's
    # own value must have been the highest of everything considered that step.
    assert xgboost_step.action_value.value == max(av.value for av in xgboost_step.candidate_scores)

    # Every subsequent step's confidence estimate reflects the model verdict
    # once it exists -- proving the ML signal actually flows forward into
    # later NBE/stopping decisions, not just into the one step that ran it.
    later_steps = [s for s in record.steps if s.step_index > xgboost_step.step_index]
    for s in later_steps:
        assert s.confidence_before.model_probabilities == xgboost_step.state_after.model_probabilities

    payload = record.to_dict()
    assert payload["final_model_probabilities"]
    import json
    json.dumps(payload)  # must be JSON-serializable end to end (API response contract)

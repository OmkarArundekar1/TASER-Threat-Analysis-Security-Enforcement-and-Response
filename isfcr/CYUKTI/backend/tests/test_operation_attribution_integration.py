"""
Realistic integration sanity check spanning Campaign Correlation,
Threat Attribution, the evidence layer, and the evidence-aware
investigation loop -- the smallest scenario that exercises all of them
together with REAL implementations, per the BUILD->INTEGRATE mandate:

    Alert 1 (new attacker/victim, no existing operation)
            -> correlate(): no match -> caller would create an operation
    Alert 2 (same attacker, overlapping techniques, shortly after)
            -> correlate(): matches the operation Alert 1 would have created
            -> campaign context now carries more techniques (chain grows)
            -> ThreatAttributionEngine.attribute() runs against real
               (mocked-Neo4j) historical campaign records
            -> the SAME real engines are wired into
               investigation.loop.run_investigation() via
               default_action_executor, so CAMPAIGN_HISTORY and
               ATTRIBUTION_MATCH produce real Evidence, not fixtures
            -> the investigation's confidence/evidence state reflects
               the real attribution signal
            -> attribution propagates to a real MISPEventGenerator call

This establishes INTEGRATION correctness -- that these engines
communicate through CYUKTI's real evidence/investigation contracts, not
just that each is independently correct. It is not an accuracy claim:
no precision/recall/F1 is computed or implied anywhere in this file.

Only genuinely external infrastructure is mocked: the Neo4j-backed
functions campaign_correlation_engine.py, operation_manager.py,
operation_feature_engine.py, and attribution_context.py each import by name.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from attribution_models import HistoricalCampaign
from campaign_context import CampaignContext
from campaign_correlation_engine import CampaignCorrelationEngine
from evidence.schema import Evidence, EvidenceSource, EvidenceType
from investigation.actions import InvestigationAction
from investigation.loop import default_action_executor, run_investigation
from threat_attribution_engine import ThreatAttributionEngine

NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _fake_evidence(source: EvidenceSource) -> list[Evidence]:
    return [Evidence(
        source=source, source_id="x", timestamp="t",
        type=EvidenceType.TECHNIQUE_KNOWLEDGE, content={}, confidence=0.9, relevance=0.9,
    )]


@pytest.fixture()
def neo4j_boundary(monkeypatch):
    """Mocks exactly the external (Neo4j) functions this scenario's
    real code paths call, keyed by module (each module imported these
    names into its own namespace via `from neo4j_client import ...` /
    `from attribution_context import context`)."""
    import attribution_context as attribution_context_module
    import campaign_correlation_engine as cce_module
    import operation_feature_engine as ofe_module
    import operation_manager as om_module
    from ml.gnn.inference import gnn_inference_service

    # Attribution and the investigation's GNN retriever both embed campaigns through
    # neo4j_client. None is what embed_campaign returns when GNN is unavailable, and
    # both callers handle it.
    monkeypatch.setattr(gnn_inference_service, "embed_campaign", lambda *a, **kw: None)

    state = {
        "active_operations": {},  # operation_id -> row dict (see test body)
        "historical_campaigns": [],
        "reopened": [],
        "next_technique_chains": [],  # chains the scenario has recorded as NEXT_TECHNIQUE edges
        "prediction_lookups": [],
    }

    def fake_prediction_distribution(previous_technique):
        """Same calculation as neo4j_client.get_prediction_distribution, over the
        edges the scenario's own chains imply instead of a database's."""
        state["prediction_lookups"].append(previous_technique)
        counts = Counter(
            following
            for chain in state["next_technique_chains"]
            for current, following in zip(chain, chain[1:])
            if current == previous_technique
        )
        total = sum(counts.values())
        return {technique: count / total for technique, count in counts.items()} if total else {}

    monkeypatch.setattr(ofe_module, "get_prediction_distribution", fake_prediction_distribution)

    monkeypatch.setattr(cce_module, "get_active_operations", lambda: list(state["active_operations"].keys()))
    monkeypatch.setattr(cce_module, "get_recent_inactive_operations", lambda: [])
    monkeypatch.setattr(cce_module, "reopen_operation_db", lambda op_id: state["reopened"].append(op_id))
    monkeypatch.setattr(om_module, "get_operation_context_data", lambda op_id: state["active_operations"].get(op_id))
    monkeypatch.setattr(
        attribution_context_module.context, "load_historical_campaigns",
        lambda: list(state["historical_campaigns"]),
    )
    return state


def test_realistic_two_alert_scenario_from_correlation_through_investigation_state(neo4j_boundary, monkeypatch):
    correlation_engine = CampaignCorrelationEngine()
    attribution_engine = ThreatAttributionEngine()

    # ---- Alert 1: brand new attacker/victim pair, nothing to correlate against yet ----
    campaign_after_alert_1 = CampaignContext(
        campaign_id="CAMP_LIVE", attacker_ip="185.220.101.7", victim_ip="10.20.0.15",
        techniques={"T1110.001"}, attack_chain=["T1110.001"], first_seen=NOW,
    )
    first_result = correlation_engine.correlate(campaign_after_alert_1)
    assert first_result.matched is False
    assert first_result.operation_id is None
    # Production (realtime_socgraph.py) would now call create_operation_db(...)
    # and attach_campaign_to_operation(...); modeled here as the operation
    # row as Alert 1 left it, matching that real call sequence's observable
    # effect without needing a live Neo4j write. Production writes Alert 2's
    # event before correlating, so its row may already include Alert 2's
    # technique, which scores higher. See
    # test_alert_two_matches_on_the_scenario_facts_alone for both states.
    neo4j_boundary["active_operations"]["OP_LIVE"] = {
        "operation": {
            "operation_id": "OP_LIVE", "primary_attacker": "185.220.101.7",
            "created_at": NOW, "last_seen": NOW,
        },
        "campaigns": [{"campaign_id": "CAMP_LIVE", "victim_ip": "10.20.0.15"}],
        "techniques": ["T1110.001"],
        "attack_chain": ["T1110.001"],
    }

    # ---- Alert 2: same attacker, 20 minutes later, chain grows by one technique ----
    campaign_after_alert_2 = CampaignContext(
        campaign_id="CAMP_LIVE", attacker_ip="185.220.101.7", victim_ip="10.20.0.15",
        techniques={"T1110.001", "T1110"}, attack_chain=["T1110.001", "T1110"],
        first_seen=NOW + timedelta(minutes=20), last_technique="T1110",
    )
    # Production's chain updater records this chain as NEXT_TECHNIQUE edges. T1110 is
    # its last technique, so nothing in this scenario follows T1110.
    neo4j_boundary["next_technique_chains"].append(campaign_after_alert_2.attack_chain)
    second_result = correlation_engine.correlate(campaign_after_alert_2)
    assert second_result.matched is True
    assert second_result.operation_id == "OP_LIVE"
    assert second_result.score == 0.75
    assert second_result.breakdown["prediction_similarity"]["value"] == 0.0
    assert neo4j_boundary["prediction_lookups"] == ["T1110"]

    # ---- historical (closed) campaigns available for attribution ----
    neo4j_boundary["historical_campaigns"] = [
        HistoricalCampaign(
            campaign_id="CAMP_PAST_1", attacker="185.220.101.7", victim="10.20.0.20",
            techniques=["T1110.001", "T1110", "T1078"], timestamps=["2026-01-01T00:00:00+00:00"] * 3,
            status="ARCHIVED",
        ),
        HistoricalCampaign(
            campaign_id="CAMP_PAST_2", attacker="203.0.113.1", victim="198.51.100.1",
            techniques=["T1595", "T1592"], timestamps=["2025-12-01T00:00:00+00:00"] * 2,
            status="INACTIVE",
        ),
    ]

    attribution = attribution_engine.attribute(campaign_after_alert_2)
    assert attribution.actors, "the enriched campaign context must produce a real attribution candidate"
    assert attribution.actors[0].actor == "CAMP_PAST_1"  # the one with real technique overlap
    assert attribution.actors[0].total_score > 0

    # ---- wire the SAME real engines into the real investigation loop ----
    import mitre_feature_engine
    import threat_intelligence_engine
    import detection_confidence_engine
    import graph_feature_engine
    import evidence.collectors.mitre_collector as mitre_collector_module
    import evidence.collectors.cti_collector as cti_collector_module
    import evidence.collectors.detection_collector as detection_collector_module
    import evidence.collectors.graph_collector as graph_collector_module
    from rag.mitre_retriever import mitre_retriever

    monkeypatch.setattr(mitre_feature_engine.engine, "extract_features", MagicMock(return_value=object()))
    monkeypatch.setattr(mitre_collector_module, "collect_mitre_evidence", lambda f: _fake_evidence(EvidenceSource.MITRE))
    monkeypatch.setattr(mitre_retriever, "query", MagicMock(return_value=_fake_evidence(EvidenceSource.MITRE)))
    monkeypatch.setattr(threat_intelligence_engine.engine, "calculate", MagicMock(return_value=object()))
    monkeypatch.setattr(cti_collector_module, "collect_cti_evidence", lambda ip, r: _fake_evidence(EvidenceSource.CTI))
    monkeypatch.setattr(detection_confidence_engine.engine, "calculate", MagicMock(return_value=object()))
    monkeypatch.setattr(detection_collector_module, "collect_detection_evidence", lambda e, r: _fake_evidence(EvidenceSource.SIEM))
    monkeypatch.setattr(graph_feature_engine.graph_analytics, "extract_features", MagicMock(return_value=object()))
    monkeypatch.setattr(graph_collector_module, "collect_graph_evidence", lambda cid, f: _fake_evidence(EvidenceSource.GRAPH))
    # NOTE: threat_attribution_engine.engine.attribute and
    # attribution_context.context.load_historical_campaigns are
    # deliberately left real -- this is the point of this test.

    executor = default_action_executor(campaign_after_alert_2, "T1110", "evt-2")
    record = run_investigation(executor, model_predictor=None, max_steps=8)

    actions_taken = [s.action_taken for s in record.steps]
    assert InvestigationAction.CAMPAIGN_HISTORY in actions_taken
    assert InvestigationAction.ATTRIBUTION_MATCH in actions_taken

    attribution_evidence = [e for e in record.evidence_store.all() if e.source == EvidenceSource.ATTRIBUTION]
    assert attribution_evidence
    assert attribution_evidence[0].content["candidate_campaign_id"] == "CAMP_PAST_1"
    assert attribution_evidence[0].relevance > 0  # real computed relevance, not a placeholder

    history_evidence = [e for e in record.evidence_store.all() if e.source == EvidenceSource.CAMPAIGN_HISTORY]
    assert history_evidence
    # Provenance: real end-to-end, not a synthetic single-item fixture.
    assert set(e.evidence_id for e in history_evidence) <= set(attribution_evidence[0].derived_from)

    # Confidence state genuinely reflects the presence of attribution evidence.
    assert record.final_confidence is not None
    assert record.final_confidence.evidence_coverage > 0

    # ---- propagation: the attribution result reaches MISP event generation ----
    from misp_event_generator import IncidentContext, MISPEventGenerator
    from types import SimpleNamespace

    incident = IncidentContext(
        campaign_id="CAMP_LIVE", operation_id=second_result.operation_id,
        attacker_ip="185.220.101.7", victim_ip="10.20.0.15", event_id="evt-2",
        technique="T1110", stage="Credential Access", prediction="", prediction_confidence=0.0,
        detection=SimpleNamespace(confidence=0.5, level="MEDIUM", breakdown={k: 1 for k in ("Wazuh", "Suricata", "Zeek", "Sigma", "YARA")}),
        threat=None,
        dynamic_risk=SimpleNamespace(risk_score=50.0, risk_level="MEDIUM", confidence=0.5),
        cti=SimpleNamespace(score=50.0, level="MEDIUM", publish=False),
        recommendations=[], investigation_payload="{}", timestamp=NOW.isoformat(),
        attribution=attribution,
    )
    misp_payload = MISPEventGenerator().generate(incident)
    tag_names = [t["name"] for t in misp_payload["Event"]["Tag"]]
    assert "actor:CAMP_PAST_1" in tag_names
    assert misp_payload["Event"]["info"] == "CYUKTI Campaign CAMP_LIVE"

    # ---- final trace is exactly what /api/investigate/<id> would serve ----
    payload = record.to_dict()
    payload["campaign_id"] = "CAMP_LIVE"
    payload["evidence"] = [e.to_dict() for e in record.evidence_store.all()]
    json.dumps(payload)


def test_scenario_chain_gives_the_transitions_the_real_query_would(neo4j_boundary):
    """The fake is derived from the chain, so it is not an empty stub: it returns
    real proportions for techniques that have successors."""
    import operation_feature_engine as ofe_module

    neo4j_boundary["next_technique_chains"].extend([
        ["T1110.001", "T1110", "T1078"],
        ["T1110.001", "T1059"],
    ])

    assert ofe_module.get_prediction_distribution("T1110.001") == {"T1110": 0.5, "T1059": 0.5}
    assert ofe_module.get_prediction_distribution("T1110") == {"T1078": 1.0}
    assert ofe_module.get_prediction_distribution("T1078") == {}


def _operation_row(techniques, attack_chain):
    return {
        "operation": {
            "operation_id": "OP_LIVE", "primary_attacker": "185.220.101.7",
            "created_at": NOW, "last_seen": NOW,
        },
        "campaigns": [{"campaign_id": "CAMP_LIVE", "victim_ip": "10.20.0.15"}],
        "techniques": techniques,
        "attack_chain": attack_chain,
    }


@pytest.mark.parametrize("techniques, attack_chain, expected_score", [
    (["T1110.001"], ["T1110.001"], 0.75),                       # as Alert 1 left it
    (["T1110.001", "T1110"], ["T1110.001", "T1110"], 0.95),     # with Alert 2's event written
])
def test_alert_two_matches_on_the_scenario_facts_alone(
    neo4j_boundary, techniques, attack_chain, expected_score,
):
    """The real scorer, with no learned transition out of T1110, in both states the
    operation row can be in: production writes Alert 2's event before correlating."""
    neo4j_boundary["active_operations"]["OP_LIVE"] = _operation_row(techniques, attack_chain)
    chain = ["T1110.001", "T1110"]
    neo4j_boundary["next_technique_chains"].append(chain)
    campaign = CampaignContext(
        campaign_id="CAMP_LIVE", attacker_ip="185.220.101.7", victim_ip="10.20.0.15",
        techniques=set(chain), attack_chain=chain,
        first_seen=NOW + timedelta(minutes=20), last_technique="T1110",
    )

    result = CampaignCorrelationEngine().correlate(campaign)

    assert result.matched is True
    assert result.score == expected_score
    assert result.breakdown["prediction_similarity"]["value"] == 0.0


@pytest.mark.parametrize("distribution, expected_similarity, expected_match", [
    ({}, 0.0, False),
    ({"T1110.001": 0.33}, 0.33, False),
    ({"T1110.001": 0.34}, 0.34, True),
    ({"T1110.001": 1.0}, 1.0, True),
])
def test_a_two_technique_jump_matches_only_with_a_learned_transition(
    neo4j_boundary, monkeypatch, distribution, expected_similarity, expected_match,
):
    """Why the scenario grows by one technique per alert. A jump of two scores 0.683
    without the prediction term, and ATTACH_THRESHOLD is 0.70, so it matches only if a
    T1078 -> T1110.001 edge holds at least 0.34 of T1078's transitions. This records
    today's behaviour. It is not a statement of what the behaviour should be."""
    import operation_feature_engine as ofe_module

    monkeypatch.setattr(ofe_module, "get_prediction_distribution", lambda previous: distribution)
    neo4j_boundary["active_operations"]["OP_LIVE"] = _operation_row(["T1110.001"], ["T1110.001"])
    campaign = CampaignContext(
        campaign_id="CAMP_LIVE", attacker_ip="185.220.101.7", victim_ip="10.20.0.15",
        techniques={"T1110.001", "T1110", "T1078"}, attack_chain=["T1110.001", "T1110", "T1078"],
        first_seen=NOW + timedelta(minutes=20), last_technique="T1078",
    )

    result = CampaignCorrelationEngine().correlate(campaign)

    assert result.matched is expected_match
    assert result.breakdown["prediction_similarity"]["value"] == expected_similarity

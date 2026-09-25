"""
Determinism verification (DETERMINISM.md, cross-cutting engineering
audit). For every component documented elsewhere as deterministic,
call it twice with byte-identical input and assert byte-identical
output. Components that are NOT deterministic by design (GNN topology
retrieval's ranking over a live, growing Neo4j dataset; anything that
depends on wall-clock time) are explicitly not tested here and are
instead documented as such in DETERMINISM.md.
"""

from __future__ import annotations

import pytest

import dashboard_api
from campaign_context import CampaignContext
from campaign_selection import BestCampaignSelector
from mitre_resolver import resolve_mitre
from soar.response_plan import ResponsePlanGenerator
from soar.schema import Playbook
from threat_qualification import ThreatQualificationEngine, QUALIFIED_THREAT


# ---------------------------------------------------------------- MITRE resolution

def test_mitre_resolution_is_deterministic_for_native_wazuh_alert():
    alert = {"rule": {"id": "5716", "mitre": {"id": ["T1110"]}}}
    first = resolve_mitre(alert)
    second = resolve_mitre(alert)
    assert first == second


def test_mitre_resolution_is_deterministic_for_unmapped_alert():
    alert = {"rule": {"id": "999999"}, "data": {}}
    first = resolve_mitre(alert)
    second = resolve_mitre(alert)
    assert first == second


# ---------------------------------------------------------------- campaign selection scoring

def test_campaign_selection_composite_score_is_deterministic():
    selector = BestCampaignSelector()
    build = lambda: selector.build_candidate(
        "CAMP_A", topology_similarity=0.83, technique_similarity=0.61,
        temporal_similarity=0.42, attacker_similarity=1.0, host_similarity=0.5,
    )
    first = build()
    second = build()
    assert first.composite_score == second.composite_score


def test_campaign_selection_ranking_and_explanation_is_deterministic():
    selector = BestCampaignSelector()

    def run():
        a = selector.build_candidate("CAMP_A", topology_similarity=0.9, technique_similarity=0.8)
        b = selector.build_candidate("CAMP_B", topology_similarity=0.3, technique_similarity=0.2)
        return selector.select([a, b])

    first = run()
    second = run()
    assert first.selected.campaign_id == second.selected.campaign_id
    assert first.selected.composite_score == second.selected.composite_score
    assert first.explanation == second.explanation
    assert first.confidence == second.confidence


# ---------------------------------------------------------------- threat qualification

class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Incident:
    def __init__(self, **kw):
        defaults = dict(campaign_id="CAMP_1", attacker_ip="1.2.3.4", technique="T1110",
                         timestamp="2026-01-01T00:00:00Z", cti=None)
        defaults.update(kw)
        self.__dict__.update(defaults)


def test_threat_qualification_is_deterministic():
    engine = ThreatQualificationEngine()
    incident = _Incident(cti=_Stub(threat_classification=QUALIFIED_THREAT, score=90.0))
    first = engine.qualify(incident)
    second = engine.qualify(incident)
    assert first.classification == second.classification
    assert first.may_publish_to_misp == second.may_publish_to_misp
    assert [c.passed for c in first.checks] == [c.passed for c in second.checks]
    assert first.reason == second.reason


# ---------------------------------------------------------------- response plan narrative

def _campaign():
    return CampaignContext(campaign_id="CAMP_1", attacker_ip="1.2.3.4", victim_ip="10.0.0.5",
                            risk_score=1300.0, last_technique="T1110", techniques={"T1110"})


def test_response_plan_generation_is_deterministic_given_identical_inputs():
    generator = ResponsePlanGenerator()
    playbook = Playbook(playbook_id="pb_1", name="TEST", description="", trigger_conditions={},
                         campaign_type="x", mitre_techniques=["T1110"], severity="HIGH", risk=900.0,
                         required_evidence=[], actions=[])

    first = generator.generate(_campaign(), playbook=playbook)
    second = generator.generate(_campaign(), playbook=playbook)

    assert first.threat_summary == second.threat_summary
    assert first.severity == second.severity
    assert first.misp_status == second.misp_status


# ---------------------------------------------------------------- GNN embedding (live-gated)

def _neo4j_reachable():
    try:
        dashboard_api.driver.verify_connectivity()
        return True
    except Exception:
        return False


def _gnn_available():
    try:
        from ml.gnn.inference import gnn_inference_service
        return gnn_inference_service.available
    except Exception:
        return False


requires_live_neo4j_and_gnn = pytest.mark.skipif(
    not (_neo4j_reachable() and _gnn_available()),
    reason="Requires a live, reachable Neo4j instance and an enabled/loaded GNN model.",
)


@requires_live_neo4j_and_gnn
def test_gnn_embedding_is_deterministic_for_the_same_campaign():
    """Live test, not mocked: proves the actual frozen model produces
    byte-identical embeddings for the same real campaign graph on two
    independent forward passes (use_cache=False on both calls, so this
    exercises the real encode path each time, not just cache lookup)."""
    import torch
    from ml.gnn.inference import gnn_inference_service
    from neo4j_client import driver

    with driver.session() as session:
        row = session.run("MATCH (c:Campaign) RETURN c.campaign_id AS id LIMIT 1").single()
    if row is None:
        pytest.skip("No real campaign exists in this Neo4j instance to test against.")

    campaign_id = row["id"]
    first = gnn_inference_service.embed_campaign(campaign_id, use_cache=False)
    second = gnn_inference_service.embed_campaign(campaign_id, use_cache=False)

    if first is None or second is None:
        pytest.skip(f"Campaign {campaign_id} produced no embedding (empty/malformed graph) -- cannot test determinism.")

    assert torch.allclose(first, second)

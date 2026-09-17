"""
Behavioral tests for CYUKTI's threat-attribution stack
(`threat_attribution_engine.py`, `attribution_context.py`,
`attribution_similarity.py`) -- previously code-present and live-called
(module_status.md rates this row **C**: "no attribution-accuracy
dataset exists... NOT MEASURED") but never behaviorally exercised
beyond the evidence-collector level (which uses hand-built fixtures,
not the real engine). These tests run the REAL
`ThreatAttributionEngine.attribute()` end to end; only the Neo4j
boundary (`attribution_context.context.load_historical_campaigns`,
`neo4j_client.driver`) is mocked.

Contract established by reading the implementation (not invented here):

- "Attribution" here means CAMPAIGN-to-CAMPAIGN similarity: candidates
  are PAST campaigns (`campaign.campaign_id` used directly as the
  `actor` identifier), scored by how similar their technique/chain
  footprint is to the CURRENT campaign -- not lookup against named
  threat-actor groups (e.g. "APT29"). That is a SEPARATE mechanism
  (`neo4j_client.update_actor_attribution` / dashboard_api.py's
  `/api/attribution/actors/<id>`, both matching against real
  `(:ThreatActor)-[:USES]->(:Technique)` nodes) -- see
  ../THREAT_ATTRIBUTION.md for why these are not unified here.
- Evidence pool: `attribution_context.load_historical_campaigns()`
  returns only campaigns with `status IN ['INACTIVE','ARCHIVED']` --
  the currently-active campaign under investigation can never attribute
  against itself or other still-open campaigns.
- Scoring: `total_score = (coverage*0.50 + precision*0.20 +
  chain_similarity*0.30) * 100`, where coverage = |observed ∩
  historical| / |observed| (asymmetric, denominator = CURRENT
  campaign's technique count), precision = |observed ∩ historical| /
  |historical|, chain_similarity = LCS(current chain, historical
  techniques) / len(current chain) (attribution_similarity.py -- note
  this LCS denominator differs from operation_feature_engine.py's
  chain_similarity, which divides by max(m, n); the two engines were
  written independently and are not meant to produce comparable chain
  scores).
- Zero-similarity candidates are dropped entirely (`if similarity == 0:
  continue`) rather than appearing as a manufactured "0% confidence"
  candidate -- the engine's insufficient-evidence signal is an empty
  (or shorter) `actors` list, not a low-confidence entry.
- Up to TOP_K=5 candidates, sorted descending by `total_score`; ties
  preserve `historical_campaigns` iteration order (Python's sort is
  stable), which is itself deterministic because
  `AttributionContext.load_historical_campaigns()`'s Cypher query has
  an explicit `ORDER BY c.campaign_id, e.first_seen`.
- No persistence: `ThreatAttributionEngine.attribute()` is read-only
  and ephemeral -- its result is consumed in-process (MISP event
  generation, evidence-aware investigation trace) but never written
  back to Neo4j. Confirmed structurally (no driver/session reference
  anywhere in threat_attribution_engine.py).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from attribution_models import HistoricalCampaign
from campaign_context import CampaignContext
from evidence.schema import Evidence, EvidenceSource, EvidenceType
from evidence.collectors.attribution_collector import collect_attribution_evidence
from evidence.collectors.campaign_history_collector import collect_campaign_history_evidence
from evidence.store import EvidenceStore
from threat_attribution_engine import ThreatAttributionEngine


def _historical(campaign_id, techniques, attacker="203.0.113.9", victim="198.51.100.9", status="INACTIVE"):
    return HistoricalCampaign(
        campaign_id=campaign_id, attacker=attacker, victim=victim,
        techniques=list(techniques), timestamps=["2026-01-01T00:00:00+00:00"] * len(techniques),
        status=status,
    )


def _campaign(techniques, attack_chain=None, campaign_id="CAMP_CURRENT"):
    return CampaignContext(
        campaign_id=campaign_id, attacker_ip="10.0.0.5", victim_ip="10.0.0.9",
        techniques=set(techniques), attack_chain=list(attack_chain if attack_chain is not None else techniques),
    )


@pytest.fixture()
def engine():
    return ThreatAttributionEngine()


@pytest.fixture()
def mock_history(monkeypatch):
    """Replaces the Neo4j-backed historical-campaign loader with a
    controllable in-memory list -- the same external boundary
    test_default_wiring.py's closed-loop test mocks."""
    import attribution_context as attribution_context_module

    state = {"campaigns": []}
    monkeypatch.setattr(
        attribution_context_module.context, "load_historical_campaigns",
        lambda: list(state["campaigns"]),
    )
    return state


# ---------------------------------------------------------------- A. positive attribution path

def test_strong_technique_overlap_produces_populated_attribution_candidate(engine, mock_history):
    mock_history["campaigns"] = [_historical("CAMP_OLD_1", ["T1110", "T1078", "T1059"])]
    campaign = _campaign(["T1110", "T1078", "T1059"])

    result = engine.attribute(campaign)

    assert len(result.actors) == 1
    actor = result.actors[0]
    assert actor.actor == "CAMP_OLD_1"
    assert actor.total_score == 100.0  # full coverage, precision, and chain match
    assert actor.confidence == 100.0
    assert actor.coverage == 100.0
    assert actor.precision == 100.0
    assert actor.matched_techniques == ["T1059", "T1078", "T1110"]  # sorted
    assert actor.evidence  # rationale strings populated
    assert any("Coverage" in e for e in actor.evidence)
    assert any("Matched 3 ATT&CK techniques" in e for e in actor.evidence)


def test_partial_overlap_computes_documented_weighted_formula(engine, mock_history):
    # observed={T1110,T1078}, historical={T1110,T1059,T1053}
    # coverage = |{T1110}| / |observed|=2 = 0.5
    # precision = |{T1110}| / |historical|=3 = 0.333
    # chain: LCS(current_chain=[T1110,T1078], historical.techniques=[T1110,T1059,T1053]) = 1 ("T1110") / len(current_chain)=2 = 0.5
    mock_history["campaigns"] = [_historical("CAMP_OLD_2", ["T1110", "T1059", "T1053"])]
    campaign = _campaign(["T1110", "T1078"])

    result = engine.attribute(campaign)

    assert len(result.actors) == 1
    actor = result.actors[0]
    expected = round((0.5 * 0.50 + (1 / 3) * 0.20 + 0.5 * 0.30) * 100, 2)
    assert actor.total_score == expected
    assert actor.coverage == 50.0
    assert actor.precision == round(100 / 3, 2)


# ---------------------------------------------------------------- B. insufficient evidence

def test_no_historical_campaigns_returns_empty_actors_not_fabricated(engine, mock_history):
    mock_history["campaigns"] = []
    result = engine.attribute(_campaign(["T1110"]))
    assert result.actors == []


def test_zero_technique_overlap_is_excluded_not_a_low_confidence_candidate(engine, mock_history):
    mock_history["campaigns"] = [_historical("CAMP_UNRELATED", ["T1595", "T1592"])]
    result = engine.attribute(_campaign(["T1110", "T1078"]))
    assert result.actors == []


def test_current_campaign_with_no_techniques_yields_no_attribution(engine, mock_history):
    mock_history["campaigns"] = [_historical("CAMP_OLD_1", ["T1110"])]
    result = engine.attribute(_campaign([]))
    assert result.actors == []


# ---------------------------------------------------------------- C. conflicting / competing evidence

def test_competing_candidates_are_ranked_by_the_real_weighted_score_not_arbitrarily(engine, mock_history):
    """Two plausible-looking historical matches that disagree about
    which is the better attribution; the ranking must follow the
    documented formula, not insertion order or technique count alone."""
    mock_history["campaigns"] = [
        # High coverage/precision (small, fully-contained match) but no chain order preserved
        _historical("CAMP_HIGH_PRECISION", ["T1110", "T1078"]),
        # Larger technique set with more noise, lower precision
        _historical("CAMP_LOW_PRECISION", ["T1110", "T1078", "T1595", "T1592", "T1590"]),
    ]
    campaign = _campaign(["T1110", "T1078"])

    result = engine.attribute(campaign)

    by_id = {a.actor: a for a in result.actors}
    # coverage identical (2/2=1.0 both), precision differs: 2/2=1.0 vs 2/5=0.4
    assert by_id["CAMP_HIGH_PRECISION"].precision == 100.0
    assert by_id["CAMP_LOW_PRECISION"].precision == 40.0
    assert by_id["CAMP_HIGH_PRECISION"].total_score > by_id["CAMP_LOW_PRECISION"].total_score
    assert result.actors[0].actor == "CAMP_HIGH_PRECISION"  # higher score ranks first
    assert result.actors == sorted(result.actors, key=lambda a: a.total_score, reverse=True)


# ---------------------------------------------------------------- D. multiple candidates / top-k

def test_more_than_top_k_candidates_truncates_deterministically(engine, mock_history):
    mock_history["campaigns"] = [
        _historical(f"CAMP_{i}", ["T1110", "T1078"] + ([f"T900{i}"] if i else []))
        for i in range(8)  # CAMP_0 is the exact match; CAMP_1..7 each add one noise technique
    ]
    campaign = _campaign(["T1110", "T1078"])

    result = engine.attribute(campaign)

    assert len(result.actors) == 5  # TOP_K
    assert result.actors[0].actor == "CAMP_0"  # exact match (no noise) scores highest
    scores = [a.total_score for a in result.actors]
    assert scores == sorted(scores, reverse=True)


def test_tied_candidates_preserve_deterministic_input_order(engine, mock_history):
    """Ties are resolved by input order (Python's stable sort);
    load_historical_campaigns' own query is ORDER BY campaign_id, so
    this composes into a fully deterministic overall ordering."""
    mock_history["campaigns"] = [
        _historical("CAMP_A", ["T1110", "T1078"]),
        _historical("CAMP_B", ["T1110", "T1078"]),
    ]
    result = engine.attribute(_campaign(["T1110", "T1078"]))

    assert result.actors[0].total_score == result.actors[1].total_score
    assert [a.actor for a in result.actors] == ["CAMP_A", "CAMP_B"]


# ---------------------------------------------------------------- E. campaign-history contribution (wiring)

def test_attribute_actually_calls_the_real_historical_campaign_loader(engine, monkeypatch):
    import attribution_context as attribution_context_module

    calls = []

    def fake_loader():
        calls.append(1)
        return [_historical("CAMP_OLD", ["T1110"])]

    monkeypatch.setattr(attribution_context_module.context, "load_historical_campaigns", fake_loader)

    engine.attribute(_campaign(["T1110"]))

    assert len(calls) == 1


def test_historical_campaign_query_excludes_active_campaigns():
    """Structural check on the real query text: only a live Neo4j
    instance could prove this behaviorally, but the WHERE clause that
    enforces "never attribute against the still-open campaign itself
    or other in-progress campaigns" is inspectable directly."""
    import inspect

    import attribution_context as attribution_context_module

    source = inspect.getsource(attribution_context_module.AttributionContext.load_historical_campaigns)
    assert "INACTIVE" in source and "ARCHIVED" in source
    assert "ACTIVE" not in source.replace("INACTIVE", "").replace("PROACTIVE", "")


# ---------------------------------------------------------------- F. technique-signal contribution

def test_chain_order_not_just_set_overlap_affects_score(engine, mock_history):
    """The engine consumes the ATT&CK-technique-derived attack_chain
    (an ordered sequence, distinct from the unordered technique set) --
    two campaigns with an IDENTICAL technique set but different
    historical technique orderings must not necessarily score
    identically, since chain_similarity is order-sensitive (LCS)."""
    mock_history["campaigns"] = [_historical("CAMP_ORDERED", ["T1110", "T1078", "T1059"])]

    in_order = engine.attribute(_campaign(["T1110", "T1078", "T1059"], attack_chain=["T1110", "T1078", "T1059"]))
    reversed_order = engine.attribute(_campaign(["T1110", "T1078", "T1059"], attack_chain=["T1059", "T1078", "T1110"]))

    assert in_order.actors[0].chain_similarity > reversed_order.actors[0].chain_similarity
    # coverage/precision (set-based) are unaffected by order
    assert in_order.actors[0].coverage == reversed_order.actors[0].coverage


# ---------------------------------------------------------------- G. persistence (intentionally absent here)

def test_attribute_engine_never_writes_to_neo4j():
    """ThreatAttributionEngine.attribute() is read-only by design --
    its result is ephemeral (consumed by MISP export and the
    evidence-aware investigation trace, see test H below), never
    persisted back to the graph. A SEPARATE mechanism
    (neo4j_client.update_actor_attribution, matching against real
    ThreatActor nodes rather than historical campaigns) does persist a
    different kind of attribution -- deliberately not unified with this
    engine (see THREAT_ATTRIBUTION.md)."""
    import inspect

    import threat_attribution_engine

    source = inspect.getsource(threat_attribution_engine)
    for forbidden in ("driver.session", "MERGE", "CREATE", ".run("):
        assert forbidden not in source


# ---------------------------------------------------------------- H. downstream propagation

def test_attribution_result_propagates_into_misp_event_generation(engine, mock_history):
    from types import SimpleNamespace

    from misp_event_generator import IncidentContext, MISPEventGenerator

    mock_history["campaigns"] = [_historical("CAMP_OLD_1", ["T1110", "T1078"])]
    campaign = _campaign(["T1110", "T1078"])
    attribution = engine.attribute(campaign)
    assert attribution.actors  # sanity: this test needs a real candidate to prove propagation

    detection = SimpleNamespace(confidence=0.5, level="MEDIUM", breakdown={
        "Wazuh": 1, "Suricata": 1, "Zeek": 1, "Sigma": 1, "YARA": 1,
    })
    dynamic_risk = SimpleNamespace(risk_score=50.0, risk_level="MEDIUM", confidence=0.5)
    cti = SimpleNamespace(score=50.0, level="MEDIUM", publish=False)

    incident = IncidentContext(
        campaign_id="CAMP_CURRENT", operation_id="OP_1", attacker_ip="10.0.0.5", victim_ip="10.0.0.9",
        event_id="evt-1", technique="T1078", stage="Initial Access", prediction="T1059",
        prediction_confidence=0.5, detection=detection, threat=None, dynamic_risk=dynamic_risk, cti=cti,
        recommendations=[], investigation_payload="{}", timestamp="2026-01-01T00:00:00+00:00",
        attribution=attribution,
    )

    misp_payload = MISPEventGenerator().generate(incident)

    best = attribution.actors[0]
    attributes = misp_payload["Event"]["Attribute"]
    tag_names = [t["name"] for t in misp_payload["Event"]["Tag"]]
    assert any(a["value"] == f"Threat Actor : {best.actor}" for a in attributes)
    assert f"actor:{best.actor}" in tag_names
    assert f"attribution:{int(best.total_score)}" in tag_names


def test_attribution_result_propagates_into_investigation_trace_via_evidence(engine, mock_history):
    """The OTHER real downstream consumer: investigation/loop.py's
    ATTRIBUTION_MATCH action wraps attribute()'s output as Evidence,
    which reaches InvestigationRecord.to_dict() -- the actual JSON
    /api/investigate/<campaign_id> serves. dashboard_api.py does NOT
    expose ThreatAttributionEngine directly through a dedicated
    endpoint (the campaign_id-scoped /api/attribution/actors/<id> route
    queries a different, ThreatActor-node-based mechanism entirely --
    see module docstring); this is the real propagation path for THIS
    engine's output."""
    mock_history["campaigns"] = [_historical("CAMP_OLD_1", ["T1110", "T1078"])]
    campaign = _campaign(["T1110", "T1078"])
    attribution = engine.attribute(campaign)

    evidence_items = collect_attribution_evidence(attribution)
    store = EvidenceStore()
    store.add_many(evidence_items)

    payload = [e.to_dict() for e in store.all()]
    json.dumps(payload)  # must be JSON-serializable, matching the real API contract
    assert payload[0]["source"] == "attribution"
    assert payload[0]["content"]["candidate_campaign_id"] == "CAMP_OLD_1"


# ---------------------------------------------------------------- evidence provenance (Section 7)

def test_attribution_evidence_derives_from_campaign_history_evidence_with_real_candidates():
    """Uses the REAL engine's multi-candidate output (not the
    hand-built single-item fixtures test_investigation_evidence_aware.py
    used) to prove investigation/loop.py's existing derived_from
    mechanism -- ActionMeta.depends_on(ATTRIBUTION_MATCH ->
    CAMPAIGN_HISTORY) -- actually engages with realistic attribution
    data, not just a synthetic one-item case."""
    from investigation.actions import ACTION_METADATA, InvestigationAction
    from investigation.loop import _tag_derived_from

    historical = [
        _historical("CAMP_OLD_1", ["T1110", "T1078"]),
        _historical("CAMP_OLD_2", ["T1110", "T1059"]),
    ]
    campaign = _campaign(["T1110", "T1078", "T1059"])

    history_evidence = collect_campaign_history_evidence(campaign.techniques, historical)
    attribution_result = ThreatAttributionEngine().attribute(campaign)
    # sanity: the real engine actually produced >1 candidate for this fixture
    assert len(attribution_result.actors) >= 2
    attribution_evidence = collect_attribution_evidence(attribution_result)

    store = EvidenceStore()
    store.add_many(history_evidence)

    _tag_derived_from(attribution_evidence, InvestigationAction.ATTRIBUTION_MATCH, store)

    history_ids = {e.evidence_id for e in history_evidence}
    for item in attribution_evidence:
        assert history_ids <= set(item.derived_from), (
            "every attribution candidate must trace back to the campaign-history "
            "evidence its computation depends on, per ActionMeta.depends_on"
        )
    assert ACTION_METADATA[InvestigationAction.ATTRIBUTION_MATCH].depends_on == frozenset(
        {InvestigationAction.CAMPAIGN_HISTORY}
    )

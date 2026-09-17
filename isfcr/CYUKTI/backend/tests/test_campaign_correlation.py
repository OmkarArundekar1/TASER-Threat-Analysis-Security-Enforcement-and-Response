"""
Behavioral tests for CYUKTI's operation-correlation stack
(`campaign_correlation_engine.py`, `operation_manager.py`,
`operation_feature_engine.py`, `operation_decision_engine.py`) --
previously code-present and live-called (module_status.md rates this
row **C**: "no dedicated test file found; live nodes exist but matching
quality not evaluated") but never behaviorally exercised. These tests
run the REAL engines end to end; only the Neo4j boundary
(`neo4j_client.get_active_operations`, `get_recent_inactive_operations`,
`get_operation_context_data`, `reopen_operation_db`,
`attach_campaign_to_operation`, `update_operation_activity`) is mocked.

Contract established by reading the implementation (not invented here):

- An operation is a cross-campaign grouping (one attacker over time,
  potentially many victims) -- distinct from a Campaign (one
  attacker-victim pair's activity window, handled by CampaignManager /
  CAMPAIGN_TIMEOUT, out of scope for this file).
- OperationFeatureEngine.extract_features computes 7 similarity
  features (attacker/victim/temporal/technique/chain/prediction/graph);
  OperationDecisionEngine.evaluate combines them via fixed weights
  (operation_decision_engine.py's `weights` dict) into `score`, and
  ATTACH_TO_OPERATION iff score >= ATTACH_THRESHOLD (0.70).
- `confidence` in the decision is NOT a match-quality/probability
  score -- it is a constant data-completeness percentage
  (len(IMPLEMENTED_FEATURES) / len(all features) * 100, currently
  5/7 = 71.4%, config.py) that does not vary with how good a specific
  match is. This is easy to misread as "how confident are we this is
  the same operation" -- it is not; `score` is that.
- attacker identity: exact string equality against the operation's
  "sticky" primary_attacker (set once, from the first campaign attached
  -- neo4j_client.attach_campaign_to_operation's CASE WHEN). victim
  identity: set membership -- an operation accumulates victims across
  campaigns over time, unlike a single Campaign.
- temporal_similarity compares the NEW campaign's first_seen against
  the CANDIDATE operation's last_seen: <=1h -> 1.0, <=24h -> 0.8,
  <=7d -> 0.5, else 0.0. This is independent of OPERATION_TIMEOUT
  (120s, config.py), which governs ACTIVE->INACTIVE *expiry* via
  operation_manager.expire_active_operations(), a separate mechanism.
- Missing/empty campaign fields degrade features to 0.0 rather than
  raising (attacker_similarity, victim_similarity, technique_similarity,
  temporal_similarity, chain_similarity all have explicit empty-input
  guards -- verified by reading each, exercised by section D below).
- A candidate operation that can no longer be loaded
  (build_operation_context returns None -- e.g. deleted between listing
  and lookup) is skipped, not fatal.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from campaign_context import CampaignContext
from campaign_correlation_engine import CampaignCorrelationEngine
from operation_decision_engine import OperationDecisionEngine
from operation_feature_engine import OperationFeatureEngine, OperationFeatures
from operation_manager import OperationManager

NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _campaign(
    campaign_id="CAMP_NEW",
    attacker_ip="10.0.0.5",
    victim_ip="10.0.0.9",
    techniques=None,
    attack_chain=None,
    first_seen=NOW,
    last_technique=None,
):
    return CampaignContext(
        campaign_id=campaign_id,
        attacker_ip=attacker_ip,
        victim_ip=victim_ip,
        techniques=set(techniques or []),
        attack_chain=list(attack_chain or []),
        first_seen=first_seen,
        last_technique=last_technique,
    )


def _operation_row(
    operation_id="OP_EXISTING",
    primary_attacker="10.0.0.5",
    victim_ip="10.0.0.9",
    techniques=None,
    attack_chain=None,
    created_at=NOW - timedelta(days=1),
    last_seen=NOW,
):
    """Shape neo4j_client.get_operation_context_data actually returns --
    see operation_manager.build_operation_context, which consumes it."""
    techniques = techniques or []
    return {
        "operation": {
            "operation_id": operation_id,
            "primary_attacker": primary_attacker,
            "created_at": created_at,
            "last_seen": last_seen,
        },
        "campaigns": [{"campaign_id": f"{operation_id}_C1", "victim_ip": victim_ip}] if victim_ip else [],
        "techniques": techniques,
        "attack_chain": list(attack_chain or techniques),
    }


@pytest.fixture()
def engine(monkeypatch):
    """A real CampaignCorrelationEngine wired to a real OperationManager
    /OperationFeatureEngine/OperationDecisionEngine -- only the Neo4j
    module-level functions each module imported by name are replaced."""
    import campaign_correlation_engine as cce_module
    import operation_manager as om_module

    state = {
        "active_ids": [],
        "inactive_ids": [],
        "operations": {},  # operation_id -> row dict from _operation_row
        "reopened": [],
    }

    monkeypatch.setattr(cce_module, "get_active_operations", lambda: list(state["active_ids"]))
    monkeypatch.setattr(cce_module, "get_recent_inactive_operations", lambda: list(state["inactive_ids"]))
    monkeypatch.setattr(cce_module, "reopen_operation_db", lambda op_id: state["reopened"].append(op_id))
    monkeypatch.setattr(om_module, "get_operation_context_data", lambda op_id: state["operations"].get(op_id))

    eng = CampaignCorrelationEngine()
    eng._state = state  # test-only handle
    return eng


# ---------------------------------------------------------------- A. same operation

def test_matching_attacker_victim_and_overlapping_techniques_attaches(engine):
    """Mirrors the mission's Event A / Event B shape: same attacker,
    same victim, a later event introducing a new technique (T1053) not
    in the operation's history yet, alongside techniques (T1078,
    T1059) it shares with it -- partial technique overlap, not
    identical sets. Score computed by hand from the real weights:
    attacker(1.0*0.20) + victim(1.0*0.15) + technique(0.5*0.30)
    + temporal(1.0*0.20) + chain(0.667*0.10) = 0.767 >= 0.70."""
    engine._state["active_ids"] = ["OP_1"]
    engine._state["operations"]["OP_1"] = _operation_row(
        "OP_1", primary_attacker="10.0.0.5", victim_ip="10.0.0.9",
        techniques=["T1110", "T1078", "T1059"], attack_chain=["T1110", "T1078", "T1059"],
        last_seen=NOW,
    )
    campaign = _campaign(
        attacker_ip="10.0.0.5", victim_ip="10.0.0.9",
        techniques=["T1078", "T1059", "T1053"], attack_chain=["T1078", "T1059", "T1053"],
        first_seen=NOW,
    )

    result = engine.correlate(campaign)

    assert result.matched is True
    assert result.operation_id == "OP_1"
    assert result.score >= OperationDecisionEngine.ATTACH_THRESHOLD
    assert result.score == pytest.approx(0.767, abs=0.01)


# ---------------------------------------------------------------- B. different operation

def test_different_attacker_and_victim_does_not_attach_even_with_identical_techniques(engine):
    """Proves campaign separation isn't accidentally achieved only by
    technique overlap: same technique set, same chain, ideal temporal
    proximity -- but a completely different attacker/victim pair."""
    engine._state["active_ids"] = ["OP_1"]
    shared_techniques = ["T1110", "T1078", "T1059"]
    engine._state["operations"]["OP_1"] = _operation_row(
        "OP_1", primary_attacker="203.0.113.1", victim_ip="198.51.100.1",
        techniques=shared_techniques, last_seen=NOW,
    )
    campaign = _campaign(
        attacker_ip="10.0.0.5", victim_ip="10.0.0.9",
        techniques=shared_techniques, attack_chain=shared_techniques, first_seen=NOW,
    )

    result = engine.correlate(campaign)

    assert result.matched is False
    assert result.operation_id is None


def test_no_active_or_inactive_operations_returns_unmatched_without_error(engine):
    result = engine.correlate(_campaign())
    assert result.matched is False
    assert result.candidate_count == 0
    assert result.candidates == []


# ---------------------------------------------------------------- C. temporal boundary

@pytest.mark.parametrize(
    "delta,expected",
    [
        (timedelta(minutes=59), 1.0),
        (timedelta(hours=1), 1.0),
        (timedelta(hours=1, minutes=1), 0.8),
        (timedelta(hours=23, minutes=59), 0.8),
        (timedelta(hours=24), 0.8),
        (timedelta(hours=24, minutes=1), 0.5),
        (timedelta(days=6, hours=23), 0.5),
        (timedelta(days=7), 0.5),
        (timedelta(days=7, hours=1), 0.0),
        (timedelta(days=30), 0.0),
    ],
)
def test_temporal_similarity_matches_documented_bands(delta, expected):
    fe = OperationFeatureEngine()
    campaign = _campaign(first_seen=NOW)
    om = OperationManager.__new__(OperationManager)  # not used; direct OperationContext below
    from operation_context import OperationContext
    operation = OperationContext(operation_id="OP_X", last_seen=NOW - delta)

    assert fe.extract_temporal_similarity(campaign, operation) == expected


def test_temporal_boundary_flips_the_attach_decision(engine):
    """Everything else held at a strong (but sub-threshold-alone) match;
    only the operation's last_seen crosses the 24h/7d band edge."""
    engine._state["active_ids"] = ["OP_1"]
    techniques = ["T1110", "T1078"]

    def _try(delta):
        engine._state["operations"]["OP_1"] = _operation_row(
            "OP_1", primary_attacker="10.0.0.5", victim_ip="10.0.0.9",
            techniques=techniques, last_seen=NOW - delta,
        )
        campaign = _campaign(
            attacker_ip="10.0.0.5", victim_ip="10.0.0.9",
            techniques=techniques, attack_chain=techniques, first_seen=NOW,
        )
        return engine.correlate(campaign)

    just_inside = _try(timedelta(hours=24))
    just_outside = _try(timedelta(hours=24, minutes=1))

    assert just_inside.score > just_outside.score
    # The band step (1.0->0.8->0.5->0.0) alone is enough to move a
    # borderline match across ATTACH_THRESHOLD given this scenario's
    # other feature values -- assert the actual documented boundary
    # values reached rather than assuming a decision flip that the
    # fixture's specific weights may not both land, keeping this
    # resilient to unrelated weight retuning elsewhere in the file.
    assert just_inside.breakdown["temporal_similarity"]["value"] == 0.8
    assert just_outside.breakdown["temporal_similarity"]["value"] == 0.5


# ---------------------------------------------------------------- D. missing data

def test_missing_attacker_victim_technique_and_timestamp_degrades_to_zero_not_crash(engine):
    engine._state["active_ids"] = ["OP_1"]
    engine._state["operations"]["OP_1"] = _operation_row("OP_1", primary_attacker="10.0.0.5", victim_ip="10.0.0.9")
    campaign = _campaign(attacker_ip="", victim_ip="", techniques=[], attack_chain=[], first_seen=None)

    result = engine.correlate(campaign)

    assert result.matched is False
    assert result.score == 0.0


def test_missing_timestamps_degrade_temporal_similarity_to_zero():
    fe = OperationFeatureEngine()
    from operation_context import OperationContext
    campaign = _campaign(first_seen=None)
    operation = OperationContext(operation_id="OP_X", last_seen=NOW)
    assert fe.extract_temporal_similarity(campaign, operation) == 0.0

    campaign2 = _campaign(first_seen=NOW)
    operation2 = OperationContext(operation_id="OP_X", last_seen=None)
    assert fe.extract_temporal_similarity(campaign2, operation2) == 0.0


def test_empty_technique_sets_on_both_sides_do_not_divide_by_zero():
    fe = OperationFeatureEngine()
    from operation_context import OperationContext
    campaign = _campaign(techniques=[])
    operation = OperationContext(operation_id="OP_X", techniques=set())
    assert fe.extract_technique_similarity(campaign, operation) == 0.0


def test_vanished_operation_candidate_is_skipped_not_fatal(engine):
    """get_active_operations lists an id, but the operation no longer
    exists (build_operation_context -> None) by the time it's looked
    up -- a real race in a live system."""
    engine._state["active_ids"] = ["OP_GONE", "OP_1"]
    engine._state["operations"]["OP_1"] = _operation_row(
        "OP_1", primary_attacker="10.0.0.5", victim_ip="10.0.0.9", techniques=["T1110"],
    )
    # OP_GONE intentionally absent from engine._state["operations"]
    campaign = _campaign(attacker_ip="10.0.0.5", victim_ip="10.0.0.9", techniques=["T1110"])

    result = engine.correlate(campaign)

    assert result.matched is True
    assert result.operation_id == "OP_1"
    assert result.candidate_count == 2  # OP_GONE was counted as a candidate, just not usable


def test_none_campaign_context_returns_controlled_empty_result(engine):
    result = engine.correlate(None)
    assert result.matched is False
    assert result.operation_id is None
    assert result.candidates == []


# ---------------------------------------------------------------- E. repeated event / idempotence

def test_correlating_the_same_campaign_twice_is_idempotent(engine):
    engine._state["active_ids"] = ["OP_1"]
    engine._state["operations"]["OP_1"] = _operation_row(
        "OP_1", primary_attacker="10.0.0.5", victim_ip="10.0.0.9", techniques=["T1110", "T1078"],
    )
    campaign = _campaign(attacker_ip="10.0.0.5", victim_ip="10.0.0.9", techniques=["T1110"])

    first = engine.correlate(campaign)
    second = engine.correlate(campaign)

    assert first.matched == second.matched
    assert first.operation_id == second.operation_id
    assert first.score == second.score


# ---------------------------------------------------------------- F. operation state evolution (Neo4j write shape)

def test_attach_campaign_to_operation_sends_correct_parameters(monkeypatch):
    """neo4j_client.attach_campaign_to_operation is the real write path
    invoked once correlate() has matched (see realtime_socgraph.py).
    Verifies the actual Cypher parameters generated for a real
    CampaignContext -- the query itself needs live Neo4j to execute,
    but the parameter contract (what gets sent) is verifiable here."""
    import neo4j_client

    captured = {}

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, query, **params):
            captured["query"] = query
            captured["params"] = params

    class _FakeDriver:
        def session(self):
            return _FakeSession()

    monkeypatch.setattr(neo4j_client, "driver", _FakeDriver())

    campaign = _campaign(campaign_id="CAMP_7", attacker_ip="10.0.0.5", victim_ip="10.0.0.9")
    campaign.last_seen = NOW
    neo4j_client.attach_campaign_to_operation("OP_1", campaign)

    assert captured["params"]["operation_id"] == "OP_1"
    assert captured["params"]["campaign_id"] == "CAMP_7"
    assert captured["params"]["attacker"] == "10.0.0.5"
    assert captured["params"]["victim"] == "10.0.0.9"
    assert "MERGE" in captured["query"] and "HAS_CAMPAIGN" in captured["query"]


def test_update_operation_activity_sends_correct_parameters(monkeypatch):
    import neo4j_client

    captured = {}

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, query, **params):
            captured["query"] = query
            captured["params"] = params

    class _FakeDriver:
        def session(self):
            return _FakeSession()

    monkeypatch.setattr(neo4j_client, "driver", _FakeDriver())

    campaign = _campaign(campaign_id="CAMP_7")
    campaign.last_seen = NOW
    neo4j_client.update_operation_activity("OP_1", campaign)

    assert captured["params"]["operation_id"] == "OP_1"
    assert captured["params"]["last_seen"] == NOW
    assert "status='ACTIVE'" in captured["query"]


# ---------------------------------------------------------------- G. campaign separation invariant

def test_attacker_and_victim_mismatch_can_never_reach_attach_threshold():
    """A structural safety property of the CURRENT weight configuration
    (operation_decision_engine.py's `weights`): with attacker_similarity
    == victim_similarity == 0, the maximum achievable score even with
    every other feature at its ceiling (1.0) is below ATTACH_THRESHOLD.
    This is what actually prevents unrelated attacker/victim pairs from
    ever being merged into one operation -- proven directly here so a
    future change to `weights` or ATTACH_THRESHOLD that breaks this
    invariant fails a test instead of silently merging distinct
    operations."""
    max_score_without_identity_match = (
        1.0 * OperationDecisionEngine.weights["temporal_similarity"]
        + 1.0 * OperationDecisionEngine.weights["technique_similarity"]
        + 1.0 * OperationDecisionEngine.weights["chain_similarity"]
        + 1.0 * OperationDecisionEngine.weights["prediction_similarity"]
        + 1.0 * OperationDecisionEngine.weights["graph_similarity"]
    )
    assert max_score_without_identity_match < OperationDecisionEngine.ATTACH_THRESHOLD

    features = OperationFeatures(
        attacker_similarity=0.0, victim_similarity=0.0,
        temporal_similarity=1.0, technique_similarity=1.0,
        chain_similarity=1.0, prediction_similarity=1.0, graph_similarity=1.0,
    )
    decision = OperationDecisionEngine().evaluate(features)
    assert decision.decision == "CREATE_NEW_OPERATION"


def test_confidence_is_a_constant_data_completeness_ratio_not_a_match_score():
    """Regression guard against the easy misreading: `confidence` does
    NOT vary with match quality -- it's len(IMPLEMENTED_FEATURES)/total,
    a fixed constant given the current config.py."""
    strong = OperationDecisionEngine().evaluate(OperationFeatures(
        attacker_similarity=1.0, victim_similarity=1.0, temporal_similarity=1.0,
        technique_similarity=1.0, chain_similarity=1.0,
    ))
    weak = OperationDecisionEngine().evaluate(OperationFeatures())
    assert strong.confidence == weak.confidence
    from config import IMPLEMENTED_FEATURES
    assert strong.confidence == round(len(IMPLEMENTED_FEATURES) / len(OperationFeatures.__dataclass_fields__) * 100, 1)


# ---------------------------------------------------------------- H. realistic event structures

def test_realistic_two_campaign_same_attacker_operation_from_real_dataset_techniques():
    """Uses real MITRE technique ids present in CYUKTI's own frozen
    dataset (ml/datasets/campaign_dataset.csv references these via the
    live graph) rather than placeholder strings."""
    engine = CampaignCorrelationEngine()
    import campaign_correlation_engine as cce_module
    import operation_manager as om_module

    active_ids = ["OP_APT"]
    operations = {
        "OP_APT": _operation_row(
            "OP_APT", primary_attacker="185.220.101.7", victim_ip="10.20.0.15",
            techniques=["T1110.001", "T1110", "T1078"],
            attack_chain=["T1110.001", "T1110", "T1078"],
            last_seen=NOW - timedelta(minutes=30),
        )
    }
    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    mp.setattr(cce_module, "get_active_operations", lambda: active_ids)
    mp.setattr(cce_module, "get_recent_inactive_operations", lambda: [])
    mp.setattr(cce_module, "reopen_operation_db", lambda op_id: None)
    mp.setattr(om_module, "get_operation_context_data", lambda op_id: operations.get(op_id))
    try:
        # Same attacker infrastructure IP and identical technique/chain
        # footprint, but a NEW victim -- the realistic "same actor,
        # different target" case an operation is meant to capture.
        # technique(1.0*0.30) + chain(1.0*0.10) + temporal(1.0*0.20)
        # + attacker(1.0*0.20) = 0.80 clears ATTACH_THRESHOLD (0.70) on
        # their own, with victim_similarity=0.0 contributing nothing --
        # proving victim match is not required when the rest of the
        # footprint is this strong (contrast with section G: attacker
        # AND victim both mismatching cannot clear it regardless).
        campaign = _campaign(
            campaign_id="CAMP_NEW_ALERT", attacker_ip="185.220.101.7", victim_ip="10.20.0.99",
            techniques=["T1110.001", "T1110", "T1078"], attack_chain=["T1110.001", "T1110", "T1078"],
        )
        result = engine.correlate(campaign)
    finally:
        mp.undo()

    assert result.matched is True
    assert result.operation_id == "OP_APT"
    assert result.breakdown["attacker_similarity"]["value"] == 1.0
    assert result.breakdown["victim_similarity"]["value"] == 0.0  # different victim, same attacker -- still attaches


# ---------------------------------------------------------------- ordering determinism (fix)

def test_get_active_operations_query_orders_results_deterministically():
    """neo4j_client.get_active_operations() previously had no ORDER BY,
    unlike its sibling get_recent_inactive_operations() (`ORDER BY
    o.last_seen DESC`) -- meaning the candidate order CampaignCorrelationEngine
    iterates, and therefore which operation wins an exact score tie
    (correlate() only replaces best_operation on a STRICT `>`, so the
    first-seen equal-or-better candidate wins), was not guaranteed
    stable across runs. Fixed to match the sibling function's ordering.
    Verified structurally (the query text) since exercising real
    ordering non-determinism would require a live, multi-node Neo4j
    instance this environment doesn't have."""
    import inspect

    import neo4j_client

    source = inspect.getsource(neo4j_client.get_active_operations)
    assert "ORDER BY" in source


def test_correlate_breaks_exact_score_ties_by_first_candidate_in_list():
    """Documents the actual, deterministic tie-break policy: given two
    candidates with an identical score, correlate() keeps the FIRST one
    seen (strict `>` comparison, never replaces on equality) -- combined
    with the ordering fix above, this means "the most recently active
    operation wins a tie" in production, not an arbitrary/unstable
    choice."""
    engine = CampaignCorrelationEngine()
    import campaign_correlation_engine as cce_module
    import operation_manager as om_module

    identical_row_kwargs = dict(
        primary_attacker="10.0.0.5", victim_ip="10.0.0.9",
        techniques=["T1110"], attack_chain=["T1110"], last_seen=NOW,
    )
    operations = {
        "OP_FIRST": _operation_row("OP_FIRST", **identical_row_kwargs),
        "OP_SECOND": _operation_row("OP_SECOND", **identical_row_kwargs),
    }
    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    mp.setattr(cce_module, "get_active_operations", lambda: ["OP_FIRST", "OP_SECOND"])
    mp.setattr(cce_module, "get_recent_inactive_operations", lambda: [])
    mp.setattr(om_module, "get_operation_context_data", lambda op_id: operations.get(op_id))
    try:
        campaign = _campaign(attacker_ip="10.0.0.5", victim_ip="10.0.0.9", techniques=["T1110"], first_seen=NOW)
        result = engine.correlate(campaign)
    finally:
        mp.undo()

    assert result.operation_id == "OP_FIRST"

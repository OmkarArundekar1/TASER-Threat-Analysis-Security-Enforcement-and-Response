from datetime import datetime, timezone, timedelta

import pytest

from campaign_context import CampaignContext
from campaign_selection import (
    BestCampaignSelector,
    _ip_similarity,
    _temporal_similarity,
)

selector = BestCampaignSelector()


# ---------------------------------------------------------------- pure helpers

def test_ip_similarity_exact_match_is_one():
    assert _ip_similarity("1.2.3.4", "1.2.3.4") == 1.0


def test_ip_similarity_same_subnet_is_half():
    assert _ip_similarity("1.2.3.4", "1.2.3.9") == 0.5


def test_ip_similarity_different_subnet_is_zero():
    assert _ip_similarity("1.2.3.4", "9.9.9.9") == 0.0


def test_ip_similarity_none_when_either_ip_missing():
    assert _ip_similarity(None, "1.2.3.4") is None
    assert _ip_similarity("1.2.3.4", None) is None


def test_temporal_similarity_same_instant_is_one():
    t = datetime.now(timezone.utc)
    assert _temporal_similarity(t, t) == 1.0


def test_temporal_similarity_a_week_apart_is_zero():
    t1 = datetime.now(timezone.utc)
    t2 = t1 + timedelta(days=8)
    assert _temporal_similarity(t1, t2) == 0.0


def test_temporal_similarity_none_when_either_missing():
    assert _temporal_similarity(None, datetime.now(timezone.utc)) is None


# ---------------------------------------------------------------- composite score / build_candidate

def test_build_candidate_composite_score_uses_only_available_signals():
    candidate = selector.build_candidate("CAMP_A", topology_similarity=1.0)
    # only one signal available -> composite score equals that signal after renormalization
    assert candidate.composite_score == pytest.approx(1.0)


def test_build_candidate_composite_score_weighted_average_of_available_signals():
    candidate = selector.build_candidate("CAMP_A", topology_similarity=1.0, technique_similarity=0.0)
    # weights: topology 0.30, technique 0.25 -> renormalized: 0.30/0.55, 0.25/0.55
    expected = (0.30 * 1.0 + 0.25 * 0.0) / 0.55
    assert candidate.composite_score == pytest.approx(expected)


def test_build_candidate_with_no_signals_has_zero_composite_score():
    candidate = selector.build_candidate("CAMP_A")
    assert candidate.composite_score == 0.0


# ---------------------------------------------------------------- select()

def test_select_with_no_candidates_returns_none_selected():
    result = selector.select([])
    assert result.selected is None
    assert result.confidence == "NONE"


def test_select_ranks_by_composite_score_highest_first():
    a = selector.build_candidate("CAMP_A", topology_similarity=0.9, technique_similarity=0.9)
    b = selector.build_candidate("CAMP_B", topology_similarity=0.1, technique_similarity=0.1)
    result = selector.select([b, a])
    assert result.selected.campaign_id == "CAMP_A"
    assert result.ranked_candidates[0].campaign_id == "CAMP_A"
    assert result.alternatives[0].campaign_id == "CAMP_B"


def test_select_confidence_high_with_large_score_gap():
    a = selector.build_candidate("CAMP_A", topology_similarity=1.0)
    b = selector.build_candidate("CAMP_B", topology_similarity=0.1)
    result = selector.select([a, b])
    assert result.confidence == "HIGH"
    assert result.score_gap == pytest.approx(0.9)


def test_select_confidence_low_with_small_score_gap():
    a = selector.build_candidate("CAMP_A", topology_similarity=0.51)
    b = selector.build_candidate("CAMP_B", topology_similarity=0.50)
    result = selector.select([a, b])
    assert result.confidence == "LOW"


def test_select_confidence_high_with_single_candidate():
    a = selector.build_candidate("CAMP_A", topology_similarity=0.5)
    result = selector.select([a])
    assert result.confidence == "HIGH"
    assert result.score_gap is None


def test_explanation_names_strongest_and_weakest_signal():
    a = selector.build_candidate("CAMP_A", topology_similarity=0.95, attacker_similarity=0.1)
    b = selector.build_candidate("CAMP_B", topology_similarity=0.2)
    result = selector.select([a, b])
    assert "CAMP_A" in result.explanation
    assert "topology" in result.explanation
    assert "attacker" in result.explanation


def test_explanation_never_fabricates_when_only_one_signal_available():
    a = selector.build_candidate("CAMP_A", topology_similarity=0.9)
    result = selector.select([a])
    assert "CAMP_A" in result.explanation
    assert "topology" in result.explanation


# ---------------------------------------------------------------- build_candidates_from_campaigns

def test_build_candidates_from_campaigns_excludes_current_and_missing_candidates(monkeypatch):
    from campaign_selection import build_candidates_from_campaigns

    current = CampaignContext(campaign_id="CAMP_CURRENT", attacker_ip="1.1.1.1", victim_ip="2.2.2.2",
                               techniques={"T1110"})
    other = CampaignContext(campaign_id="CAMP_OTHER", attacker_ip="1.1.1.1", victim_ip="9.9.9.9",
                             techniques={"T1110"})

    def fake_load(session, campaign_id):
        if campaign_id == "CAMP_OTHER":
            return other
        return None  # CAMP_MISSING doesn't exist

    monkeypatch.setattr("dashboard_api._load_campaign_context", fake_load)
    monkeypatch.setattr(
        "ml.gnn.topology_similarity.gnn_topology_similarity_between_campaigns", lambda a, b: 0.8
    )

    candidates = build_candidates_from_campaigns(
        current, ["CAMP_CURRENT", "CAMP_OTHER", "CAMP_MISSING"], neo4j_session=object(),
    )
    assert len(candidates) == 1
    assert candidates[0].campaign_id == "CAMP_OTHER"
    assert candidates[0].signals.technique_similarity == 1.0
    assert candidates[0].signals.attacker_similarity == 1.0
    assert candidates[0].signals.topology_similarity == 0.8

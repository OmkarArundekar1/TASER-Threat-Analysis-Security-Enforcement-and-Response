import pytest

from campaign_context import CampaignContext
from soar.matcher import PlaybookMatcher, _technique_jaccard
from soar.memory import PlaybookMemoryStore
from soar.schema import ExecutionStatus, Playbook, PlaybookAction, PlaybookExecution


@pytest.fixture()
def store(tmp_path):
    return PlaybookMemoryStore(db_path=str(tmp_path / "matcher_test.db"))


def _playbook(playbook_id, source_campaign_id, techniques):
    return Playbook(
        playbook_id=playbook_id, name=f"PB_{playbook_id}", description="", trigger_conditions={},
        campaign_type="Credential Access", mitre_techniques=techniques, severity="HIGH", risk=900.0,
        required_evidence=[], actions=[PlaybookAction(action_type="block_ip", name="x", description="", order=1)],
        source_campaign_id=source_campaign_id,
    )


def _current_campaign(techniques=None):
    return CampaignContext(
        campaign_id="CAMP_CURRENT", attacker_ip="1.2.3.4", victim_ip="10.0.0.5",
        techniques=techniques or {"T1110"},
    )


def test_technique_jaccard_full_overlap_is_one():
    assert _technique_jaccard({"T1110"}, {"T1110"}) == 1.0


def test_technique_jaccard_no_overlap_is_zero():
    assert _technique_jaccard({"T1110"}, {"T1059"}) == 0.0


def test_technique_jaccard_both_empty_is_zero():
    assert _technique_jaccard(set(), set()) == 0.0


def test_find_matches_excludes_playbooks_from_the_current_campaign(store):
    store.save_playbook(_playbook("pb_1", "CAMP_CURRENT", ["T1110"]))
    matcher = PlaybookMatcher(store)
    matches = matcher.find_matches(_current_campaign(), neo4j_session=None)
    assert matches == []


def test_find_matches_scores_technique_overlap_without_neo4j_session(store):
    store.save_playbook(_playbook("pb_1", "CAMP_OLD", ["T1110"]))
    matcher = PlaybookMatcher(store)
    matches = matcher.find_matches(_current_campaign(techniques={"T1110"}), neo4j_session=None)
    assert len(matches) == 1
    assert matches[0].technique_similarity == 1.0
    assert matches[0].playbook_id == "pb_1"


def test_find_matches_filters_out_playbooks_with_no_real_signal(store, monkeypatch):
    store.save_playbook(_playbook("pb_1", "CAMP_OLD", ["T1059"]))  # no technique overlap with current
    monkeypatch.setattr(
        "ml.gnn.topology_similarity.gnn_topology_similarity_between_campaigns", lambda a, b: None
    )
    matcher = PlaybookMatcher(store)
    matches = matcher.find_matches(_current_campaign(techniques={"T1110"}), neo4j_session=None)
    assert matches == []


def test_find_matches_includes_historical_success_rate_in_reason(store):
    store.save_playbook(_playbook("pb_1", "CAMP_OLD", ["T1110"]))
    store.save_execution(PlaybookExecution(
        playbook_id="pb_1", playbook_version=1, campaign_id="CAMP_OLD",
        operation_id=None, investigation_id=None, status=ExecutionStatus.SUCCESS,
    ))
    matcher = PlaybookMatcher(store)
    matches = matcher.find_matches(_current_campaign(techniques={"T1110"}), neo4j_session=None)
    assert matches[0].historical_success_rate == 1.0
    assert "success rate" in matches[0].recommendation_reason


def test_find_matches_respects_top_k(store):
    for i in range(10):
        store.save_playbook(_playbook(f"pb_{i}", f"CAMP_OLD_{i}", ["T1110"]))
    matcher = PlaybookMatcher(store)
    matches = matcher.find_matches(_current_campaign(techniques={"T1110"}), neo4j_session=None, top_k=3)
    assert len(matches) == 3


def test_find_matches_uses_topology_similarity_when_gnn_available(store, monkeypatch):
    store.save_playbook(_playbook("pb_1", "CAMP_OLD", []))  # no technique overlap at all
    monkeypatch.setattr(
        "ml.gnn.topology_similarity.gnn_topology_similarity_between_campaigns", lambda a, b: 0.75
    )
    matcher = PlaybookMatcher(store)
    matches = matcher.find_matches(_current_campaign(techniques={"T1110"}), neo4j_session=None)
    assert len(matches) == 1
    assert matches[0].topology_similarity == 0.75

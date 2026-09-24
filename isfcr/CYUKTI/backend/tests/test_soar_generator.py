from campaign_context import CampaignContext
from soar.generator import PlaybookGenerator
from soar.schema import ExecutionPolicy


def _campaign(risk_score=200.0, techniques=None, victim_ip="10.0.0.5"):
    return CampaignContext(
        campaign_id="CAMP_1", attacker_ip="1.2.3.4", victim_ip=victim_ip,
        risk_score=risk_score, last_technique="T1110", techniques=techniques or {"T1110"},
    )


def test_generate_always_includes_enrichment_and_notification_actions(monkeypatch):
    monkeypatch.setattr("soar.generator.get_recommendations", lambda tech: [])
    pb = PlaybookGenerator().generate(_campaign())
    action_types = [a.action_type for a in pb.actions]
    assert "enrich_ip" in action_types
    assert "threat_intel_lookup" in action_types
    assert "historical_campaign_search" in action_types
    assert "notify_soc" in action_types
    assert action_types[-1] == "notify_soc"  # always last


def test_generate_every_action_has_a_nonempty_reason(monkeypatch):
    monkeypatch.setattr("soar.generator.get_recommendations", lambda tech: [])
    pb = PlaybookGenerator().generate(_campaign())
    assert all(a.reason for a in pb.actions)


def test_generate_low_severity_has_no_destructive_actions(monkeypatch):
    monkeypatch.setattr("soar.generator.get_recommendations", lambda tech: [])
    pb = PlaybookGenerator().generate(_campaign(risk_score=10.0))
    assert pb.severity == "LOW"
    assert not pb.has_destructive_action


def test_generate_high_severity_adds_destructive_block_and_isolate_actions(monkeypatch):
    monkeypatch.setattr("soar.generator.get_recommendations", lambda tech: [])
    pb = PlaybookGenerator().generate(_campaign(risk_score=1300.0))
    assert pb.severity == "CRITICAL"
    action_types = [a.action_type for a in pb.actions]
    assert "block_ip" in action_types
    assert "isolate_host" in action_types
    assert pb.has_destructive_action is True
    block_action = next(a for a in pb.actions if a.action_type == "block_ip")
    assert block_action.requires_approval is True


def test_generate_includes_real_mitre_mitigations_with_traceable_reason(monkeypatch):
    monkeypatch.setattr(
        "soar.generator.get_recommendations",
        lambda tech: [{"recommendation": "Enforce MFA", "mitre_mitigation": "M1032", "reason": "MFA mitigates brute force"}],
    )
    pb = PlaybookGenerator().generate(_campaign())
    mitigation_actions = [a for a in pb.actions if a.action_type == "apply_mitigation"]
    assert len(mitigation_actions) == 1
    assert "M1032" in mitigation_actions[0].reason
    assert mitigation_actions[0].requires_approval is True


def test_generate_sets_recommend_only_policy_when_nothing_requires_approval(monkeypatch):
    monkeypatch.setattr("soar.generator.get_recommendations", lambda tech: [])
    pb = PlaybookGenerator().generate(_campaign(risk_score=5.0, techniques=set()))
    assert pb.execution_policy == ExecutionPolicy.RECOMMEND_ONLY


def test_generate_sets_analyst_approval_policy_when_any_action_requires_approval(monkeypatch):
    monkeypatch.setattr(
        "soar.generator.get_recommendations",
        lambda tech: [{"recommendation": "x", "mitre_mitigation": "M1", "reason": "r"}],
    )
    pb = PlaybookGenerator().generate(_campaign())
    assert pb.execution_policy == ExecutionPolicy.ANALYST_APPROVAL


def test_generate_source_campaign_id_is_set(monkeypatch):
    monkeypatch.setattr("soar.generator.get_recommendations", lambda tech: [])
    pb = PlaybookGenerator().generate(_campaign())
    assert pb.source_campaign_id == "CAMP_1"


def test_generate_skips_victim_evidence_action_when_no_victim_ip(monkeypatch):
    monkeypatch.setattr("soar.generator.get_recommendations", lambda tech: [])
    pb = PlaybookGenerator().generate(_campaign(victim_ip=""))
    assert "collect_evidence" not in [a.action_type for a in pb.actions]

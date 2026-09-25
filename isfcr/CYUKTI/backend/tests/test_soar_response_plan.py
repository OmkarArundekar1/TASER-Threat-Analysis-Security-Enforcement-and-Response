from campaign_context import CampaignContext
from campaign_selection import BestCampaignSelector
from soar.response_plan import ResponsePlanGenerator
from soar.schema import Playbook
from threat_qualification import ThreatQualificationEngine, QUALIFIED_THREAT, NOT_THREAT


class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Incident:
    def __init__(self, **kw):
        defaults = dict(campaign_id="CAMP_1", attacker_ip="1.2.3.4", technique="T1110",
                         timestamp="2026-01-01T00:00:00Z", cti=None)
        defaults.update(kw)
        self.__dict__.update(defaults)


selector = BestCampaignSelector()
qualification_engine = ThreatQualificationEngine()
generator = ResponsePlanGenerator()


def _campaign(risk_score=1300.0):
    return CampaignContext(campaign_id="CAMP_1", attacker_ip="1.2.3.4", victim_ip="10.0.0.5",
                            risk_score=risk_score, last_technique="T1110", techniques={"T1110"})


def test_response_plan_degrades_honestly_with_no_qualification_or_selection():
    plan = generator.generate(_campaign())
    assert plan.misp_status == "NOT_APPLICABLE"
    assert "No CTI confidence" in plan.why_threat[0]
    assert plan.selected_historical_campaign is None
    assert plan.why_selected is None


def test_response_plan_misp_ready_when_qualification_passes():
    incident = _Incident(cti=_Stub(threat_classification=QUALIFIED_THREAT, score=90.0))
    qualification = qualification_engine.qualify(incident)
    plan = generator.generate(_campaign(), qualification=qualification)
    assert plan.misp_status == "READY"
    assert any("Threat classification: QUALIFIED_THREAT" in w for w in plan.why_threat)


def test_response_plan_misp_blocked_when_qualification_fails():
    incident = _Incident(cti=_Stub(threat_classification=NOT_THREAT, score=5.0))
    qualification = qualification_engine.qualify(incident)
    plan = generator.generate(_campaign(), qualification=qualification)
    assert plan.misp_status == "BLOCKED"


def test_response_plan_includes_selected_campaign_and_explanation():
    a = selector.build_candidate("CAMP_OLD", topology_similarity=0.9, technique_similarity=0.8)
    selection = selector.select([a])
    plan = generator.generate(_campaign(), selection=selection)
    assert plan.selected_historical_campaign == "CAMP_OLD"
    assert "CAMP_OLD" in plan.why_selected
    assert plan.selection_confidence == "HIGH"


def test_response_plan_includes_historical_success_rate_from_selection():
    a = selector.build_candidate(
        "CAMP_OLD", topology_similarity=0.9,
        historical_playbook_success_rate=0.83, historical_playbook_executions=6,
    )
    selection = selector.select([a])
    plan = generator.generate(_campaign(), selection=selection)
    assert plan.historical_playbook_success_rate == 0.83
    assert plan.historical_playbook_executions == 6


def test_response_plan_embeds_the_real_playbook_unmodified():
    playbook = Playbook(playbook_id="pb_1", name="TEST", description="", trigger_conditions={},
                         campaign_type="x", mitre_techniques=["T1110"], severity="HIGH", risk=900.0,
                         required_evidence=[], actions=[])
    plan = generator.generate(_campaign(), playbook=playbook)
    assert plan.playbook is playbook
    assert plan.to_dict()["playbook"]["playbook_id"] == "pb_1"


def test_response_plan_to_dict_is_fully_serializable():
    plan = generator.generate(_campaign())
    d = plan.to_dict()
    assert d["campaign_id"] == "CAMP_1"
    assert d["severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert d["playbook"] is None

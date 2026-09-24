from campaign_context import CampaignContext
from soar.adapter import PlaybookAdaptation
from soar.schema import Playbook, PlaybookAction


def _historical_playbook():
    return Playbook(
        playbook_id="pb_old", name="SSH_BRUTE_FORCE_RESPONSE", description="", trigger_conditions={"campaign_type": "x"},
        campaign_type="Credential Access", mitre_techniques=["T1110"], severity="HIGH", risk=900.0,
        required_evidence=[],
        actions=[
            PlaybookAction(action_type="enrich_ip", name="Enrich", description="", order=1,
                            inputs={"ip": "9.9.9.9"}, reason="old reason"),
            PlaybookAction(action_type="collect_evidence", name="Collect", description="", order=2,
                            inputs={"ip": "8.8.8.8"}, reason="old reason"),
            PlaybookAction(action_type="block_ip", name="Block", description="", order=3,
                            inputs={"ip": "9.9.9.9"}, destructive=True, requires_approval=True, reason="old reason"),
            PlaybookAction(action_type="create_incident", name="Ticket", description="", order=4,
                            inputs={"campaign_id": "CAMP_OLD"}, reason="old reason"),
        ],
        source_campaign_id="CAMP_OLD",
    )


def _current_campaign():
    return CampaignContext(campaign_id="CAMP_NEW", attacker_ip="1.1.1.1", victim_ip="2.2.2.2")


def test_adapt_assigns_new_playbook_id_and_lineage():
    historical = _historical_playbook()
    adapted = PlaybookAdaptation().adapt(historical, _current_campaign())
    assert adapted.playbook_id != historical.playbook_id
    assert adapted.adapted_from_playbook_id == historical.playbook_id
    assert adapted.source_campaign_id == "CAMP_NEW"


def test_adapt_retargets_attacker_directed_actions_to_current_attacker():
    historical = _historical_playbook()
    adapted = PlaybookAdaptation().adapt(historical, _current_campaign())
    enrich = next(a for a in adapted.actions if a.action_type == "enrich_ip")
    block = next(a for a in adapted.actions if a.action_type == "block_ip")
    assert enrich.inputs["ip"] == "1.1.1.1"
    assert block.inputs["ip"] == "1.1.1.1"


def test_adapt_retargets_victim_directed_actions_to_current_victim():
    historical = _historical_playbook()
    adapted = PlaybookAdaptation().adapt(historical, _current_campaign())
    collect = next(a for a in adapted.actions if a.action_type == "collect_evidence")
    assert collect.inputs["ip"] == "2.2.2.2"


def test_adapt_retargets_campaign_id_inputs():
    historical = _historical_playbook()
    adapted = PlaybookAdaptation().adapt(historical, _current_campaign())
    ticket = next(a for a in adapted.actions if a.action_type == "create_incident")
    assert ticket.inputs["campaign_id"] == "CAMP_NEW"


def test_adapt_preserves_action_semantics_not_just_identity():
    historical = _historical_playbook()
    adapted = PlaybookAdaptation().adapt(historical, _current_campaign())
    block = next(a for a in adapted.actions if a.action_type == "block_ip")
    assert block.destructive is True
    assert block.requires_approval is True
    assert len(adapted.actions) == len(historical.actions)


def test_adapt_marks_reason_as_adapted():
    historical = _historical_playbook()
    adapted = PlaybookAdaptation().adapt(historical, _current_campaign())
    assert all("[Adapted from pb_old]" in a.reason for a in adapted.actions)


def test_adapt_does_not_mutate_the_original_historical_playbook():
    historical = _historical_playbook()
    original_ip = historical.actions[0].inputs["ip"]
    PlaybookAdaptation().adapt(historical, _current_campaign())
    assert historical.actions[0].inputs["ip"] == original_ip

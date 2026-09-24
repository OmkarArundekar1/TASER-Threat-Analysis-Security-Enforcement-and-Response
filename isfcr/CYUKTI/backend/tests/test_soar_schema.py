from soar.schema import (
    ExecutionPolicy,
    ExecutionStatus,
    Playbook,
    PlaybookAction,
    PlaybookActionResult,
    PlaybookExecution,
)


def _make_action(destructive=False, requires_approval=False):
    return PlaybookAction(
        action_type="block_ip", name="Block IP", description="Block attacker",
        order=1, inputs={"ip": "1.2.3.4"}, destructive=destructive,
        requires_approval=requires_approval, reason="test reason",
    )


def test_playbook_action_round_trips_through_dict():
    action = _make_action(destructive=True, requires_approval=True)
    restored = PlaybookAction.from_dict(action.to_dict())
    assert restored.action_type == action.action_type
    assert restored.destructive is True
    assert restored.requires_approval is True
    assert restored.reason == "test reason"


def test_playbook_has_destructive_action_true_when_any_action_destructive():
    pb = Playbook(
        name="TEST", description="", trigger_conditions={}, campaign_type="x",
        mitre_techniques=[], severity="HIGH", risk=90.0, required_evidence=[],
        actions=[_make_action(destructive=False), _make_action(destructive=True)],
    )
    assert pb.has_destructive_action is True


def test_playbook_has_destructive_action_false_when_no_action_destructive():
    pb = Playbook(
        name="TEST", description="", trigger_conditions={}, campaign_type="x",
        mitre_techniques=[], severity="LOW", risk=10.0, required_evidence=[],
        actions=[_make_action(destructive=False)],
    )
    assert pb.has_destructive_action is False


def test_playbook_to_dict_includes_execution_policy_value_not_enum():
    pb = Playbook(
        name="TEST", description="", trigger_conditions={}, campaign_type="x",
        mitre_techniques=[], severity="LOW", risk=10.0, required_evidence=[],
        actions=[], execution_policy=ExecutionPolicy.ANALYST_APPROVAL,
    )
    d = pb.to_dict()
    assert d["execution_policy"] == "analyst_approval"
    assert isinstance(d["execution_policy"], str)


def test_playbook_execution_to_dict_serializes_action_results():
    execution = PlaybookExecution(
        playbook_id="pb_1", playbook_version=1, campaign_id="CAMP_1",
        operation_id=None, investigation_id=None, status=ExecutionStatus.SUCCESS,
        action_results=[PlaybookActionResult(action_id="act_1", status=ExecutionStatus.SUCCESS)],
    )
    d = execution.to_dict()
    assert d["status"] == "success"
    assert d["action_results"][0]["status"] == "success"

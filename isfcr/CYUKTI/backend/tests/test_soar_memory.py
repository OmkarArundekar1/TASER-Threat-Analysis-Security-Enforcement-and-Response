import pytest

from soar.memory import PlaybookMemoryStore
from soar.schema import (
    ExecutionPolicy,
    ExecutionStatus,
    Playbook,
    PlaybookAction,
    PlaybookActionResult,
    PlaybookExecution,
)


@pytest.fixture()
def store(tmp_path):
    return PlaybookMemoryStore(db_path=str(tmp_path / "test_playbook_memory.db"))


def _playbook(playbook_id="pb_1", campaign_id="CAMP_1"):
    return Playbook(
        playbook_id=playbook_id, name="SSH_BRUTE_FORCE_RESPONSE", description="test",
        trigger_conditions={"campaign_type": "Credential Access"}, campaign_type="Credential Access",
        mitre_techniques=["T1110"], severity="HIGH", risk=80.0,
        required_evidence=["campaign_context"],
        actions=[PlaybookAction(action_type="block_ip", name="Block IP", description="", order=1,
                                 destructive=True, requires_approval=True, reason="high severity")],
        execution_policy=ExecutionPolicy.ANALYST_APPROVAL,
        source_campaign_id=campaign_id,
    )


def test_save_and_get_playbook_round_trips(store):
    pb = _playbook()
    store.save_playbook(pb)
    restored = store.get_playbook("pb_1")
    assert restored is not None
    assert restored.name == "SSH_BRUTE_FORCE_RESPONSE"
    assert restored.actions[0].destructive is True
    assert restored.execution_policy == ExecutionPolicy.ANALYST_APPROVAL


def test_get_playbook_returns_none_for_unknown_id(store):
    assert store.get_playbook("does_not_exist") is None


def test_list_playbooks_returns_all_saved(store):
    store.save_playbook(_playbook("pb_1"))
    store.save_playbook(_playbook("pb_2"))
    ids = {p.playbook_id for p in store.list_playbooks()}
    assert ids == {"pb_1", "pb_2"}


def test_save_execution_persists_action_results(store):
    store.save_playbook(_playbook())
    execution = PlaybookExecution(
        playbook_id="pb_1", playbook_version=1, campaign_id="CAMP_1",
        operation_id="OP_1", investigation_id=None, status=ExecutionStatus.SUCCESS,
        action_results=[PlaybookActionResult(action_id="act_1", status=ExecutionStatus.SUCCESS, output={"ok": True})],
    )
    store.save_execution(execution)
    restored = store.get_execution(execution.execution_id)
    assert restored is not None
    assert restored.status == ExecutionStatus.SUCCESS
    assert len(restored.action_results) == 1
    assert restored.action_results[0].output == {"ok": True}


def test_save_execution_overwrites_action_results_on_update(store):
    store.save_playbook(_playbook())
    execution = PlaybookExecution(playbook_id="pb_1", playbook_version=1, campaign_id="CAMP_1",
                                   operation_id=None, investigation_id=None)
    store.save_execution(execution)

    execution.action_results = [PlaybookActionResult(action_id="act_1", status=ExecutionStatus.SUCCESS)]
    execution.status = ExecutionStatus.SUCCESS
    store.save_execution(execution)

    restored = store.get_execution(execution.execution_id)
    assert len(restored.action_results) == 1


def test_effectiveness_computes_success_rate_over_terminal_executions(store):
    store.save_playbook(_playbook())
    for status in [ExecutionStatus.SUCCESS, ExecutionStatus.SUCCESS, ExecutionStatus.FAILED, ExecutionStatus.PENDING_APPROVAL]:
        store.save_execution(PlaybookExecution(
            playbook_id="pb_1", playbook_version=1, campaign_id="CAMP_1",
            operation_id=None, investigation_id=None, status=status,
        ))
    eff = store.effectiveness("pb_1")
    assert eff.executions == 4
    assert eff.successful_executions == 2
    assert eff.failed_executions == 1
    assert eff.success_rate == pytest.approx(2 / 3)


def test_effectiveness_success_rate_none_when_no_terminal_executions(store):
    store.save_playbook(_playbook())
    store.save_execution(PlaybookExecution(
        playbook_id="pb_1", playbook_version=1, campaign_id="CAMP_1",
        operation_id=None, investigation_id=None, status=ExecutionStatus.PENDING_APPROVAL,
    ))
    eff = store.effectiveness("pb_1")
    assert eff.success_rate is None


def test_historical_match_stats_matches_effectiveness(store):
    store.save_playbook(_playbook())
    store.save_execution(PlaybookExecution(
        playbook_id="pb_1", playbook_version=1, campaign_id="CAMP_1",
        operation_id=None, investigation_id=None, status=ExecutionStatus.SUCCESS,
    ))
    executions, successes, failures, rate = store.historical_match_stats("pb_1")
    assert (executions, successes, failures, rate) == (1, 1, 0, 1.0)


def test_audit_events_are_recorded_and_filterable_by_execution(store):
    store.log_audit_event("PLAYBOOK_GENERATED", campaign_id="CAMP_1", playbook_id="pb_1")
    store.log_audit_event("PLAYBOOK_EXECUTION_STARTED", campaign_id="CAMP_1", playbook_id="pb_1", execution_id="exec_1")
    store.log_audit_event("PLAYBOOK_EXECUTION_STARTED", campaign_id="CAMP_1", playbook_id="pb_1", execution_id="exec_2")

    all_events = store.list_audit_events()
    assert len(all_events) == 3

    scoped = store.list_audit_events(execution_id="exec_1")
    assert len(scoped) == 1
    assert scoped[0]["event_type"] == "PLAYBOOK_EXECUTION_STARTED"


def test_executions_for_campaign_filters_correctly(store):
    store.save_playbook(_playbook())
    store.save_execution(PlaybookExecution(playbook_id="pb_1", playbook_version=1, campaign_id="CAMP_1",
                                            operation_id=None, investigation_id=None))
    store.save_execution(PlaybookExecution(playbook_id="pb_1", playbook_version=1, campaign_id="CAMP_2",
                                            operation_id=None, investigation_id=None))
    assert len(store.executions_for_campaign("CAMP_1")) == 1

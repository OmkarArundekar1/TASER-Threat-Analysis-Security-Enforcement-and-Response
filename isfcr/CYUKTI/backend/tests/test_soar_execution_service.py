import pytest

from soar.execution_service import PlaybookExecutionService, PolicyError
from soar.memory import PlaybookMemoryStore
from soar.schema import ExecutionPolicy, ExecutionStatus, Playbook, PlaybookAction
from soar.shuffle_client import ShuffleClient, ShuffleTriggerOutcome, ShuffleTriggerResult


class _FakeShuffleClient:
    def __init__(self, result: ShuffleTriggerResult):
        self._result = result
        self.triggered_payloads = []

    def trigger(self, payload):
        self.triggered_payloads.append(payload)
        return self._result

    def get_execution_status(self, workflow_id, execution_id):
        raise NotImplementedError


@pytest.fixture()
def store(tmp_path):
    return PlaybookMemoryStore(db_path=str(tmp_path / "exec_service_test.db"))


def _playbook(policy, destructive=False):
    return Playbook(
        playbook_id="pb_1", name="TEST", description="", trigger_conditions={}, campaign_type="x",
        mitre_techniques=[], severity="HIGH", risk=900.0, required_evidence=[],
        actions=[PlaybookAction(action_type="block_ip", name="Block", description="", order=1,
                                 destructive=destructive, requires_approval=destructive)],
        execution_policy=policy,
    )


def test_recommend_only_policy_raises_policy_error(store):
    pb = _playbook(ExecutionPolicy.RECOMMEND_ONLY)
    store.save_playbook(pb)
    service = PlaybookExecutionService(store, _FakeShuffleClient(ShuffleTriggerResult(ShuffleTriggerOutcome.TRIGGERED)))
    with pytest.raises(PolicyError):
        service.request_execution(pb, campaign_id="CAMP_1")


def test_analyst_approval_policy_creates_pending_approval_without_triggering_shuffle(store):
    pb = _playbook(ExecutionPolicy.ANALYST_APPROVAL, destructive=True)
    store.save_playbook(pb)
    shuffle = _FakeShuffleClient(ShuffleTriggerResult(ShuffleTriggerOutcome.TRIGGERED))
    service = PlaybookExecutionService(store, shuffle)

    execution = service.request_execution(pb, campaign_id="CAMP_1")
    assert execution.status == ExecutionStatus.PENDING_APPROVAL
    assert shuffle.triggered_payloads == []


def test_automatic_policy_with_destructive_action_is_downgraded_to_approval(store):
    """Safety rule: destructive + AUTOMATIC must never auto-execute."""
    pb = _playbook(ExecutionPolicy.AUTOMATIC, destructive=True)
    store.save_playbook(pb)
    shuffle = _FakeShuffleClient(ShuffleTriggerResult(ShuffleTriggerOutcome.TRIGGERED))
    service = PlaybookExecutionService(store, shuffle)

    execution = service.request_execution(pb, campaign_id="CAMP_1")
    assert execution.status == ExecutionStatus.PENDING_APPROVAL
    assert shuffle.triggered_payloads == []


def test_automatic_policy_without_destructive_action_triggers_immediately(store):
    pb = _playbook(ExecutionPolicy.AUTOMATIC, destructive=False)
    store.save_playbook(pb)
    shuffle = _FakeShuffleClient(ShuffleTriggerResult(ShuffleTriggerOutcome.SYNCHRONOUS_RESULT, output={"done": True}))
    service = PlaybookExecutionService(store, shuffle)

    execution = service.request_execution(pb, campaign_id="CAMP_1")
    assert execution.status == ExecutionStatus.SUCCESS
    assert len(shuffle.triggered_payloads) == 1


def test_approve_triggers_shuffle_and_marks_success_on_synchronous_result(store):
    pb = _playbook(ExecutionPolicy.ANALYST_APPROVAL, destructive=True)
    store.save_playbook(pb)
    shuffle = _FakeShuffleClient(ShuffleTriggerResult(ShuffleTriggerOutcome.SYNCHRONOUS_RESULT, output={"ok": 1}))
    service = PlaybookExecutionService(store, shuffle)

    execution = service.request_execution(pb, campaign_id="CAMP_1")
    approved = service.approve(execution.execution_id, approved_by="analyst1")

    assert approved.status == ExecutionStatus.SUCCESS
    assert approved.approved_by == "analyst1"
    assert len(shuffle.triggered_payloads) == 1


def test_approve_fails_if_execution_not_pending_approval(store):
    pb = _playbook(ExecutionPolicy.AUTOMATIC, destructive=False)
    store.save_playbook(pb)
    shuffle = _FakeShuffleClient(ShuffleTriggerResult(ShuffleTriggerOutcome.SYNCHRONOUS_RESULT))
    service = PlaybookExecutionService(store, shuffle)
    execution = service.request_execution(pb, campaign_id="CAMP_1")  # already SUCCESS

    with pytest.raises(ValueError):
        service.approve(execution.execution_id, approved_by="analyst1")


def test_reject_marks_rejected_and_never_calls_shuffle(store):
    pb = _playbook(ExecutionPolicy.ANALYST_APPROVAL, destructive=True)
    store.save_playbook(pb)
    shuffle = _FakeShuffleClient(ShuffleTriggerResult(ShuffleTriggerOutcome.TRIGGERED))
    service = PlaybookExecutionService(store, shuffle)

    execution = service.request_execution(pb, campaign_id="CAMP_1")
    rejected = service.reject(execution.execution_id, reason="Not warranted")

    assert rejected.status == ExecutionStatus.REJECTED
    assert rejected.rejection_reason == "Not warranted"
    assert shuffle.triggered_payloads == []


def test_trigger_marks_failed_when_shuffle_not_configured(store):
    pb = _playbook(ExecutionPolicy.AUTOMATIC, destructive=False)
    store.save_playbook(pb)
    shuffle = _FakeShuffleClient(ShuffleTriggerResult(ShuffleTriggerOutcome.NOT_CONFIGURED))
    service = PlaybookExecutionService(store, shuffle)

    execution = service.request_execution(pb, campaign_id="CAMP_1")
    assert execution.status == ExecutionStatus.FAILED
    assert execution.action_results[0].error is not None


def test_trigger_leaves_execution_running_when_shuffle_only_acknowledges(store):
    pb = _playbook(ExecutionPolicy.AUTOMATIC, destructive=False)
    store.save_playbook(pb)
    shuffle = _FakeShuffleClient(
        ShuffleTriggerResult(ShuffleTriggerOutcome.TRIGGERED, shuffle_execution_id="shuffle_exec_1")
    )
    service = PlaybookExecutionService(store, shuffle)

    execution = service.request_execution(pb, campaign_id="CAMP_1")
    assert execution.status == ExecutionStatus.RUNNING
    assert execution.shuffle_execution_id == "shuffle_exec_1"


def test_every_lifecycle_transition_is_audit_logged(store):
    pb = _playbook(ExecutionPolicy.ANALYST_APPROVAL, destructive=True)
    store.save_playbook(pb)
    shuffle = _FakeShuffleClient(ShuffleTriggerResult(ShuffleTriggerOutcome.SYNCHRONOUS_RESULT))
    service = PlaybookExecutionService(store, shuffle)

    execution = service.request_execution(pb, campaign_id="CAMP_1")
    service.approve(execution.execution_id, approved_by="analyst1")

    events = [e["event_type"] for e in store.list_audit_events(execution_id=execution.execution_id)]
    assert "PLAYBOOK_APPROVAL_REQUESTED" in events
    assert "PLAYBOOK_APPROVED" in events
    assert "PLAYBOOK_EXECUTION_STARTED" in events
    assert "PLAYBOOK_EXECUTION_COMPLETED" in events


def test_poll_status_finalizes_running_execution_on_success(store):
    pb = _playbook(ExecutionPolicy.AUTOMATIC, destructive=False)
    store.save_playbook(pb)
    shuffle = _FakeShuffleClient(
        ShuffleTriggerResult(ShuffleTriggerOutcome.TRIGGERED, shuffle_execution_id="shuffle_exec_1")
    )

    class _PollableShuffleClient(_FakeShuffleClient):
        def get_execution_status(self, workflow_id, execution_id):
            from soar.shuffle_client import ShuffleStatusResult
            return ShuffleStatusResult(outcome="success", raw={"status": "FINISHED"})

    pollable = _PollableShuffleClient(shuffle._result)
    service = PlaybookExecutionService(store, pollable)

    execution = service.request_execution(pb, campaign_id="CAMP_1")
    assert execution.status == ExecutionStatus.RUNNING

    finalized = service.poll_status(execution.execution_id)
    assert finalized.status == ExecutionStatus.SUCCESS


def test_poll_status_leaves_execution_running_when_shuffle_status_not_configured(store):
    pb = _playbook(ExecutionPolicy.AUTOMATIC, destructive=False)
    store.save_playbook(pb)

    class _NotConfiguredShuffleClient(_FakeShuffleClient):
        def get_execution_status(self, workflow_id, execution_id):
            from soar.shuffle_client import ShuffleStatusResult
            return ShuffleStatusResult(outcome="not_configured")

    shuffle = _NotConfiguredShuffleClient(
        ShuffleTriggerResult(ShuffleTriggerOutcome.TRIGGERED, shuffle_execution_id="shuffle_exec_1")
    )
    service = PlaybookExecutionService(store, shuffle)

    execution = service.request_execution(pb, campaign_id="CAMP_1")
    finalized = service.poll_status(execution.execution_id)
    assert finalized.status == ExecutionStatus.RUNNING

"""
soar/execution_service.py
============================
PlaybookExecutionService: the state machine between "analyst/system
wants this playbook to run" and "Shuffle has (or hasn't) actually run
it". Owns every transition PENDING/PENDING_APPROVAL -> RUNNING ->
SUCCESS/FAILED/TIMEOUT, persists each one to soar.memory (CYUKTI's own
record, independent of Shuffle), and writes an audit event for every
lifecycle transition (Phase 16).

Safety rule enforced here, not just at the UI layer: a playbook
containing any destructive action can NEVER execute under AUTOMATIC
policy -- it is silently downgraded to ANALYST_APPROVAL. A historical
incident looking similar is never sufficient justification for
unattended destructive action.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from soar.memory import PlaybookMemoryStore
from soar.schema import (
    ExecutionPolicy,
    ExecutionStatus,
    Playbook,
    PlaybookActionResult,
    PlaybookExecution,
)
from soar.shuffle_client import ShuffleClient, ShuffleTriggerOutcome

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PolicyError(Exception):
    """Raised when an execution is attempted under RECOMMEND_ONLY --
    the API layer should never let this happen (no EXECUTE button is
    shown for that policy), but the service enforces it too."""


class PlaybookExecutionService:
    def __init__(self, memory_store: PlaybookMemoryStore, shuffle_client: ShuffleClient):
        self.memory_store = memory_store
        self.shuffle_client = shuffle_client

    def effective_policy(self, playbook: Playbook) -> ExecutionPolicy:
        if playbook.has_destructive_action and playbook.execution_policy == ExecutionPolicy.AUTOMATIC:
            return ExecutionPolicy.ANALYST_APPROVAL
        return playbook.execution_policy

    def request_execution(
        self,
        playbook: Playbook,
        campaign_id: str,
        operation_id: str | None = None,
        investigation_id: str | None = None,
    ) -> PlaybookExecution:
        policy = self.effective_policy(playbook)
        if policy == ExecutionPolicy.RECOMMEND_ONLY:
            raise PolicyError(f"Playbook {playbook.playbook_id}'s policy is recommend_only; it cannot be executed.")

        execution = PlaybookExecution(
            playbook_id=playbook.playbook_id,
            playbook_version=playbook.version,
            campaign_id=campaign_id,
            operation_id=operation_id,
            investigation_id=investigation_id,
        )

        if policy == ExecutionPolicy.ANALYST_APPROVAL:
            execution.status = ExecutionStatus.PENDING_APPROVAL
            self.memory_store.save_execution(execution)
            self.memory_store.log_audit_event(
                "PLAYBOOK_APPROVAL_REQUESTED", campaign_id=campaign_id, operation_id=operation_id,
                investigation_id=investigation_id, playbook_id=playbook.playbook_id,
                execution_id=execution.execution_id,
            )
            return execution

        # AUTOMATIC and no destructive action present -- trigger immediately.
        self.memory_store.save_execution(execution)
        return self._trigger(execution, playbook)

    def approve(self, execution_id: str, approved_by: str) -> PlaybookExecution:
        execution = self.memory_store.get_execution(execution_id)
        if execution is None:
            raise ValueError(f"Unknown execution {execution_id}")
        if execution.status != ExecutionStatus.PENDING_APPROVAL:
            raise ValueError(f"Execution {execution_id} is not pending approval (status={execution.status.value}).")

        execution.approved_by = approved_by
        execution.approval_decision_at = _now_iso()
        self.memory_store.save_execution(execution)
        self.memory_store.log_audit_event(
            "PLAYBOOK_APPROVED", campaign_id=execution.campaign_id, playbook_id=execution.playbook_id,
            execution_id=execution.execution_id, detail={"approved_by": approved_by},
        )

        playbook = self.memory_store.get_playbook(execution.playbook_id)
        if playbook is None:
            execution.status = ExecutionStatus.FAILED
            self.memory_store.save_execution(execution)
            return execution
        return self._trigger(execution, playbook)

    def reject(self, execution_id: str, reason: str, rejected_by: str | None = None) -> PlaybookExecution:
        execution = self.memory_store.get_execution(execution_id)
        if execution is None:
            raise ValueError(f"Unknown execution {execution_id}")
        if execution.status != ExecutionStatus.PENDING_APPROVAL:
            raise ValueError(f"Execution {execution_id} is not pending approval (status={execution.status.value}).")

        execution.status = ExecutionStatus.REJECTED
        execution.rejection_reason = reason
        execution.approval_decision_at = _now_iso()
        self.memory_store.save_execution(execution)
        self.memory_store.log_audit_event(
            "PLAYBOOK_REJECTED", campaign_id=execution.campaign_id, playbook_id=execution.playbook_id,
            execution_id=execution.execution_id, detail={"reason": reason, "rejected_by": rejected_by},
        )
        return execution

    def _trigger(self, execution: PlaybookExecution, playbook: Playbook) -> PlaybookExecution:
        execution.status = ExecutionStatus.RUNNING
        execution.started_at = _now_iso()
        self.memory_store.save_execution(execution)
        self.memory_store.log_audit_event(
            "PLAYBOOK_EXECUTION_STARTED", campaign_id=execution.campaign_id, playbook_id=playbook.playbook_id,
            execution_id=execution.execution_id,
        )

        payload = {
            "execution_id": execution.execution_id,
            "playbook_id": playbook.playbook_id,
            "playbook_name": playbook.name,
            "campaign_id": execution.campaign_id,
            "operation_id": execution.operation_id,
            "actions": [a.to_dict() for a in playbook.actions],
        }
        result = self.shuffle_client.trigger(payload)
        execution.shuffle_execution_id = result.shuffle_execution_id

        if result.outcome == ShuffleTriggerOutcome.NOT_CONFIGURED:
            execution.status = ExecutionStatus.FAILED
            execution.completed_at = _now_iso()
            execution.action_results = [
                PlaybookActionResult(
                    action_id=a.action_id, status=ExecutionStatus.FAILED,
                    error="Shuffle is not configured (SHUFFLE_WEBHOOK is empty) -- see SOAR_PLAYBOOK_INTEGRATION.md.",
                )
                for a in playbook.actions
            ]
            self.memory_store.save_execution(execution)
            self.memory_store.log_audit_event(
                "PLAYBOOK_EXECUTION_FAILED", campaign_id=execution.campaign_id, playbook_id=playbook.playbook_id,
                execution_id=execution.execution_id, detail={"reason": "shuffle_not_configured"},
            )
            return execution

        if result.outcome == ShuffleTriggerOutcome.SYNCHRONOUS_RESULT:
            execution.status = ExecutionStatus.SUCCESS
            execution.completed_at = _now_iso()
            execution.action_results = [
                PlaybookActionResult(
                    action_id=a.action_id, status=ExecutionStatus.SUCCESS, output=result.output,
                    started_at=execution.started_at, completed_at=execution.completed_at,
                )
                for a in playbook.actions
            ]
            self.memory_store.save_execution(execution)
            self.memory_store.log_audit_event(
                "PLAYBOOK_EXECUTION_COMPLETED", campaign_id=execution.campaign_id, playbook_id=playbook.playbook_id,
                execution_id=execution.execution_id,
            )
            return execution

        if result.outcome == ShuffleTriggerOutcome.TRIGGERED:
            # Fire-and-forget: Shuffle acknowledged the trigger but the
            # workflow runs asynchronously. Stays RUNNING until
            # poll_status() (if SHUFFLE_BASE_URL/SHUFFLE_API_KEY are
            # configured) or a manual finalization confirms the outcome.
            self.memory_store.save_execution(execution)
            return execution

        # AUTH_FAILED / TIMEOUT / ERROR
        execution.status = (
            ExecutionStatus.TIMEOUT if result.outcome == ShuffleTriggerOutcome.TIMEOUT else ExecutionStatus.FAILED
        )
        execution.completed_at = _now_iso()
        execution.action_results = [
            PlaybookActionResult(
                action_id=a.action_id, status=execution.status,
                error=result.error or result.outcome.value,
            )
            for a in playbook.actions
        ]
        self.memory_store.save_execution(execution)
        self.memory_store.log_audit_event(
            "PLAYBOOK_EXECUTION_FAILED", campaign_id=execution.campaign_id, playbook_id=playbook.playbook_id,
            execution_id=execution.execution_id, detail={"reason": result.outcome.value, "error": result.error},
        )
        return execution

    def poll_status(self, execution_id: str) -> PlaybookExecution:
        """Finalizes a RUNNING execution if Shuffle's REST API is
        configured and reachable; otherwise returns it unchanged --
        never fabricates a status this environment cannot actually
        observe."""
        execution = self.memory_store.get_execution(execution_id)
        if execution is None:
            raise ValueError(f"Unknown execution {execution_id}")
        if execution.status != ExecutionStatus.RUNNING or not execution.shuffle_execution_id:
            return execution

        playbook = self.memory_store.get_playbook(execution.playbook_id)
        workflow_id = playbook.shuffle_workflow_id if playbook else None
        status_result = self.shuffle_client.get_execution_status(workflow_id or "", execution.shuffle_execution_id)

        if status_result.outcome == "success":
            execution.status = ExecutionStatus.SUCCESS
            execution.completed_at = _now_iso()
            self.memory_store.save_execution(execution)
            self.memory_store.log_audit_event(
                "PLAYBOOK_EXECUTION_COMPLETED", campaign_id=execution.campaign_id,
                playbook_id=execution.playbook_id, execution_id=execution.execution_id,
            )
        elif status_result.outcome == "failed":
            execution.status = ExecutionStatus.FAILED
            execution.completed_at = _now_iso()
            self.memory_store.save_execution(execution)
            self.memory_store.log_audit_event(
                "PLAYBOOK_EXECUTION_FAILED", campaign_id=execution.campaign_id,
                playbook_id=execution.playbook_id, execution_id=execution.execution_id,
                detail={"error": status_result.error},
            )
        # "running" / "not_configured" / "error": leave RUNNING -- an
        # unreachable status endpoint is not evidence of failure.
        return execution

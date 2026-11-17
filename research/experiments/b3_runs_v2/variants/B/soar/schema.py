"""
soar/schema.py
================
Strongly typed schema for CYUKTI's SOAR/Playbook layer (see
SOAR_PLAYBOOK_INTEGRATION.md for the full architecture writeup).

CYUKTI is the intelligence/investigation/memory layer; Shuffle is the
execution layer. Everything in this module describes CYUKTI's OWN
record of what a playbook is and what happened when it ran -- Shuffle
is never the source of truth for this state, only the thing that
executes actions and reports results back.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionPolicy(str, Enum):
    RECOMMEND_ONLY = "recommend_only"
    ANALYST_APPROVAL = "analyst_approval"
    AUTOMATIC = "automatic"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    PENDING_APPROVAL = "pending_approval"
    REJECTED = "rejected"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class PlaybookStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclass
class PlaybookAction:
    """One step in a playbook. `destructive` and `requires_approval` are
    the two fields the execution policy engine (soar/execution_service.py)
    actually reads to decide whether an action can ever run under
    AUTOMATIC policy -- see PlaybookExecutionService.effective_policy()."""

    action_type: str
    name: str
    description: str
    order: int
    inputs: dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    destructive: bool = False
    requires_approval: bool = False
    timeout_seconds: int = 60
    action_id: str = field(default_factory=lambda: _new_id("act"))
    # Non-fabrication trail: which real evidence/signal justified this
    # action being included (e.g. "MITRE T1110 mitigation M1032",
    # "attacker_ip present in CampaignContext"). Never left empty by the
    # generator -- an action with no reason is a fabricated action.
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "name": self.name,
            "description": self.description,
            "order": self.order,
            "inputs": self.inputs,
            "expected_output": self.expected_output,
            "destructive": self.destructive,
            "requires_approval": self.requires_approval,
            "timeout_seconds": self.timeout_seconds,
            "reason": self.reason,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PlaybookAction":
        return PlaybookAction(
            action_id=d.get("action_id") or _new_id("act"),
            action_type=d["action_type"],
            name=d["name"],
            description=d.get("description", ""),
            order=d.get("order", 0),
            inputs=d.get("inputs", {}),
            expected_output=d.get("expected_output", ""),
            destructive=d.get("destructive", False),
            requires_approval=d.get("requires_approval", False),
            timeout_seconds=d.get("timeout_seconds", 60),
            reason=d.get("reason", ""),
        )


@dataclass
class Playbook:
    name: str
    description: str
    trigger_conditions: dict[str, Any]
    campaign_type: str
    mitre_techniques: list[str]
    severity: str
    risk: float
    required_evidence: list[str]
    actions: list[PlaybookAction]
    execution_policy: ExecutionPolicy = ExecutionPolicy.ANALYST_APPROVAL
    shuffle_workflow_id: str | None = None
    shuffle_workflow_version: str | None = None
    playbook_id: str = field(default_factory=lambda: _new_id("pb"))
    version: int = 1
    status: PlaybookStatus = PlaybookStatus.ACTIVE
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    # Real provenance: which campaign/investigation produced this
    # playbook, and whether it was freshly generated or adapted from a
    # prior one -- never blank for a generated playbook.
    source_campaign_id: str | None = None
    adapted_from_playbook_id: str | None = None

    @property
    def has_destructive_action(self) -> bool:
        return any(a.destructive for a in self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "playbook_id": self.playbook_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "trigger_conditions": self.trigger_conditions,
            "campaign_type": self.campaign_type,
            "mitre_techniques": self.mitre_techniques,
            "severity": self.severity,
            "risk": self.risk,
            "required_evidence": self.required_evidence,
            "actions": [a.to_dict() for a in self.actions],
            "execution_policy": self.execution_policy.value,
            "shuffle_workflow_id": self.shuffle_workflow_id,
            "shuffle_workflow_version": self.shuffle_workflow_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "source_campaign_id": self.source_campaign_id,
            "adapted_from_playbook_id": self.adapted_from_playbook_id,
            "has_destructive_action": self.has_destructive_action,
        }


@dataclass
class PlaybookActionResult:
    action_id: str
    status: ExecutionStatus
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] | str | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    action_result_id: str = field(default_factory=lambda: _new_id("actres"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_result_id": self.action_result_id,
            "action_id": self.action_id,
            "status": self.status.value if isinstance(self.status, ExecutionStatus) else self.status,
            "input": self.input,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class PlaybookExecution:
    playbook_id: str
    playbook_version: int
    campaign_id: str
    operation_id: str | None
    investigation_id: str | None
    status: ExecutionStatus = ExecutionStatus.PENDING
    shuffle_execution_id: str | None = None
    execution_id: str = field(default_factory=lambda: _new_id("exec"))
    started_at: str | None = None
    completed_at: str | None = None
    action_results: list[PlaybookActionResult] = field(default_factory=list)
    approved_by: str | None = None
    approval_decision_at: str | None = None
    rejection_reason: str | None = None
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "playbook_id": self.playbook_id,
            "playbook_version": self.playbook_version,
            "campaign_id": self.campaign_id,
            "operation_id": self.operation_id,
            "investigation_id": self.investigation_id,
            "status": self.status.value if isinstance(self.status, ExecutionStatus) else self.status,
            "shuffle_execution_id": self.shuffle_execution_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "action_results": [r.to_dict() for r in self.action_results],
            "approved_by": self.approved_by,
            "approval_decision_at": self.approval_decision_at,
            "rejection_reason": self.rejection_reason,
            "created_at": self.created_at,
        }


@dataclass
class HistoricalPlaybookMatch:
    """Output of PlaybookMatcher -- every signal kept separate on
    purpose (Phase 7's explicit requirement: never blend into one
    similarity score)."""

    playbook_id: str
    playbook_name: str
    source_campaign_id: str
    technique_similarity: float
    topology_similarity: float | None
    attacker_ip_match: bool
    victim_ip_match: bool
    historical_executions: int
    historical_successes: int
    historical_failures: int
    historical_success_rate: float | None
    recommendation_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "playbook_id": self.playbook_id,
            "playbook_name": self.playbook_name,
            "source_campaign_id": self.source_campaign_id,
            "technique_similarity": self.technique_similarity,
            "topology_similarity": self.topology_similarity,
            "attacker_ip_match": self.attacker_ip_match,
            "victim_ip_match": self.victim_ip_match,
            "historical_executions": self.historical_executions,
            "historical_successes": self.historical_successes,
            "historical_failures": self.historical_failures,
            "historical_success_rate": self.historical_success_rate,
            "recommendation_reason": self.recommendation_reason,
        }


@dataclass
class PlaybookEffectiveness:
    playbook_id: str
    playbook_name: str
    executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float | None
    average_execution_seconds: float | None
    analyst_approvals: int
    analyst_rejections: int
    last_execution_at: str | None
    failure_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "playbook_id": self.playbook_id,
            "playbook_name": self.playbook_name,
            "executions": self.executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": self.success_rate,
            "average_execution_seconds": self.average_execution_seconds,
            "analyst_approvals": self.analyst_approvals,
            "analyst_rejections": self.analyst_rejections,
            "last_execution_at": self.last_execution_at,
            "failure_reasons": self.failure_reasons,
        }

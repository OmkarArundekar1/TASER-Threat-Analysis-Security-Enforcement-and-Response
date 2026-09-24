"""
soar/memory.py
================
Playbook Memory: CYUKTI's own persisted record of every playbook,
execution, and per-action result -- independent of Shuffle, which only
executes and reports results back (see module docstring in
soar/schema.py and SOAR_PLAYBOOK_INTEGRATION.md).

Backed by SQLite (stdlib, no new dependency) rather than Neo4j:
Playbook -> PlaybookExecution -> PlaybookActionResult is a strictly
relational one-to-many-to-many shape with no graph-traversal
requirement of its own (campaign/operation/investigation association is
stored as plain foreign-key-style columns, which is all this data model
needs) -- Neo4j is reserved for the parts of CYUKTI that are genuinely
graph-shaped. Mirrors misp_cache.py's existing pattern of a small,
thread-safe, file-backed store next to the module that owns it, just
with SQLite instead of a JSON blob because the "list executions for
this playbook, compute success rate" access pattern is a real query,
not a full-file scan.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

from soar.schema import (
    ExecutionStatus,
    HistoricalPlaybookMatch,
    Playbook,
    PlaybookAction,
    PlaybookActionResult,
    PlaybookEffectiveness,
    PlaybookExecution,
    PlaybookStatus,
    ExecutionPolicy,
)

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "playbook_memory.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS playbooks (
    playbook_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    trigger_conditions TEXT,
    campaign_type TEXT,
    mitre_techniques TEXT,
    severity TEXT,
    risk REAL,
    required_evidence TEXT,
    actions TEXT,
    execution_policy TEXT,
    shuffle_workflow_id TEXT,
    shuffle_workflow_version TEXT,
    status TEXT,
    created_at TEXT,
    updated_at TEXT,
    source_campaign_id TEXT,
    adapted_from_playbook_id TEXT
);

CREATE TABLE IF NOT EXISTS playbook_executions (
    execution_id TEXT PRIMARY KEY,
    playbook_id TEXT NOT NULL,
    playbook_version INTEGER,
    campaign_id TEXT,
    operation_id TEXT,
    investigation_id TEXT,
    status TEXT,
    shuffle_execution_id TEXT,
    started_at TEXT,
    completed_at TEXT,
    approved_by TEXT,
    approval_decision_at TEXT,
    rejection_reason TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS playbook_action_results (
    action_result_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    action_id TEXT,
    status TEXT,
    input TEXT,
    output TEXT,
    error TEXT,
    started_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS soar_audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    campaign_id TEXT,
    operation_id TEXT,
    investigation_id TEXT,
    playbook_id TEXT,
    execution_id TEXT,
    detail TEXT
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlaybookMemoryStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------------------------------------------------------------- playbooks

    def save_playbook(self, playbook: Playbook) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO playbooks
                (playbook_id, version, name, description, trigger_conditions,
                 campaign_type, mitre_techniques, severity, risk, required_evidence,
                 actions, execution_policy, shuffle_workflow_id, shuffle_workflow_version,
                 status, created_at, updated_at, source_campaign_id, adapted_from_playbook_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    playbook.playbook_id, playbook.version, playbook.name, playbook.description,
                    json.dumps(playbook.trigger_conditions), playbook.campaign_type,
                    json.dumps(playbook.mitre_techniques), playbook.severity, playbook.risk,
                    json.dumps(playbook.required_evidence),
                    json.dumps([a.to_dict() for a in playbook.actions]),
                    playbook.execution_policy.value, playbook.shuffle_workflow_id,
                    playbook.shuffle_workflow_version, playbook.status.value,
                    playbook.created_at, playbook.updated_at,
                    playbook.source_campaign_id, playbook.adapted_from_playbook_id,
                ),
            )

    @staticmethod
    def _row_to_playbook(row: sqlite3.Row) -> Playbook:
        return Playbook(
            playbook_id=row["playbook_id"],
            version=row["version"],
            name=row["name"],
            description=row["description"] or "",
            trigger_conditions=json.loads(row["trigger_conditions"] or "{}"),
            campaign_type=row["campaign_type"] or "",
            mitre_techniques=json.loads(row["mitre_techniques"] or "[]"),
            severity=row["severity"] or "",
            risk=row["risk"] or 0.0,
            required_evidence=json.loads(row["required_evidence"] or "[]"),
            actions=[PlaybookAction.from_dict(a) for a in json.loads(row["actions"] or "[]")],
            execution_policy=ExecutionPolicy(row["execution_policy"]),
            shuffle_workflow_id=row["shuffle_workflow_id"],
            shuffle_workflow_version=row["shuffle_workflow_version"],
            status=PlaybookStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            source_campaign_id=row["source_campaign_id"],
            adapted_from_playbook_id=row["adapted_from_playbook_id"],
        )

    def get_playbook(self, playbook_id: str) -> Playbook | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM playbooks WHERE playbook_id = ?", (playbook_id,)).fetchone()
        return self._row_to_playbook(row) if row else None

    def list_playbooks(self) -> list[Playbook]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM playbooks ORDER BY created_at DESC").fetchall()
        return [self._row_to_playbook(r) for r in rows]

    def playbooks_for_campaign_type(self, campaign_type: str) -> list[Playbook]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM playbooks WHERE campaign_type = ? AND status = ? ORDER BY created_at DESC",
                (campaign_type, PlaybookStatus.ACTIVE.value),
            ).fetchall()
        return [self._row_to_playbook(r) for r in rows]

    # ---------------------------------------------------------------- executions

    def save_execution(self, execution: PlaybookExecution) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO playbook_executions
                (execution_id, playbook_id, playbook_version, campaign_id, operation_id,
                 investigation_id, status, shuffle_execution_id, started_at, completed_at,
                 approved_by, approval_decision_at, rejection_reason, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    execution.execution_id, execution.playbook_id, execution.playbook_version,
                    execution.campaign_id, execution.operation_id, execution.investigation_id,
                    execution.status.value if isinstance(execution.status, ExecutionStatus) else execution.status,
                    execution.shuffle_execution_id, execution.started_at, execution.completed_at,
                    execution.approved_by, execution.approval_decision_at, execution.rejection_reason,
                    execution.created_at,
                ),
            )
            conn.execute("DELETE FROM playbook_action_results WHERE execution_id = ?", (execution.execution_id,))
            for r in execution.action_results:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO playbook_action_results
                    (action_result_id, execution_id, action_id, status, input, output, error,
                     started_at, completed_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        r.action_result_id, execution.execution_id, r.action_id,
                        r.status.value if isinstance(r.status, ExecutionStatus) else r.status,
                        json.dumps(r.input),
                        json.dumps(r.output) if isinstance(r.output, dict) else r.output,
                        r.error, r.started_at, r.completed_at,
                    ),
                )

    def _row_to_execution(self, row: sqlite3.Row, conn: sqlite3.Connection) -> PlaybookExecution:
        action_rows = conn.execute(
            "SELECT * FROM playbook_action_results WHERE execution_id = ?", (row["execution_id"],)
        ).fetchall()
        action_results = []
        for ar in action_rows:
            output = ar["output"]
            try:
                output = json.loads(output) if output else None
            except (TypeError, ValueError):
                pass  # a plain string output (e.g. raw Shuffle text), not JSON -- keep as-is
            action_results.append(
                PlaybookActionResult(
                    action_result_id=ar["action_result_id"],
                    action_id=ar["action_id"],
                    status=ExecutionStatus(ar["status"]),
                    input=json.loads(ar["input"] or "{}"),
                    output=output,
                    error=ar["error"],
                    started_at=ar["started_at"],
                    completed_at=ar["completed_at"],
                )
            )
        return PlaybookExecution(
            execution_id=row["execution_id"],
            playbook_id=row["playbook_id"],
            playbook_version=row["playbook_version"],
            campaign_id=row["campaign_id"],
            operation_id=row["operation_id"],
            investigation_id=row["investigation_id"],
            status=ExecutionStatus(row["status"]),
            shuffle_execution_id=row["shuffle_execution_id"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            action_results=action_results,
            approved_by=row["approved_by"],
            approval_decision_at=row["approval_decision_at"],
            rejection_reason=row["rejection_reason"],
            created_at=row["created_at"],
        )

    def get_execution(self, execution_id: str) -> PlaybookExecution | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM playbook_executions WHERE execution_id = ?", (execution_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_execution(row, conn)

    def list_executions(self, limit: int = 100) -> list[PlaybookExecution]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM playbook_executions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_to_execution(r, conn) for r in rows]

    def list_executions_for_playbook(self, playbook_id: str) -> list[PlaybookExecution]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM playbook_executions WHERE playbook_id = ? ORDER BY created_at DESC",
                (playbook_id,),
            ).fetchall()
            return [self._row_to_execution(r, conn) for r in rows]

    def executions_for_campaign(self, campaign_id: str) -> list[PlaybookExecution]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM playbook_executions WHERE campaign_id = ? ORDER BY created_at DESC",
                (campaign_id,),
            ).fetchall()
            return [self._row_to_execution(r, conn) for r in rows]

    # ---------------------------------------------------------------- effectiveness

    def effectiveness(self, playbook_id: str) -> PlaybookEffectiveness:
        playbook = self.get_playbook(playbook_id)
        executions = self.list_executions_for_playbook(playbook_id)
        return self._compute_effectiveness(playbook_id, playbook.name if playbook else playbook_id, executions)

    def effectiveness_all(self) -> list[PlaybookEffectiveness]:
        return [self.effectiveness(p.playbook_id) for p in self.list_playbooks()]

    @staticmethod
    def _compute_effectiveness(playbook_id: str, name: str, executions: list[PlaybookExecution]) -> PlaybookEffectiveness:
        terminal = [e for e in executions if e.status in (ExecutionStatus.SUCCESS, ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT)]
        successes = [e for e in terminal if e.status == ExecutionStatus.SUCCESS]
        failures = [e for e in terminal if e.status != ExecutionStatus.SUCCESS]
        approvals = [e for e in executions if e.approved_by]
        rejections = [e for e in executions if e.status == ExecutionStatus.REJECTED]

        durations = []
        for e in terminal:
            if e.started_at and e.completed_at:
                try:
                    start = datetime.fromisoformat(e.started_at)
                    end = datetime.fromisoformat(e.completed_at)
                    durations.append((end - start).total_seconds())
                except ValueError:
                    continue

        failure_reasons = [e.rejection_reason for e in failures if e.rejection_reason]
        failure_reasons += [
            r.error for e in failures for r in e.action_results if r.error
        ]

        last_execution_at = executions[0].created_at if executions else None

        return PlaybookEffectiveness(
            playbook_id=playbook_id,
            playbook_name=name,
            executions=len(executions),
            successful_executions=len(successes),
            failed_executions=len(failures),
            success_rate=(len(successes) / len(terminal)) if terminal else None,
            average_execution_seconds=(sum(durations) / len(durations)) if durations else None,
            analyst_approvals=len(approvals),
            analyst_rejections=len(rejections),
            last_execution_at=last_execution_at,
            failure_reasons=failure_reasons[:10],
        )

    def historical_match_stats(self, playbook_id: str) -> tuple[int, int, int, float | None]:
        """(executions, successes, failures, success_rate) -- the subset
        of effectiveness() PlaybookMatcher needs, without recomputing
        the whole PlaybookEffectiveness object."""
        eff = self.effectiveness(playbook_id)
        return eff.executions, eff.successful_executions, eff.failed_executions, eff.success_rate

    # ---------------------------------------------------------------- audit events

    def log_audit_event(
        self,
        event_type: str,
        campaign_id: str | None = None,
        operation_id: str | None = None,
        investigation_id: str | None = None,
        playbook_id: str | None = None,
        execution_id: str | None = None,
        detail: dict | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO soar_audit_events
                (event_type, timestamp, campaign_id, operation_id, investigation_id,
                 playbook_id, execution_id, detail)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    event_type, _now_iso(), campaign_id, operation_id, investigation_id,
                    playbook_id, execution_id, json.dumps(detail or {}),
                ),
            )

    def list_audit_events(self, execution_id: str | None = None, limit: int = 200) -> list[dict]:
        with self._connect() as conn:
            if execution_id:
                rows = conn.execute(
                    "SELECT * FROM soar_audit_events WHERE execution_id = ? ORDER BY id DESC LIMIT ?",
                    (execution_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM soar_audit_events ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        return [
            {
                "event_type": r["event_type"],
                "timestamp": r["timestamp"],
                "campaign_id": r["campaign_id"],
                "operation_id": r["operation_id"],
                "investigation_id": r["investigation_id"],
                "playbook_id": r["playbook_id"],
                "execution_id": r["execution_id"],
                "detail": json.loads(r["detail"] or "{}"),
            }
            for r in rows
        ]


memory_store = PlaybookMemoryStore()

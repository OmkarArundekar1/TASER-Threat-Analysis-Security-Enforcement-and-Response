"""WP-M1: Marker reads, startup checks, and constraint verification tests.

Tests pure logic and fake-driver boundaries for:
- read_applied_write: all four statuses, client deadline, no retries, no secret logging
- read_marker_schema: every row of the constraint table, online index checks, label checks
- verify_startup: 6-stage startup check sequence, version/edition validation, alarm triggers
- WriteCoordinator integration with marker_reader protocol
"""
from __future__ import annotations

import ast
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from neo4j_client import (
    ConstraintStatus,
    MarkerLookup,
    MarkerSchemaReport,
    MarkerStatus,
    StartupReport,
    read_applied_write,
    read_marker_schema,
    verify_startup,
)
from neo4j_write_coordinator import (
    CoordinatorError,
    OperationRegistry,
    RegisteredOperation,
    WriteCoordinator,
)

BACKEND = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- Fake Drivers for Testing

class FakeTx:
    def __init__(self, run_fn: Callable):
        self._run_fn = run_fn

    def run(self, query: str, **params: Any) -> Any:
        return self._run_fn(query, params)


class FakeSession:
    def __init__(self, run_fn: Callable, delay: float = 0.0):
        self._run_fn = run_fn
        self._delay = delay
        self.call_count = 0

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def execute_read(self, work_fn: Callable) -> Any:
        self.call_count += 1
        if self._delay > 0:
            time.sleep(self._delay)
        tx = FakeTx(self._run_fn)
        return work_fn(tx)

    def run(self, query: str, **params: Any) -> Any:
        self.call_count += 1
        if self._delay > 0:
            time.sleep(self._delay)
        return self._run_fn(query, params)


class FakeDriver:
    def __init__(self, run_fn: Callable, delay: float = 0.0):
        self._run_fn = run_fn
        self._delay = delay
        self.sessions: List[FakeSession] = []

    def session(self, **kwargs: Any) -> FakeSession:
        s = FakeSession(self._run_fn, delay=self._delay)
        self.sessions.append(s)
        return s

    def close(self) -> None:
        pass


class FakeResult:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def data(self) -> List[Dict[str, Any]]:
        return self._rows

    def single(self) -> Optional[Dict[str, Any]]:
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


# ================================================================ read_applied_write tests

def test_lookup_applied_status_one_row():
    """Exactly one marker returned -> APPLIED with operation, fingerprint, and applied_at."""
    def run_fn(query, params):
        assert "request_key" in params
        return FakeResult([{
            "request_key": params["request_key"],
            "applied_at": "2026-11-13T06:00:00Z",
            "operation": "test_event.update",
            "fingerprint": "a1b2c3d4e5f60000",
        }])

    driver = FakeDriver(run_fn)
    lookup = read_applied_write("wazuh:mgr/1.0", driver=driver)
    assert lookup.status == MarkerStatus.APPLIED
    assert lookup.operation == "test_event.update"
    assert lookup.fingerprint == "a1b2c3d4e5f60000"
    assert lookup.applied_at == "2026-11-13T06:00:00Z"


def test_lookup_absent_status_zero_rows():
    """Zero rows returned -> ABSENT."""
    driver = FakeDriver(lambda q, p: FakeResult([]))
    lookup = read_applied_write("wazuh:mgr/1.0", driver=driver)
    assert lookup.status == MarkerStatus.ABSENT
    assert lookup.operation is None
    assert lookup.fingerprint is None


def test_lookup_inconsistent_status_multiple_rows():
    """Multiple rows returned -> INCONSISTENT (duplicate markers on missing constraint)."""
    driver = FakeDriver(lambda q, p: FakeResult([
        {"request_key": "k", "applied_at": "t1", "operation": "op", "fingerprint": "f1"},
        {"request_key": "k", "applied_at": "t2", "operation": "op", "fingerprint": "f2"},
    ]))
    lookup = read_applied_write("wazuh:mgr/1.0", driver=driver)
    assert lookup.status == MarkerStatus.INCONSISTENT
    assert "2 markers" in lookup.diagnostic


def test_lookup_unknown_status_on_driver_exception():
    """Any exception during lookup -> UNKNOWN (never ABSENT)."""
    def run_fn(query, params):
        raise ConnectionResetError("Neo4j socket reset by peer")

    driver = FakeDriver(run_fn)
    lookup = read_applied_write("wazuh:mgr/1.0", driver=driver)
    assert lookup.status == MarkerStatus.UNKNOWN
    assert lookup.error_type == "ConnectionResetError"
    assert "socket reset" in lookup.diagnostic


def test_lookup_deadline_stalled_read_is_unknown():
    """A stalled read exceeding deadline returns UNKNOWN with ClientDeadline."""
    driver = FakeDriver(lambda q, p: FakeResult([]), delay=1.0)
    lookup = read_applied_write("wazuh:mgr/1.0", driver=driver, deadline=0.1)
    assert lookup.status == MarkerStatus.UNKNOWN
    assert lookup.error_type == "ClientDeadline"
    assert "deadline" in lookup.diagnostic


def test_lookup_never_retries_on_failure():
    """Failed reads do NOT retry - exactly one call to execute_read / session."""
    call_records = []

    def run_fn(query, params):
        call_records.append(1)
        raise TimeoutError("connection timeout")

    driver = FakeDriver(run_fn)
    lookup = read_applied_write("wazuh:mgr/1.0", driver=driver)
    assert lookup.status == MarkerStatus.UNKNOWN
    assert len(call_records) == 1
    assert sum(s.call_count for s in driver.sessions) == 1


def test_lookup_invalid_key_input_rejected():
    """Non-string or empty keys return UNKNOWN with ValueError."""
    assert read_applied_write("").status == MarkerStatus.UNKNOWN
    assert read_applied_write(None).status == MarkerStatus.UNKNOWN
    assert read_applied_write(123).status == MarkerStatus.UNKNOWN


def test_lookup_no_driver_returns_unknown(monkeypatch):
    """When driver is None and no global driver exists, returns UNKNOWN without network calls."""
    import neo4j_client
    monkeypatch.setattr(neo4j_client, "driver", None)
    lookup = read_applied_write("wazuh:mgr/1.0", driver=None)
    assert lookup.status == MarkerStatus.UNKNOWN
    assert lookup.error_type == "DriverError"
    assert "no Neo4j driver" in lookup.diagnostic


def test_lookup_canary_secrets_never_logged(caplog):
    """Canary values in key or parameters never appear in logs."""
    canary_key = "wazuh:SECRET_MGR_CANARY/SECRET_ID_CANARY_9999"

    def run_fn(query, params):
        raise RuntimeError("database exploded with SECRET_MGR_CANARY")

    driver = FakeDriver(run_fn)
    with caplog.at_level(logging.DEBUG):
        lookup = read_applied_write(canary_key, driver=driver)

    assert "SECRET_MGR_CANARY" not in caplog.text
    assert "SECRET_ID_CANARY" not in caplog.text
    assert lookup.status == MarkerStatus.UNKNOWN


# ================================================================ read_marker_schema tests

def _schema_driver(constraints: List[Dict], indexes: List[Dict], labels: List[str]):
    def run_fn(query, params):
        if "SHOW CONSTRAINTS" in query:
            return FakeResult(constraints)
        elif "SHOW INDEXES" in query:
            return FakeResult(indexes)
        elif "db.labels" in query:
            return FakeResult([{"label": l} for l in labels])
        return FakeResult([])
    return FakeDriver(run_fn)


def test_schema_row_1_present_valid_with_label():
    """Constraint present, index ONLINE, label present -> PRESENT_VALID."""
    driver = _schema_driver(
        constraints=[{
            "name": "applied_write_key",
            "type": "UNIQUENESS",
            "entityType": "NODE",
            "labelsOrTypes": ["AppliedWrite"],
            "properties": ["request_key"],
            "ownedIndexName": "applied_write_key",
        }],
        indexes=[{
            "name": "applied_write_key",
            "state": "ONLINE",
            "labelsOrTypes": ["AppliedWrite"],
            "properties": ["request_key"],
        }],
        labels=["AppliedWrite", "Campaign", "AttackEvent"],
    )
    report = read_marker_schema(driver=driver)
    assert report.status == ConstraintStatus.PRESENT_VALID
    assert report.valid is True
    assert report.alarm is False
    assert report.has_label is True
    assert report.index_state == "ONLINE"


def test_schema_row_2_present_valid_without_label():
    """Constraint present, index ONLINE, label not yet in db.labels() -> PRESENT_VALID."""
    driver = _schema_driver(
        constraints=[{
            "name": "applied_write_key",
            "type": "UNIQUENESS",
            "entityType": "NODE",
            "labelsOrTypes": ["AppliedWrite"],
            "properties": ["request_key"],
            "ownedIndexName": "applied_write_key",
        }],
        indexes=[{
            "name": "applied_write_key",
            "state": "ONLINE",
            "labelsOrTypes": ["AppliedWrite"],
            "properties": ["request_key"],
        }],
        labels=["Campaign", "AttackEvent"],
    )
    report = read_marker_schema(driver=driver)
    assert report.status == ConstraintStatus.PRESENT_VALID
    assert report.valid is True
    assert report.alarm is False
    assert report.has_label is False


def test_schema_row_3_absent_constraint_and_absent_label():
    """Constraint absent, label absent -> ABSENT (rollout hasn't happened)."""
    driver = _schema_driver(
        constraints=[],
        indexes=[],
        labels=["Campaign", "AttackEvent"],
    )
    report = read_marker_schema(driver=driver)
    assert report.status == ConstraintStatus.ABSENT
    assert report.valid is False
    assert report.alarm is False
    assert "rollout has not happened" in report.diagnostic


def test_schema_row_4_dropped_constraint_with_label_present():
    """Constraint absent, but label AppliedWrite exists -> DROPPED_LABEL_PRESENT (ALARM!)."""
    driver = _schema_driver(
        constraints=[],
        indexes=[],
        labels=["AppliedWrite", "Campaign"],
    )
    report = read_marker_schema(driver=driver)
    assert report.status == ConstraintStatus.DROPPED_LABEL_PRESENT
    assert report.valid is False
    assert report.alarm is True
    assert "dropped" in report.reasons[0].lower() or "alarm" in report.reasons[0].lower()


def test_schema_row_5_wrong_properties():
    """Constraint exists on AppliedWrite but on property other than request_key -> WRONG_PROPERTIES."""
    driver = _schema_driver(
        constraints=[{
            "name": "applied_write_key",
            "type": "UNIQUENESS",
            "entityType": "NODE",
            "labelsOrTypes": ["AppliedWrite"],
            "properties": ["wrong_prop"],
            "ownedIndexName": "applied_write_key",
        }],
        indexes=[{
            "name": "applied_write_key",
            "state": "ONLINE",
        }],
        labels=["AppliedWrite"],
    )
    report = read_marker_schema(driver=driver)
    assert report.status == ConstraintStatus.WRONG_PROPERTIES
    assert report.valid is False


def test_schema_row_6_wrong_constraint_name():
    """Constraint exists on AppliedWrite with wrong name -> WRONG_PROPERTIES."""
    driver = _schema_driver(
        constraints=[{
            "name": "other_constraint_name",
            "type": "UNIQUENESS",
            "entityType": "NODE",
            "labelsOrTypes": ["AppliedWrite"],
            "properties": ["request_key"],
            "ownedIndexName": "other_constraint_name",
        }],
        indexes=[{
            "name": "other_constraint_name",
            "state": "ONLINE",
        }],
        labels=["AppliedWrite"],
    )
    report = read_marker_schema(driver=driver)
    assert report.status == ConstraintStatus.WRONG_PROPERTIES
    assert report.valid is False


def test_schema_row_7_wrong_constraint_type():
    """Constraint exists on AppliedWrite but is not UNIQUENESS -> WRONG_PROPERTIES."""
    driver = _schema_driver(
        constraints=[{
            "name": "applied_write_key",
            "type": "NODE_PROPERTY_EXISTENCE",
            "entityType": "NODE",
            "labelsOrTypes": ["AppliedWrite"],
            "properties": ["request_key"],
            "ownedIndexName": "applied_write_key",
        }],
        indexes=[],
        labels=["AppliedWrite"],
    )
    report = read_marker_schema(driver=driver)
    assert report.status == ConstraintStatus.WRONG_PROPERTIES
    assert report.valid is False


def test_schema_row_8_index_not_online():
    """Constraint exists but owned index state is POPULATING or FAILED -> INDEX_NOT_ONLINE."""
    driver = _schema_driver(
        constraints=[{
            "name": "applied_write_key",
            "type": "UNIQUENESS",
            "entityType": "NODE",
            "labelsOrTypes": ["AppliedWrite"],
            "properties": ["request_key"],
            "ownedIndexName": "applied_write_key",
        }],
        indexes=[{
            "name": "applied_write_key",
            "state": "POPULATING",
            "labelsOrTypes": ["AppliedWrite"],
            "properties": ["request_key"],
        }],
        labels=["AppliedWrite"],
    )
    report = read_marker_schema(driver=driver)
    assert report.status == ConstraintStatus.INDEX_NOT_ONLINE
    assert report.valid is False
    assert "ONLINE" in report.diagnostic


def test_schema_row_9_driver_failure_returns_unknown():
    """Driver query failure during read_marker_schema returns UNKNOWN."""
    def run_fn(query, params):
        raise RuntimeError("database communication failure")

    driver = FakeDriver(run_fn)
    report = read_marker_schema(driver=driver)
    assert report.status == ConstraintStatus.UNKNOWN
    assert report.valid is False


# ================================================================ verify_startup tests

def _startup_driver(version="5.26.27", edition="community", constraints=None, indexes=None, labels=None, invalid_markers=0):
    if constraints is None:
        constraints = [{
            "name": "applied_write_key",
            "type": "UNIQUENESS",
            "entityType": "NODE",
            "labelsOrTypes": ["AppliedWrite"],
            "properties": ["request_key"],
            "ownedIndexName": "applied_write_key",
        }]
    if indexes is None:
        indexes = [{
            "name": "applied_write_key",
            "state": "ONLINE",
            "labelsOrTypes": ["AppliedWrite"],
            "properties": ["request_key"],
        }]
    if labels is None:
        labels = ["AppliedWrite"]

    def run_fn(query, params):
        if "dbms.components" in query:
            return FakeResult([{"name": "Neo4j Kernel", "versions": [version], "edition": edition}])
        elif "SHOW CONSTRAINTS" in query:
            return FakeResult(constraints)
        elif "SHOW INDEXES" in query:
            return FakeResult(indexes)
        elif "db.labels" in query:
            return FakeResult([{"label": l} for l in labels])
        elif "WHERE m.operation IS NULL OR m.fingerprint IS NULL" in query:
            return FakeResult([{"invalid_count": invalid_markers}])
        return FakeResult([])

    return FakeDriver(run_fn)


def test_verify_startup_all_checks_pass():
    """All 6 checks pass -> marker_mode_allowed=True, valid=True, alarm=False."""
    driver = _startup_driver()
    report = verify_startup(driver=driver)
    assert report.valid is True
    assert report.marker_mode_allowed is True
    assert report.alarm is False
    assert report.server_version == "5.26.27"
    assert report.server_edition == "community"
    assert report.schema.status == ConstraintStatus.PRESENT_VALID


def test_verify_startup_check_1_components_fail():
    """Check 1: dbms.components query fails -> UNKNOWN, marker mode refused."""
    def run_fn(query, params):
        raise ConnectionError("cannot reach Neo4j")

    driver = FakeDriver(run_fn)
    report = verify_startup(driver=driver)
    assert report.valid is False
    assert report.marker_mode_allowed is False
    assert "UNKNOWN" in report.diagnostic


def test_verify_startup_check_2_version_mismatch():
    """Check 2: Server version does not match validated version -> Refused."""
    driver = _startup_driver(version="4.4.12")
    report = verify_startup(driver=driver, allowed_versions=("5.26.27",))
    assert report.valid is False
    assert report.marker_mode_allowed is False
    assert "4.4.12" in report.reasons[0]


def test_verify_startup_check_2_edition_mismatch():
    """Check 2: Server edition does not match validated editions -> Refused."""
    driver = _startup_driver(edition="aura")
    report = verify_startup(driver=driver, allowed_editions=("community",))
    assert report.valid is False
    assert report.marker_mode_allowed is False
    assert "aura" in report.reasons[0]


def test_verify_startup_check_3_constraint_missing():
    """Check 3: Constraint is missing -> marker mode refused."""
    driver = _startup_driver(constraints=[], indexes=[], labels=[])
    report = verify_startup(driver=driver)
    assert report.valid is False
    assert report.marker_mode_allowed is False
    assert report.schema.status == ConstraintStatus.ABSENT


def test_verify_startup_check_5_dropped_constraint_triggers_alarm():
    """Check 5: Constraint dropped while label exists -> Refused with ALARM."""
    driver = _startup_driver(constraints=[], indexes=[], labels=["AppliedWrite"])
    report = verify_startup(driver=driver)
    assert report.valid is False
    assert report.marker_mode_allowed is False
    assert report.alarm is True
    assert report.schema.status == ConstraintStatus.DROPPED_LABEL_PRESENT


def test_verify_startup_check_6_invalid_markers_sampled_triggers_alarm():
    """Check 6: Sample found markers missing operation or fingerprint -> Refused with ALARM."""
    driver = _startup_driver(invalid_markers=3)
    report = verify_startup(driver=driver)
    assert report.valid is False
    assert report.marker_mode_allowed is False
    assert report.alarm is True
    assert report.sample_invalid_count == 3
    assert "Alarm" in report.reasons[0]


# ================================================================ Coordinator Integration Tests

def test_coordinator_accepts_marker_reader(tmp_path):
    """WriteCoordinator accepts marker_reader and delegates verify_startup."""
    from neo4j_idempotency_ledger import IdempotencyLedger
    from neo4j_write_journal import NeoWriteJournal
    from tests._neo4j_fake_writer import FakeWriter

    journal = NeoWriteJournal(str(tmp_path / "j.db"))
    ledger = IdempotencyLedger(str(tmp_path / "l.db"))
    reg = OperationRegistry([RegisteredOperation("test.op", "MATCH (n) SET n.v = $v", frozenset({"v"}))])

    driver = _startup_driver()

    class Adapter:
        def verify_startup(self, **kwargs):
            return verify_startup(driver=driver, **kwargs)

    coord = WriteCoordinator(
        journal, ledger,
        writer=FakeWriter(),
        registry=reg,
        marker_reader=Adapter(),
    )
    report = coord.verify_startup()
    assert report.marker_mode_allowed is True
    journal.close()
    ledger.close()


def test_coordinator_verify_startup_without_marker_reader_raises(tmp_path):
    """WriteCoordinator.verify_startup without marker_reader raises CoordinatorError."""
    from neo4j_idempotency_ledger import IdempotencyLedger
    from neo4j_write_journal import NeoWriteJournal
    from tests._neo4j_fake_writer import FakeWriter

    journal = NeoWriteJournal(str(tmp_path / "j.db"))
    ledger = IdempotencyLedger(str(tmp_path / "l.db"))
    reg = OperationRegistry([RegisteredOperation("test.op", "MATCH (n) SET n.v = $v", frozenset({"v"}))])

    coord = WriteCoordinator(journal, ledger, writer=FakeWriter(), registry=reg)
    with pytest.raises(CoordinatorError, match="no marker_reader"):
        coord.verify_startup()
    journal.close()
    ledger.close()


# ================================================================ Guard Tests

def test_coordinator_source_has_no_forbidden_marker_strings():
    """neo4j_write_coordinator.py does not contain AppliedWrite, CREATE CONSTRAINT, etc outside docstrings."""
    source = (BACKEND / "neo4j_write_coordinator.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.split('"""', 2)[2].splitlines() if not line.lstrip().startswith("#"))
    for word in ("AppliedWrite", "CREATE CONSTRAINT", "investigation_payload", "SHOW CONSTRAINTS"):
        assert word not in code, f"Forbidden word '{word}' found in coordinator source"


def test_neo4j_client_does_not_import_coordinator_or_journal_or_ledger():
    """neo4j_client.py remains independent and does not import the coordinator, journal, or ledger."""
    source = (BACKEND / "neo4j_client.py").read_text(encoding="utf-8")
    assert "neo4j_write_coordinator" not in source
    assert "neo4j_write_journal" not in source
    assert "neo4j_idempotency_ledger" not in source

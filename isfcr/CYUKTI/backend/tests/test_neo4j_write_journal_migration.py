"""Schema version 1 to 2. Every old file here is made by the previous implementation itself, kept byte for byte in
_neo4j_journal_v1_frozen.py, so these don't depend on anyone's memory of what version 1 wrote."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import neo4j_write_journal as module
from neo4j_write_journal import JournalCorruptionError, JournalSchemaError, JournalState as S, NeoWriteJournal, Resolution
from tests import _neo4j_journal_v1_frozen as v1
from tests._neo4j_child_tempdirs import child_processes

CHILD = str(Path(__file__).resolve().parent / "_neo4j_journal_lifecycle_child.py")
FROZEN = Path(v1.__file__)
KEY = "cyukti.v1|wazuh:manager-01/1773199644.1234|duplicate_event.update"
ONE = [("CREATE (n:X {v:$v})", {"v": 1})]
V1_RECORD_COLUMNS = ", ".join(v1._RECORD_COLUMNS) + ", row_sha256"
V1_HISTORY_COLUMNS = ", ".join(v1._HISTORY_COLUMNS) + ", prev_sha256, entry_sha256"


@pytest.fixture
def path(tmp_path):
    return tmp_path / "journal.db"


def _make_v1(path):
    """One record in each state, written and closed by the previous code. Returns their ids by state."""
    ids = {}
    with v1.NeoWriteJournal(path) as journal:
        ids[S.PENDING] = journal.create_intent("op", ONE).record_id
        record = journal.mark_in_flight(journal.create_intent("op", ONE).record_id, worker_id="w1")
        ids[S.IN_FLIGHT] = record.record_id
        record = journal.create_intent("op", [("Q", {"token": "value", "n": 2 ** 53 + 1})])
        journal.mark_in_flight(record.record_id)
        ids[S.COMMITTED] = journal.mark_committed(record.record_id).record_id
        record = journal.create_intent("op", ONE)
        journal.mark_in_flight(record.record_id)
        ids[S.NOT_COMMITTED] = journal.mark_not_committed(record.record_id, error_type="ServiceUnavailable").record_id
        record = journal.create_intent("op", ONE)
        journal.mark_in_flight(record.record_id)
        ids[S.AMBIGUOUS] = journal.mark_ambiguous(record.record_id, error_type="IncompleteCommit").record_id
        record = journal.create_intent("op", ONE)
        journal.mark_in_flight(record.record_id)
        journal.mark_ambiguous(record.record_id)
        ids[S.RESOLVED] = journal.resolve_ambiguous(record.record_id, resolution=v1.Resolution.ABANDONED, resolved_by="a", reason="r").record_id
    return ids


def _v1_content(path):
    """The version 1 columns of every row, checksums included, read without either implementation."""
    conn = sqlite3.connect(path)
    try:
        return (conn.execute(f"SELECT {V1_RECORD_COLUMNS} FROM journal_records ORDER BY seq").fetchall(),
                conn.execute(f"SELECT {V1_HISTORY_COLUMNS} FROM journal_history ORDER BY seq").fetchall())
    finally:
        conn.close()


def _version(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _schema(path):
    conn = sqlite3.connect(path)
    try:
        return sorted(row[0] for row in conn.execute("SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL"))
    finally:
        conn.close()


def test_the_frozen_copy_really_is_the_previous_implementation():
    """Recorded before Phase 2.6B changed anything. If this fails, these tests no longer test what they claim."""
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest().startswith("c69834f8475dc742")
    assert v1.SCHEMA_VERSION == 1


def test_an_old_file_opens_with_every_record_readable_and_nothing_rewritten(path):
    ids = _make_v1(path)
    records_before, events_before = _v1_content(path)
    with v1.NeoWriteJournal(path) as old:
        expected = {state: old.get_record(i) for state, i in ids.items()}
        expected_history = {state: old.get_history(i) for state, i in ids.items()}
    with NeoWriteJournal(path) as journal:
        report = journal.check_integrity()
        assert report.ok, report
        assert report.schema_version == 2 and report.records_checked == 6
        for state, record_id in ids.items():
            record = journal.get_record(record_id)
            assert record.state is state
            assert (record.correlation_key, record.epoch, record.ledger_attempt) == (None, None, None)
            old_record = expected[state]
            for name in ("record_id", "operation", "created_at", "updated_at", "attempt_count", "worker_id", "owner_session", "last_outcome",
                         "error_type", "error_code", "diagnostic", "intent_sha256", "version"):
                assert getattr(record, name) == getattr(old_record, name), name
            assert module._identical(record.statements, old_record.statements)
            events = journal.get_history(record_id)
            assert [(e.transition_no, e.to_state.value, e.cause, e.at, e.actor, e.attempt) for e in events] == [
                (e.transition_no, e.to_state.value, e.cause, e.at, e.actor, e.attempt) for e in expected_history[state]]
    assert _v1_content(path) == (records_before, events_before), "not one existing value or checksum was rewritten"


def test_new_columns_of_old_rows_are_empty_and_the_file_stays_private(path):
    _make_v1(path)
    NeoWriteJournal(path).close()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT count(*) FROM journal_records WHERE correlation_key IS NOT NULL OR epoch IS NOT NULL OR ledger_attempt IS NOT NULL").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM journal_history WHERE correlation_key IS NOT NULL OR epoch IS NOT NULL OR ledger_attempt IS NOT NULL").fetchone()[0] == 0
    conn.close()


def test_a_migrated_file_has_the_same_schema_as_a_new_one(path, tmp_path):
    _make_v1(path)
    NeoWriteJournal(path).close()
    fresh = tmp_path / "fresh.db"
    NeoWriteJournal(fresh).close()
    assert _schema(path) == _schema(fresh)
    assert _version(path) == _version(fresh) == 2


def test_old_records_carry_on_under_the_old_rules(path):
    ids = _make_v1(path)
    with NeoWriteJournal(path) as journal:
        report = journal.recover_unfinished()
        assert report.recovered == (ids[S.IN_FLIGHT],), "the old process is gone, so its record becomes unknown"
        retried = journal.authorize_retry(ids[S.NOT_COMMITTED], authorized_by="op", reason="r")
        assert retried.state is S.IN_FLIGHT and retried.ledger_attempt is None
        assert journal.mark_committed(ids[S.NOT_COMMITTED]).state is S.COMMITTED
        assert journal.resolve_ambiguous(ids[S.AMBIGUOUS], resolution=Resolution.FOUND_NOT_COMMITTED, resolved_by="a", reason="r").state is S.RESOLVED
        assert journal.mark_in_flight(ids[S.PENDING]).state is S.IN_FLIGHT
        assert journal.check_integrity().ok, "the history chain extends across the migration"
    records, events = _v1_content(path)
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT count(*) FROM journal_history WHERE ledger_attempt IS NOT NULL").fetchone()[0] == 0
    conn.close()
    assert len(events) > 0 and len(records) == 6


def test_correlated_records_can_be_created_next_to_old_ones(path):
    ids = _make_v1(path)
    with NeoWriteJournal(path) as journal:
        record = journal.create_intent("op", ONE, correlation_key=KEY, epoch=1)
        journal.mark_in_flight(record.record_id, ledger_attempt=1)
        journal.mark_ambiguous(record.record_id, expected_ledger_attempt=1)
        journal.resolve_ambiguous(record.record_id, resolution=Resolution.FOUND_NOT_COMMITTED, resolved_by="a", reason="r")
        second = journal.create_intent("op", ONE, correlation_key=KEY, epoch=2, authorized_by="a", reason="r")
        assert [r.epoch for r in journal.find_by_correlation_key(KEY)] == [1, 2]
        assert journal.get_record(ids[S.COMMITTED]).correlation_key is None
        assert journal.check_integrity().ok
    with NeoWriteJournal(path) as reopened:
        assert reopened.get_record(second.record_id).state is S.PENDING
        assert reopened.check_integrity().ok


def test_an_empty_old_file_migrates(path):
    v1.NeoWriteJournal(path).close()
    with NeoWriteJournal(path) as journal:
        report = journal.check_integrity()
        assert report.ok and report.records_checked == 0 and report.schema_version == 2
        assert journal.create_intent("op", ONE, correlation_key=KEY, epoch=1).epoch == 1


def test_reopening_after_migration_changes_nothing_and_migrates_once(path, caplog):
    _make_v1(path)
    caplog.set_level(logging.INFO, logger="neo4j_write_journal")
    NeoWriteJournal(path).close()
    assert caplog.text.count("migrated") == 1
    first_schema, first_content = _schema(path), _v1_content(path)
    caplog.clear()
    for _ in range(3):
        with NeoWriteJournal(path) as journal:
            assert journal.check_integrity().ok
    assert "migrated" not in caplog.text
    assert (_schema(path), _v1_content(path)) == (first_schema, first_content)


def test_the_previous_implementation_refuses_a_migrated_file_and_cannot_harm_it(path):
    """There's no way back through the old code, and it fails closed rather than misreading."""
    _make_v1(path)
    NeoWriteJournal(path).close()
    content = _v1_content(path)
    with pytest.raises(v1.JournalSchemaError, match="version 2"):
        v1.NeoWriteJournal(path)
    assert _v1_content(path) == content


def test_an_old_process_still_running_through_the_migration_keeps_writing_validly(path):
    """It already passed its version check. It names its columns, so its writes stay correct and uncorrelated."""
    with v1.NeoWriteJournal(path) as old:
        record = old.create_intent("op", ONE)
        old.mark_in_flight(record.record_id)
        with NeoWriteJournal(path) as journal:
            assert journal.recover_unfinished().still_running == (record.record_id,), "its lock still shows it's alive"
            assert old.mark_committed(record.record_id).state is v1.JournalState.COMMITTED
            late = old.create_intent("op", ONE)
            assert journal.get_record(late.record_id).correlation_key is None
            assert journal.check_integrity().ok


def test_read_only_never_migrates(path):
    _make_v1(path)
    content, schema = _v1_content(path), _schema(path)
    with pytest.raises(JournalSchemaError, match="version 1"):
        NeoWriteJournal(path, read_only=True)
    assert (_version(path), _schema(path), _v1_content(path)) == (1, schema, content)
    NeoWriteJournal(path).close()
    with NeoWriteJournal(path, read_only=True) as reader:
        assert reader.check_integrity().ok


def test_a_failure_during_migration_rolls_all_of_it_back(path, monkeypatch):
    ids = _make_v1(path)
    content, schema = _v1_content(path), _schema(path)
    monkeypatch.setattr(module, "_SCHEMA_V2", (*module._SCHEMA_V2, "CREATE TABLE journal_records (x)"))
    with pytest.raises(module.JournalError):
        NeoWriteJournal(path)
    monkeypatch.undo()
    assert (_version(path), _schema(path), _v1_content(path)) == (1, schema, content), "no column, index or trigger survived"
    with v1.NeoWriteJournal(path) as old:
        assert old.check_integrity().ok, "the old code still works on it"
    with NeoWriteJournal(path) as journal:
        assert journal.get_record(ids[S.COMMITTED]).state is S.COMMITTED and journal.check_integrity().ok


def test_a_non_sqlite_error_during_migration_also_rolls_back(path, monkeypatch):
    _make_v1(path)
    content = _v1_content(path)

    class _Explodes(tuple):
        def __iter__(self):
            yield module._SCHEMA_V2_REAL[0]
            raise RuntimeError("injected")

    monkeypatch.setattr(module, "_SCHEMA_V2_REAL", module._SCHEMA_V2, raising=False)
    monkeypatch.setattr(module, "_SCHEMA_V2", _Explodes())
    with pytest.raises(RuntimeError):
        NeoWriteJournal(path)
    monkeypatch.undo()
    assert (_version(path), _v1_content(path)) == (1, content)
    conn = sqlite3.connect(path)
    assert "correlation_key" not in {row[1] for row in conn.execute("PRAGMA table_info(journal_records)")}
    conn.close()


def test_a_lock_held_too_long_fails_the_open_and_leaves_the_file_as_it_was(path):
    _make_v1(path)
    content = _v1_content(path)
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(module.JournalBusyError):
            NeoWriteJournal(path, busy_timeout=0.1)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert (_version(path), _v1_content(path)) == (1, content)
    with NeoWriteJournal(path) as journal:
        assert journal.check_integrity().ok


def test_killed_at_the_migrations_commit_it_stays_version_1_and_migrates_next_time(path, tmp_path):
    ids = _make_v1(path)
    content, schema = _v1_content(path), _schema(path)
    with child_processes(str(tmp_path)) as children:
        result = children.run([sys.executable, CHILD, "migrate_dying", str(path), str(tmp_path / "unused")], capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    assert result.returncode == -signal.SIGKILL, result.stderr[-800:]
    assert "survived" not in result.stdout
    assert (_version(path), _schema(path), _v1_content(path)) == (1, schema, content)
    with v1.NeoWriteJournal(path, read_only=True) as old:
        assert old.check_integrity().ok, "still a sound version 1 file, even for the old code"
    with NeoWriteJournal(path) as journal:
        assert journal.check_integrity().ok and journal.check_integrity().schema_version == 2
        assert journal.get_record(ids[S.RESOLVED]).state is S.RESOLVED
    assert _v1_content(path) == (content[0], content[1])


def test_processes_migrating_at_once_do_it_once_and_lose_nothing(path, tmp_path):
    _make_v1(path)
    content = _v1_content(path)
    go = tmp_path / "go"
    with child_processes(str(tmp_path)) as children:
        processes = [children.spawn([sys.executable, CHILD, "open_when_told", str(path), str(go)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(tmp_path))
                     for _ in range(4)]
        time.sleep(1.5)
        go.touch()
        reports = []
        for process in processes:
            out, err = process.communicate(timeout=90)
            assert process.returncode == 0, err[-800:]
            reports.append(json.loads(out.strip().splitlines()[-1]))
    assert reports == [{"version": 2, "ok": True, "records": 6}] * 4
    assert _v1_content(path) == content
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT count(*) FROM sqlite_schema WHERE name = 'journal_records_correlation'").fetchone()[0] == 1
    conn.close()


def test_an_open_that_waited_for_the_lock_sees_the_migration_was_done_and_does_not_repeat_it(path):
    """The race above, made deterministic: this open reads version 1, then waits while another finishes migrating."""
    _make_v1(path)
    content = _v1_content(path)
    other = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    other.execute("BEGIN IMMEDIATE")
    for statement in module._SCHEMA_V2:
        other.execute(statement)
    other.execute(f"PRAGMA user_version = {module.SCHEMA_VERSION}")
    outcome = []

    def opens():
        try:
            with NeoWriteJournal(path, busy_timeout=30) as journal:
                outcome.append(journal.check_integrity().ok)
        except Exception as error:
            outcome.append(type(error).__name__)

    thread = threading.Thread(target=opens)
    thread.start()
    thread.join(1.0)
    assert thread.is_alive(), "it has to be waiting on the lock, or this isn't the race"
    other.execute("COMMIT")
    other.close()
    thread.join(30)
    assert outcome == [True]
    assert _v1_content(path) == content


@pytest.mark.parametrize("sabotage, error", [
    ("DROP TRIGGER journal_history_no_delete", JournalCorruptionError),
    ("DROP INDEX journal_records_state", JournalCorruptionError),
    ("ALTER TABLE journal_records ADD COLUMN surprise TEXT", JournalSchemaError),
    ("ALTER TABLE journal_history ADD COLUMN correlation_key TEXT", JournalSchemaError),
], ids=["missing trigger", "missing index", "unexpected column", "half migrated by hand"])
def test_a_version_1_file_that_is_not_what_it_should_be_is_refused_and_unchanged(path, sabotage, error):
    _make_v1(path)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute(sabotage)
    conn.close()
    schema = _schema(path)
    with pytest.raises(error):
        NeoWriteJournal(path)
    assert (_version(path), _schema(path)) == (1, schema)


def test_damage_to_an_old_record_is_still_found_after_migration(path):
    ids = _make_v1(path)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("UPDATE journal_records SET attempt_count = 9 WHERE record_id = ?", (ids[S.COMMITTED],))
    conn.close()
    with NeoWriteJournal(path) as journal:
        report = journal.check_integrity()
        assert not report.ok and report.corrupt_record_ids == (ids[S.COMMITTED],)
        with pytest.raises(JournalCorruptionError):
            journal.get_record(ids[S.COMMITTED])
        assert journal.get_record(ids[S.PENDING]).state is S.PENDING


def test_forging_correlation_onto_an_old_record_is_found_after_migration(path):
    ids = _make_v1(path)
    NeoWriteJournal(path).close()
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("DROP TRIGGER journal_records_correlation_fixed")
    conn.execute("UPDATE journal_records SET correlation_key = ?, epoch = 1 WHERE record_id = ?", (KEY, ids[S.COMMITTED]))
    conn.execute(module._SCHEMA_V2[7])
    conn.close()
    with NeoWriteJournal(path) as journal:
        assert journal.check_integrity().corrupt_record_ids == (ids[S.COMMITTED],)


def test_a_missing_v2_object_after_migration_is_refused(path):
    _make_v1(path)
    NeoWriteJournal(path).close()
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("DROP INDEX journal_records_correlation")
    conn.close()
    with pytest.raises(JournalCorruptionError, match="schema objects"):
        NeoWriteJournal(path)


@pytest.mark.parametrize("table, column", [("journal_records", "ledger_attempt"), ("journal_history", "ledger_attempt")])
def test_a_missing_v2_column_after_migration_is_refused(path, table, column):
    _make_v1(path)
    NeoWriteJournal(path).close()
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("DROP TRIGGER IF EXISTS journal_records_attempt_rises")
    conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    conn.execute(module._SCHEMA_V2[8] if table == "journal_history" else "CREATE TRIGGER journal_records_attempt_rises BEFORE DELETE ON journal_records BEGIN SELECT 1; END")
    conn.close()
    with pytest.raises(JournalCorruptionError, match="missing columns"):
        NeoWriteJournal(path)


def test_a_file_sqlite_reports_as_damaged_is_not_migrated(path, monkeypatch):
    """The check itself, reported as failing: real page damage in a file this size can land outside what quick_check reads."""
    _make_v1(path)
    schema = _schema(path)
    real_connect = sqlite3.connect

    class _Failing(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql.startswith("PRAGMA quick_check"):
                return super().execute("SELECT 'page 2 is damaged'")
            return super().execute(sql, *args)

    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: real_connect(*args, factory=_Failing, **kwargs))
    with pytest.raises(JournalCorruptionError, match="quick check"):
        NeoWriteJournal(path)
    monkeypatch.undo()
    assert (_version(path), _schema(path)) == (1, schema)


def test_a_version_1_file_still_in_rollback_mode_migrates(path):
    """A version 1 open that died after initializing, before it turned on WAL."""
    _make_v1(path)
    conn = sqlite3.connect(path, isolation_level=None)
    assert conn.execute("PRAGMA journal_mode = DELETE").fetchone()[0].lower() == "delete"
    conn.close()
    with NeoWriteJournal(path) as journal:
        assert journal._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert journal.check_integrity().ok

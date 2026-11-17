"""Fault injection: failing storage, lock contention, damaged and foreign files. Whatever fails,
the write must not be acknowledged, no false COMMITTED may appear, and existing evidence must be
left exactly as it was."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import sys
from pathlib import Path

import pytest

from neo4j_write_journal import (
    InvalidTransitionError,
    JournalBusyError,
    JournalCorruptionError,
    JournalSchemaError,
    JournalState as S,
    JournalWriteError,
    NeoWriteJournal,
    SCHEMA_VERSION,
    UnsupportedParameterError,
    _RECORD_COLUMNS,
)
from tests._neo4j_child_tempdirs import child_processes

# Calls install_child_guards before importing backend code, as the crash tests check.
CHILD = str(Path(__file__).resolve().parent / "_neo4j_journal_child.py")
ONE = [("CREATE (n:X {v:$v})", {"v": 1})]
SECRET = "s3cret-parameter-value"


@pytest.fixture
def path(tmp_path):
    return tmp_path / "journal.db"


def _fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inject_history_failure(journal):
    """Any write of history now fails, inside the transition's transaction."""
    journal._conn.execute("CREATE TEMP TRIGGER fail_history BEFORE INSERT ON main.journal_history "
                          "BEGIN SELECT RAISE(ABORT, 'injected'); END")


def _heal_history_failure(journal):
    journal._conn.execute("DROP TRIGGER temp.fail_history")


def _fill_the_disk(journal):
    """SQLite's own limit: the next page it needs gets the same error as a full disk."""
    pages = journal._conn.execute("PRAGMA page_count").fetchone()[0]
    journal._conn.execute(f"PRAGMA max_page_count = {pages}")


class _Path:
    """The contract a later integration must follow, as test code. No production code does this yet."""

    def __init__(self, journal):
        self.journal = journal
        self.executed = 0
        self.acknowledged = 0
        self.fail_commit_transition = False

    def write(self):
        record = self.journal.create_intent("flush", ONE)
        self.journal.mark_in_flight(record.record_id)
        self.executed += 1
        if self.fail_commit_transition:
            _inject_history_failure(self.journal)
        self.journal.mark_committed(record.record_id)
        self.acknowledged += 1
        return record


# ================================================================ scenario 9: the journal fails

def test_scenario_9_failure_to_record_the_intent_means_no_execution_and_no_acknowledgement(path):
    with NeoWriteJournal(path) as journal:
        _inject_history_failure(journal)
        write = _Path(journal)
        with pytest.raises(JournalWriteError):
            write.write()
        assert (write.executed, write.acknowledged) == (0, 0)
        assert journal.list_records() == ()


def test_scenario_9_failure_to_record_in_flight_means_no_execution(path):
    with NeoWriteJournal(path) as journal:
        record = journal.create_intent("flush", ONE)
        _inject_history_failure(journal)
        with pytest.raises(JournalWriteError):
            journal.mark_in_flight(record.record_id)
        _heal_history_failure(journal)
        assert journal.get_record(record.record_id).state is S.PENDING
        assert len(journal.get_history(record.record_id)) == 1


def test_scenario_9_failure_after_the_commit_gives_no_false_state_no_ack_and_no_retry(path):
    with NeoWriteJournal(path) as journal:
        write = _Path(journal)
        write.fail_commit_transition = True
        with pytest.raises(JournalWriteError):
            write.write()
        assert write.executed == 1, "no retry as a workaround for the journal failing"
        assert write.acknowledged == 0
        _heal_history_failure(journal)
        record = journal.list_records()[0]
        assert record.state is S.IN_FLIGHT, "still unresolved, never a false COMMITTED"
    with NeoWriteJournal(path) as reopened:
        assert reopened.recover_unfinished().recovered == (record.record_id,)
        assert reopened.get_record(record.record_id).state is S.AMBIGUOUS


@pytest.mark.parametrize("move", [
    lambda j, i: j.mark_committed(i),
    lambda j, i: j.mark_not_committed(i),
    lambda j, i: j.mark_ambiguous(i),
], ids=["committed", "not committed", "ambiguous"])
def test_a_failed_transition_changes_neither_row_nor_history(path, move):
    with NeoWriteJournal(path) as journal:
        record = journal.create_intent("op", ONE)
        record = journal.mark_in_flight(record.record_id)
        history = journal.get_history(record.record_id)
        _inject_history_failure(journal)
        with pytest.raises(JournalWriteError):
            move(journal, record.record_id)
        _heal_history_failure(journal)
        assert journal.get_record(record.record_id) == record
        assert journal.get_history(record.record_id) == history
        assert journal.check_integrity().ok
        assert move(journal, record.record_id).state is not S.IN_FLIGHT, "and it works once storage recovers"


def test_a_full_disk_on_create_is_surfaced_and_creates_nothing(path):
    with NeoWriteJournal(path) as journal:
        existing = journal.create_intent("op", ONE)
        _fill_the_disk(journal)
        with pytest.raises(JournalWriteError):
            for _ in range(2000):
                journal.create_intent("op", [("Q", {"v": "x" * 20000})])
        count = len(journal.list_records())
        assert journal.get_record(existing.record_id).state is S.PENDING
        assert journal.check_integrity().ok
    with NeoWriteJournal(path) as reopened:
        assert len(reopened.list_records()) == count


def test_a_full_disk_on_a_transition_is_surfaced_and_changes_nothing(path):
    with NeoWriteJournal(path) as journal:
        record = journal.create_intent("op", ONE)
        record = journal.mark_in_flight(record.record_id)
        _fill_the_disk(journal)
        failures = 0
        with pytest.raises(JournalWriteError):
            for _ in range(2000):
                journal.create_intent("op", [("Q", {"v": "x" * 20000})])
        for _ in range(500):
            try:
                other = journal.create_intent("op", ONE)
                journal.mark_in_flight(other.record_id)
                journal.mark_ambiguous(other.record_id, diagnostic="y" * 500)
            except JournalWriteError:
                failures += 1
                break
        assert failures == 1
        assert journal.check_integrity().ok
        assert journal.get_record(record.record_id).state is S.IN_FLIGHT


def test_a_real_kernel_write_refusal_is_surfaced_and_the_journal_survives(tmp_path):
    """RLIMIT_FSIZE in a child: the kernel really refuses the write, as with no space left."""
    path = tmp_path / "journal.db"
    with child_processes(str(tmp_path)) as children:
        result = children.run([sys.executable, CHILD, "file_size_limit", str(path), str(tmp_path / "effects")], capture_output=True, text=True,
                              timeout=120, cwd=str(tmp_path))
    assert result.returncode == -signal.SIGKILL, result.stderr[-800:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["error"] == "JournalWriteError"
    assert report["created"], "some writes fitted before the limit"
    assert report["marked"] is False
    assert report["state_seen"] == "IN_FLIGHT", "no false COMMITTED"
    with NeoWriteJournal(path) as journal:
        stored = {r.record_id: r for r in journal.list_records()}
        assert set(report["created"]) <= set(stored), "everything reported as stored is stored"
        assert stored[report["in_flight_id"]].state is S.IN_FLIGHT
        assert journal.recover_unfinished().recovered == (report["in_flight_id"],)
        assert journal.check_integrity().ok


def test_failed_serialization_touches_nothing(path):
    with NeoWriteJournal(path) as journal:
        with pytest.raises(UnsupportedParameterError) as caught:
            journal.create_intent("op", [("Q", {"good": 1, "bad": {SECRET}})])
        assert SECRET not in str(caught.value)
        assert journal._conn.execute("SELECT count(*) FROM journal_history").fetchone()[0] == 0


def test_an_error_partway_through_a_transition_undoes_the_history_entry_too(path):
    """Not a SQLite error: anything raised between the two writes of one transition."""
    with NeoWriteJournal(path) as journal:
        record = journal.mark_in_flight(journal.create_intent("op", ONE).record_id)
        history = journal.get_history(record.record_id)
        real, calls = journal._row_hash, []

        def fails_on_the_update(values):
            calls.append(1)
            if len(calls) == 2:  # 1 is reading the record, 2 is after the history entry is written
                raise RuntimeError("injected")
            return real(values)

        journal._row_hash = fails_on_the_update
        with pytest.raises(RuntimeError):
            journal.mark_committed(record.record_id)
        del journal._row_hash
        assert journal.get_record(record.record_id) == record
        assert journal.get_history(record.record_id) == history
        assert journal.check_integrity().ok


@pytest.mark.parametrize("statement, replacement", [
    ("PRAGMA journal_mode = WAL", "PRAGMA journal_mode"),
    ("PRAGMA synchronous", "SELECT 1"),
], ids=["WAL unavailable", "FULL sync not in force"])
def test_a_journal_that_cannot_be_made_durable_is_refused(path, monkeypatch, statement, replacement):
    real_connect = sqlite3.connect

    class _Limited(sqlite3.Connection):
        def execute(self, sql, *args):
            return super().execute(replacement if sql == statement else sql, *args)

    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: real_connect(*args, factory=_Limited, **kwargs))
    with pytest.raises(JournalWriteError, match="durable settings"):
        NeoWriteJournal(path)
    monkeypatch.undo()
    with NeoWriteJournal(path) as journal:
        assert journal.check_integrity().ok, "and the refusal left nothing behind that blocks a later open"


# ================================================================ lock contention

def test_a_lock_held_too_long_is_an_error_and_changes_nothing(path):
    with NeoWriteJournal(path, busy_timeout=0.1) as journal:
        record = journal.create_intent("op", ONE)
        blocker = sqlite3.connect(path, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(JournalBusyError):
                journal.mark_in_flight(record.record_id)
            with pytest.raises(JournalBusyError):
                journal.create_intent("op", ONE)
            assert isinstance(JournalBusyError("x"), JournalWriteError), "callers treat it as a failed journal write"
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        assert journal.get_record(record.record_id).state is S.PENDING
        assert len(journal.list_records()) == 1
        assert journal.mark_in_flight(record.record_id).state is S.IN_FLIGHT


def test_reading_works_while_another_connection_holds_the_write_lock(path):
    with NeoWriteJournal(path, busy_timeout=0.1) as journal:
        record = journal.create_intent("op", ONE)
        blocker = sqlite3.connect(path, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            assert journal.get_record(record.record_id).state is S.PENDING
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()


def test_a_transition_waits_for_a_short_lock_rather_than_failing(path):
    import threading
    with NeoWriteJournal(path, busy_timeout=5.0) as journal:
        record = journal.create_intent("op", ONE)
        blocker = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        blocker.execute("BEGIN IMMEDIATE")
        timer = threading.Timer(0.2, lambda: blocker.execute("ROLLBACK"))
        timer.start()
        assert journal.mark_in_flight(record.record_id).state is S.IN_FLIGHT
        timer.join()
        blocker.close()


# ================================================================ foreign files: refused and untouched

def test_a_foreign_sqlite_database_is_refused_and_left_byte_for_byte(path, tmp_path):
    """The shape of soar/playbook_memory.db: another application's SQLite file."""
    foreign = sqlite3.connect(path)
    foreign.execute("CREATE TABLE playbooks (playbook_id TEXT PRIMARY KEY)")
    foreign.execute("INSERT INTO playbooks VALUES ('p1')")
    foreign.commit()
    foreign.close()
    before = _fingerprint(path)
    with pytest.raises(JournalSchemaError):
        NeoWriteJournal(path)
    with pytest.raises(JournalSchemaError):
        NeoWriteJournal(path, read_only=True)
    assert _fingerprint(path) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["journal.db"], "no -wal, -shm or sessions directory"


def test_sqlite_never_opens_a_foreign_file(path, monkeypatch):
    """Opening another application's WAL database can change it, so the header is checked first."""
    foreign = sqlite3.connect(path)
    foreign.execute("CREATE TABLE other (x)")
    foreign.commit()
    foreign.close()

    def must_not_connect(*args, **kwargs):
        raise AssertionError("SQLite was asked to open a foreign file")

    monkeypatch.setattr(sqlite3, "connect", must_not_connect)
    with pytest.raises(JournalSchemaError):
        NeoWriteJournal(path)


@pytest.mark.parametrize("content", [b"not a database at all", b"x" * 4096, b"SQLite format 3\x00", b"\x00" * 200],
                         ids=["text", "garbage", "short header", "zeros"])
def test_a_file_that_is_not_sqlite_is_refused_and_left_alone(path, content):
    path.write_bytes(content)
    with pytest.raises(JournalCorruptionError):
        NeoWriteJournal(path)
    assert path.read_bytes() == content


# ================================================================ damage

def _closed_journal(path, count=3):
    with NeoWriteJournal(path) as journal:
        ids = []
        for index in range(count):
            record = journal.create_intent("op", [("CREATE (n:X {v:$v})", {"v": f"marker-value-{index}"})])
            journal.mark_in_flight(record.record_id)
            journal.mark_committed(record.record_id)
            ids.append(record.record_id)
    return ids


def _raw(path):
    conn = sqlite3.connect(path, isolation_level=None)
    for name in ("journal_history_no_update", "journal_history_no_delete", "journal_records_no_delete", "journal_records_intent_fixed"):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    return conn


def _restore_triggers(conn):
    from neo4j_write_journal import _SCHEMA
    for statement in _SCHEMA:
        if statement.startswith("CREATE TRIGGER"):
            conn.execute(statement)
    conn.close()


@pytest.mark.parametrize("damage, column", [
    ("UPDATE journal_records SET state = 'PENDING' WHERE record_id = ?", "state changed to look replayable"),
    ("UPDATE journal_records SET attempt_count = 9 WHERE record_id = ?", "attempts changed"),
    ("UPDATE journal_records SET intent_json = replace(intent_json, 'marker', 'MARKER') WHERE record_id = ?", "intent changed"),
    ("UPDATE journal_records SET last_outcome = 'NOT_COMMITTED' WHERE record_id = ?", "outcome changed"),
    ("UPDATE journal_records SET row_sha256 = 'x' WHERE record_id = ?", "checksum changed"),
], ids=lambda value: value if " " in value and "UPDATE" not in value else "")
def test_a_changed_row_is_detected_and_refused(path, damage, column):
    ids = _closed_journal(path)
    conn = _raw(path)
    conn.execute(damage, (ids[1],))
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        report = journal.check_integrity()
        assert not report.ok and report.corrupt_record_ids == (ids[1],)
        with pytest.raises(JournalCorruptionError):
            journal.get_record(ids[1])
        with pytest.raises(JournalCorruptionError):
            journal.mark_in_flight(ids[1])
        with pytest.raises(JournalCorruptionError):
            journal.list_records()
        assert journal.get_record(ids[0]).state is S.COMMITTED, "the others remain readable"
        assert journal.recover_unfinished().corrupt == (ids[1],), "and recovery reports it"


def test_an_intent_changed_with_a_matching_row_checksum_is_still_detected(path):
    """The intent has a checksum of its own, so fixing up the row's isn't enough."""
    ids = _closed_journal(path)
    conn = _raw(path)
    row = conn.execute(f"SELECT {', '.join(_RECORD_COLUMNS)} FROM journal_records WHERE record_id = ?", (ids[0],)).fetchone()
    values = dict(zip(_RECORD_COLUMNS, row))
    values["intent_json"] = values["intent_json"].replace("marker", "MARKER")
    conn.execute("UPDATE journal_records SET intent_json = ?, row_sha256 = ? WHERE record_id = ?",
                 (values["intent_json"], NeoWriteJournal._row_hash(values), ids[0]))
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        with pytest.raises(JournalCorruptionError):
            journal.get_record(ids[0])
        assert journal.check_integrity().corrupt_record_ids == (ids[0],)


def test_a_damaged_in_flight_record_is_not_converted_or_trusted(path):
    with NeoWriteJournal(path) as first:
        good = first.mark_in_flight(first.create_intent("op", ONE).record_id)
        bad = first.mark_in_flight(first.create_intent("op", ONE).record_id)
    conn = _raw(path)
    conn.execute("UPDATE journal_records SET attempt_count = 9 WHERE record_id = ?", (bad.record_id,))
    _restore_triggers(conn)
    with NeoWriteJournal(path) as second:
        report = second.recover_unfinished()
        assert report.recovered == (good.record_id,) and report.corrupt == (bad.record_id,)
        with pytest.raises(JournalCorruptionError):
            second.mark_ambiguous(bad.record_id)


@pytest.mark.parametrize("damage", [
    "UPDATE journal_history SET to_state = 'COMMITTED' WHERE record_id = ? AND transition_no = 2",
    "UPDATE journal_history SET actor = 'someone else' WHERE record_id = ? AND transition_no = 1",
    "DELETE FROM journal_history WHERE record_id = ? AND transition_no = 2",
    "DELETE FROM journal_history WHERE record_id = ? AND transition_no = 3",
], ids=["state rewritten", "actor rewritten", "middle entry removed", "last entry removed"])
def test_changed_history_is_detected(path, damage):
    ids = _closed_journal(path)
    conn = _raw(path)
    conn.execute(damage, (ids[0],))
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        report = journal.check_integrity()
        assert not report.ok and report.corrupt_record_ids == (ids[0],)


def test_a_deleted_record_is_detected(path):
    ids = _closed_journal(path)
    conn = _raw(path)
    conn.execute("DELETE FROM journal_records WHERE record_id = ?", (ids[2],))
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        report = journal.check_integrity()
        assert not report.ok
        assert any("3 records were created but 2" in p for p in report.problems)
        assert any("history exists for 1" in p for p in report.problems)


def test_bytes_flipped_inside_a_value_are_detected_and_the_file_is_kept(path):
    ids = _closed_journal(path, count=1)
    data = bytearray(path.read_bytes())
    position = data.find(b"marker-value-0")
    assert position > 0
    data[position] ^= 0x01
    path.write_bytes(bytes(data))
    with NeoWriteJournal(path) as journal:
        report = journal.check_integrity()
        assert not report.ok and report.corrupt_record_ids == tuple(ids)
    assert len(path.read_bytes()) == len(data), "no recreation, no truncation"


def test_a_truncated_file_is_refused_and_never_recreated(path):
    _closed_journal(path, count=50)
    data = path.read_bytes()
    truncated = data[:len(data) // 2]
    path.write_bytes(truncated)
    with pytest.raises((JournalCorruptionError, JournalWriteError)):
        with NeoWriteJournal(path) as journal:
            assert not journal.check_integrity().ok
            raise JournalCorruptionError("reported through the integrity check")
    assert path.read_bytes() == truncated


def test_overwritten_pages_are_refused_or_reported_and_the_file_is_kept(path):
    _closed_journal(path, count=50)
    data = bytearray(path.read_bytes())
    data[4096:8192] = b"\xff" * 4096
    path.write_bytes(bytes(data))
    with pytest.raises(JournalCorruptionError):
        with NeoWriteJournal(path) as journal:
            report = journal.check_integrity()
            assert not report.ok
            raise JournalCorruptionError("reported through the integrity check")
    assert path.stat().st_size == len(data)


@pytest.mark.parametrize("version, words", [(0, "version 0"), (SCHEMA_VERSION + 1, f"version {SCHEMA_VERSION + 1}"), (-1, "version -1")],
                         ids=["missing", "newer", "invalid"])
def test_an_unusable_schema_version_is_refused_without_changing_anything(path, version, words):
    _closed_journal(path)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute(f"PRAGMA user_version = {version}")
    conn.close()
    before = _fingerprint(path)
    with pytest.raises(JournalSchemaError, match=words):
        NeoWriteJournal(path)
    assert _fingerprint(path) == before


def test_missing_schema_objects_are_refused(path):
    _closed_journal(path)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("DROP TRIGGER journal_history_no_delete")
    conn.close()
    with pytest.raises(JournalCorruptionError, match="schema objects"):
        NeoWriteJournal(path)


def test_refusing_a_damaged_journal_preserves_unresolved_records_for_a_repair_attempt(path):
    """Nothing about an open failure removes evidence: restoring the bytes restores the record."""
    with NeoWriteJournal(path) as journal:
        record = journal.create_intent("op", ONE)
        journal.mark_in_flight(record.record_id)
    good = path.read_bytes()
    path.write_bytes(b"garbage" * 100)
    with pytest.raises(JournalCorruptionError):
        NeoWriteJournal(path)
    path.write_bytes(good)
    with NeoWriteJournal(path) as journal:
        assert journal.recover_unfinished().recovered == (record.record_id,)


def test_no_fault_above_ever_makes_an_ambiguous_record_replayable(path):
    with NeoWriteJournal(path) as journal:
        record = journal.create_intent("op", ONE)
        journal.mark_in_flight(record.record_id)
        journal.mark_ambiguous(record.record_id)
        _inject_history_failure(journal)
        with pytest.raises(InvalidTransitionError):
            journal.authorize_retry(record.record_id, authorized_by="op", reason="r")
        _heal_history_failure(journal)
        assert journal.get_record(record.record_id).state is S.AMBIGUOUS
        assert not journal.get_record(record.record_id).retry_eligible

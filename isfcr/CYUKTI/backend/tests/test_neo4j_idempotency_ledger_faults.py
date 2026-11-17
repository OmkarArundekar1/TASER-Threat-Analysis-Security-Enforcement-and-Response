"""Fault injection: failing storage, lock contention, damaged and foreign files. Whatever fails, no
false state may appear, an existing identity may not change, and evidence stays exactly as it was."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import neo4j_idempotency_ledger as module
from neo4j_idempotency_ledger import (
    CheckStatus,
    FingerprintVersionError,
    IdempotencyConflictError,
    IdempotencyLedger,
    InvalidTransitionError,
    LedgerBusyError,
    LedgerCorruptionError,
    LedgerSchemaError,
    LedgerState as S,
    LedgerWriteError,
    SCHEMA_VERSION,
    _REQUEST_COLUMNS,
    _SCHEMA,
    fingerprint_payload,
)
from neo4j_write_journal import JournalSchemaError, NeoWriteJournal

# Calls install_child_guards before importing backend code, as the crash tests check.
CHILD = str(Path(__file__).resolve().parent / "_neo4j_ledger_child.py")
KEY = "campaign-42/flush/7"
ONE = [("CREATE (n:X {v:$v})", {"v": 1})]
OTHER = [("Q", None)]
FP = fingerprint_payload("op", ONE).value
SECRET = "s3cret-parameter-value"


@pytest.fixture
def path(tmp_path):
    return tmp_path / "ledger.db"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fail_inserts(ledger, table="ledger_history"):
    """Any write of that audit table now fails, inside the caller's transaction."""
    ledger._conn.execute(f"CREATE TEMP TRIGGER fail_{table} BEFORE INSERT ON main.{table} BEGIN SELECT RAISE(ABORT, 'injected'); END")


def _heal(ledger, table="ledger_history"):
    ledger._conn.execute(f"DROP TRIGGER temp.fail_{table}")


def _fill_the_disk(ledger):
    pages = ledger._conn.execute("PRAGMA page_count").fetchone()[0]
    ledger._conn.execute(f"PRAGMA max_page_count = {pages}")


# ================================================================ 17 durability settings

@pytest.mark.parametrize("statement, replacement", [
    ("PRAGMA journal_mode = WAL", "PRAGMA journal_mode"),
    ("PRAGMA synchronous", "SELECT 1"),
], ids=["WAL unavailable", "FULL sync not in force"])
def test_a_ledger_that_cannot_be_made_durable_is_refused(path, monkeypatch, statement, replacement):
    real_connect = sqlite3.connect

    class _Limited(sqlite3.Connection):
        def execute(self, sql, *args):
            return super().execute(replacement if sql == statement else sql, *args)

    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: real_connect(*args, factory=_Limited, **kwargs))
    with pytest.raises(LedgerWriteError, match="durable settings"):
        IdempotencyLedger(path)
    monkeypatch.undo()
    with IdempotencyLedger(path) as ledger:
        assert ledger.check_integrity().ok, "and the refusal left nothing behind that blocks a later open"


def test_a_failed_durability_check_leaves_an_existing_ledger_untouched(path, monkeypatch):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", ONE)
    before = _sha(path)
    real_connect = sqlite3.connect

    class _Limited(sqlite3.Connection):
        def execute(self, sql, *args):
            return super().execute("SELECT 1" if sql == "PRAGMA synchronous" else sql, *args)

    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: real_connect(*args, factory=_Limited, **kwargs))
    with pytest.raises(LedgerWriteError):
        IdempotencyLedger(path)
    monkeypatch.undo()
    assert _sha(path) == before
    assert not [p for p in Path(str(path) + ".sessions").glob("*.lock")], "no session was left behind"


# ================================================================ 18 storage errors and rollback

def test_failure_to_register_leaves_no_request(path):
    with IdempotencyLedger(path) as ledger:
        _fail_inserts(ledger)
        with pytest.raises(LedgerWriteError):
            ledger.register(KEY, "op", ONE)
        _heal(ledger)
        assert ledger.list_requests() == ()
        assert ledger.register(KEY, "op", ONE).status is CheckStatus.NEW, "the key is still free"


@pytest.mark.parametrize("move", [
    lambda l: l.record_completed(KEY, attempt=1),
    lambda l: l.record_not_completed(KEY, attempt=1),
    lambda l: l.record_uncertain(KEY, attempt=1),
], ids=["completed", "not completed", "uncertain"])
def test_a_failed_transition_changes_neither_row_nor_history(path, move):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", ONE)
        request = ledger.begin_execution(KEY, FP)
        history = ledger.get_history(KEY)
        _fail_inserts(ledger)
        with pytest.raises(LedgerWriteError):
            move(ledger)
        _heal(ledger)
        assert ledger.get_request(KEY) == request and ledger.get_history(KEY) == history
        assert ledger.check_integrity().ok
        assert move(ledger).state is not S.IN_PROGRESS, "and it works once storage recovers"


def test_failure_to_claim_grants_nothing(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", ONE)
        _fail_inserts(ledger)
        with pytest.raises(LedgerWriteError):
            ledger.begin_execution(KEY, FP)
        _heal(ledger)
        assert ledger.get_request(KEY).state is S.REGISTERED and ledger.get_request(KEY).attempt_count == 0


def test_failure_to_authorize_grants_nothing(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", ONE)
        ledger.begin_execution(KEY, FP)
        ledger.record_not_completed(KEY, attempt=1)
        _fail_inserts(ledger)
        with pytest.raises(LedgerWriteError):
            ledger.authorize_retry(KEY, authorized_by="op", reason="r")
        _heal(ledger)
        assert ledger.get_request(KEY).state is S.NOT_COMPLETED


def test_failure_to_audit_a_conflict_still_refuses_and_overwrites_nothing(path):
    with IdempotencyLedger(path) as ledger:
        original = ledger.register(KEY, "op", ONE).request
        _fail_inserts(ledger, "ledger_conflicts")
        with pytest.raises(LedgerWriteError):
            ledger.register(KEY, "op", OTHER)
        _heal(ledger, "ledger_conflicts")
        assert ledger.get_request(KEY) == original and ledger.get_conflicts(KEY) == ()


def test_an_error_partway_through_a_transition_undoes_the_history_entry_too(path):
    """Not a SQLite error: anything raised between the two writes of one transition."""
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", ONE)
        request = ledger.begin_execution(KEY, FP)
        history = ledger.get_history(KEY)
        real, calls = ledger._row_hash, []

        def fails_on_the_update(values):
            calls.append(1)
            if len(calls) == 2:  # 1 is reading the request, 2 is after the history entry is written
                raise RuntimeError("injected")
            return real(values)

        ledger._row_hash = fails_on_the_update
        with pytest.raises(RuntimeError):
            ledger.record_completed(KEY, attempt=1)
        del ledger._row_hash
        assert ledger.get_request(KEY) == request and ledger.get_history(KEY) == history
        assert ledger.check_integrity().ok


def test_an_error_partway_through_a_conflict_undoes_its_audit_row_too(path):
    with IdempotencyLedger(path) as ledger:
        original = ledger.register(KEY, "op", ONE).request
        real, calls = ledger._row_hash, []

        def fails_on_the_update(values):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("injected")
            return real(values)

        ledger._row_hash = fails_on_the_update
        with pytest.raises(RuntimeError):
            ledger.register(KEY, "op", OTHER)
        del ledger._row_hash
        assert ledger.get_request(KEY) == original and ledger.get_conflicts(KEY) == ()
        assert ledger.check_integrity().ok


def test_a_full_disk_on_register_is_surfaced_and_creates_nothing(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", ONE)
        _fill_the_disk(ledger)
        with pytest.raises(LedgerWriteError):
            for number in range(5000):
                ledger.register(f"fill/{number}", "op", ONE)
        count = len(ledger.list_requests())
        assert ledger.get_request(KEY).state is S.REGISTERED and ledger.check_integrity().ok
    with IdempotencyLedger(path) as reopened:
        assert len(reopened.list_requests()) == count


def test_a_real_kernel_write_refusal_is_surfaced_and_the_ledger_survives(tmp_path):
    """RLIMIT_FSIZE in a child: the kernel really refuses the write, as with no space left."""
    path = tmp_path / "ledger.db"
    result = subprocess.run([sys.executable, CHILD, "file_size_limit", str(path), str(tmp_path / "effects")], capture_output=True, text=True,
                            timeout=120, cwd=str(tmp_path), env={**os.environ, "TMPDIR": str(tmp_path)})
    assert result.returncode == -signal.SIGKILL, result.stderr[-800:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["error"] == "LedgerWriteError"
    assert report["created"], "some writes fitted before the limit"
    assert report["marked"] is False and report["state_seen"] == "IN_PROGRESS", "no false COMPLETED"
    with IdempotencyLedger(path) as ledger:
        stored = {r.idempotency_key for r in ledger.list_requests()}
        assert set(report["created"]) <= stored, "everything reported as stored is stored"
        assert ledger.recover_unfinished().recovered == (KEY,)
        assert ledger.check_integrity().ok


def test_a_bad_payload_touches_nothing(path):
    with IdempotencyLedger(path) as ledger:
        with pytest.raises(module.UnsupportedPayloadError) as caught:
            ledger.register(KEY, "op", [("Q", {"good": 1, "bad": {SECRET}})])
        assert SECRET not in str(caught.value)
        assert ledger._conn.execute("SELECT count(*) FROM ledger_history").fetchone()[0] == 0


# ================================================================ lock contention

def test_a_lock_held_too_long_is_an_error_and_changes_nothing(path):
    with IdempotencyLedger(path, busy_timeout=0.1) as ledger:
        ledger.register(KEY, "op", ONE)
        blocker = sqlite3.connect(path, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            for call in (lambda: ledger.begin_execution(KEY, FP), lambda: ledger.register("new", "op", ONE)):
                with pytest.raises(LedgerBusyError):
                    call()
            assert isinstance(LedgerBusyError("x"), LedgerWriteError)
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        assert ledger.get_request(KEY).state is S.REGISTERED and len(ledger.list_requests()) == 1
        assert ledger.begin_execution(KEY, FP).state is S.IN_PROGRESS


def test_reading_works_while_another_connection_holds_the_write_lock(path):
    with IdempotencyLedger(path, busy_timeout=0.1) as ledger:
        ledger.register(KEY, "op", ONE)
        blocker = sqlite3.connect(path, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            assert ledger.get_request(KEY).state is S.REGISTERED
            assert ledger.check(KEY, "op", ONE).status is CheckStatus.DUPLICATE
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()


def test_a_transition_waits_for_a_short_lock_rather_than_failing(path):
    with IdempotencyLedger(path, busy_timeout=5.0) as ledger:
        ledger.register(KEY, "op", ONE)
        blocker = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        blocker.execute("BEGIN IMMEDIATE")
        timer = threading.Timer(0.2, lambda: blocker.execute("ROLLBACK"))
        timer.start()
        assert ledger.begin_execution(KEY, FP).state is S.IN_PROGRESS
        timer.join()
        blocker.close()


# ================================================================ 16 foreign files

def test_a_foreign_sqlite_database_is_refused_and_left_byte_for_byte(path, tmp_path):
    """The shape of soar/playbook_memory.db: another application's SQLite file."""
    foreign = sqlite3.connect(path)
    foreign.execute("CREATE TABLE playbooks (playbook_id TEXT PRIMARY KEY)")
    foreign.execute("INSERT INTO playbooks VALUES ('p1')")
    foreign.commit()
    foreign.close()
    before = _sha(path)
    for read_only in (False, True):
        with pytest.raises(LedgerSchemaError):
            IdempotencyLedger(path, read_only=read_only)
    assert _sha(path) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ledger.db"], "no -wal, -shm or sessions directory"


def test_sqlite_never_opens_a_foreign_file(path, monkeypatch):
    foreign = sqlite3.connect(path)
    foreign.execute("CREATE TABLE other (x)")
    foreign.commit()
    foreign.close()

    def must_not_connect(*args, **kwargs):
        raise AssertionError("SQLite was asked to open a foreign file")

    monkeypatch.setattr(sqlite3, "connect", must_not_connect)
    with pytest.raises(LedgerSchemaError):
        IdempotencyLedger(path)


def test_a_journal_file_is_refused_as_a_ledger_and_the_reverse(tmp_path):
    journal_path, ledger_path = tmp_path / "journal.db", tmp_path / "ledger.db"
    with NeoWriteJournal(journal_path) as journal:
        journal.create_intent("op", ONE)
    with IdempotencyLedger(ledger_path) as ledger:
        ledger.register(KEY, "op", ONE)
    journal_before, ledger_before = _sha(journal_path), _sha(ledger_path)
    with pytest.raises(LedgerSchemaError):
        IdempotencyLedger(journal_path)
    with pytest.raises(JournalSchemaError):
        NeoWriteJournal(ledger_path)
    assert (_sha(journal_path), _sha(ledger_path)) == (journal_before, ledger_before)


@pytest.mark.parametrize("content", [b"not a database at all", b"x" * 4096, b"SQLite format 3\x00", b"\x00" * 200],
                         ids=["text", "garbage", "short header", "zeros"])
def test_a_file_that_is_not_sqlite_is_refused_and_left_alone(path, content):
    path.write_bytes(content)
    with pytest.raises(LedgerCorruptionError):
        IdempotencyLedger(path)
    assert path.read_bytes() == content


@pytest.mark.parametrize("version, words", [(0, "version 0"), (SCHEMA_VERSION + 1, f"version {SCHEMA_VERSION + 1}"), (-1, "version -1")],
                         ids=["missing", "newer", "invalid"])
def test_an_unusable_schema_version_is_refused_without_changing_anything(path, version, words):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", ONE)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute(f"PRAGMA user_version = {version}")
    conn.close()
    before = _sha(path)
    with pytest.raises(LedgerSchemaError, match=words):
        IdempotencyLedger(path)
    assert _sha(path) == before


def test_missing_schema_objects_are_refused(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", ONE)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("DROP TRIGGER ledger_history_no_delete")
    conn.close()
    with pytest.raises(LedgerCorruptionError, match="schema objects"):
        IdempotencyLedger(path)


# ================================================================ 15 integrity and checksums

def _closed_ledger(path, count=3):
    keys = []
    with IdempotencyLedger(path) as ledger:
        for number in range(count):
            key = f"key/{number}"
            ledger.register(key, "op", [("CREATE (n:X {v:$v})", {"v": number})])
            fingerprint = ledger.get_request(key).fingerprint
            ledger.begin_execution(key, fingerprint)
            ledger.record_completed(key, attempt=1)
            with pytest.raises(IdempotencyConflictError):
                ledger.register(key, "op", OTHER)
            keys.append(key)
    return keys


def _raw(path):
    conn = sqlite3.connect(path, isolation_level=None)
    for row in conn.execute("SELECT name FROM sqlite_schema WHERE type = 'trigger'").fetchall():
        conn.execute(f"DROP TRIGGER {row[0]}")
    return conn


def _restore_triggers(conn):
    for statement in _SCHEMA:
        if statement.startswith("CREATE TRIGGER"):
            conn.execute(statement)
    conn.close()


@pytest.mark.parametrize("damage", [
    "UPDATE ledger_requests SET state = 'REGISTERED' WHERE idempotency_key = ?",
    "UPDATE ledger_requests SET attempt_count = 9 WHERE idempotency_key = ?",
    "UPDATE ledger_requests SET fingerprint = 'x' WHERE idempotency_key = ?",
    "UPDATE ledger_requests SET last_outcome = 'NOT_COMPLETED' WHERE idempotency_key = ?",
    "UPDATE ledger_requests SET conflict_count = 0 WHERE idempotency_key = ?",
    "UPDATE ledger_requests SET row_sha256 = 'x' WHERE idempotency_key = ?",
], ids=["state made claimable", "attempts", "fingerprint", "outcome", "conflict count", "checksum"])
def test_a_changed_row_is_detected_and_refused(path, damage):
    keys = _closed_ledger(path)
    conn = _raw(path)
    conn.execute(damage, (keys[1],))
    _restore_triggers(conn)
    with IdempotencyLedger(path) as ledger:
        report = ledger.check_integrity()
        assert not report.ok and report.corrupt_keys == (module._key_ref(keys[1]),)
        for call in (lambda: ledger.get_request(keys[1]), lambda: ledger.begin_execution(keys[1], FP), ledger.list_requests,
                     lambda: ledger.register(keys[1], "op", OTHER), lambda: ledger.check(keys[1], "op", OTHER)):
            with pytest.raises(LedgerCorruptionError):
                call()
        assert ledger.get_request(keys[0]).state is S.COMPLETED, "the others remain readable"
        assert ledger.recover_unfinished().corrupt == (module._key_ref(keys[1]),), "and recovery reports it"


def test_a_damaged_in_progress_request_is_not_converted_or_trusted(path):
    with IdempotencyLedger(path) as first:
        for key in ("good", "bad"):
            first.register(key, "op", ONE)
            first.begin_execution(key, FP)
    conn = _raw(path)
    conn.execute("UPDATE ledger_requests SET attempt_count = 9 WHERE idempotency_key = 'bad'")
    _restore_triggers(conn)
    with IdempotencyLedger(path) as second:
        report = second.recover_unfinished()
        assert report.recovered == ("good",) and report.corrupt == (module._key_ref("bad"),)
        with pytest.raises(LedgerCorruptionError):
            second.record_uncertain("bad", attempt=9)


@pytest.mark.parametrize("damage", [
    "UPDATE ledger_history SET to_state = 'COMPLETED' WHERE idempotency_key = ? AND transition_no = 2",
    "UPDATE ledger_history SET actor = 'someone else' WHERE idempotency_key = ? AND transition_no = 1",
    "DELETE FROM ledger_history WHERE idempotency_key = ? AND transition_no = 2",
    "DELETE FROM ledger_history WHERE idempotency_key = ? AND transition_no = 3",
    "UPDATE ledger_conflicts SET presented_fingerprint = 'x' WHERE idempotency_key = ?",
    "DELETE FROM ledger_conflicts WHERE idempotency_key = ?",
], ids=["history state", "history actor", "middle entry removed", "last entry removed", "conflict rewritten", "conflict removed"])
def test_changed_audit_records_are_detected(path, damage):
    keys = _closed_ledger(path)
    conn = _raw(path)
    conn.execute(damage, (keys[0],))
    _restore_triggers(conn)
    with IdempotencyLedger(path) as ledger:
        report = ledger.check_integrity()
        assert not report.ok and report.corrupt_keys == (module._key_ref(keys[0]),)


def test_a_deleted_request_is_detected(path):
    keys = _closed_ledger(path)
    conn = _raw(path)
    conn.execute("DELETE FROM ledger_requests WHERE idempotency_key = ?", (keys[2],))
    _restore_triggers(conn)
    with IdempotencyLedger(path) as ledger:
        report = ledger.check_integrity()
        assert not report.ok
        assert any("3 requests were registered but 2" in p for p in report.problems)
        assert any("audit entries exist for 1" in p for p in report.problems)


def test_bytes_flipped_inside_a_value_are_detected_and_the_file_is_kept(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register("marker-key-value", "operation-marker", ONE)
    data = bytearray(path.read_bytes())
    position = data.find(b"operation-marker")
    assert position > 0
    data[position] ^= 0x01
    path.write_bytes(bytes(data))
    with IdempotencyLedger(path) as ledger:
        report = ledger.check_integrity()
        assert not report.ok and len(report.corrupt_keys) == 1
    assert len(path.read_bytes()) == len(data), "no recreation, no truncation"


def test_a_truncated_file_is_refused_on_open_and_never_recreated(path):
    _closed_ledger(path, count=60)
    truncated = path.read_bytes()
    truncated = truncated[:len(truncated) // 2]
    path.write_bytes(truncated)
    with pytest.raises(LedgerCorruptionError):
        IdempotencyLedger(path)
    assert path.read_bytes() == truncated


def test_overwritten_pages_are_refused_on_open_and_the_file_is_kept(path):
    """The schema page is intact, so only the quick check on open can see this."""
    _closed_ledger(path, count=60)
    data = bytearray(path.read_bytes())
    data[4096:] = b"\xff" * (len(data) - 4096)
    path.write_bytes(bytes(data))
    before = _sha(path)
    with pytest.raises(LedgerCorruptionError, match="quick check|damaged"):
        IdempotencyLedger(path)
    assert _sha(path) == before


def test_a_damaged_schema_page_raises_a_real_sqlite_corruption_error_and_is_reported_as_corruption(path):
    """Header intact, so the file check passes and SQLite itself reports the damage."""
    _closed_ledger(path, count=3)
    data = bytearray(path.read_bytes())
    data[100:4096] = b"\xff" * (4096 - 100)
    path.write_bytes(bytes(data))
    before = _sha(path)
    with pytest.raises(LedgerCorruptionError, match="SQLite reports"):
        IdempotencyLedger(path)
    assert _sha(path) == before


def test_invalid_text_in_a_value_is_reported_as_corruption_not_a_storage_failure(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register("marker-key-value", "operation-marker", ONE)
    data = bytearray(path.read_bytes())
    position = data.find(b"operation-marker")
    assert position > 0
    data[position] = 0xFF
    path.write_bytes(bytes(data))
    with IdempotencyLedger(path) as ledger:
        with pytest.raises(LedgerCorruptionError, match="value in the ledger file is damaged"):
            ledger.get_request("marker-key-value")
        assert not ledger.check_integrity().ok


def test_a_damaged_index_is_found_by_the_full_integrity_check(path):
    """The table's rows are untouched, so their checksums pass, and quick check doesn't compare
    an index with its table. Only PRAGMA integrity_check sees it."""
    keys = _closed_ledger(path, count=3)
    conn = sqlite3.connect(path)
    root = conn.execute("SELECT rootpage FROM sqlite_schema WHERE name = 'ledger_requests_state'").fetchone()[0]
    size = conn.execute("PRAGMA page_size").fetchone()[0]
    conn.close()
    data = bytearray(path.read_bytes())
    start, end = (root - 1) * size, root * size
    page = bytes(data[start:end])
    assert b"COMPLETED" in page
    data[start:end] = page.replace(b"COMPLETED", b"COMPLETEX")
    path.write_bytes(bytes(data))
    with IdempotencyLedger(path) as ledger:
        report = ledger.check_integrity()
        assert not report.ok and report.corrupt_keys == ()
        assert any("SQLite integrity check" in p for p in report.problems)
        assert ledger.get_request(keys[0]).state is S.COMPLETED


def test_an_application_id_changed_in_the_wal_is_refused(path):
    """The header check reads the main file, so it can't see this: only the check after connecting can."""
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", ONE)
    other = sqlite3.connect(path, isolation_level=None)
    try:
        other.execute("PRAGMA application_id = 5")
        with pytest.raises(LedgerSchemaError, match="unexpected application id"):
            IdempotencyLedger(path)
    finally:
        other.execute(f"PRAGMA application_id = {module._APPLICATION_ID}")
        other.close()


def test_the_directory_is_synced_when_a_ledger_is_created_and_only_then(path, monkeypatch):
    """Without it, a new file's name might not survive a power loss. The sync itself can't be
    proved here, only that it's asked for."""
    synced = []
    monkeypatch.setattr(IdempotencyLedger, "_sync_directory", staticmethod(lambda directory: synced.append(directory)))
    IdempotencyLedger(path).close()
    assert synced == [str(path.parent)]
    IdempotencyLedger(path).close()
    assert synced == [str(path.parent)]


def test_refusing_a_damaged_ledger_preserves_it_for_a_repair_attempt(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", ONE)
        ledger.begin_execution(KEY, FP)
    good = path.read_bytes()
    path.write_bytes(b"garbage" * 100)
    with pytest.raises(LedgerCorruptionError):
        IdempotencyLedger(path)
    path.write_bytes(good)
    with IdempotencyLedger(path) as ledger:
        assert ledger.recover_unfinished().recovered == (KEY,)


def _forge_version(path, key, version):
    """A consistent row from a future format: only the version differs, with a matching checksum."""
    conn = _raw(path)
    row = conn.execute(f"SELECT {', '.join(_REQUEST_COLUMNS)} FROM ledger_requests WHERE idempotency_key = ?", (key,)).fetchone()
    values = dict(zip(_REQUEST_COLUMNS, row))
    values["fingerprint_version"] = version
    conn.execute("UPDATE ledger_requests SET fingerprint_version = ?, row_sha256 = ? WHERE idempotency_key = ?",
                 (version, IdempotencyLedger._row_hash(values), key))
    _restore_triggers(conn)


def test_a_request_under_another_fingerprint_version_is_never_treated_as_new(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", ONE)
    _forge_version(path, KEY, 2)
    with IdempotencyLedger(path) as ledger:
        for call in (lambda: ledger.register(KEY, "op", ONE), lambda: ledger.check(KEY, "op", ONE), lambda: ledger.begin_execution(KEY, FP)):
            with pytest.raises(FingerprintVersionError):
                call()
        assert ledger.get_request(KEY).state is S.REGISTERED and len(ledger.list_requests()) == 1


# ================================================================ 20-21 no replay, whatever fails

def test_no_fault_makes_an_uncertain_request_runnable(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", ONE)
        ledger.begin_execution(KEY, FP)
        ledger.record_uncertain(KEY, attempt=1)
        _fail_inserts(ledger)
        for call in (lambda: ledger.begin_execution(KEY, FP), lambda: ledger.authorize_retry(KEY, authorized_by="op", reason="r")):
            with pytest.raises(InvalidTransitionError):
                call()
        _heal(ledger)
        assert ledger.get_request(KEY).state is S.UNCERTAIN and not ledger.get_request(KEY).claimable


def test_a_damaged_row_can_never_be_claimed_even_if_it_says_it_can(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", ONE)
        ledger.begin_execution(KEY, FP)
        ledger.record_uncertain(KEY, attempt=1)
    conn = _raw(path)
    conn.execute("UPDATE ledger_requests SET state = 'REGISTERED' WHERE idempotency_key = ?", (KEY,))
    _restore_triggers(conn)
    with IdempotencyLedger(path) as ledger:
        with pytest.raises(LedgerCorruptionError):
            ledger.begin_execution(KEY, FP)

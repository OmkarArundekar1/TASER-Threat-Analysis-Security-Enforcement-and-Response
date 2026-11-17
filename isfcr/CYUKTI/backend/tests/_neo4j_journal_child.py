"""Runs journal steps in a real process that really dies. Started by the crash and fault tests.

usage: _neo4j_journal_child.py ACTION JOURNAL_PATH EFFECTS_PATH [ARG]

EFFECTS_PATH stands in for Neo4j: a step that would reach the database appends a line, so
a test can tell what "happened" there and that recovery never adds to it. Most actions end in
SIGKILL, which runs no cleanup. One JSON line goes to stdout before the process dies.
"""
import datetime as dt
import json
import os
import signal
import sqlite3
import sys
import threading
import time

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[0] = BACKEND

from tests._production_safety import install_child_guards  # noqa: E402

install_child_guards()

from neo4j_write_journal import (  # noqa: E402
    JournalClosedError,
    JournalError,
    JournalState,
    NeoWriteJournal,
    Resolution,
    _identical,
)

ORIGINAL_STATEMENTS = (
    ("UNWIND $records AS r MATCH (e:AttackEvent {event_id:r.event_id}) SET e.tps = r.total_tps", {
        "records": [{"event_id": "E1", "total_tps": 0.1, "occurrences": 2 ** 53 + 1, "payload": None,
                     "last_seen": "2026-01-01T00:10:00+00:00"}],
        "event_time": dt.datetime(2026, 1, 1, 0, 10, 0, 123456, tzinfo=dt.timezone.utc),
    }),
    ("MATCH (c:Campaign {campaign_id:$c}) SET c.last_seen = $t", {"c": "C1", "t": None}),
)


def emit(**values):
    print(json.dumps(values), flush=True)


def die():
    os.kill(os.getpid(), signal.SIGKILL)
    time.sleep(60)


def effect(path, what):
    """Reaches "the database": durable, and separate from the journal, as Neo4j would be."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(what + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def kill_at_commit(conn, nth=1):
    """Dies as the nth COMMIT is about to run, so that transaction never completes."""
    seen = [0]

    def trace(statement):
        if statement.strip().upper().startswith("COMMIT"):
            seen[0] += 1
            if seen[0] == nth:
                die()
    conn.set_trace_callback(trace)


def create(journal):
    return journal.create_intent("flush_duplicates", ORIGINAL_STATEMENTS)


def main():
    action, path, effects = sys.argv[1:4]
    arg = sys.argv[4] if len(sys.argv) > 4 else None

    if action == "dies_during_initialization":
        real_connect = sqlite3.connect

        def connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            kill_at_commit(conn)
            return conn
        sqlite3.connect = connect
        NeoWriteJournal(path)
        emit(unexpected="survived")
        return

    if action == "peek":
        with NeoWriteJournal(path, read_only=True) as journal:
            emit(states={r.record_id: r.state.value for r in journal.list_records()}, integrity_ok=journal.check_integrity().ok)
        return

    if action == "inspect":
        with NeoWriteJournal(path) as journal:
            report = journal.recover_unfinished()
            records = {}
            for record in journal.list_records():
                records[record.record_id] = {
                    "state": record.state.value, "attempts": record.attempt_count, "retry_eligible": record.retry_eligible,
                    "last_outcome": record.last_outcome, "error_type": record.error_type,
                    "intent_intact": _identical(record.statements, ORIGINAL_STATEMENTS),
                    "recovery": record.recovery_metadata,
                    "history": [(e.from_state.value if e.from_state else None, e.to_state.value, e.cause) for e in journal.get_history(record.record_id)],
                }
            integrity = journal.check_integrity()
            emit(recovered=report.recovered, still_running=report.still_running, pending=report.pending,
                 not_committed=report.not_committed, ambiguous=report.ambiguous, corrupt=report.corrupt, records=records,
                 integrity_ok=integrity.ok, problems=integrity.problems, corrupt_ids=integrity.corrupt_record_ids)
        return

    journal = NeoWriteJournal(path)

    if action == "after_create":
        record = create(journal)
        emit(id=record.record_id)
        die()
    if action == "after_in_flight":
        record = create(journal)
        journal.mark_in_flight(record.record_id, worker_id="w1")
        emit(id=record.record_id)
        die()
    if action in ("during_run", "during_pull", "during_commit", "after_database_commit"):
        record = create(journal)
        journal.mark_in_flight(record.record_id, worker_id="w1")
        effect(effects, action)
        emit(id=record.record_id)
        die()
    if action == "after_committed":
        record = create(journal)
        journal.mark_in_flight(record.record_id)
        effect(effects, "committed")
        journal.mark_committed(record.record_id)
        emit(id=record.record_id)
        die()
    if action == "not_committed":
        record = create(journal)
        journal.mark_in_flight(record.record_id)
        journal.mark_not_committed(record.record_id, error_type="ServiceUnavailable", diagnostic="ServiceUnavailable: failure before commit")
        emit(id=record.record_id)
        die()
    if action == "retry":
        record = journal.authorize_retry(arg, authorized_by="operator", reason="confirmed safe")
        emit(state=record.state.value, attempts=record.attempt_count)
        journal.close()
        return
    if action == "worker_outlives_caller":
        record = create(journal)
        journal.mark_in_flight(record.record_id, worker_id="w1")
        never = threading.Event()

        def worker():
            effect(effects, "query sent")
            never.wait()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=0.2)
        emit(id=record.record_id, worker_alive=thread.is_alive(), state=journal.get_record(record.record_id).state.value)
        die()
    if action == "holds_in_flight":
        record = create(journal)
        journal.mark_in_flight(record.record_id, worker_id="w1")
        emit(id=record.record_id)
        sys.stdin.read()
        die()
    if action == "many_in_flight":
        ids = []
        for _ in range(3):
            record = create(journal)
            journal.mark_in_flight(record.record_id)
            ids.append(record.record_id)
        emit(ids=ids)
        die()
    if action == "dies_in_create":
        kill_at_commit(journal._conn)
        create(journal)
    if action == "dies_in_mark_in_flight":
        record = create(journal)
        emit(id=record.record_id)
        kill_at_commit(journal._conn)
        journal.mark_in_flight(record.record_id)
    if action == "dies_in_mark_committed":
        record = create(journal)
        journal.mark_in_flight(record.record_id)
        effect(effects, "committed")
        emit(id=record.record_id)
        kill_at_commit(journal._conn)
        journal.mark_committed(record.record_id)
    if action == "dies_in_retry":
        record = create(journal)
        journal.mark_in_flight(record.record_id)
        journal.mark_not_committed(record.record_id)
        emit(id=record.record_id)
        kill_at_commit(journal._conn)
        journal.authorize_retry(record.record_id, authorized_by="operator", reason="test")
    if action == "dies_in_resolve":
        record = create(journal)
        journal.mark_in_flight(record.record_id)
        journal.mark_ambiguous(record.record_id, error_type="IncompleteCommit")
        emit(id=record.record_id)
        kill_at_commit(journal._conn)
        journal.resolve_ambiguous(record.record_id, resolution=Resolution.ABANDONED, resolved_by="operator", reason="test")
    if action == "dies_in_recovery":
        kill_at_commit(journal._conn, nth=2)
        journal.recover_unfinished()
    if action == "transition_loop":
        emit(ready=True)
        while True:
            record = create(journal)
            journal.mark_in_flight(record.record_id)
            journal.mark_committed(record.record_id)
    if action == "race_mark_in_flight":
        while not os.path.exists(effects):
            time.sleep(0.001)
        try:
            journal.mark_in_flight(arg)
            emit(won=True)
        except JournalError as error:
            emit(won=False, error=type(error).__name__)
        journal.close()
        return
    if action == "fork_check":
        # Here rather than in the test: fork() in pytest's multi-threaded process isn't safe.
        record = create(journal)
        pid = os.fork()
        if pid == 0:
            try:
                journal.get_record(record.record_id)
                os._exit(1)
            except JournalClosedError:
                journal.close()
                os._exit(0)
            except BaseException:
                os._exit(2)
        status = os.waitpid(pid, 0)[1]
        emit(child_status=status, state_after=journal.get_record(record.record_id).state.value,
             lock_kept=os.path.exists(journal._session_path))
        journal.close()
        return
    if action == "create_many":
        ids = [create(journal).record_id for _ in range(10)]
        emit(ids=ids)
        journal.close()
        return
    if action == "file_size_limit":
        # A real write failure: the kernel refuses writes past this size.
        import resource
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        record = create(journal)
        journal.mark_in_flight(record.record_id)
        limit = os.path.getsize(path + "-wal") + 200000
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
        created, error = [], None
        large = [("CREATE (n:X {v:$v})", {"v": "x" * 3000})]
        try:
            for _ in range(200):
                created.append(journal.create_intent("fill", large).record_id)
        except JournalError as caught:
            error = type(caught).__name__
        # A smaller write could reuse the space the failed one left in the WAL, so refuse them all.
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, limit))
        try:
            journal.mark_committed(record.record_id)
            marked = True
        except JournalError:
            marked = False
        emit(error=error, created=created, in_flight_id=record.record_id, marked=marked,
             state_seen=journal.get_record(record.record_id).state.value)
        die()
    emit(unexpected=f"unknown or finished action {action}", state=JournalState.PENDING.value)


main()

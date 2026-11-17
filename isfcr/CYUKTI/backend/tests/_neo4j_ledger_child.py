"""Runs ledger steps in a real process that really dies. Started by the crash and concurrency tests.

usage: _neo4j_ledger_child.py ACTION LEDGER_PATH EFFECTS_PATH [ARG]

EFFECTS_PATH stands in for Neo4j: a step that would reach the database appends a line, so a test
can show recovery never adds to it. Most actions end in SIGKILL, which runs no cleanup. One JSON
line goes to stdout before the process dies.
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

from neo4j_idempotency_ledger import (  # noqa: E402
    IdempotencyConflictError,
    IdempotencyLedger,
    InvalidTransitionError,
    LedgerClosedError,
    LedgerError,
    Resolution,
    fingerprint_payload,
)

KEY = "campaign-42/flush/7"
OPERATION = "flush_duplicates"
STATEMENTS = [
    ("UNWIND $records AS r MATCH (e:AttackEvent {event_id:r.event_id}) SET e.tps = r.total_tps", {
        "records": [{"event_id": "E1", "total_tps": 0.1, "occurrences": 2 ** 53 + 1, "payload": None}],
        "event_time": dt.datetime(2026, 1, 1, 0, 10, 0, 123456, tzinfo=dt.timezone.utc),
    }),
    ("MATCH (c:Campaign {campaign_id:$c}) SET c.last_seen = $t", {"c": "C1", "t": None}),
]
OTHER_STATEMENTS = [("CREATE (n:Other {v:$v})", {"v": 1})]
FINGERPRINT = fingerprint_payload(OPERATION, STATEMENTS).value


def emit(**values):
    print(json.dumps(values), flush=True)


def die():
    os.kill(os.getpid(), signal.SIGKILL)
    time.sleep(60)


def effect(path, what):
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


def wait_for_start(path):
    while not os.path.exists(path):
        time.sleep(0.001)


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
        IdempotencyLedger(path)
        emit(unexpected="survived")
        return

    if action == "print_fingerprint":
        emit(fingerprint=FINGERPRINT, mixed=fingerprint_payload("op", [("Q", {"z": 1, "a": {"y": [1.5], "b": None}})]).value)
        return

    if action == "peek":
        with IdempotencyLedger(path, read_only=True) as ledger:
            emit(states={r.key_ref: r.state.value for r in ledger.list_requests()}, integrity_ok=ledger.check_integrity().ok)
        return

    if action == "inspect":
        with IdempotencyLedger(path) as ledger:
            report = ledger.recover_unfinished()
            requests = {}
            for request in ledger.list_requests():
                requests[request.idempotency_key] = {
                    "state": request.state.value, "attempts": request.attempt_count, "claimable": request.claimable,
                    "last_outcome": request.last_outcome, "error_type": request.error_type, "fingerprint": request.fingerprint,
                    "matches_original": request.fingerprint == FINGERPRINT, "recovery": request.recovery_metadata,
                    "resolution": request.resolution.value if request.resolution else None,
                    "conflicts": request.conflict_count,
                    "history": [(e.from_state.value if e.from_state else None, e.to_state.value, e.cause) for e in ledger.get_history(request.idempotency_key)],
                }
            integrity = ledger.check_integrity()
            emit(recovered=report.recovered, still_running=report.still_running, registered=report.registered,
                 not_completed=report.not_completed, retry_authorized=report.retry_authorized, uncertain=report.uncertain,
                 corrupt=report.corrupt, requests=requests, integrity_ok=integrity.ok, problems=integrity.problems,
                 corrupt_keys=integrity.corrupt_keys)
        return

    ledger = IdempotencyLedger(path)

    def in_progress():
        ledger.register(KEY, OPERATION, STATEMENTS)
        return ledger.begin_execution(KEY, FINGERPRINT, worker_id="w1")

    if action == "after_register":
        ledger.register(KEY, OPERATION, STATEMENTS)
        emit(done=True)
        die()
    if action == "register_and_exit":
        status = ledger.register(KEY, OPERATION, STATEMENTS).status.value
        ledger.close()
        emit(status=status, fingerprint=FINGERPRINT)
        return
    if action == "dies_in_register":
        kill_at_commit(ledger._conn)
        ledger.register(KEY, OPERATION, STATEMENTS)
    if action == "dies_in_conflict_audit":
        ledger.register(KEY, OPERATION, STATEMENTS)
        kill_at_commit(ledger._conn)
        ledger.register(KEY, OPERATION, OTHER_STATEMENTS)
    if action == "after_begin":
        in_progress()
        emit(done=True)
        die()
    if action == "dies_in_begin":
        ledger.register(KEY, OPERATION, STATEMENTS)
        kill_at_commit(ledger._conn)
        ledger.begin_execution(KEY, FINGERPRINT)
    if action in ("during_run", "during_commit", "after_database_commit"):
        in_progress()
        effect(effects, action)
        emit(done=True)
        die()
    if action == "dies_in_record_completed":
        request = in_progress()
        effect(effects, "committed")
        kill_at_commit(ledger._conn)
        ledger.record_completed(KEY, attempt=request.attempt_count)
    if action == "after_completed":
        request = in_progress()
        effect(effects, "committed")
        ledger.record_completed(KEY, attempt=request.attempt_count)
        emit(done=True)
        die()
    if action == "not_completed":
        request = in_progress()
        ledger.record_not_completed(KEY, attempt=request.attempt_count, error_type="ServiceUnavailable")
        emit(done=True)
        die()
    if action == "dies_in_authorize":
        request = in_progress()
        ledger.record_not_completed(KEY, attempt=request.attempt_count)
        kill_at_commit(ledger._conn)
        ledger.authorize_retry(KEY, authorized_by="operator", reason="test")
    if action == "after_authorize":
        request = in_progress()
        ledger.record_not_completed(KEY, attempt=request.attempt_count)
        ledger.authorize_retry(KEY, authorized_by="operator", reason="test")
        emit(done=True)
        die()
    if action == "authorize_existing":
        ledger.authorize_retry(KEY, authorized_by="operator", reason="confirmed safe")
        ledger.close()
        emit(done=True)
        return
    if action == "dies_in_resolve":
        request = in_progress()
        ledger.record_uncertain(KEY, attempt=request.attempt_count, error_type="IncompleteCommit")
        kill_at_commit(ledger._conn)
        ledger.resolve_uncertain(KEY, resolution=Resolution.FOUND_NOT_COMPLETED, resolved_by="operator", reason="test")
    if action == "worker_outlives_caller":
        in_progress()
        never = threading.Event()

        def worker():
            effect(effects, "query sent")
            never.wait()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=0.2)
        emit(worker_alive=thread.is_alive(), state=ledger.get_request(KEY).state.value)
        die()
    if action == "holds_in_progress":
        in_progress()
        emit(ready=True)
        sys.stdin.read()
        die()
    if action == "many_in_progress":
        keys = []
        for number in range(3):
            key = f"many/{number}"
            ledger.register(key, OPERATION, STATEMENTS)
            ledger.begin_execution(key, FINGERPRINT)
            keys.append(key)
        emit(keys=keys)
        die()
    if action == "dies_in_recovery":
        kill_at_commit(ledger._conn, nth=2)
        ledger.recover_unfinished()
    if action == "transition_loop":
        emit(ready=True)
        number = 0
        while True:
            key = f"{arg}/{number}"
            number += 1
            ledger.register(key, OPERATION, STATEMENTS)
            attempt = ledger.begin_execution(key, FINGERPRINT).attempt_count
            ledger.record_completed(key, attempt=attempt)
    if action == "race_register":
        wait_for_start(effects)
        statements = STATEMENTS if arg == "same" or arg == "a" else OTHER_STATEMENTS
        try:
            emit(status=ledger.register(KEY, OPERATION, statements).status.value, variant=arg)
        except IdempotencyConflictError as error:
            emit(status="CONFLICT", variant=arg, conflict_no=error.conflict_no)
        ledger.close()
        return
    if action == "race_begin":
        wait_for_start(effects)
        try:
            ledger.begin_execution(KEY, FINGERPRINT)
            emit(won=True)
        except InvalidTransitionError:
            emit(won=False)
        ledger.close()
        return
    if action == "race_authorize":
        wait_for_start(effects)
        try:
            ledger.authorize_retry(KEY, authorized_by=f"operator-{os.getpid()}", reason="race")
            emit(won=True)
        except InvalidTransitionError:
            emit(won=False)
        ledger.close()
        return
    if action == "many_keys":
        wait_for_start(effects)
        keys = [f"{arg}/{number}" for number in range(10)]
        for key in keys:
            ledger.register(key, OPERATION, STATEMENTS)
        emit(keys=keys)
        ledger.close()
        return
    if action == "fork_check":
        ledger.register(KEY, OPERATION, STATEMENTS)
        pid = os.fork()
        if pid == 0:
            try:
                ledger.get_request(KEY)
                os._exit(1)
            except LedgerClosedError:
                ledger.close()
                os._exit(0)
            except BaseException:
                os._exit(2)
        status = os.waitpid(pid, 0)[1]
        emit(child_status=status, state_after=ledger.get_request(KEY).state.value, lock_kept=os.path.exists(ledger._session_path))
        ledger.close()
        return
    if action == "file_size_limit":
        # A real write failure: the kernel refuses every write past the limit.
        import resource
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        request = in_progress()
        limit = os.path.getsize(path + "-wal") + 200000
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
        created, error = [], None
        large = [("CREATE (n:X {v:$v})", {"v": "x" * 3000})]
        try:
            for number in range(5000):
                key = f"fill/{number}"
                ledger.register(key, OPERATION, large)
                created.append(key)
        except LedgerError as caught:
            error = type(caught).__name__
        # A smaller write could reuse the space the failed one left in the WAL, so refuse them all.
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, limit))
        try:
            ledger.record_completed(KEY, attempt=request.attempt_count)
            marked = True
        except LedgerError:
            marked = False
        emit(error=error, created=created, marked=marked, state_seen=ledger.get_request(KEY).state.value)
        die()
    emit(unexpected=f"unknown or finished action {action}")


main()

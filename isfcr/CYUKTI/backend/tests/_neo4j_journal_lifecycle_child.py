"""Runs journal and ledger steps in a real process that really dies. Started by test_neo4j_write_journal_lifecycle.py
and test_neo4j_write_journal_migration.py. Stops at one durable step, then SIGKILL: nothing from this process can help
what comes after.

usage: _neo4j_journal_lifecycle_child.py ACTION JOURNAL_PATH LEDGER_PATH [ARG]

The steps below are the order the Phase 2.5 and 2.6A contract sets out, written as test code. They aren't a coordinator:
they exist to reach each pair of store states, and nothing is ever executed against a database.
"""
import json
import os
import signal
import sys
import time

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[0] = BACKEND

from tests._production_safety import install_child_guards  # noqa: E402

install_child_guards()

import neo4j_idempotency_ledger as ledger_module  # noqa: E402
from neo4j_idempotency_ledger import IdempotencyLedger  # noqa: E402
from neo4j_write_journal import InvalidTransitionError, JournalError, NeoWriteJournal, Resolution  # noqa: E402

KEY = "cyukti.v1|wazuh:manager-01/1773199644.1234|duplicate_event.update"
OP = "duplicate_event.update"
STATEMENTS = [("MATCH (e:AttackEvent {event_id:$e}) SET e.occurrences = e.occurrences + 1", {"e": "E1", "tps": 0})]


def emit(**values):
    print(json.dumps(values), flush=True)


def die():
    os.kill(os.getpid(), signal.SIGKILL)
    time.sleep(60)


def kill_at_commit(conn, nth=1):
    """Dies as the nth COMMIT is about to run, so that transaction never completes."""
    seen = [0]

    def trace(statement):
        if statement.strip().upper().startswith("COMMIT"):
            seen[0] += 1
            if seen[0] == nth:
                die()
    conn.set_trace_callback(trace)


class Run:
    def __init__(self, journal, ledger):
        self.journal, self.ledger = journal, ledger
        self.fingerprint = None
        self.first = None

    def first_attempt(self):
        """Registered, intent created, claimed as attempt 1, IN_FLIGHT."""
        self.fingerprint = self.ledger.register(KEY, OP, STATEMENTS).request.fingerprint
        record = self.journal.create_intent(OP, STATEMENTS, correlation_key=KEY, epoch=1)
        attempt = self.ledger.begin_execution(KEY, self.fingerprint).attempt_count
        self.first = self.journal.mark_in_flight(record.record_id, ledger_attempt=attempt)

    def ambiguous(self):
        self.first_attempt()
        self.journal.mark_ambiguous(self.first.record_id, expected_ledger_attempt=1)
        self.ledger.record_uncertain(KEY, attempt=1)

    def journal_resolved(self):
        self.ambiguous()
        self.first = self.journal.resolve_ambiguous(self.first.record_id, resolution=Resolution.FOUND_NOT_COMMITTED, resolved_by="analyst", reason="checked")

    def authorized(self):
        self.journal_resolved()
        self.ledger.resolve_uncertain(KEY, resolution=ledger_module.Resolution.FOUND_NOT_COMPLETED, resolved_by="analyst", reason="checked")
        self.ledger.authorize_retry(KEY, authorized_by="operator", reason="retry")

    def new_intent(self):
        self.authorized()
        self.second = self.journal.create_intent(OP, self.first.statements, correlation_key=KEY, epoch=2, authorized_by="operator", reason="retry")

    def claimed(self):
        self.new_intent()
        self.attempt = self.ledger.begin_execution(KEY, self.fingerprint).attempt_count

    def second_in_flight(self):
        self.claimed()
        self.journal.mark_in_flight(self.second.record_id, ledger_attempt=self.attempt)

    def confirmed_not_committed_then_claimed(self):
        self.first_attempt()
        self.journal.mark_not_committed(self.first.record_id, expected_ledger_attempt=1, error_type="ServiceUnavailable")
        self.ledger.record_not_completed(KEY, attempt=1, error_type="ServiceUnavailable")
        self.ledger.authorize_retry(KEY, authorized_by="operator", reason="retry")
        self.ledger.begin_execution(KEY, self.fingerprint)

    def claim_whose_in_flight_failed(self):
        """In this process, as the coordinator's handler would: the claim is released through the ledger alone."""
        self.fingerprint = self.ledger.register(KEY, OP, STATEMENTS).request.fingerprint
        self.first = self.journal.create_intent(OP, STATEMENTS, correlation_key=KEY, epoch=1)
        attempt = self.ledger.begin_execution(KEY, self.fingerprint).attempt_count
        self.ledger.record_not_completed(KEY, attempt=attempt, diagnostic="the journal could not record IN_FLIGHT")


def stop_after(run, step):
    getattr(run, step)()
    emit(step=step)
    die()


def main():
    action, journal_path, ledger_path = sys.argv[1:4]
    if action == "migrate_dying":
        # A version 1 file. Dies as the migration's COMMIT is about to run.
        import sqlite3
        real_connect = sqlite3.connect

        def connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            kill_at_commit(conn)
            return conn
        sqlite3.connect = connect
        NeoWriteJournal(journal_path)
        emit(unexpected="survived")
        return
    if action == "open_when_told":
        # Waits for the go file, opens (which migrates), and reports.
        while not os.path.exists(ledger_path):
            time.sleep(0.001)
        with NeoWriteJournal(journal_path, busy_timeout=30) as journal:
            emit(version=journal.check_integrity().schema_version, ok=journal.check_integrity().ok, records=len(journal.list_records()))
        return
    if action == "race_next_epoch":
        while not os.path.exists(ledger_path):
            time.sleep(0.001)
        with NeoWriteJournal(journal_path, busy_timeout=30) as journal:
            first = journal.find_by_correlation_key(KEY)[0]
            try:
                record = journal.create_intent(OP, first.statements, correlation_key=KEY, epoch=2, authorized_by="operator", reason="retry")
                emit(won=True, id=record.record_id)
            except JournalError as error:
                emit(won=False, error=type(error).__name__, existing=getattr(error, "record_id", None))
        return
    if action == "race_outcome":
        while not os.path.exists(ledger_path):
            time.sleep(0.001)
        record_id, attempt = sys.argv[4], int(sys.argv[5])
        with NeoWriteJournal(journal_path, busy_timeout=30) as journal:
            try:
                journal.mark_committed(record_id, expected_ledger_attempt=attempt)
                emit(won=True, attempt=attempt)
            except InvalidTransitionError as error:
                emit(won=False, attempt=attempt, error=type(error).__name__)
        return

    journal = NeoWriteJournal(journal_path)
    ledger = IdempotencyLedger(ledger_path)
    run = Run(journal, ledger)

    if action in ("journal_resolved", "authorized", "new_intent", "claimed", "second_in_flight", "confirmed_not_committed_then_claimed", "claim_whose_in_flight_failed"):
        stop_after(run, action)
    if action.startswith("outcome_"):
        run.second_in_flight()
        outcome = action.removeprefix("outcome_")
        call = {"committed": journal.mark_committed, "not_committed": journal.mark_not_committed, "ambiguous": journal.mark_ambiguous}[outcome]
        call(run.second.record_id, expected_ledger_attempt=run.attempt)
        emit(step=action)
        die()

    # Dying inside one journal transaction. The commit never happens, so the step must leave no trace.
    if action == "dies_creating_first":
        run.fingerprint = ledger.register(KEY, OP, STATEMENTS).request.fingerprint
        kill_at_commit(journal._conn)
        journal.create_intent(OP, STATEMENTS, correlation_key=KEY, epoch=1)
    if action == "dies_marking_in_flight":
        run.fingerprint = ledger.register(KEY, OP, STATEMENTS).request.fingerprint
        record = journal.create_intent(OP, STATEMENTS, correlation_key=KEY, epoch=1)
        attempt = ledger.begin_execution(KEY, run.fingerprint).attempt_count
        emit(id=record.record_id)
        kill_at_commit(journal._conn)
        journal.mark_in_flight(record.record_id, ledger_attempt=attempt)
    if action in ("dies_marking_committed", "dies_marking_not_committed", "dies_marking_ambiguous"):
        run.first_attempt()
        emit(id=run.first.record_id)
        kill_at_commit(journal._conn)
        {"dies_marking_committed": journal.mark_committed, "dies_marking_not_committed": journal.mark_not_committed,
         "dies_marking_ambiguous": journal.mark_ambiguous}[action](run.first.record_id, expected_ledger_attempt=1)
    if action == "dies_authorizing_retry":
        run.first_attempt()
        journal.mark_not_committed(run.first.record_id, expected_ledger_attempt=1)
        ledger.record_not_completed(KEY, attempt=1)
        ledger.authorize_retry(KEY, authorized_by="operator", reason="retry")
        attempt = ledger.begin_execution(KEY, run.fingerprint).attempt_count
        emit(id=run.first.record_id)
        kill_at_commit(journal._conn)
        journal.authorize_retry(run.first.record_id, authorized_by="operator", reason="retry", ledger_attempt=attempt)
    if action == "dies_resolving":
        run.ambiguous()
        emit(id=run.first.record_id)
        kill_at_commit(journal._conn)
        journal.resolve_ambiguous(run.first.record_id, resolution=Resolution.FOUND_NOT_COMMITTED, resolved_by="analyst", reason="checked")
    if action == "dies_creating_next_epoch":
        run.authorized()
        emit(id=run.first.record_id)
        kill_at_commit(journal._conn)
        journal.create_intent(OP, run.first.statements, correlation_key=KEY, epoch=2, authorized_by="operator", reason="retry")
    if action == "loop":
        emit(ready=True)
        number = 0
        while True:
            number += 1
            key = f"{KEY}|{os.getpid()}|{number}"
            record = journal.create_intent(OP, STATEMENTS, correlation_key=key, epoch=1)
            journal.mark_in_flight(record.record_id, ledger_attempt=1)
            journal.mark_ambiguous(record.record_id, expected_ledger_attempt=1)
            journal.resolve_ambiguous(record.record_id, resolution=Resolution.FOUND_NOT_COMMITTED, resolved_by="a", reason="r")
            second = journal.create_intent(OP, STATEMENTS, correlation_key=key, epoch=2, authorized_by="a", reason="r")
            journal.mark_in_flight(second.record_id, ledger_attempt=2)
            journal.mark_committed(second.record_id, expected_ledger_attempt=2)
    emit(unexpected=f"unknown or finished action {action}")


main()

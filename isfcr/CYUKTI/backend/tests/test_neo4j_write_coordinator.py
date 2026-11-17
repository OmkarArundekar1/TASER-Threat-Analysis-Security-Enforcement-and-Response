"""WP-C1: the coordinator against the real journal and ledger, on real files, with a fake writer.

Nothing here contacts Neo4j, and a fake writer says nothing about how Neo4j behaves. Every lifecycle
test asserts the caller-visible result, the persisted state of both stores, and how often the writer
ran. Stores are never mocked: a fault is either a real SQLite lock, a really closed store, or a real
store subclass that raises the store's own error at one named call, and otherwise delegates.
"""
from __future__ import annotations

import ast
import datetime as dt
import inspect
import logging
import sqlite3
import threading
from pathlib import Path

import pytest

from listener.utils import extract_source_identity, request_key
from neo4j_idempotency_ledger import IdempotencyLedger, LedgerState as L, LedgerWriteError, fingerprint_payload
from neo4j_write_coordinator import (
    CombinedState as C,
    CoordinatedResult,
    CoordinationInconsistency,
    Decision,
    Disposition as D,
    IdentityRefusal,
    InvalidRequest,
    NotAuthorized,
    OperationRegistry,
    PayloadMismatch,
    PayloadNotJournalable,
    RecoveryCategory as R,
    RegisteredOperation,
    ResolutionConflict,
    WriteCoordinator,
    WriteRequest,
    check_source_key,
    combined_state,
)
from neo4j_write_journal import JournalState as J, JournalWriteError, NeoWriteJournal, Resolution as JournalResolution
from tests._neo4j_fake_writer import FakeOutcome, FakeWriter, FakeWriteResult, Script

BACKEND = Path(__file__).resolve().parent.parent
SOURCE = "wazuh:manager-01/1700000000.1234"
OTHER_SOURCE = "wazuh:manager-01/1700000000.5678"
OPERATION = RegisteredOperation("test_event.update", "MATCH (n:CoordinatorTestOnly {id:$id}) SET n.v = $v", frozenset({"id", "v"}))
OPERATION_B = RegisteredOperation("test_event.create", "CREATE (n:CoordinatorTestOnly {id:$id, v:$v})", frozenset({"id", "v"}))
REGISTRY = OperationRegistry([OPERATION, OPERATION_B])
PARAMETERS = {"id": "E1", "v": 1}
KEY = request_key(SOURCE, OPERATION.name)


def request(source=SOURCE, parameters=None, operation=OPERATION.name):
    return WriteRequest(source, operation, dict(PARAMETERS) if parameters is None else parameters)


class Env:
    """Real stores on real files. `reopen` gives a second instance sharing the same files."""

    def __init__(self, directory, *, journal_class=NeoWriteJournal, ledger_class=IdempotencyLedger, busy_timeout=5.0):
        self.journal_path = str(directory / "journal.db")
        self.ledger_path = str(directory / "ledger.db")
        self.journal_class, self.ledger_class = journal_class, ledger_class
        self.busy_timeout = busy_timeout
        self.opened = []
        self.journal, self.ledger = self.open_stores()

    def open_stores(self, journal_class=None, ledger_class=None):
        journal = (journal_class or self.journal_class)(self.journal_path, busy_timeout=self.busy_timeout)
        ledger = (ledger_class or self.ledger_class)(self.ledger_path, busy_timeout=self.busy_timeout)
        self.opened += [journal, ledger]
        return journal, ledger

    def coordinator(self, writer, *, stores=None, **options):
        journal, ledger = stores or (self.journal, self.ledger)
        options.setdefault("store_retry_delay", 0.0)
        return WriteCoordinator(journal, ledger, writer=writer, registry=REGISTRY, **options)

    def close(self):
        for store in self.opened:
            store.close()


@pytest.fixture
def env(tmp_path):
    environment = Env(tmp_path)
    yield environment
    environment.close()


def assert_settled(env, key=KEY, *, ledger, journal, attempt=None, writer=None, calls=None, effects=None, combined=None, consistent=True):
    """The persisted state, read back through fresh read-only connections, and the writer's count."""
    reader_journal = NeoWriteJournal(env.journal_path, read_only=True)
    reader_ledger = IdempotencyLedger(env.ledger_path, read_only=True)
    try:
        found = reader_ledger.get_request(key)
        chain = reader_journal.find_by_correlation_key(key)
        assert found.state is ledger
        assert [record.state for record in chain] == list(journal)
        if attempt is not None:
            assert found.attempt_count == attempt
        state = combined_state(found, chain)
        assert (state is not C.INCONSISTENT) is consistent, state
        if combined is not None:
            assert state is combined
        assert reader_journal.check_integrity().ok and reader_ledger.check_integrity().ok
    finally:
        reader_journal.close()
        reader_ledger.close()
    if writer is not None:
        if calls is not None:
            assert len(writer.calls) == calls
        if effects is not None:
            assert len(writer.effects) == effects


class FailingJournal(NeoWriteJournal):
    """The real journal, with the store's own error raised at a named call."""
    failures: dict = {}

    def _fail(self, name):
        left = self.failures.get(name, 0)
        if left:
            self.failures[name] = left - 1 if left > 0 else left
            raise JournalWriteError(f"injected failure in {name}")

    def preflight_intent(self, *a, **k):
        self._fail("preflight_intent")
        return super().preflight_intent(*a, **k)

    def create_intent(self, *a, **k):
        self._fail("create_intent")
        return super().create_intent(*a, **k)

    def mark_in_flight(self, *a, **k):
        self._fail("mark_in_flight")
        return super().mark_in_flight(*a, **k)

    def authorize_retry(self, *a, **k):
        self._fail("authorize_retry")
        return super().authorize_retry(*a, **k)

    def mark_committed(self, *a, **k):
        self._fail("mark_committed")
        return super().mark_committed(*a, **k)

    def mark_not_committed(self, *a, **k):
        self._fail("mark_not_committed")
        return super().mark_not_committed(*a, **k)

    def mark_ambiguous(self, *a, **k):
        self._fail("mark_ambiguous")
        return super().mark_ambiguous(*a, **k)


class FailingLedger(IdempotencyLedger):
    """The real ledger, with the store's own error raised at a named call."""
    failures: dict = {}

    def _fail(self, name):
        left = self.failures.get(name, 0)
        if left:
            self.failures[name] = left - 1 if left > 0 else left
            raise LedgerWriteError(f"injected failure in {name}")

    def register(self, *a, **k):
        self._fail("register")
        return super().register(*a, **k)

    def begin_execution(self, *a, **k):
        self._fail("begin_execution")
        return super().begin_execution(*a, **k)

    def record_completed(self, *a, **k):
        self._fail("record_completed")
        return super().record_completed(*a, **k)

    def record_not_completed(self, *a, **k):
        self._fail("record_not_completed")
        return super().record_not_completed(*a, **k)

    def record_uncertain(self, *a, **k):
        self._fail("record_uncertain")
        return super().record_uncertain(*a, **k)


@pytest.fixture
def faulty(tmp_path):
    """Faults per store method: a positive count fails that many calls, and -1 fails every call."""
    journal_class = type("J", (FailingJournal,), {"failures": {}})
    ledger_class = type("L", (FailingLedger,), {"failures": {}})
    environment = Env(tmp_path, journal_class=journal_class, ledger_class=ledger_class)
    environment.journal_faults, environment.ledger_faults = journal_class.failures, ledger_class.failures
    yield environment
    environment.close()


@pytest.fixture
def restart(env):
    """A second coordinator on new store instances, as after a restart of a process that closed cleanly."""
    def start(writer, **options):
        return env.coordinator(writer, stores=env.open_stores(), **options)
    return start


# ================================================================ A. new request

def test_a_new_request_runs_once_and_finishes_in_both_stores(env):
    writer = FakeWriter()
    result = env.coordinator(writer).submit(request())
    assert result.disposition is D.EXECUTED_COMMITTED
    assert result.applied and result.executed and result.finalized and result.writer_invoked and not result.must_hold
    assert result.ledger_attempt == 1 and result.journal_record_id
    assert_settled(env, ledger=L.COMPLETED, journal=[J.COMMITTED], attempt=1, writer=writer, calls=1, effects=1, combined=C.COMPLETED)


def test_the_writer_is_given_the_statements_the_journal_stored(env):
    writer = FakeWriter()
    env.coordinator(writer).submit(request())
    (operation, statements, limits) = writer.calls[0]
    record = env.journal.find_by_correlation_key(KEY)[0]
    assert operation == OPERATION.name and tuple(statements) == record.statements
    assert statements[0][0] == OPERATION.query and statements[0][1] == PARAMETERS
    assert limits is None


def test_limits_are_passed_to_the_writer_unchanged(env):
    writer = FakeWriter()
    limits = object()
    env.coordinator(writer, limits=limits).submit(request())
    assert writer.calls[0][2] is limits


def test_by_the_time_the_writer_runs_the_claim_and_in_flight_are_durable(env):
    seen = {}

    def during():
        reader_journal = NeoWriteJournal(env.journal_path, read_only=True)
        reader_ledger = IdempotencyLedger(env.ledger_path, read_only=True)
        seen["ledger"] = reader_ledger.get_request(KEY).state
        seen["journal"] = reader_journal.find_by_correlation_key(KEY)[0]
        reader_journal.close()
        reader_ledger.close()

    env.coordinator(FakeWriter(during=during)).submit(request())
    assert seen["ledger"] is L.IN_PROGRESS
    assert seen["journal"].state is J.IN_FLIGHT and seen["journal"].ledger_attempt == 1


def test_the_journal_record_names_the_key_epoch_and_the_ledgers_attempt(env):
    env.coordinator(FakeWriter()).submit(request())
    record = env.journal.find_by_correlation_key(KEY)[0]
    assert record.correlation_key == KEY and record.epoch == 1 and record.ledger_attempt == 1
    assert env.ledger.get_request(KEY).fingerprint == fingerprint_payload(OPERATION.name, record.statements).value


def test_success_is_not_reported_until_the_ledger_has_finalized(faulty):
    faulty.ledger_faults["record_completed"] = -1
    writer = FakeWriter()
    result = faulty.coordinator(writer).submit(request())
    assert result.disposition is D.COMMITTED_UNRECORDED and not result.applied and not result.finalized
    assert_settled(faulty, ledger=L.IN_PROGRESS, journal=[J.COMMITTED], writer=writer, calls=1, effects=1, combined=C.OUTCOME_NOT_MIRRORED)


# ================================================================ B. same key, same payload

def test_a_repeat_after_completion_doesnt_run_again(env):
    writer = FakeWriter()
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    result = coordinator.submit(request())
    assert result.disposition is D.ALREADY_COMPLETED and result.applied and not result.writer_invoked and not result.executed
    assert result.ledger_attempt == 1
    assert_settled(env, ledger=L.COMPLETED, journal=[J.COMMITTED], attempt=1, writer=writer, calls=1, effects=1)


def test_a_repeat_while_another_holds_the_claim_doesnt_run(env):
    writer = FakeWriter()
    coordinator = env.coordinator(writer)
    answers = []
    writer.during = lambda: answers.append(coordinator.submit(request()))
    coordinator.submit(request())
    assert answers[0].disposition is D.IN_PROGRESS_ELSEWHERE and not answers[0].writer_invoked
    assert_settled(env, ledger=L.COMPLETED, journal=[J.COMMITTED], writer=writer, calls=1)


@pytest.mark.parametrize("step, disposition, ledger, journal", [
    (Script.FAILED_BEFORE_COMMIT, D.AWAITING_AUTHORIZATION, L.NOT_COMPLETED, J.NOT_COMMITTED),
    (Script.NOT_EXECUTED, D.AWAITING_AUTHORIZATION, L.NOT_COMPLETED, J.NOT_COMMITTED),
    (Script.AMBIGUOUS_APPLIED, D.BLOCKED_UNCERTAIN, L.UNCERTAIN, J.AMBIGUOUS),
])
def test_a_repeat_answers_from_the_state_and_runs_nothing(env, step, disposition, ledger, journal):
    writer = FakeWriter(step)
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    effects = len(writer.effects)
    result = coordinator.submit(request())
    assert result.disposition is disposition and not result.writer_invoked and not result.applied
    assert_settled(env, ledger=ledger, journal=[journal], attempt=1, writer=writer, calls=1, effects=effects)


def test_a_repeat_doesnt_spend_an_authorization(env):
    writer = FakeWriter(Script.FAILED_BEFORE_COMMIT)
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    coordinator.authorize_retry(KEY, authorized_by="analyst", reason="transient error confirmed")
    result = coordinator.submit(request())
    assert result.disposition is D.AWAITING_AUTHORIZATION and not result.writer_invoked
    assert_settled(env, ledger=L.RETRY_AUTHORIZED, journal=[J.NOT_COMMITTED], attempt=1, writer=writer, calls=1)


# ================================================================ C. same key, different payload

def test_a_different_payload_is_an_explicit_conflict_and_changes_nothing_of_the_original(env):
    writer = FakeWriter()
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    before = env.ledger.get_request(KEY)
    history = env.ledger.get_history(KEY)
    record = env.journal.find_by_correlation_key(KEY)[0]
    result = coordinator.submit(request(parameters={"id": "E1", "v": 2}))
    assert result.disposition is D.CONFLICT and not result.applied and not result.writer_invoked
    assert result.differences == ("statement 0 parameters",)
    after = env.ledger.get_request(KEY)
    assert after.fingerprint == before.fingerprint and after.state is L.COMPLETED and after.attempt_count == 1
    assert env.ledger.get_history(KEY) == history
    assert env.ledger.get_conflicts(KEY)[0].differences == ("statement 0 parameters",)
    assert env.journal.find_by_correlation_key(KEY) == (record,)
    assert_settled(env, ledger=L.COMPLETED, journal=[J.COMMITTED], attempt=1, writer=writer, calls=1, effects=1)


def test_a_conflict_is_refused_at_every_state_of_the_original(env):
    writer = FakeWriter(Script.AMBIGUOUS_NOT_APPLIED)
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    result = coordinator.submit(request(parameters={"id": "E1", "v": 2}))
    assert result.disposition is D.CONFLICT
    assert_settled(env, ledger=L.UNCERTAIN, journal=[J.AMBIGUOUS], attempt=1, writer=writer, calls=1)


def test_a_conflict_creates_no_journal_record(env):
    env.ledger.register(KEY, OPERATION.name, OPERATION.statements({"id": "E1", "v": 9}))
    writer = FakeWriter()
    result = env.coordinator(writer).submit(request())
    assert result.disposition is D.CONFLICT and writer.calls == []
    assert env.journal.find_by_correlation_key(KEY) == ()


# ================================================================ D. different keys, same payload

def test_different_keys_with_the_same_payload_are_separate_requests(env):
    writer = FakeWriter()
    coordinator = env.coordinator(writer)
    first = coordinator.submit(request(SOURCE))
    second = coordinator.submit(request(OTHER_SOURCE))
    assert first.disposition is second.disposition is D.EXECUTED_COMMITTED
    assert first.key_ref != second.key_ref
    other = request_key(OTHER_SOURCE, OPERATION.name)
    assert env.ledger.get_request(KEY).fingerprint == env.ledger.get_request(other).fingerprint
    assert_settled(env, ledger=L.COMPLETED, journal=[J.COMMITTED], writer=writer, calls=2, effects=2)
    assert_settled(env, other, ledger=L.COMPLETED, journal=[J.COMMITTED])


def test_one_alert_through_two_operations_is_two_requests(env):
    writer = FakeWriter()
    coordinator = env.coordinator(writer)
    assert coordinator.submit(request()).disposition is D.EXECUTED_COMMITTED
    assert coordinator.submit(request(operation=OPERATION_B.name)).disposition is D.EXECUTED_COMMITTED
    assert len(writer.effects) == 2
    assert_settled(env, ledger=L.COMPLETED, journal=[J.COMMITTED])
    assert_settled(env, request_key(SOURCE, OPERATION_B.name), ledger=L.COMPLETED, journal=[J.COMMITTED])


# ================================================================ E. confirmed failure before execution

def test_a_writer_that_established_it_sent_nothing_isnt_called_executed_or_ambiguous(env):
    writer = FakeWriter(Script.NOT_EXECUTED)
    result = env.coordinator(writer).submit(request())
    assert result.disposition is D.NOT_EXECUTED_CONFIRMED
    assert not result.executed and not result.applied and not result.must_hold and result.writer_attempts == 0 and result.finalized
    assert_settled(env, ledger=L.NOT_COMPLETED, journal=[J.NOT_COMMITTED], attempt=1, writer=writer, calls=1, effects=0, combined=C.AWAITING_AUTHORIZATION)


def test_a_journal_refusing_in_flight_is_established_as_not_run_not_uncertain(faulty):
    faulty.journal_faults["mark_in_flight"] = 1
    writer = FakeWriter()
    result = faulty.coordinator(writer).submit(request())
    assert result.disposition is D.NOT_DISPATCHED_UNAVAILABLE and result.must_hold and result.unavailable == "journal"
    assert not result.writer_invoked and not result.executed and result.finalized and result.ledger_attempt == 1
    assert_settled(faulty, ledger=L.NOT_COMPLETED, journal=[J.PENDING], attempt=1, writer=writer, calls=0, effects=0, combined=C.AWAITING_AUTHORIZATION)
    assert faulty.ledger.get_request(KEY).last_outcome == "NOT_COMPLETED"


def test_after_a_refused_in_flight_the_same_record_runs_once_authorized(faulty):
    faulty.journal_faults["mark_in_flight"] = 1
    writer = FakeWriter()
    coordinator = faulty.coordinator(writer)
    coordinator.submit(request())
    record_id = faulty.journal.find_by_correlation_key(KEY)[0].record_id
    assert coordinator.submit(request()).disposition is D.AWAITING_AUTHORIZATION and writer.calls == []
    coordinator.authorize_retry(KEY, authorized_by="analyst", reason="journal restored")
    result = coordinator.retry(KEY)
    assert result.disposition is D.EXECUTED_COMMITTED and result.ledger_attempt == 2 and result.journal_record_id == record_id
    assert_settled(faulty, ledger=L.COMPLETED, journal=[J.COMMITTED], attempt=2, writer=writer, calls=1, effects=1)


def test_if_the_claim_cant_be_released_either_recovery_reports_it_as_never_dispatched(faulty):
    faulty.journal_faults["mark_in_flight"] = 1
    faulty.ledger_faults["record_not_completed"] = -1
    writer = FakeWriter()
    result = faulty.coordinator(writer).submit(request())
    assert result.disposition is D.NOT_DISPATCHED_UNAVAILABLE and not result.finalized and writer.calls == []
    faulty.ledger_faults.clear()
    assert_settled(faulty, ledger=L.IN_PROGRESS, journal=[J.PENDING], attempt=1, combined=C.CLAIMED)


# ================================================================ writer fails or is unclear

def test_a_failure_after_execution_began_is_a_confirmed_non_commit_held_for_authorization(env):
    writer = FakeWriter(Script.FAILED_BEFORE_COMMIT)
    result = env.coordinator(writer).submit(request())
    assert result.disposition is D.EXECUTED_NOT_COMMITTED and result.executed and not result.applied and result.writer_attempts == 1
    assert_settled(env, ledger=L.NOT_COMPLETED, journal=[J.NOT_COMMITTED], attempt=1, writer=writer, calls=1, effects=0)
    record = env.journal.find_by_correlation_key(KEY)[0]
    assert record.error_type == "TransientError" and record.error_code == "Neo.TransientError.Test"
    assert env.ledger.get_request(KEY).error_code == "Neo.TransientError.Test"


@pytest.mark.parametrize("step, effects", [(Script.AMBIGUOUS_APPLIED, 1), (Script.AMBIGUOUS_NOT_APPLIED, 0)])
def test_f_an_ambiguous_outcome_is_recorded_and_never_replayed_or_called_success(env, step, effects):
    writer = FakeWriter(step)
    coordinator = env.coordinator(writer)
    result = coordinator.submit(request())
    assert result.disposition is D.EXECUTED_UNCERTAIN and not result.applied and result.finalized and result.executed
    assert_settled(env, ledger=L.UNCERTAIN, journal=[J.AMBIGUOUS], attempt=1, writer=writer, calls=1, effects=effects, combined=C.UNCERTAIN)
    for _ in range(3):
        assert coordinator.submit(request()).disposition is D.BLOCKED_UNCERTAIN
    with pytest.raises(NotAuthorized):
        coordinator.retry(KEY)
    with pytest.raises(NotAuthorized):
        coordinator.authorize_retry(KEY, authorized_by="analyst", reason="not allowed before resolving")
    assert_settled(env, ledger=L.UNCERTAIN, journal=[J.AMBIGUOUS], attempt=1, writer=writer, calls=1, effects=effects)


@pytest.mark.parametrize("step, effects", [(Script.RAISES, 0), (Script.RAISES_AFTER_APPLYING, 1), (Script.UNKNOWN_OUTCOME, 0), (Script.NO_OUTCOME, 0)])
def test_a_writer_that_raises_or_reports_nonsense_is_uncertain(env, step, effects):
    writer = FakeWriter(step)
    result = env.coordinator(writer).submit(request())
    assert result.disposition is D.EXECUTED_UNCERTAIN and not result.applied and result.finalized
    assert_settled(env, ledger=L.UNCERTAIN, journal=[J.AMBIGUOUS], attempt=1, writer=writer, calls=1, effects=effects)


def test_an_interrupt_during_the_writer_propagates_and_leaves_it_uncertain(env):
    writer = FakeWriter(Script.INTERRUPTED)
    with pytest.raises(KeyboardInterrupt):
        env.coordinator(writer).submit(request())
    assert_settled(env, ledger=L.UNCERTAIN, journal=[J.AMBIGUOUS], attempt=1, writer=writer, calls=1)


def test_an_interrupt_with_the_journal_down_still_propagates_and_stays_unrunnable(faulty):
    faulty.journal_faults["mark_ambiguous"] = -1
    writer = FakeWriter(Script.INTERRUPTED)
    with pytest.raises(KeyboardInterrupt):
        faulty.coordinator(writer).submit(request())
    assert_settled(faulty, ledger=L.IN_PROGRESS, journal=[J.IN_FLIGHT], writer=writer, calls=1, combined=C.DISPATCHED)


# ================================================================ finalization failures after the writer ran

def test_a_commit_the_journal_cant_record_is_reported_and_never_run_again(faulty):
    faulty.journal_faults["mark_committed"] = -1
    writer = FakeWriter()
    coordinator = faulty.coordinator(writer)
    result = coordinator.submit(request())
    assert result.disposition is D.COMMITTED_UNRECORDED and not result.applied and not result.finalized and result.must_hold
    assert_settled(faulty, ledger=L.IN_PROGRESS, journal=[J.IN_FLIGHT], writer=writer, calls=1, effects=1, combined=C.DISPATCHED)
    assert coordinator.submit(request()).disposition is D.IN_PROGRESS_ELSEWHERE
    assert len(writer.calls) == 1


def test_a_brief_journal_failure_after_a_commit_is_retried_locally_without_running_again(faulty):
    faulty.journal_faults["mark_committed"] = 2
    writer = FakeWriter()
    result = faulty.coordinator(writer).submit(request())
    assert result.disposition is D.EXECUTED_COMMITTED and result.applied
    assert_settled(faulty, ledger=L.COMPLETED, journal=[J.COMMITTED], writer=writer, calls=1, effects=1)


def test_local_store_retries_are_bounded(faulty):
    faulty.journal_faults["mark_committed"] = 3
    writer = FakeWriter()
    result = faulty.coordinator(writer, store_write_attempts=3).submit(request())
    assert result.disposition is D.COMMITTED_UNRECORDED and faulty.journal_faults["mark_committed"] == 0


def test_a_non_commit_the_journal_cant_record_becomes_uncertain_not_a_confirmed_failure(faulty):
    faulty.journal_faults["mark_not_committed"] = -1
    writer = FakeWriter(Script.FAILED_BEFORE_COMMIT)
    result = faulty.coordinator(writer).submit(request())
    assert result.disposition is D.EXECUTED_UNCERTAIN and not result.finalized
    assert_settled(faulty, ledger=L.IN_PROGRESS, journal=[J.IN_FLIGHT], writer=writer, calls=1)


def test_a_non_commit_the_ledger_cant_record_is_left_for_recovery(faulty):
    faulty.ledger_faults["record_not_completed"] = -1
    writer = FakeWriter(Script.FAILED_BEFORE_COMMIT)
    result = faulty.coordinator(writer).submit(request())
    assert result.disposition is D.EXECUTED_NOT_COMMITTED and not result.finalized
    assert_settled(faulty, ledger=L.IN_PROGRESS, journal=[J.NOT_COMMITTED], combined=C.OUTCOME_NOT_MIRRORED)


def test_an_ambiguous_outcome_the_stores_cant_record_is_still_reported_as_uncertain(faulty):
    faulty.journal_faults["mark_ambiguous"] = -1
    writer = FakeWriter(Script.AMBIGUOUS_APPLIED)
    result = faulty.coordinator(writer).submit(request())
    assert result.disposition is D.EXECUTED_UNCERTAIN and not result.finalized and not result.applied
    assert_settled(faulty, ledger=L.IN_PROGRESS, journal=[J.IN_FLIGHT], writer=writer, calls=1, effects=1)


def test_a_stale_attempts_report_is_dropped_and_never_overwrites(env):
    """A stalled writer reports after recovery and a person have moved the request on."""
    second = {}

    def during():
        # While the first attempt's writer is stuck, the stores are moved on as recovery and a person would.
        record = env.journal.find_by_correlation_key(KEY)[0]
        env.journal.mark_not_committed(record.record_id, expected_ledger_attempt=1)
        env.ledger.record_not_completed(KEY, attempt=1)
        env.ledger.authorize_retry(KEY, authorized_by="analyst", reason="moved on")
        claim = env.ledger.begin_execution(KEY, env.ledger.get_request(KEY).fingerprint)
        env.journal.authorize_retry(record.record_id, authorized_by="analyst", reason="moved on", ledger_attempt=claim.attempt_count)
        second["attempt"] = claim.attempt_count

    writer = FakeWriter(during=during)
    result = env.coordinator(writer).submit(request())
    assert second["attempt"] == 2
    assert result.disposition is D.COMMITTED_UNRECORDED and not result.applied and not result.finalized
    assert_settled(env, ledger=L.IN_PROGRESS, journal=[J.IN_FLIGHT], attempt=2)


# ================================================================ G. retry, resolve, authorize

@pytest.mark.parametrize("step", [Script.FAILED_BEFORE_COMMIT, Script.NOT_EXECUTED])
def test_retry_without_authorization_is_refused_and_runs_nothing(env, step):
    writer = FakeWriter(step)
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    with pytest.raises(NotAuthorized):
        coordinator.retry(KEY)
    assert_settled(env, ledger=L.NOT_COMPLETED, journal=[J.NOT_COMMITTED], attempt=1, writer=writer, calls=1, effects=0)


def test_retry_of_a_completed_or_unknown_request_is_refused(env):
    writer = FakeWriter()
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    with pytest.raises(NotAuthorized):
        coordinator.retry(KEY)
    with pytest.raises(InvalidRequest):
        coordinator.retry(request_key(OTHER_SOURCE, OPERATION.name))
    assert len(writer.calls) == 1


@pytest.mark.parametrize("by, reason", [(None, "because"), ("", "because"), ("   ", "because"), (7, "because"),
                                        ("analyst", None), ("analyst", ""), ("analyst", "  "), ("analyst", 7)])
def test_authorization_needs_a_named_person_and_a_reason(env, by, reason):
    writer = FakeWriter(Script.FAILED_BEFORE_COMMIT)
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    with pytest.raises(InvalidRequest):
        coordinator.authorize_retry(KEY, authorized_by=by, reason=reason)
    assert_settled(env, ledger=L.NOT_COMPLETED, journal=[J.NOT_COMMITTED], attempt=1, writer=writer, calls=1)
    with pytest.raises(NotAuthorized):
        coordinator.retry(KEY)


def test_authorization_arguments_are_mandatory(env):
    coordinator = env.coordinator(FakeWriter())
    with pytest.raises(TypeError):
        coordinator.authorize_retry(KEY, reason="no person")
    with pytest.raises(TypeError):
        coordinator.authorize_retry(KEY, authorized_by="no reason")


def test_authorizing_runs_nothing(env):
    writer = FakeWriter(Script.FAILED_BEFORE_COMMIT)
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    view = coordinator.authorize_retry(KEY, authorized_by="analyst", reason="confirmed transient")
    assert view.combined is C.AUTHORIZED and view.ledger_state is L.RETRY_AUTHORIZED
    assert_settled(env, ledger=L.RETRY_AUTHORIZED, journal=[J.NOT_COMMITTED], attempt=1, writer=writer, calls=1)
    event = env.ledger.get_history(KEY)[-1]
    assert event.actor == "analyst" and event.reason == "confirmed transient"


def test_path_a_a_non_commit_retries_on_the_same_record_with_the_ledgers_next_attempt(env):
    writer = FakeWriter(Script.FAILED_BEFORE_COMMIT, Script.COMMIT)
    coordinator = env.coordinator(writer)
    first = coordinator.submit(request())
    coordinator.authorize_retry(KEY, authorized_by="analyst", reason="confirmed transient")
    second = coordinator.retry(KEY)
    assert second.disposition is D.EXECUTED_COMMITTED and second.applied and second.ledger_attempt == 2
    assert second.journal_record_id == first.journal_record_id
    assert_settled(env, ledger=L.COMPLETED, journal=[J.COMMITTED], attempt=2, writer=writer, calls=2, effects=1)
    record = env.journal.find_by_correlation_key(KEY)[0]
    assert record.ledger_attempt == 2 and record.epoch == 1 and record.attempt_count == 2
    history = env.journal.get_history(record.record_id)
    assert [event.to_state for event in history] == [J.PENDING, J.IN_FLIGHT, J.NOT_COMMITTED, J.IN_FLIGHT, J.COMMITTED]
    retried = [event for event in history if event.cause == "RETRY"][0]
    assert retried.actor == "analyst" and retried.reason == "confirmed transient" and retried.ledger_attempt == 2


def test_the_retry_sends_the_stored_payload_exactly(env):
    payload = {"id": "E1", "v": dt.datetime(2026, 1, 1, 0, 0, 0, 5, tzinfo=dt.timezone.utc)}
    writer = FakeWriter(Script.FAILED_BEFORE_COMMIT, Script.COMMIT)
    coordinator = env.coordinator(writer)
    coordinator.submit(request(parameters=payload))
    coordinator.authorize_retry(KEY, authorized_by="analyst", reason="retry")
    coordinator.retry(KEY)
    assert writer.calls[0][1] == writer.calls[1][1]
    assert writer.calls[1][1][0][1] == payload


def test_one_authorization_allows_one_attempt(env):
    writer = FakeWriter(Script.FAILED_BEFORE_COMMIT, Script.FAILED_BEFORE_COMMIT)
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    coordinator.authorize_retry(KEY, authorized_by="analyst", reason="first")
    assert coordinator.retry(KEY).disposition is D.EXECUTED_NOT_COMMITTED
    with pytest.raises(NotAuthorized):
        coordinator.retry(KEY)
    assert_settled(env, ledger=L.NOT_COMPLETED, journal=[J.NOT_COMMITTED], attempt=2, writer=writer, calls=2)


def test_path_b_found_not_committed_needs_authorization_then_makes_a_linked_next_epoch(env):
    writer = FakeWriter(Script.AMBIGUOUS_NOT_APPLIED, Script.COMMIT)
    coordinator = env.coordinator(writer)
    first = coordinator.submit(request())
    view = coordinator.resolve(KEY, decision=Decision.FOUND_NOT_COMMITTED, resolved_by="analyst", reason="checked the graph")
    assert view.combined is C.AWAITING_AUTHORIZATION
    with pytest.raises(NotAuthorized):
        coordinator.retry(KEY)
    assert coordinator.submit(request()).disposition is D.AWAITING_AUTHORIZATION
    assert len(writer.calls) == 1

    coordinator.authorize_retry(KEY, authorized_by="lead", reason="reviewed the finding")
    result = coordinator.retry(KEY)
    assert result.disposition is D.EXECUTED_COMMITTED and result.ledger_attempt == 2
    assert result.journal_record_id != first.journal_record_id
    assert_settled(env, ledger=L.COMPLETED, journal=[J.RESOLVED, J.COMMITTED], attempt=2, writer=writer, calls=2, effects=1)
    old, new = env.journal.find_by_correlation_key(KEY)
    assert old.resolution is JournalResolution.FOUND_NOT_COMMITTED and old.ledger_attempt == 1
    assert (new.epoch, new.ledger_attempt, new.intent_sha256) == (2, 2, old.intent_sha256)
    created = env.journal.get_history(new.record_id)[0]
    assert created.actor == "lead" and created.reason == "reviewed the finding"
    assert [event.to_state for event in env.journal.get_history(old.record_id)] == [J.PENDING, J.IN_FLIGHT, J.AMBIGUOUS, J.RESOLVED]


def test_path_b_can_repeat_and_previous_attempts_stay_in_history(env):
    writer = FakeWriter(Script.AMBIGUOUS_NOT_APPLIED, Script.AMBIGUOUS_NOT_APPLIED, Script.COMMIT)
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    for round_number in (1, 2):
        coordinator.resolve(KEY, decision=Decision.FOUND_NOT_COMMITTED, resolved_by="analyst", reason=f"checked {round_number}")
        coordinator.authorize_retry(KEY, authorized_by="lead", reason=f"approved {round_number}")
        coordinator.retry(KEY)
    assert_settled(env, ledger=L.COMPLETED, journal=[J.RESOLVED, J.RESOLVED, J.COMMITTED], attempt=3, writer=writer, calls=3, effects=1)
    assert [record.ledger_attempt for record in env.journal.find_by_correlation_key(KEY)] == [1, 2, 3]
    assert [record.epoch for record in env.journal.find_by_correlation_key(KEY)] == [1, 2, 3]
    attempts = [event.attempt for event in env.ledger.get_history(KEY) if event.to_state is L.IN_PROGRESS]
    assert attempts == [1, 2, 3]


def test_found_committed_completes_without_running(env):
    writer = FakeWriter(Script.AMBIGUOUS_APPLIED)
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    view = coordinator.resolve(KEY, decision=Decision.FOUND_COMMITTED, resolved_by="analyst", reason="found in the graph")
    assert view.combined is C.COMPLETED
    assert coordinator.submit(request()).disposition is D.ALREADY_COMPLETED
    assert_settled(env, ledger=L.COMPLETED, journal=[J.RESOLVED], writer=writer, calls=1, effects=1)
    with pytest.raises(NotAuthorized):
        coordinator.retry(KEY)


def test_abandoned_is_final(env):
    writer = FakeWriter(Script.AMBIGUOUS_APPLIED)
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    view = coordinator.resolve(KEY, decision=Decision.ABANDONED, resolved_by="analyst", reason="no longer wanted")
    assert view.combined is C.ABANDONED
    assert coordinator.submit(request()).disposition is D.BLOCKED_ABANDONED
    with pytest.raises(NotAuthorized):
        coordinator.authorize_retry(KEY, authorized_by="analyst", reason="change of mind")
    with pytest.raises(NotAuthorized):
        coordinator.retry(KEY)
    assert_settled(env, ledger=L.ABANDONED, journal=[J.RESOLVED], writer=writer, calls=1)


@pytest.mark.parametrize("by, reason", [(None, "x"), ("", "x"), ("analyst", None), ("analyst", " ")])
def test_resolving_needs_a_named_person_and_a_reason(env, by, reason):
    writer = FakeWriter(Script.AMBIGUOUS_APPLIED)
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    with pytest.raises(InvalidRequest):
        coordinator.resolve(KEY, decision=Decision.FOUND_NOT_COMMITTED, resolved_by=by, reason=reason)
    assert_settled(env, ledger=L.UNCERTAIN, journal=[J.AMBIGUOUS], writer=writer, calls=1)


def test_a_different_decision_on_a_resolved_request_is_refused(env):
    coordinator = env.coordinator(FakeWriter(Script.AMBIGUOUS_APPLIED))
    coordinator.submit(request())
    coordinator.resolve(KEY, decision=Decision.FOUND_COMMITTED, resolved_by="analyst", reason="found")
    with pytest.raises(ResolutionConflict):
        coordinator.resolve(KEY, decision=Decision.FOUND_NOT_COMMITTED, resolved_by="analyst", reason="changed my mind")
    assert_settled(env, ledger=L.COMPLETED, journal=[J.RESOLVED])


def test_repeating_the_same_decision_changes_nothing(env):
    coordinator = env.coordinator(FakeWriter(Script.AMBIGUOUS_APPLIED))
    coordinator.submit(request())
    coordinator.resolve(KEY, decision=Decision.FOUND_COMMITTED, resolved_by="analyst", reason="found")
    history = env.ledger.get_history(KEY)
    assert coordinator.resolve(KEY, decision=Decision.FOUND_COMMITTED, resolved_by="analyst", reason="found").combined is C.COMPLETED
    assert env.ledger.get_history(KEY) == history


def test_a_request_that_isnt_uncertain_cant_be_resolved(env):
    coordinator = env.coordinator(FakeWriter())
    coordinator.submit(request())
    with pytest.raises(NotAuthorized):
        coordinator.resolve(KEY, decision=Decision.FOUND_NOT_COMMITTED, resolved_by="analyst", reason="nothing to resolve")
    assert_settled(env, ledger=L.COMPLETED, journal=[J.COMMITTED])


def test_a_half_done_resolution_can_be_finished_by_repeating_it(env):
    coordinator = env.coordinator(FakeWriter(Script.AMBIGUOUS_NOT_APPLIED))
    coordinator.submit(request())
    record = env.journal.find_by_correlation_key(KEY)[0]
    env.journal.resolve_ambiguous(record.record_id, resolution=JournalResolution.FOUND_NOT_COMMITTED, resolved_by="analyst", reason="found")
    assert coordinator.inspect(KEY).combined is C.RESOLUTION_INCOMPLETE
    with pytest.raises(ResolutionConflict):
        coordinator.resolve(KEY, decision=Decision.ABANDONED, resolved_by="analyst", reason="different")
    assert coordinator.resolve(KEY, decision=Decision.FOUND_NOT_COMMITTED, resolved_by="analyst", reason="found").combined is C.AWAITING_AUTHORIZATION


def test_a_stored_payload_that_doesnt_match_the_ledger_stops_the_retry(env):
    """Reached with a real, deliberately mismatched pair of stores: the journal's copy was never the ledger's."""
    writer = FakeWriter()
    statements = OPERATION.statements(PARAMETERS)
    env.ledger.register(KEY, OPERATION.name, statements)
    env.journal.create_intent(OPERATION.name, OPERATION.statements({"id": "E1", "v": 999}), correlation_key=KEY, epoch=1)
    with pytest.raises(PayloadMismatch):
        env.coordinator(writer).submit(request())
    assert writer.calls == []
    assert_settled(env, ledger=L.REGISTERED, journal=[J.PENDING], attempt=0)


# ================================================================ H. store failure

@pytest.fixture
def briefly(tmp_path):
    environment = Env(tmp_path, busy_timeout=0.05)
    yield environment
    environment.close()


class RealLock:
    """A real SQLite write lock held on a store's file by another connection."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.connection = sqlite3.connect(self.path, isolation_level=None)
        self.connection.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, *exc):
        self.connection.execute("ROLLBACK")
        self.connection.close()


def test_a_locked_ledger_holds_the_alert_and_runs_nothing(briefly):
    writer = FakeWriter()
    coordinator = briefly.coordinator(writer)
    with RealLock(briefly.ledger_path):
        result = coordinator.submit(request())
    assert result.disposition is D.NOT_DISPATCHED_UNAVAILABLE and result.must_hold and result.unavailable == "ledger"
    assert not result.applied and not result.writer_invoked and writer.calls == []
    assert briefly.journal.find_by_correlation_key(KEY) == ()
    assert coordinator.submit(request()).disposition is D.EXECUTED_COMMITTED  # held, then delivered again
    assert_settled(briefly, ledger=L.COMPLETED, journal=[J.COMMITTED], attempt=1, writer=writer, calls=1, effects=1)


def test_a_locked_journal_before_the_intent_holds_the_alert_and_a_resubmit_continues(briefly):
    writer = FakeWriter()
    coordinator = briefly.coordinator(writer)
    with RealLock(briefly.journal_path):
        result = coordinator.submit(request())
    assert result.disposition is D.NOT_DISPATCHED_UNAVAILABLE and result.must_hold and result.unavailable == "journal"
    assert writer.calls == []
    assert_settled(briefly, ledger=L.REGISTERED, journal=[], attempt=0, combined=C.REGISTERED_NO_INTENT)
    assert coordinator.submit(request()).disposition is D.EXECUTED_COMMITTED
    assert_settled(briefly, ledger=L.COMPLETED, journal=[J.COMMITTED], attempt=1, writer=writer, calls=1, effects=1)


def test_closed_stores_hold_the_alert_and_never_reach_the_writer(tmp_path):
    for close in ("journal", "ledger"):
        directory = tmp_path / close
        directory.mkdir()
        environment = Env(directory)
        writer = FakeWriter()
        coordinator = environment.coordinator(writer)
        getattr(environment, close).close()
        result = coordinator.submit(request())
        assert result.disposition is D.NOT_DISPATCHED_UNAVAILABLE and result.must_hold and result.unavailable == close
        assert writer.calls == [] and not result.applied
        environment.close()


@pytest.mark.parametrize("store, method, combined", [
    ("journal", "preflight_intent", None),
    ("ledger", "register", None),
    ("journal", "create_intent", C.REGISTERED_NO_INTENT),
    ("ledger", "begin_execution", C.REGISTERED),
])
def test_a_store_failure_at_each_step_before_the_claim_holds_the_alert_and_a_resubmit_continues(faulty, store, method, combined):
    getattr(faulty, f"{store}_faults")[method] = 1
    writer = FakeWriter()
    coordinator = faulty.coordinator(writer)
    result = coordinator.submit(request())
    assert result.disposition is D.NOT_DISPATCHED_UNAVAILABLE and result.must_hold and result.unavailable == store
    assert writer.calls == []
    if combined is not None:
        assert coordinator.inspect(KEY).combined is combined
    again = coordinator.submit(request())
    assert again.disposition is D.EXECUTED_COMMITTED and again.ledger_attempt == 1
    assert_settled(faulty, ledger=L.COMPLETED, journal=[J.COMMITTED], attempt=1, writer=writer, calls=1, effects=1)


def test_a_store_failure_on_retry_keeps_the_authorization_unused(faulty):
    writer = FakeWriter(Script.AMBIGUOUS_NOT_APPLIED, Script.COMMIT)
    coordinator = faulty.coordinator(writer)
    coordinator.submit(request())
    coordinator.resolve(KEY, decision=Decision.FOUND_NOT_COMMITTED, resolved_by="analyst", reason="found")
    coordinator.authorize_retry(KEY, authorized_by="lead", reason="approved")
    faulty.journal_faults["create_intent"] = 1
    result = coordinator.retry(KEY)
    assert result.disposition is D.NOT_DISPATCHED_UNAVAILABLE and result.must_hold and len(writer.calls) == 1
    assert_settled(faulty, ledger=L.RETRY_AUTHORIZED, journal=[J.RESOLVED], attempt=1)
    assert coordinator.retry(KEY).disposition is D.EXECUTED_COMMITTED
    assert_settled(faulty, ledger=L.COMPLETED, journal=[J.RESOLVED, J.COMMITTED], attempt=2, writer=writer, calls=2, effects=1)


def test_a_journal_failure_after_the_retrys_claim_is_established_as_not_run(faulty):
    writer = FakeWriter(Script.FAILED_BEFORE_COMMIT, Script.COMMIT)
    coordinator = faulty.coordinator(writer)
    coordinator.submit(request())
    coordinator.authorize_retry(KEY, authorized_by="lead", reason="approved")
    faulty.journal_faults["authorize_retry"] = 1
    result = coordinator.retry(KEY)
    assert result.disposition is D.NOT_DISPATCHED_UNAVAILABLE and result.finalized and result.ledger_attempt == 2 and len(writer.calls) == 1
    assert_settled(faulty, ledger=L.NOT_COMPLETED, journal=[J.NOT_COMMITTED], attempt=2, combined=C.AWAITING_AUTHORIZATION)
    with pytest.raises(NotAuthorized):
        coordinator.retry(KEY)  # the authorization was spent by the claim, so a person decides again
    coordinator.authorize_retry(KEY, authorized_by="lead", reason="approved again")
    assert coordinator.retry(KEY).disposition is D.EXECUTED_COMMITTED
    assert_settled(faulty, ledger=L.COMPLETED, journal=[J.COMMITTED], attempt=3, writer=writer, calls=2, effects=1)


def test_a_closed_health_gate_stores_nothing_and_runs_nothing(env):
    class Closed:
        def allow_dispatch(self):
            return False

    writer = FakeWriter()
    result = env.coordinator(writer, health=Closed()).submit(request())
    assert result.disposition is D.NOT_DISPATCHED_UNAVAILABLE and result.must_hold and result.unavailable == "neo4j"
    assert writer.calls == [] and env.journal.find_by_correlation_key(KEY) == ()
    assert env.ledger.check(KEY, OPERATION.name, OPERATION.statements(PARAMETERS)).request is None


@pytest.mark.parametrize("answer", [None, 1, "yes", "raise"])
def test_a_health_gate_only_opens_on_true(env, answer):
    class Gate:
        def allow_dispatch(self):
            if answer == "raise":
                raise RuntimeError("no idea")
            return answer

    writer = FakeWriter()
    assert env.coordinator(writer, health=Gate()).submit(request()).must_hold and writer.calls == []


def test_a_closed_health_gate_keeps_a_retrys_authorization(env):
    open_now = {"value": True}

    class Gate:
        def allow_dispatch(self):
            return open_now["value"]

    writer = FakeWriter(Script.FAILED_BEFORE_COMMIT)
    coordinator = env.coordinator(writer, health=Gate())
    coordinator.submit(request())
    coordinator.authorize_retry(KEY, authorized_by="lead", reason="approved")
    open_now["value"] = False
    assert coordinator.retry(KEY).must_hold
    assert_settled(env, ledger=L.RETRY_AUTHORIZED, journal=[J.NOT_COMMITTED], attempt=1, writer=writer, calls=1)


def test_a_damaged_journal_raises_before_anything_runs(env):
    writer = FakeWriter()
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    connection = sqlite3.connect(env.journal_path, isolation_level=None)
    connection.execute("UPDATE journal_records SET diagnostic = 'tampered'")
    connection.close()
    with pytest.raises(CoordinationInconsistency):
        coordinator.inspect(KEY)
    with pytest.raises(CoordinationInconsistency):
        coordinator.submit(request())
    assert len(writer.calls) == 1


def test_a_damaged_ledger_raises_before_anything_runs(env):
    writer = FakeWriter()
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    connection = sqlite3.connect(env.ledger_path, isolation_level=None)
    connection.execute("UPDATE ledger_requests SET diagnostic = 'tampered'")
    connection.close()
    with pytest.raises(CoordinationInconsistency):
        coordinator.submit(request())
    assert len(writer.calls) == 1


# ================================================================ I. identity

@pytest.mark.parametrize("source_key, reason", [
    (None, IdentityRefusal.MISSING),
    ("", IdentityRefusal.MISSING),
    (7, IdentityRefusal.NOT_A_STRING),
    (b"wazuh:m/1.2", IdentityRefusal.NOT_A_STRING),
    (["wazuh:m/1.2"], IdentityRefusal.NOT_A_STRING),
    ("not-a-key", IdentityRefusal.MALFORMED),
    ("wazuh:", IdentityRefusal.MALFORMED),
    ("wazuh:manager", IdentityRefusal.MALFORMED),
    ("wazuh:/1.2", IdentityRefusal.MALFORMED),
    ("wazuh:manager/", IdentityRefusal.MALFORMED),
    (" wazuh:manager/1.2", IdentityRefusal.MALFORMED),
    ("splunk:manager/1.2", IdentityRefusal.MALFORMED),
    ("wazuh:manager/abc", IdentityRefusal.INVALID),
    ("wazuh:manager/1", IdentityRefusal.INVALID),
    ("wazuh:manager/1.2 ", IdentityRefusal.INVALID),
    ("wazuh:manager/1.2\n", IdentityRefusal.INVALID),
    ("wazuh:manager/1.2|other", IdentityRefusal.INVALID),
    ("wazuh:a/b/1.2", IdentityRefusal.INVALID),      # an unencoded slash: the listener never writes this
    ("wazuh:man ager/1.2", IdentityRefusal.INVALID),  # an unencoded space
    ("wazuh:man%20ager/1.2x", IdentityRefusal.INVALID),
    ("wazuh:%ff/1.2", IdentityRefusal.MALFORMED),
    ("wazuh:manager/" + "1" * 64 + ".2", IdentityRefusal.INVALID),
])
def test_an_unusable_identity_is_refused_with_its_reason_and_nothing_runs_or_is_stored(env, source_key, reason):
    writer = FakeWriter()
    coordinator = env.coordinator(writer)
    result = coordinator.submit(WriteRequest(source_key, OPERATION.name, dict(PARAMETERS)))
    assert result.disposition is D.REFUSED_IDENTITY and result.refusal is reason
    assert not result.applied and not result.writer_invoked and not result.executed and not result.must_hold and result.key_ref is None
    assert writer.calls == [] and env.ledger.list_requests() == () and env.journal.list_records() == ()
    assert coordinator.counters()[f"identity_refusal.{reason.value}"] == 1


def test_each_refusal_reason_is_counted_separately(env):
    coordinator = env.coordinator(FakeWriter())
    for source_key in (None, None, 7, "nope", "wazuh:m/abc"):
        coordinator.submit(WriteRequest(source_key, OPERATION.name, dict(PARAMETERS)))
    counts = coordinator.counters()
    assert counts["identity_refusal.MISSING"] == 2 and counts["identity_refusal.NOT_A_STRING"] == 1
    assert counts["identity_refusal.MALFORMED"] == 1 and counts["identity_refusal.INVALID"] == 1
    assert counts["disposition.REFUSED_IDENTITY"] == 5


def test_a_refusal_is_logged_with_its_reason_and_not_the_value(env, caplog):
    with caplog.at_level(logging.WARNING, logger="neo4j_write_coordinator"):
        env.coordinator(FakeWriter()).submit(WriteRequest("SECRET-CANARY-identity", OPERATION.name, dict(PARAMETERS)))
    assert "MALFORMED" in caplog.text and "SECRET-CANARY" not in caplog.text


@pytest.mark.parametrize("alert", [
    {"id": "1700000000.1234", "manager": {"name": "manager-01"}},
    {"id": "1.2", "manager": {"name": "host/with:odd chars|and%signs"}},
    {"id": "1700000000.1234", "manager": {"name": "élève"}},
])
def test_what_the_listener_produces_is_accepted(env, alert):
    source_key = extract_source_identity(alert).source_key
    assert check_source_key(source_key) is None
    assert env.coordinator(FakeWriter()).submit(WriteRequest(source_key, OPERATION.name, dict(PARAMETERS))).disposition is D.EXECUTED_COMMITTED


@pytest.mark.parametrize("alert", [{}, {"id": "1.2"}, {"id": "bad", "manager": {"name": "m"}}, {"id": 1.2, "manager": {"name": "m"}}])
def test_an_alert_the_listener_gives_no_identity_is_refused(env, alert):
    writer = FakeWriter()
    source_key = extract_source_identity(alert).source_key
    assert env.coordinator(writer).submit(WriteRequest(source_key, OPERATION.name, dict(PARAMETERS))).refusal is IdentityRefusal.MISSING
    assert writer.calls == []


def test_a_refused_identity_never_reaches_any_writer_including_another_route(env):
    """There's no second route: the only callable the coordinator holds is the one it was given."""
    routes = {name: value for name, value in vars(env.coordinator(FakeWriter())).items() if callable(value)}
    assert list(routes) == ["_writer"]


# ================================================================ validation

@pytest.mark.parametrize("operation", ["unregistered", "", None, 7, "test_event.update|x"])
def test_an_unregistered_operation_is_refused(env, operation):
    writer = FakeWriter()
    with pytest.raises(InvalidRequest):
        env.coordinator(writer).submit(WriteRequest(SOURCE, operation, dict(PARAMETERS)))
    assert writer.calls == [] and env.ledger.list_requests() == ()


@pytest.mark.parametrize("parameters", [{}, {"id": "E1"}, {"id": "E1", "v": 1, "extra": 2}, None, [("id", 1)], "x"])
def test_parameters_must_match_the_registered_names_exactly(env, parameters):
    writer = FakeWriter()
    with pytest.raises(InvalidRequest):
        env.coordinator(writer).submit(WriteRequest(SOURCE, OPERATION.name, parameters))
    assert writer.calls == [] and env.ledger.list_requests() == ()


@pytest.mark.parametrize("value", [object(), {1, 2}, 2 ** 70])
def test_a_value_that_cant_be_stored_exactly_registers_nothing(env, value):
    writer = FakeWriter()
    with pytest.raises(PayloadNotJournalable):
        env.coordinator(writer).submit(request(parameters={"id": "E1", "v": value}))
    assert writer.calls == [] and env.ledger.list_requests() == () and env.journal.list_records() == ()


def test_a_value_only_the_ledger_refuses_registers_nothing_and_isnt_called_damage(tmp_path):
    from neo4j_idempotency_ledger import UnsupportedPayloadError

    class Refusing(IdempotencyLedger):
        def register(self, *args, **kwargs):
            raise UnsupportedPayloadError("no canonical form")

    environment = Env(tmp_path, ledger_class=Refusing)
    writer = FakeWriter()
    with pytest.raises(PayloadNotJournalable):
        environment.coordinator(writer).submit(request())
    assert writer.calls == [] and environment.journal.list_records() == ()
    environment.close()


def test_submit_takes_only_a_write_request(env):
    with pytest.raises(InvalidRequest):
        env.coordinator(FakeWriter()).submit({"source_key": SOURCE})


@pytest.mark.parametrize("name", ["", "has space", "a|b", "a/b", None])
def test_a_registered_operation_needs_a_plain_name(name):
    with pytest.raises(ValueError):
        RegisteredOperation(name, "RETURN 1", frozenset())


def test_a_registry_refuses_a_duplicate_name():
    with pytest.raises(ValueError):
        OperationRegistry([OPERATION, OPERATION])


@pytest.mark.parametrize("arguments", [
    {"journal": None}, {"ledger": None}, {"writer": None}, {"registry": None}, {"store_write_attempts": 0},
])
def test_the_constructor_refuses_what_it_cant_use(env, arguments):
    options = {"journal": env.journal, "ledger": env.ledger, "writer": FakeWriter(), "registry": REGISTRY, **arguments}
    with pytest.raises((TypeError, ValueError)):
        WriteCoordinator(options.pop("journal"), options.pop("ledger"), **options)


def test_the_writer_has_no_default():
    signature = inspect.signature(WriteCoordinator.__init__)
    assert signature.parameters["writer"].default is inspect.Parameter.empty
    assert signature.parameters["registry"].default is inspect.Parameter.empty


def test_the_coordinator_opens_and_closes_no_store(env):
    env.coordinator(FakeWriter()).submit(request())
    assert env.journal.find_by_correlation_key(KEY) and env.ledger.get_request(KEY)  # both still open
    tree = ast.parse((BACKEND / "neo4j_write_coordinator.py").read_text(encoding="utf-8"))
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not [func for func in calls if isinstance(func, ast.Attribute) and func.attr == "close"]
    assert not [func for func in calls if isinstance(func, ast.Name) and func.id in ("NeoWriteJournal", "IdempotencyLedger")]


# ================================================================ separate instances and threads

def test_separate_instances_on_the_same_files_share_the_outcome(env, restart):
    first = FakeWriter()
    second = FakeWriter()
    env.coordinator(first).submit(request())
    result = restart(second).submit(request())
    assert result.disposition is D.ALREADY_COMPLETED and second.calls == []
    assert len(first.effects) == 1


def test_concurrent_same_key_runs_once_across_threads_on_one_instance(env):
    writer = FakeWriter()
    coordinator = env.coordinator(writer)
    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(coordinator.submit(request()))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    assert len(writer.effects) == 1
    executed = [result for result in results if result.disposition is D.EXECUTED_COMMITTED]
    assert len(executed) == 1 and len(results) == 8
    assert {result.disposition for result in results} <= {D.EXECUTED_COMMITTED, D.ALREADY_COMPLETED, D.IN_PROGRESS_ELSEWHERE}
    assert_settled(env, ledger=L.COMPLETED, journal=[J.COMMITTED], attempt=1)


def test_concurrent_same_key_runs_once_across_instances(env):
    effects = []
    lock = threading.Lock()

    def effect(operation, statements):
        with lock:
            effects.append(operation)

    coordinators = []
    for _ in range(6):
        coordinators.append(env.coordinator(FakeWriter(effect=effect), stores=env.open_stores()))
    results = []
    barrier = threading.Barrier(len(coordinators))

    def worker(coordinator):
        barrier.wait()
        results.append(coordinator.submit(request()))

    threads = [threading.Thread(target=worker, args=(coordinator,)) for coordinator in coordinators]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    assert len(effects) == 1 and len(results) == 6
    assert sum(result.disposition is D.EXECUTED_COMMITTED for result in results) == 1
    assert_settled(env, ledger=L.COMPLETED, journal=[J.COMMITTED], attempt=1)


def test_a_retry_that_read_the_ledger_just_before_another_retry_finished_isnt_called_damage(tmp_path):
    """Deterministic: the other retry runs exactly between this one's read of the ledger and of the journal."""
    interrupt = {}

    class Racing(NeoWriteJournal):
        def find_by_correlation_key(self, key):
            action = interrupt.pop("run", None)
            if action is not None:
                action()
            return super().find_by_correlation_key(key)

    environment = Env(tmp_path, journal_class=Racing)
    first = FakeWriter(Script.FAILED_BEFORE_COMMIT)
    environment.coordinator(first).submit(request())
    environment.coordinator(first).authorize_retry(KEY, authorized_by="lead", reason="approved")
    other = FakeWriter()
    other_coordinator = environment.coordinator(other, stores=environment.open_stores(journal_class=NeoWriteJournal))
    late = FakeWriter()
    interrupt["run"] = lambda: other_coordinator.retry(KEY)
    result = environment.coordinator(late).retry(KEY)
    assert result.disposition is D.ALREADY_COMPLETED and late.calls == [] and len(other.effects) == 1
    assert_settled(environment, ledger=L.COMPLETED, journal=[J.COMMITTED], attempt=2)
    environment.close()


def test_a_resolution_race_lost_at_either_store_is_a_conflict_not_a_raw_store_error(tmp_path):
    from neo4j_write_coordinator import ResolutionConflict as Conflict

    class Racing(IdempotencyLedger):
        def resolve_uncertain(self, key, **kwargs):
            super().resolve_uncertain(key, **kwargs)  # someone else got there first
            return super().resolve_uncertain(key, **kwargs)

    environment = Env(tmp_path, ledger_class=Racing)
    coordinator = environment.coordinator(FakeWriter(Script.AMBIGUOUS_APPLIED))
    coordinator.submit(request())
    with pytest.raises(Conflict):
        coordinator.resolve(KEY, decision=Decision.ABANDONED, resolved_by="analyst", reason="x")
    environment.close()


def test_concurrent_retries_of_one_authorization_run_once(env):
    calls = []
    lock = threading.Lock()

    def effect(operation, statements):
        with lock:
            calls.append(operation)

    first_writer = FakeWriter(Script.AMBIGUOUS_NOT_APPLIED)
    coordinator = env.coordinator(first_writer)
    coordinator.submit(request())
    coordinator.resolve(KEY, decision=Decision.FOUND_NOT_COMMITTED, resolved_by="analyst", reason="found")
    coordinator.authorize_retry(KEY, authorized_by="lead", reason="approved")
    coordinators = [env.coordinator(FakeWriter(effect=effect), stores=env.open_stores()) for _ in range(5)]
    results = []
    errors = []
    barrier = threading.Barrier(len(coordinators))

    def worker(one):
        barrier.wait()
        try:
            results.append(one.retry(KEY))
        except NotAuthorized as error:
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(one,)) for one in coordinators]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    assert len(calls) == 1
    assert sum(result.disposition is D.EXECUTED_COMMITTED for result in results) == 1
    assert_settled(env, ledger=L.COMPLETED, journal=[J.RESOLVED, J.COMMITTED], attempt=2)


# ================================================================ recovery, in one process

def test_recovery_on_clean_stores_reports_nothing_and_runs_nothing(env):
    writer = FakeWriter()
    result = env.coordinator(writer).recover()
    assert result.counts() == {} and result.integrity_ok and writer.calls == []


def test_recovery_never_calls_the_writer_and_repeating_it_changes_nothing(faulty):
    faulty.ledger_faults["record_completed"] = 1
    writer = FakeWriter()
    coordinator = faulty.coordinator(writer, store_write_attempts=1)
    coordinator.submit(request())
    assert len(writer.calls) == 1
    first = coordinator.recover()
    assert first.of(R.RECONCILED) == (KEY,)
    assert_settled(faulty, ledger=L.COMPLETED, journal=[J.COMMITTED], attempt=1, writer=writer, calls=1, effects=1)
    history = faulty.ledger.get_history(KEY)
    assert coordinator.recover().of(R.RECONCILED) == ()
    assert faulty.ledger.get_history(KEY) == history and len(writer.calls) == 1


@pytest.mark.parametrize("step, ledger, journal, category", [
    (Script.FAILED_BEFORE_COMMIT, L.NOT_COMPLETED, J.NOT_COMMITTED, R.RECONCILED),
    (Script.AMBIGUOUS_APPLIED, L.UNCERTAIN, J.AMBIGUOUS, R.NEWLY_UNCERTAIN),  # needs a person, so it isn't "reconciled"
])
def test_recovery_finishes_the_ledger_from_the_journals_outcome(faulty, step, ledger, journal, category):
    method = "record_not_completed" if ledger is L.NOT_COMPLETED else "record_uncertain"
    faulty.ledger_faults[method] = 1
    writer = FakeWriter(step)
    coordinator = faulty.coordinator(writer, store_write_attempts=1)
    assert not coordinator.submit(request()).finalized
    found = coordinator.recover()
    assert found.of(category) == (KEY,)
    if category is R.NEWLY_UNCERTAIN:
        assert found.of(R.RECONCILED) == () and found.of(R.UNCERTAIN) == ()
    assert_settled(faulty, ledger=ledger, journal=[journal], attempt=1, writer=writer, calls=1)


def test_recovery_reports_each_waiting_state(env):
    writer = FakeWriter(Script.FAILED_BEFORE_COMMIT, Script.AMBIGUOUS_APPLIED, Script.FAILED_BEFORE_COMMIT)
    coordinator = env.coordinator(writer)
    keys = {}
    for name, source in (("not_completed", "1.1"), ("uncertain", "1.2"), ("authorized", "1.3")):
        keys[name] = request_key(f"wazuh:m/{source}", OPERATION.name)
        coordinator.submit(request(f"wazuh:m/{source}"))
    coordinator.authorize_retry(keys["authorized"], authorized_by="lead", reason="approved")
    env.ledger.register(request_key("wazuh:m/1.4", OPERATION.name), OPERATION.name, OPERATION.statements(PARAMETERS))
    found = coordinator.recover()
    assert found.of(R.AWAITING_AUTHORIZATION) == (keys["not_completed"],)
    assert found.of(R.UNCERTAIN) == (keys["uncertain"],)
    assert found.of(R.RETRY_AUTHORIZED) == (keys["authorized"],)
    assert found.of(R.REGISTERED_NO_INTENT) == (request_key("wazuh:m/1.4", OPERATION.name),)
    assert len(writer.calls) == 3


def test_recovery_leaves_this_instances_own_running_request_alone(env):
    seen = []
    coordinator = None

    def during():
        seen.append(coordinator.recover())

    writer = FakeWriter(during=during)
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    assert seen[0].of(R.STILL_RUNNING) == (KEY,)
    assert_settled(env, ledger=L.COMPLETED, journal=[J.COMMITTED], writer=writer, calls=1, effects=1)


def test_recovery_reports_a_journal_record_with_no_ledger_request(env):
    env.journal.create_intent(OPERATION.name, OPERATION.statements(PARAMETERS), correlation_key=KEY, epoch=1)
    found = env.coordinator(FakeWriter()).recover()
    assert found.of(R.INCONSISTENT) == (KEY,)


def test_recovery_excludes_and_reports_damaged_rows_and_never_repairs_them(env):
    writer = FakeWriter()
    coordinator = env.coordinator(writer)
    coordinator.submit(request())
    connection = sqlite3.connect(env.journal_path, isolation_level=None)
    connection.execute("UPDATE journal_records SET diagnostic = 'tampered'")
    connection.close()
    found = coordinator.recover()
    assert not found.integrity_ok and len(found.of(R.CORRUPT)) == 1
    assert len(writer.calls) == 1


def test_the_recovery_result_prints_only_counts(env):
    coordinator = env.coordinator(FakeWriter(Script.AMBIGUOUS_APPLIED))
    coordinator.submit(request())
    text = repr(coordinator.recover())
    assert "UNCERTAIN" in text and "manager-01" not in text and "wazuh:" not in text


def test_recovery_with_a_store_down_raises_and_concludes_nothing(tmp_path):
    from neo4j_write_coordinator import JournalUnavailable, LedgerUnavailable
    for close, error in (("journal", JournalUnavailable), ("ledger", LedgerUnavailable)):
        directory = tmp_path / close
        directory.mkdir()
        environment = Env(directory)
        coordinator = environment.coordinator(FakeWriter())
        getattr(environment, close).close()
        with pytest.raises(error):
            coordinator.recover()
        environment.close()


# ================================================================ combined state, every pairing

class _Req:
    def __init__(self, state, attempt):
        self.state, self.attempt_count = state, attempt


class _Rec:
    def __init__(self, state, tag=None, resolution=None):
        self.state, self.ledger_attempt, self.resolution = state, tag, resolution


LEGAL = {
    (L.REGISTERED, J.PENDING, None, None): C.REGISTERED,
    (L.IN_PROGRESS, J.PENDING, None, None): C.CLAIMED,
    (L.IN_PROGRESS, J.NOT_COMMITTED, 1, None): C.CLAIMED,
    (L.IN_PROGRESS, J.IN_FLIGHT, 2, None): C.DISPATCHED,
    (L.IN_PROGRESS, J.COMMITTED, 2, None): C.OUTCOME_NOT_MIRRORED,
    (L.IN_PROGRESS, J.NOT_COMMITTED, 2, None): C.OUTCOME_NOT_MIRRORED,
    (L.IN_PROGRESS, J.AMBIGUOUS, 2, None): C.OUTCOME_NOT_MIRRORED,
    (L.COMPLETED, J.COMMITTED, 2, None): C.COMPLETED,
    (L.COMPLETED, J.RESOLVED, 2, JournalResolution.FOUND_COMMITTED): C.COMPLETED,
    (L.NOT_COMPLETED, J.PENDING, None, None): C.AWAITING_AUTHORIZATION,
    (L.NOT_COMPLETED, J.NOT_COMMITTED, 1, None): C.AWAITING_AUTHORIZATION,  # a released retry claim
    (L.NOT_COMPLETED, J.NOT_COMMITTED, 2, None): C.AWAITING_AUTHORIZATION,
    (L.NOT_COMPLETED, J.RESOLVED, 2, JournalResolution.FOUND_NOT_COMMITTED): C.AWAITING_AUTHORIZATION,
    (L.RETRY_AUTHORIZED, J.PENDING, None, None): C.AUTHORIZED,
    (L.RETRY_AUTHORIZED, J.NOT_COMMITTED, 1, None): C.AUTHORIZED,
    (L.RETRY_AUTHORIZED, J.NOT_COMMITTED, 2, None): C.AUTHORIZED,
    (L.RETRY_AUTHORIZED, J.RESOLVED, 2, JournalResolution.FOUND_NOT_COMMITTED): C.AUTHORIZED,
    (L.UNCERTAIN, J.AMBIGUOUS, 2, None): C.UNCERTAIN,
    (L.UNCERTAIN, J.PENDING, None, None): C.CLAIMED_NOT_DISPATCHED,
    (L.UNCERTAIN, J.NOT_COMMITTED, 1, None): C.CLAIMED_NOT_DISPATCHED,
    (L.UNCERTAIN, J.RESOLVED, 2, JournalResolution.FOUND_NOT_COMMITTED): C.RESOLUTION_INCOMPLETE,
    (L.UNCERTAIN, J.RESOLVED, 2, JournalResolution.FOUND_COMMITTED): C.RESOLUTION_INCOMPLETE,
    (L.UNCERTAIN, J.RESOLVED, 2, JournalResolution.ABANDONED): C.RESOLUTION_INCOMPLETE,
    (L.ABANDONED, J.PENDING, None, None): C.ABANDONED,
    (L.ABANDONED, J.NOT_COMMITTED, 1, None): C.ABANDONED,
    (L.ABANDONED, J.RESOLVED, 2, JournalResolution.ABANDONED): C.ABANDONED,
}


def test_every_ledger_and_journal_pairing_is_legal_only_where_the_table_says():
    """An attempt of 2 in the ledger, against a journal tagged with an earlier attempt, the same one, a
    later one, or none. Mocks are fine here: combined_state is pure."""
    checked = 0
    for ledger in L:
        attempt = 0 if ledger is L.REGISTERED else 2
        for journal in J:
            for tag in (None, 1, 2, 3):
                for resolution in ((None,) if journal is not J.RESOLVED else tuple(JournalResolution)):
                    wanted = LEGAL.get((ledger, journal, tag, resolution), C.INCONSISTENT)
                    assert combined_state(_Req(ledger, attempt), [_Rec(journal, tag, resolution)]) is wanted, (ledger, journal, tag, resolution)
                    checked += 1
    assert checked == 7 * (5 * 4 + 3 * 4)


def test_no_journal_record_is_legal_only_when_registered():
    for ledger in L:
        wanted = C.REGISTERED_NO_INTENT if ledger is L.REGISTERED else C.INCONSISTENT
        assert combined_state(_Req(ledger, 0), []) is wanted
    assert combined_state(None, []) is C.NOT_FOUND
    assert combined_state(None, [_Rec(J.PENDING)]) is C.INCONSISTENT


def test_inspect_of_an_unknown_key_says_so(env):
    view = env.coordinator(FakeWriter()).inspect(KEY)
    assert view.combined is C.NOT_FOUND and view.ledger_state is None and view.consistent


# ================================================================ secrets and logs

def test_no_key_or_value_reaches_a_log_a_result_or_a_store_detail(env, caplog):
    source = "wazuh:SECRET-CANARY-manager/1.2"
    parameters = {"id": "SECRET-CANARY-value", "v": 1}
    key = request_key(source, OPERATION.name)
    coordinator = env.coordinator(FakeWriter(Script.FAILED_BEFORE_COMMIT, Script.AMBIGUOUS_APPLIED, Script.RAISES))
    with caplog.at_level(logging.DEBUG):
        results = [coordinator.submit(WriteRequest(source, OPERATION.name, parameters))]
        results.append(coordinator.submit(WriteRequest(source, OPERATION.name, {"id": "SECRET-CANARY-other", "v": 2})))
        coordinator.authorize_retry(key, authorized_by="analyst", reason="retry")
        results.append(coordinator.retry(key))
        coordinator.resolve(key, decision=Decision.FOUND_NOT_COMMITTED, resolved_by="analyst", reason="checked")
        coordinator.authorize_retry(key, authorized_by="analyst", reason="retry")
        results.append(coordinator.retry(key))
        recovered = coordinator.recover()
    assert "SECRET-CANARY" not in caplog.text
    assert "SECRET-CANARY" not in repr(results) and "SECRET-CANARY" not in repr(recovered)
    assert "SECRET-CANARY" not in repr(WriteRequest(source, OPERATION.name, parameters))
    ledger_row = env.ledger.get_request(key)
    assert "SECRET-CANARY" not in " ".join(str(value) for value in (ledger_row.error_type, ledger_row.error_code, ledger_row.diagnostic))
    for record in env.journal.find_by_correlation_key(key):
        assert "SECRET-CANARY" not in " ".join(str(value) for value in (record.error_type, record.error_code, record.diagnostic))


# ================================================================ the boundary with later packages

def _coordinator_source():
    return (BACKEND / "neo4j_write_coordinator.py").read_text(encoding="utf-8")


def _imported_modules(source):
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.add((node.module or "").split(".")[0])
    return found


def test_the_coordinator_imports_no_neo4j_client_driver_or_config():
    imported = _imported_modules(_coordinator_source())
    assert not imported & {"neo4j", "neo4j_client", "config", "dashboard_api", "realtime_socgraph"}
    assert {"neo4j_write_journal", "neo4j_idempotency_ledger"} <= imported


def test_the_coordinator_has_no_unprotected_write_path():
    source = _coordinator_source()
    tree = ast.parse(source)
    called = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not called & {"run", "execute_write", "write_statements", "write_query", "write_query_rows", "session"}
    writer_calls = [node for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_writer"]
    assert len(writer_calls) == 1, "the writer has exactly one call site"


def test_no_marker_or_later_package_behaviour_was_improvised():
    source = _coordinator_source()
    code = "\n".join(line for line in source.split('"""', 2)[2].splitlines() if not line.lstrip().startswith("#"))
    for word in ("AppliedWrite", "CREATE CONSTRAINT", "investigation_payload", "SHOW CONSTRAINTS"):
        assert word not in code


def test_no_production_module_imports_the_coordinator_yet():
    """Using the journal or ledger through the coordinator would be integration, which WP-C1 doesn't include."""
    users = []
    for source in BACKEND.rglob("*.py"):
        if "tests" in source.parts or source.name == "neo4j_write_coordinator.py" or "__pycache__" in source.parts:
            continue
        if "neo4j_write_coordinator" in source.read_text(encoding="utf-8", errors="replace"):
            users.append(str(source))
    assert users == []


def test_the_coordinator_registers_no_production_operation():
    tree = ast.parse(_coordinator_source())
    built = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("RegisteredOperation", "OperationRegistry")]
    assert built == []


def test_the_fake_matches_the_real_helpers_boundary():
    """Only the shape. It doesn't show the fake behaves like Neo4j."""
    import dataclasses

    from neo4j_client import WriteOutcome, WriteResult, write_statements
    assert {outcome.value for outcome in WriteOutcome} == {outcome.value for outcome in FakeOutcome}
    assert [f.name for f in dataclasses.fields(WriteResult)] == [f.name for f in dataclasses.fields(FakeWriteResult)]
    parameters = inspect.signature(write_statements).parameters
    assert list(parameters)[:2] == ["operation", "statements"]
    assert parameters["limits"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_real_write_result_is_read_the_same_way(env):
    """A real WriteResult, built by hand: nothing connects. Its exception text must go nowhere."""
    from neo4j_client import WriteOutcome, WriteResult

    def writer(operation, statements, *, limits):
        return WriteResult(operation, WriteOutcome.NOT_COMMITTED, 0, 0.0, exception_type="ServiceUnavailable",
                           diagnostic="ServiceUnavailable: failure before any query was submitted",
                           exception=RuntimeError("SECRET-CANARY"))

    result = env.coordinator(writer).submit(request())
    assert result.disposition is D.NOT_EXECUTED_CONFIRMED
    record = env.journal.find_by_correlation_key(KEY)[0]
    assert record.error_type == "ServiceUnavailable" and "SECRET-CANARY" not in repr(result)


def test_results_name_a_request_the_way_the_ledger_does(env):
    coordinator = env.coordinator(FakeWriter())
    result = coordinator.submit(request())
    assert isinstance(result, CoordinatedResult)
    assert result.key_ref == env.ledger.get_request(KEY).key_ref == coordinator.inspect(KEY).key_ref

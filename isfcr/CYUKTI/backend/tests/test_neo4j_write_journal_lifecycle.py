"""D4's crash windows and AC-3 across processes: a child really dies by SIGKILL at each durable step, using the real
journal and the real ledger on separate files, and a fresh process looks at what's left. No distributed transaction is
simulated: these stores commit separately, and each window is the gap between two of those commits.

Nothing here executes anything against a database. The test code stands in for a future coordinator's order of calls,
to show the journal supports it."""
from __future__ import annotations

import json
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import neo4j_idempotency_ledger as ledger_module
from neo4j_idempotency_ledger import IdempotencyLedger, LedgerState as L
from neo4j_write_journal import (
    DuplicateCorrelationError,
    InvalidTransitionError,
    JournalState as J,
    NeoWriteJournal,
    PreflightCode as P,
    Resolution,
    StaleAttemptError,
)
from tests._neo4j_child_tempdirs import child_processes

CHILD = str(Path(__file__).resolve().parent / "_neo4j_journal_lifecycle_child.py")
KEY = "cyukti.v1|wazuh:manager-01/1773199644.1234|duplicate_event.update"
OP = "duplicate_event.update"


@pytest.fixture
def env(tmp_path):
    with child_processes(str(tmp_path)) as children:
        yield {"journal": str(tmp_path / "journal.db"), "ledger": str(tmp_path / "ledger.db"), "cwd": str(tmp_path), "children": children, "tmp": tmp_path}


def _die_at(env, action):
    result = env["children"].run([sys.executable, CHILD, action, env["journal"], env["ledger"]], capture_output=True, text=True, timeout=60, cwd=env["cwd"])
    assert result.returncode == -signal.SIGKILL, (action, result.returncode, result.stderr[-800:])
    lines = result.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def _stores(env):
    return NeoWriteJournal(env["journal"]), IdempotencyLedger(env["ledger"])


def _states(journal, ledger):
    return [r.state for r in journal.find_by_correlation_key(KEY)], ledger.get_request(KEY).state, ledger.get_request(KEY).attempt_count


def _claim_is_refused(ledger):
    fingerprint = ledger.get_request(KEY).fingerprint
    with pytest.raises(ledger_module.InvalidTransitionError):
        ledger.begin_execution(KEY, fingerprint)


def _resolve_ledger(ledger):
    ledger.resolve_uncertain(KEY, resolution=ledger_module.Resolution.FOUND_NOT_COMPLETED, resolved_by="analyst", reason="evidence")


def _authorize_and_claim(ledger):
    ledger.authorize_retry(KEY, authorized_by="operator", reason="retry")
    return ledger.begin_execution(KEY, ledger.get_request(KEY).fingerprint).attempt_count


def _next_epoch(journal, epoch=2):
    chain = journal.find_by_correlation_key(KEY)
    return journal.create_intent(OP, chain[-1].statements, correlation_key=KEY, epoch=epoch, authorized_by="operator", reason="retry")


def _sound(journal, ledger):
    assert journal.check_integrity().ok
    assert ledger.check_integrity().ok


# ================================================================ the seven crash windows of D4

def test_window_1_journal_resolved_ledger_not(env):
    _die_at(env, "journal_resolved")
    journal, ledger = _stores(env)
    with journal, ledger:
        journal.recover_unfinished(), ledger.recover_unfinished()
        assert _states(journal, ledger) == ([J.RESOLVED], L.UNCERTAIN, 1)
        _claim_is_refused(ledger)
        first = journal.find_by_correlation_key(KEY)[0]
        with pytest.raises(InvalidTransitionError):
            journal.resolve_ambiguous(first.record_id, resolution=Resolution.FOUND_COMMITTED, resolved_by="other", reason="different")
        assert journal.find_by_correlation_key(KEY)[0].resolution is Resolution.FOUND_NOT_COMMITTED, "the first decision stands"
        _resolve_ledger(ledger)
        assert ledger.get_request(KEY).state is L.NOT_COMPLETED
        _claim_is_refused(ledger)
        _sound(journal, ledger)


def test_window_2_authorized_with_no_new_intent_yet(env):
    _die_at(env, "authorized")
    journal, ledger = _stores(env)
    with journal, ledger:
        journal.recover_unfinished(), ledger.recover_unfinished()
        assert _states(journal, ledger) == ([J.RESOLVED], L.RETRY_AUTHORIZED, 1)
        assert len(journal.find_by_correlation_key(KEY)) == 1, "the crash created no intent, and recovery doesn't"
        second = _next_epoch(journal)
        attempt = ledger.begin_execution(KEY, ledger.get_request(KEY).fingerprint).attempt_count
        assert journal.mark_in_flight(second.record_id, ledger_attempt=attempt).ledger_attempt == 2
        journal.mark_committed(second.record_id, expected_ledger_attempt=2)
        ledger.record_completed(KEY, attempt=2)
        assert _states(journal, ledger) == ([J.RESOLVED, J.COMMITTED], L.COMPLETED, 2)
        _sound(journal, ledger)


def test_window_3_new_intent_with_no_claim_is_reused_and_not_duplicated(env):
    """Also: a crash between intent creation and the next ledger step."""
    _die_at(env, "new_intent")
    journal, ledger = _stores(env)
    with journal, ledger:
        report = journal.recover_unfinished()
        ledger.recover_unfinished()
        assert _states(journal, ledger) == ([J.RESOLVED, J.PENDING], L.RETRY_AUTHORIZED, 1)
        second = journal.find_by_correlation_key(KEY)[1]
        assert report.pending == (second.record_id,), "reported, and not run"
        with pytest.raises(DuplicateCorrelationError) as caught:
            _next_epoch(journal)
        assert caught.value.record_id == second.record_id, "whoever comes next finds it"
        first = journal.find_by_correlation_key(KEY)[0]
        answer = journal.preflight_intent(OP, first.statements, correlation_key=KEY, epoch=2, authorized_by="o", reason="r")
        assert (answer.code, answer.existing_record_id, answer.intent_sha256) == (P.DUPLICATE_CORRELATION, second.record_id, second.intent_sha256)
        attempt = ledger.begin_execution(KEY, ledger.get_request(KEY).fingerprint).attempt_count
        assert journal.mark_in_flight(second.record_id, ledger_attempt=attempt).state is J.IN_FLIGHT
        assert len(journal.find_by_correlation_key(KEY)) == 2
        _sound(journal, ledger)


def test_window_4_claimed_before_in_flight_uses_the_same_epoch_again(env):
    _die_at(env, "claimed")
    journal, ledger = _stores(env)
    with journal, ledger:
        journal.recover_unfinished(), ledger.recover_unfinished()
        assert _states(journal, ledger) == ([J.RESOLVED, J.PENDING], L.UNCERTAIN, 2)
        second = journal.find_by_correlation_key(KEY)[1]
        assert second.ledger_attempt is None, "IN_FLIGHT never happened, which is what shows the write wasn't sent"
        _claim_is_refused(ledger)
        with pytest.raises(InvalidTransitionError):
            journal.mark_not_committed(second.record_id, expected_ledger_attempt=2)
        _resolve_ledger(ledger)  # the ledger alone: the journal has nothing to resolve
        attempt = _authorize_and_claim(ledger)
        assert attempt == 3
        flying = journal.mark_in_flight(second.record_id, ledger_attempt=attempt)
        assert (flying.epoch, flying.ledger_attempt, flying.attempt_count) == (2, 3, 1), "attempt 2 left a gap, which is normal"
        assert len(journal.find_by_correlation_key(KEY)) == 2
        _sound(journal, ledger)


def test_window_5_claimed_before_the_journal_retry_on_path_a(env):
    """A legal pair Phase 2.5 didn't list: the lower attempt on the record shows this claim never ran."""
    _die_at(env, "confirmed_not_committed_then_claimed")
    journal, ledger = _stores(env)
    with journal, ledger:
        journal.recover_unfinished(), ledger.recover_unfinished()
        assert _states(journal, ledger) == ([J.NOT_COMMITTED], L.UNCERTAIN, 2)
        record = journal.find_by_correlation_key(KEY)[0]
        assert record.ledger_attempt == 1 < ledger.get_request(KEY).attempt_count
        _claim_is_refused(ledger)
        _resolve_ledger(ledger)
        attempt = _authorize_and_claim(ledger)
        retried = journal.authorize_retry(record.record_id, authorized_by="operator", reason="retry", ledger_attempt=attempt)
        assert (retried.record_id, retried.epoch, retried.ledger_attempt, retried.attempt_count) == (record.record_id, 1, 3, 2)
        _sound(journal, ledger)


def test_window_6_in_flight_when_it_died_is_unknown_to_both(env):
    _die_at(env, "second_in_flight")
    journal, ledger = _stores(env)
    with journal, ledger:
        report = journal.recover_unfinished()
        assert len(report.recovered) == 1
        ledger.recover_unfinished()
        assert _states(journal, ledger) == ([J.RESOLVED, J.AMBIGUOUS], L.UNCERTAIN, 2)
        second = journal.find_by_correlation_key(KEY)[1]
        assert second.ledger_attempt == 2 and second.recovery_metadata["ledger_attempt"] == 2
        _claim_is_refused(ledger)
        with pytest.raises(InvalidTransitionError):
            _next_epoch(journal, 3)
        journal.resolve_ambiguous(second.record_id, resolution=Resolution.FOUND_NOT_COMMITTED, resolved_by="analyst", reason="evidence")
        _resolve_ledger(ledger)
        attempt = _authorize_and_claim(ledger)
        third = _next_epoch(journal, 3)
        assert journal.mark_in_flight(third.record_id, ledger_attempt=attempt).epoch == 3
        assert [r.epoch for r in journal.find_by_correlation_key(KEY)] == [1, 2, 3]
        _sound(journal, ledger)


@pytest.mark.parametrize("action, state, finish, ledger_state", [
    ("outcome_committed", J.COMMITTED, lambda l, a: l.record_completed(KEY, attempt=a), L.COMPLETED),
    ("outcome_not_committed", J.NOT_COMMITTED, lambda l, a: l.record_not_completed(KEY, attempt=a), L.NOT_COMPLETED),
    ("outcome_ambiguous", J.AMBIGUOUS, lambda l, a: l.record_uncertain(KEY, attempt=a), L.UNCERTAIN),
], ids=["committed", "not committed", "ambiguous"])
def test_window_7_the_journal_has_the_outcome_and_can_finish_the_ledger(env, action, state, finish, ledger_state):
    _die_at(env, action)
    journal, ledger = _stores(env)
    with journal, ledger:
        journal.recover_unfinished()
        assert _states(journal, ledger) == ([J.RESOLVED, state], L.IN_PROGRESS, 2)
        second = journal.find_by_correlation_key(KEY)[1]
        request = ledger.get_request(KEY)
        assert second.ledger_attempt == request.attempt_count, "the journal says which attempt this outcome belongs to"
        finish(ledger, second.ledger_attempt)
        assert _states(journal, ledger) == ([J.RESOLVED, state], ledger_state, 2)
        _sound(journal, ledger)


def test_legal_pair_ledger_not_completed_with_journal_pending(env):
    """A legal pair Phase 2.5 didn't list: the claim was released in-process when IN_FLIGHT failed, then the process died."""
    _die_at(env, "claim_whose_in_flight_failed")
    journal, ledger = _stores(env)
    with journal, ledger:
        journal.recover_unfinished(), ledger.recover_unfinished()
        assert _states(journal, ledger) == ([J.PENDING], L.NOT_COMPLETED, 1)
        record = journal.find_by_correlation_key(KEY)[0]
        _claim_is_refused(ledger)
        attempt = _authorize_and_claim(ledger)
        assert attempt == 2
        flying = journal.mark_in_flight(record.record_id, ledger_attempt=attempt)
        assert (flying.record_id, flying.epoch, flying.ledger_attempt, flying.attempt_count) == (record.record_id, 1, 2, 1)
        journal.mark_committed(record.record_id, expected_ledger_attempt=2)
        ledger.record_completed(KEY, attempt=2)
        _sound(journal, ledger)


def test_a_stale_outcome_from_before_a_crash_cannot_reach_the_newer_epoch(env):
    _die_at(env, "second_in_flight")
    journal, ledger = _stores(env)
    with journal, ledger:
        first, second = journal.find_by_correlation_key(KEY)
        before = journal.get_record(second.record_id), journal.get_history(second.record_id), journal.get_history(first.record_id)
        with pytest.raises(StaleAttemptError):
            journal.mark_committed(second.record_id, expected_ledger_attempt=1)
        with pytest.raises(InvalidTransitionError):
            journal.mark_committed(first.record_id, expected_ledger_attempt=1)
        assert (journal.get_record(second.record_id), journal.get_history(second.record_id), journal.get_history(first.record_id)) == before


# ================================================================ dying inside one journal transaction

@pytest.mark.parametrize("action, expected", [
    ("dies_marking_in_flight", ([J.PENDING], L.UNCERTAIN, 1)),
    ("dies_marking_committed", ([J.AMBIGUOUS], L.UNCERTAIN, 1)),
    ("dies_marking_not_committed", ([J.AMBIGUOUS], L.UNCERTAIN, 1)),
    ("dies_marking_ambiguous", ([J.AMBIGUOUS], L.UNCERTAIN, 1)),
    ("dies_authorizing_retry", ([J.NOT_COMMITTED], L.UNCERTAIN, 2)),
    ("dies_resolving", ([J.AMBIGUOUS], L.UNCERTAIN, 1)),
    ("dies_creating_next_epoch", ([J.RESOLVED], L.RETRY_AUTHORIZED, 1)),
], ids=["in flight", "committed", "not committed", "ambiguous", "journal retry", "resolve", "next epoch"])
def test_dying_inside_a_transaction_leaves_the_step_undone_and_nothing_runnable(env, action, expected):
    _die_at(env, action)
    journal, ledger = _stores(env)
    with journal, ledger:
        journal.recover_unfinished(), ledger.recover_unfinished()
        assert _states(journal, ledger) == expected
        request = ledger.get_request(KEY)
        if request.state is not L.RETRY_AUTHORIZED:
            _claim_is_refused(ledger)
        assert not any(r.state is J.IN_FLIGHT for r in journal.find_by_correlation_key(KEY))
        _sound(journal, ledger)


def test_dying_while_creating_the_first_intent_leaves_no_record(env):
    _die_at(env, "dies_creating_first")
    journal, ledger = _stores(env)
    with journal, ledger:
        assert journal.find_by_correlation_key(KEY) == () and journal.list_records() == ()
        assert ledger.get_request(KEY).state is L.REGISTERED, "registered first, by the ordering rule"
        assert journal.preflight_intent(OP, [("Q", None)], correlation_key=KEY, epoch=1).accepted
        _sound(journal, ledger)


def test_killed_at_arbitrary_moments_every_chain_stays_sound(env):
    """Not aimed at any statement, so it lands wherever it lands."""
    generator = random.Random(20261106)
    for round_number in range(8):
        process = env["children"].spawn([sys.executable, CHILD, "loop", env["journal"], env["ledger"] + str(round_number)], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, cwd=env["cwd"])
        assert json.loads(process.stdout.readline())["ready"]
        time.sleep(generator.uniform(0.05, 0.4))
        process.kill()
        assert process.wait(30) == -signal.SIGKILL
        process.stdout.close()
        process.stderr.close()
        with NeoWriteJournal(env["journal"]) as journal:
            journal.recover_unfinished()
            report = journal.check_integrity()
            assert report.ok, report.problems
            records = journal.list_records()
            assert all(r.correlation_key is not None for r in records)
            assert all(r.state is not J.IN_FLIGHT for r in records), "recovery leaves nothing that looks like it's running"
    assert len(records) > 8, "the children have to have written something for this to mean anything"
    assert any(r.epoch == 2 for r in records), "and to have reached the new-intent step"


# ================================================================ several real processes

def _race(env, argvs):
    go = env["tmp"] / "go"
    processes = [env["children"].spawn([sys.executable, CHILD, *argv[:1], env["journal"], str(go), *argv[1:]], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, cwd=env["cwd"]) for argv in argvs]
    time.sleep(1.5)
    go.touch()
    results = []
    for process in processes:
        out, err = process.communicate(timeout=90)
        assert process.returncode == 0, err[-800:]
        results.append(json.loads(out.strip().splitlines()[-1]))
    return results


def test_processes_creating_the_same_next_epoch_produce_one_record(env):
    with NeoWriteJournal(env["journal"]) as journal:
        first = journal.create_intent(OP, [("Q", {"v": 1})], correlation_key=KEY, epoch=1)
        journal.mark_in_flight(first.record_id, ledger_attempt=1)
        journal.mark_ambiguous(first.record_id, expected_ledger_attempt=1)
        journal.resolve_ambiguous(first.record_id, resolution=Resolution.FOUND_NOT_COMMITTED, resolved_by="a", reason="r")
    results = _race(env, [["race_next_epoch"]] * 4)
    winners = [r for r in results if r["won"]]
    assert len(winners) == 1
    assert all(r["error"] == "DuplicateCorrelationError" and r["existing"] == winners[0]["id"] for r in results if not r["won"])
    with NeoWriteJournal(env["journal"]) as journal:
        assert [r.epoch for r in journal.find_by_correlation_key(KEY)] == [1, 2]
        assert journal.check_integrity().ok


def test_processes_reporting_different_attempts_cannot_bypass_the_fence(env):
    with NeoWriteJournal(env["journal"]) as journal:
        record = journal.create_intent(OP, [("Q", {"v": 1})], correlation_key=KEY, epoch=1)
        journal.mark_in_flight(record.record_id, ledger_attempt=1)
        journal.mark_not_committed(record.record_id, expected_ledger_attempt=1)
        journal.authorize_retry(record.record_id, authorized_by="o", reason="r", ledger_attempt=2)
    results = _race(env, [["race_outcome", record.record_id, str(attempt)] for attempt in (1, 2, 1, 2, 1)])
    assert [r["attempt"] for r in results if r["won"]] == [2]
    assert all(r["error"] == "StaleAttemptError" for r in results if r["attempt"] == 1)
    with NeoWriteJournal(env["journal"]) as journal:
        assert journal.get_record(record.record_id).state is J.COMMITTED
        assert len(journal.get_history(record.record_id)) == 5
        assert journal.check_integrity().ok


def test_the_child_installs_its_guards_before_importing_any_backend_code():
    source = Path(CHILD).read_text(encoding="utf-8")
    assert source.index("install_child_guards()") < source.index("import neo4j_idempotency_ledger")

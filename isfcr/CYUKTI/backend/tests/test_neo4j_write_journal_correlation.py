"""AC-1, AC-2, AC-3 and D4 of neo4j_write_journal: correlation, preflight, attempt fencing, and the new
intent after a resolved request is retried. Everything lives in pytest's temporary directories. Process
death and the real ledger are in test_neo4j_write_journal_lifecycle.py, and version 1 files in
test_neo4j_write_journal_migration.py."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import random
import sqlite3
import threading
import unicodedata

import pytest

import neo4j_idempotency_ledger as ledger_module
import neo4j_write_journal as module
from neo4j_write_journal import (
    DuplicateCorrelationError,
    EpochChainError,
    InvalidTransitionError,
    JournalClosedError,
    JournalCorruptionError,
    JournalState as S,
    NeoWriteJournal,
    PreflightCode as P,
    Resolution,
    StaleAttemptError,
    UnsupportedParameterError,
    _HISTORY_COLUMNS,
    _RECORD_COLUMNS,
    _identical,
)

KEY = "cyukti.v1|wazuh:manager-01/1773199644.1234|duplicate_event.update"
OTHER = "cyukti.v1|wazuh:manager-01/1773199651.9876|duplicate_event.update"
OP = "duplicate_event.update"
AT = dt.datetime(2026, 3, 11, 3, 29, 24, 123000, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
INTENT = [("MATCH (e:AttackEvent {event_id:$e}) SET e.seen = $t", {"e": "E1", "t": AT, "tps": 0, "payload": '{"a": 1}', "pair": (1, 2.5, None)})]
SECRET = "s3cret-parameter-value"


@pytest.fixture
def path(tmp_path):
    return tmp_path / "journal.db"


@pytest.fixture
def journal(path):
    with NeoWriteJournal(path) as instance:
        yield instance


def _first(journal, key=KEY, statements=INTENT, **arguments):
    return journal.create_intent(OP, statements, correlation_key=key, epoch=1, **arguments)


def _flying(journal, key=KEY, attempt=1):
    return journal.mark_in_flight(_first(journal, key).record_id, ledger_attempt=attempt)


def _resolved(journal, resolution=Resolution.FOUND_NOT_COMMITTED, key=KEY, attempt=1):
    record = _flying(journal, key, attempt)
    journal.mark_ambiguous(record.record_id, expected_ledger_attempt=attempt)
    return journal.resolve_ambiguous(record.record_id, resolution=resolution, resolved_by="analyst", reason="checked")


def _next(journal, previous, epoch=2, key=KEY, **overrides):
    arguments = {"correlation_key": key, "epoch": epoch, "authorized_by": "analyst", "reason": "retry authorized", **overrides}
    return journal.create_intent(previous.operation, previous.statements, return_rows=previous.return_rows, **arguments)


def _state_of(journal, path):
    """Everything durable: every row, every history entry, and the file bytes other than the shared-memory index."""
    conn = journal._conn
    rows = conn.execute("SELECT * FROM journal_records ORDER BY seq").fetchall()
    events = conn.execute("SELECT * FROM journal_history ORDER BY seq").fetchall()
    files = {}
    for suffix in ("", "-wal"):
        target = str(path) + suffix
        with open(target, "rb") as handle:
            files[suffix] = hashlib.sha256(handle.read()).hexdigest()
    return rows, events, files


# ================================================================ AC-1: validation

@pytest.mark.parametrize("arguments, error", [
    ({"correlation_key": KEY}, TypeError),
    ({"epoch": 1}, TypeError),
    ({"correlation_key": KEY, "epoch": None}, TypeError),
    ({"correlation_key": None, "epoch": 1}, TypeError),
    ({"correlation_key": KEY, "epoch": True}, TypeError),
    ({"correlation_key": KEY, "epoch": False}, TypeError),
    ({"correlation_key": KEY, "epoch": 1.0}, TypeError),
    ({"correlation_key": KEY, "epoch": "1"}, TypeError),
    ({"correlation_key": KEY, "epoch": 0}, ValueError),
    ({"correlation_key": KEY, "epoch": -1}, ValueError),
    ({"correlation_key": KEY, "epoch": 2 ** 63}, ValueError),
    ({"correlation_key": "", "epoch": 1}, ValueError),
    ({"correlation_key": " k", "epoch": 1}, ValueError),
    ({"correlation_key": "k\n", "epoch": 1}, ValueError),
    ({"correlation_key": "a\x00b", "epoch": 1}, ValueError),
    ({"correlation_key": "\ud800", "epoch": 1}, ValueError),
    ({"correlation_key": "k" * 513, "epoch": 1}, ValueError),
    ({"correlation_key": b"k", "epoch": 1}, TypeError),
    ({"correlation_key": 5, "epoch": 1}, TypeError),
    ({"authorized_by": "a"}, TypeError),
    ({"reason": "r"}, TypeError),
    ({"correlation_key": KEY, "epoch": 1, "authorized_by": "a", "reason": "r"}, TypeError),
], ids=lambda value: None if isinstance(value, dict) else value.__name__)
def test_partial_or_malformed_correlation_is_refused_and_creates_nothing(journal, path, arguments, error):
    before = _state_of(journal, path)
    with pytest.raises(error):
        journal.create_intent(OP, INTENT, **arguments)
    assert _state_of(journal, path) == before


class _StrKey(str):
    pass


class _IntEpoch(int):
    pass


@pytest.mark.parametrize("arguments", [{"correlation_key": _StrKey(KEY), "epoch": 1}, {"correlation_key": KEY, "epoch": _IntEpoch(1)}],
                         ids=["str subclass", "int subclass"])
def test_subclasses_are_refused_like_everywhere_else_in_the_journal(journal, arguments):
    with pytest.raises(TypeError):
        journal.create_intent(OP, INTENT, **arguments)
    assert journal.list_records() == ()


KEYS_TO_COMPARE = ["", " k", "k ", "k\n", "\tk", "a\x00b", "a\x1fb", "a\x7fb", "\ud800", "k" * 512, "k" * 513, "a b", "A|b/c:d", "é☃𝒳", " k", "k "]


@pytest.mark.parametrize("key", KEYS_TO_COMPARE, ids=[repr(k)[:20] for k in KEYS_TO_COMPARE])
def test_the_journal_refuses_exactly_the_keys_the_ledger_refuses(key):
    def verdict(check):
        try:
            check(key)
            return "accepted"
        except (TypeError, ValueError) as error:
            return type(error).__name__
    assert verdict(module._checked_correlation_key) == verdict(ledger_module._checked_key)


@pytest.mark.parametrize("key", ["A|b/c:d", "a b", "é☃𝒳", "k" * 512, unicodedata.normalize("NFD", "é"), unicodedata.normalize("NFC", "é")],
                         ids=["separators", "inner space", "unicode", "longest", "NFD", "NFC"])
def test_a_key_is_stored_and_returned_exactly(path, key):
    with NeoWriteJournal(path) as journal:
        record = _first(journal, key)
    with NeoWriteJournal(path) as reopened:
        found = reopened.find_by_correlation_key(key)
        assert [r.record_id for r in found] == [record.record_id]
        assert found[0].correlation_key == key and type(found[0].correlation_key) is str
        assert reopened.get_history(record.record_id)[0].correlation_key == key


def test_keys_differing_only_in_form_or_case_are_different_requests(journal):
    for key in (unicodedata.normalize("NFD", "é"), unicodedata.normalize("NFC", "é"), "KEY", "key"):
        assert _first(journal, key).epoch == 1
    assert len(journal.find_by_correlation_key("key")) == 1


def test_correlation_survives_reopening_and_reaches_every_read(path):
    with NeoWriteJournal(path) as journal:
        created = _first(journal)
        assert (created.correlation_key, created.epoch, created.ledger_attempt, created.attempt_count) == (KEY, 1, None, 0)
        flying = journal.mark_in_flight(created.record_id, ledger_attempt=3)
        assert (flying.ledger_attempt, flying.attempt_count) == (3, 1), "the ledger's number and the record's own are separate"
    with NeoWriteJournal(path) as reopened:
        assert reopened.get_record(created.record_id) == flying
        assert reopened.list_records(S.IN_FLIGHT) == (flying,)
        assert reopened.find_by_correlation_key(KEY) == (flying,)
        events = reopened.get_history(created.record_id)
        assert [(e.correlation_key, e.epoch, e.ledger_attempt, e.attempt) for e in events] == [(KEY, 1, None, 0), (KEY, 1, 3, 1)]
    with NeoWriteJournal(path, read_only=True) as reader:
        assert reader.find_by_correlation_key(KEY) == (flying,)


def test_lookup_is_by_exact_key_and_refuses_a_malformed_one(journal):
    assert journal.find_by_correlation_key(KEY) == ()
    _first(journal)
    assert len(journal.find_by_correlation_key(KEY)) == 1
    assert journal.find_by_correlation_key(KEY + "x") == ()
    assert journal.find_by_correlation_key(KEY[:-1]) == ()
    for bad, error in ((None, TypeError), (5, TypeError), ("", ValueError), (" k", ValueError)):
        with pytest.raises(error):
            journal.find_by_correlation_key(bad)


def test_lookup_uses_the_index_rather_than_scanning(journal):
    plan = journal._conn.execute(f"EXPLAIN QUERY PLAN {module._SELECT_RECORD} WHERE correlation_key = ? ORDER BY epoch", (KEY,)).fetchall()
    text = " ".join(str(row[3]) for row in plan)
    assert "USING INDEX journal_records_correlation" in text and "SCAN" not in text and "TEMP B-TREE" not in text


def test_lookup_returns_every_epoch_oldest_first(journal):
    previous = _resolved(journal, attempt=1)
    second = _next(journal, previous, 2)
    journal.mark_in_flight(second.record_id, ledger_attempt=2)
    journal.mark_ambiguous(second.record_id, expected_ledger_attempt=2)
    journal.resolve_ambiguous(second.record_id, resolution=Resolution.FOUND_NOT_COMMITTED, resolved_by="a", reason="r")
    third = _next(journal, second, 3)
    _first(journal, OTHER)
    assert [r.epoch for r in journal.find_by_correlation_key(KEY)] == [1, 2, 3]
    assert journal.find_by_correlation_key(KEY)[2].record_id == third.record_id
    assert [r.epoch for r in journal.find_by_correlation_key(OTHER)] == [1]
    assert journal.check_integrity().ok


def test_one_record_per_key_and_epoch_even_with_the_same_intent(journal):
    record = _first(journal)
    with pytest.raises(DuplicateCorrelationError) as caught:
        _first(journal)
    assert caught.value.record_id == record.record_id
    assert KEY not in str(caught.value), "keys are named by reference"
    assert len(journal.list_records()) == 1
    assert _first(journal, OTHER).epoch == 1, "the same epoch under another key is a different request"


def test_the_database_itself_refuses_a_second_record_for_a_key_and_epoch(journal):
    """The check before the insert isn't the only protection."""
    _first(journal)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        journal._conn.execute(
            "INSERT INTO journal_records (record_id, format_version, operation, intent_json, intent_sha256, created_at, updated_at, state, "
            "attempt_count, version, history_head, row_sha256, correlation_key, epoch) "
            "VALUES ('x', 1, 'o', '{}', 's', 't', 't', 'PENDING', 0, 1, 'h', 'r', ?, 1)", (KEY,))


def test_uncorrelated_records_do_not_collide_under_the_index(journal):
    for _ in range(3):
        assert journal.create_intent(OP, INTENT).correlation_key is None
    assert len(journal.list_records()) == 3


# ================================================================ AC-1: legacy compatibility

def test_an_uncorrelated_record_is_exactly_what_it_always_was(journal):
    created = journal.create_intent(OP, INTENT)
    assert (created.correlation_key, created.epoch, created.ledger_attempt) == (None, None, None)
    journal.mark_not_committed(created.record_id, error_type="X")  # PENDING to NOT_COMMITTED remains allowed here
    flying = journal.authorize_retry(created.record_id, authorized_by="op", reason="r")
    assert flying.ledger_attempt is None and flying.attempt_count == 1
    done = journal.mark_committed(created.record_id)
    assert done.state is S.COMMITTED
    assert all((e.correlation_key, e.epoch, e.ledger_attempt) == (None, None, None) for e in journal.get_history(created.record_id))
    assert journal.check_integrity().ok


@pytest.mark.parametrize("move", [
    lambda j, i: j.mark_in_flight(i, ledger_attempt=1),
    lambda j, i: j.mark_committed(i, expected_ledger_attempt=1),
    lambda j, i: j.mark_not_committed(i, expected_ledger_attempt=1),
    lambda j, i: j.mark_ambiguous(i, expected_ledger_attempt=1),
    lambda j, i: j.authorize_retry(i, authorized_by="o", reason="r", ledger_attempt=1),
], ids=["in flight", "committed", "not committed", "ambiguous", "retry"])
def test_an_uncorrelated_record_refuses_attempt_arguments(journal, path, move):
    record = journal.create_intent(OP, INTENT)
    before = _state_of(journal, path)
    with pytest.raises(TypeError):
        move(journal, record.record_id)
    assert _state_of(journal, path) == before


# ================================================================ AC-2: preflight

def _setup_nothing(journal):
    return {}


def _setup_epoch_1_pending(journal):
    return {"record": _first(journal)}


def _setup_epoch_1_in_flight(journal):
    return {"record": _flying(journal)}


def _setup_resolved(resolution):
    def setup(journal):
        return {"record": _resolved(journal, resolution)}
    return setup


def _setup_committed(journal):
    record = _flying(journal)
    return {"record": journal.mark_committed(record.record_id, expected_ledger_attempt=1)}


def _setup_not_committed(journal):
    record = _flying(journal)
    return {"record": journal.mark_not_committed(record.record_id, expected_ledger_attempt=1)}


CASES = {
    "standalone": (_setup_nothing, {}, P.ACCEPTED),
    "first epoch": (_setup_nothing, {"correlation_key": KEY, "epoch": 1}, P.ACCEPTED),
    "first epoch and an attempt": (_setup_nothing, {"correlation_key": KEY, "epoch": 1, "ledger_attempt": 1}, P.ACCEPTED),
    "partial, key only": (_setup_nothing, {"correlation_key": KEY}, P.INVALID_CORRELATION),
    "partial, epoch only": (_setup_nothing, {"epoch": 1}, P.INVALID_CORRELATION),
    "boolean epoch": (_setup_nothing, {"correlation_key": KEY, "epoch": True}, P.INVALID_CORRELATION),
    "zero epoch": (_setup_nothing, {"correlation_key": KEY, "epoch": 0}, P.INVALID_CORRELATION),
    "empty key": (_setup_nothing, {"correlation_key": "", "epoch": 1}, P.INVALID_CORRELATION),
    "boolean attempt": (_setup_nothing, {"correlation_key": KEY, "epoch": 1, "ledger_attempt": True}, P.INVALID_CORRELATION),
    "zero attempt": (_setup_nothing, {"correlation_key": KEY, "epoch": 1, "ledger_attempt": 0}, P.INVALID_CORRELATION),
    "attempt with no correlation": (_setup_nothing, {"ledger_attempt": 1}, P.INVALID_CORRELATION),
    "duplicate first epoch": (_setup_epoch_1_pending, {"correlation_key": KEY, "epoch": 1}, P.DUPLICATE_CORRELATION),
    "skipped epoch, nothing there": (_setup_nothing, {"correlation_key": KEY, "epoch": 2, "authorized_by": "a", "reason": "r"}, P.EPOCH_OUT_OF_ORDER),
    "skipped epoch": (_setup_resolved(Resolution.FOUND_NOT_COMMITTED), {"correlation_key": KEY, "epoch": 3, "authorized_by": "a", "reason": "r"}, P.EPOCH_OUT_OF_ORDER),
    "regressed epoch": (_setup_resolved(Resolution.FOUND_NOT_COMMITTED), {"correlation_key": KEY, "epoch": 1}, P.DUPLICATE_CORRELATION),
    "next epoch, no authorization": (_setup_resolved(Resolution.FOUND_NOT_COMMITTED), {"correlation_key": KEY, "epoch": 2}, P.AUTHORIZATION_REQUIRED),
    "next epoch, no reason": (_setup_resolved(Resolution.FOUND_NOT_COMMITTED), {"correlation_key": KEY, "epoch": 2, "authorized_by": "a"}, P.AUTHORIZATION_REQUIRED),
    "next epoch, blank authorizer": (_setup_resolved(Resolution.FOUND_NOT_COMMITTED), {"correlation_key": KEY, "epoch": 2, "authorized_by": " ", "reason": "r"}, P.AUTHORIZATION_REQUIRED),
    "authorization on the first epoch": (_setup_nothing, {"correlation_key": KEY, "epoch": 1, "authorized_by": "a", "reason": "r"}, P.AUTHORIZATION_REQUIRED),
    "next epoch": (_setup_resolved(Resolution.FOUND_NOT_COMMITTED), {"correlation_key": KEY, "epoch": 2, "authorized_by": "a", "reason": "r"}, P.ACCEPTED),
    "next epoch, higher attempt": (_setup_resolved(Resolution.FOUND_NOT_COMMITTED), {"correlation_key": KEY, "epoch": 2, "authorized_by": "a", "reason": "r", "ledger_attempt": 2}, P.ACCEPTED),
    "next epoch, stale attempt": (_setup_resolved(Resolution.FOUND_NOT_COMMITTED), {"correlation_key": KEY, "epoch": 2, "authorized_by": "a", "reason": "r", "ledger_attempt": 1}, P.STALE_ATTEMPT),
    "next epoch, previous found committed": (_setup_resolved(Resolution.FOUND_COMMITTED), {"correlation_key": KEY, "epoch": 2, "authorized_by": "a", "reason": "r"}, P.PREVIOUS_NOT_RETRYABLE),
    "next epoch, previous abandoned": (_setup_resolved(Resolution.ABANDONED), {"correlation_key": KEY, "epoch": 2, "authorized_by": "a", "reason": "r"}, P.PREVIOUS_NOT_RETRYABLE),
    "next epoch, previous pending": (_setup_epoch_1_pending, {"correlation_key": KEY, "epoch": 2, "authorized_by": "a", "reason": "r"}, P.PREVIOUS_NOT_RESOLVED),
    "next epoch, previous in flight": (_setup_epoch_1_in_flight, {"correlation_key": KEY, "epoch": 2, "authorized_by": "a", "reason": "r"}, P.PREVIOUS_NOT_RESOLVED),
    "next epoch, previous committed": (_setup_committed, {"correlation_key": KEY, "epoch": 2, "authorized_by": "a", "reason": "r"}, P.PREVIOUS_NOT_RESOLVED),
    "next epoch, previous only not committed": (_setup_not_committed, {"correlation_key": KEY, "epoch": 2, "authorized_by": "a", "reason": "r"}, P.PREVIOUS_NOT_RESOLVED),
}


@pytest.mark.parametrize("name", CASES)
def test_preflight_answers_as_specified_changes_nothing_and_agrees_with_create(journal, path, name):
    setup, arguments, expected = CASES[name]
    setup(journal)
    before = _state_of(journal, path)
    result = journal.preflight_intent(OP, INTENT, **arguments)
    assert _state_of(journal, path) == before, "preflight must leave no trace, accepted or refused"
    assert journal.check_integrity().ok
    assert result.code is expected and result.accepted is (expected is P.ACCEPTED)
    assert result.reason and KEY not in result.reason
    create_arguments = {k: v for k, v in arguments.items() if k != "ledger_attempt"}
    if result.accepted:
        record = journal.create_intent(OP, INTENT, **create_arguments)
        assert record.intent_sha256 == result.intent_sha256
    elif expected is not P.STALE_ATTEMPT and "ledger_attempt" not in arguments:
        with pytest.raises((TypeError, ValueError, module.JournalError)):
            journal.create_intent(OP, INTENT, **create_arguments)
        assert _state_of(journal, path)[:2] == before[:2]


@pytest.mark.parametrize("operation, statements, arguments, code", [
    ("", INTENT, {}, P.INVALID_ARGUMENT),
    (None, INTENT, {}, P.INVALID_ARGUMENT),
    (OP, INTENT, {"return_rows": 1}, P.INVALID_ARGUMENT),
    (OP, [], {}, P.INVALID_ARGUMENT),
    (OP, None, {}, P.INVALID_ARGUMENT),
    (OP, "Q", {}, P.INVALID_ARGUMENT),
    (OP, [("", None)], {}, P.INVALID_ARGUMENT),
    (OP, [("Q", {"bad": {1, 2}})], {}, P.UNSUPPORTED_PARAMETER),
    (OP, [("Q", {"bad": {1, 2}})], {"correlation_key": KEY, "epoch": 1}, P.UNSUPPORTED_PARAMETER),
], ids=["empty operation", "no operation", "return_rows not a bool", "no statements", "statements None", "bare string", "empty query", "set", "set, correlated"])
def test_preflight_refuses_what_create_refuses_in_the_payload(journal, path, operation, statements, arguments, code):
    before = _state_of(journal, path)
    result = journal.preflight_intent(operation, statements, **arguments)
    assert (result.accepted, result.code) == (False, code)
    with pytest.raises((TypeError, ValueError)):
        journal.create_intent(operation, statements, **arguments)
    assert _state_of(journal, path)[:2] == before[:2]


def test_preflight_does_not_quote_a_value_it_refuses(journal):
    result = journal.preflight_intent(OP, [("Q", {"token": {SECRET}})])
    assert SECRET not in result.reason


def test_preflight_refuses_a_one_shot_iterator_because_reading_it_would_empty_it(journal):
    statements = iter(INTENT)
    result = journal.preflight_intent(OP, statements)
    assert (result.accepted, result.code) == (False, P.INVALID_ARGUMENT)
    assert list(statements) == INTENT, "it wasn't consumed"


def test_a_duplicate_names_the_existing_record_and_hash_so_a_caller_can_compare(journal):
    record = _first(journal)
    same = journal.preflight_intent(OP, INTENT, correlation_key=KEY, epoch=1)
    assert same.existing_record_id == record.record_id and same.intent_sha256 == record.intent_sha256
    different = journal.preflight_intent(OP, [("OTHER", None)], correlation_key=KEY, epoch=1)
    assert different.existing_record_id == record.record_id and different.intent_sha256 != record.intent_sha256


def test_preflight_catches_a_changed_payload_across_epochs(journal):
    previous = _resolved(journal)
    arguments = {"correlation_key": KEY, "epoch": 2, "authorized_by": "a", "reason": "r"}
    changed = journal.preflight_intent(OP, [("OTHER", None)], **arguments)
    assert changed.code is P.INTENT_MISMATCH
    assert journal.preflight_intent("other.op", previous.statements, **arguments).code is P.INTENT_MISMATCH
    assert journal.preflight_intent(OP, previous.statements, return_rows=True, **arguments).code is P.INTENT_MISMATCH
    assert journal.preflight_intent(OP, previous.statements, **arguments).accepted


def test_acceptance_guarantees_nothing_later_and_create_checks_again(journal):
    accepted = journal.preflight_intent(OP, INTENT, correlation_key=KEY, epoch=1)
    assert accepted.accepted
    _first(journal)
    with pytest.raises(DuplicateCorrelationError):
        _first(journal)


def test_preflight_on_a_read_only_instance_says_so(path):
    with NeoWriteJournal(path) as writer:
        before = _state_of(writer, path)
        with NeoWriteJournal(path, read_only=True) as reader:
            result = reader.preflight_intent(OP, INTENT, correlation_key=KEY, epoch=1)
            assert (result.accepted, result.code) == (False, P.READ_ONLY)
        assert _state_of(writer, path)[:2] == before[:2]


def test_preflight_on_a_closed_journal_raises_because_nothing_can_be_said(path):
    journal = NeoWriteJournal(path)
    journal.close()
    with pytest.raises(JournalClosedError):
        journal.preflight_intent(OP, INTENT)


def test_preflight_takes_no_write_lock(path):
    with NeoWriteJournal(path, busy_timeout=0.1) as journal:
        _first(journal)
        blocker = sqlite3.connect(path, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            assert journal.preflight_intent(OP, INTENT, correlation_key=OTHER, epoch=1).accepted
            assert journal.preflight_intent(OP, INTENT, correlation_key=KEY, epoch=1).code is P.DUPLICATE_CORRELATION
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()


def test_preflight_leaves_no_open_transaction_after_any_answer(journal):
    _first(journal)
    journal.preflight_intent(OP, INTENT, correlation_key=KEY, epoch=1)
    assert not journal._conn.in_transaction
    journal.preflight_intent(OP, INTENT, correlation_key=OTHER, epoch=1)
    assert not journal._conn.in_transaction


def test_preflight_reports_a_damaged_chain_rather_than_accepting(path):
    with NeoWriteJournal(path) as journal:
        record = _first(journal)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("UPDATE journal_records SET attempt_count = 9 WHERE record_id = ?", (record.record_id,))
    conn.close()
    with NeoWriteJournal(path) as journal:
        assert journal.preflight_intent(OP, INTENT, correlation_key=KEY, epoch=1).code is P.JOURNAL_DAMAGED
        with pytest.raises(JournalCorruptionError):
            _first(journal)


def test_preflight_is_safe_under_concurrent_calls_and_changes_nothing(journal, path):
    previous = _resolved(journal)
    before = _state_of(journal, path)
    answers, problems = [], []

    def ask():
        try:
            for _ in range(40):
                answers.append(journal.preflight_intent(OP, previous.statements, correlation_key=KEY, epoch=2, authorized_by="a", reason="r").code)
                answers.append(journal.preflight_intent(OP, previous.statements, correlation_key=KEY, epoch=1).code)
        except Exception as error:
            problems.append(type(error).__name__)

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert problems == []
    assert set(answers) == {P.ACCEPTED, P.DUPLICATE_CORRELATION} and len(answers) == 640
    assert _state_of(journal, path) == before


def test_random_proposals_preflight_and_create_never_disagree(journal):
    """Every accepted proposal is created, and every refused one is refused again, from whatever state a random walk reaches."""
    generator = random.Random(20261106)
    keys = [KEY, OTHER]
    created, refused = 0, 0
    highest = {key: 0 for key in keys}
    for _ in range(600):
        key = generator.choice(keys)
        chain = journal.find_by_correlation_key(key)
        latest = chain[-1] if chain else None
        statements = latest.statements if latest is not None and generator.random() < 0.8 else [("Q", {"v": generator.randint(1, 2)})]
        epoch = generator.choice([1, 2, 3, len(chain) + 1, len(chain) + 1, len(chain) + 1])
        authorization = {"authorized_by": "a", "reason": "r"} if epoch > 1 and generator.random() < 0.9 else {}
        arguments = {"correlation_key": key, "epoch": epoch, **authorization}
        result = journal.preflight_intent(OP, statements, **arguments)
        if result.accepted:
            assert journal.create_intent(OP, statements, **arguments).intent_sha256 == result.intent_sha256
            created += 1
        else:
            with pytest.raises((TypeError, ValueError, module.JournalError)):
                journal.create_intent(OP, statements, **arguments)
            refused += 1
        chain = journal.find_by_correlation_key(key)
        latest = chain[-1] if chain else None
        if latest is not None and latest.state is S.PENDING and generator.random() < 0.7:
            highest[key] += generator.randint(1, 3)
            journal.mark_in_flight(latest.record_id, ledger_attempt=highest[key])
            journal.mark_ambiguous(latest.record_id, expected_ledger_attempt=highest[key])
            resolution = generator.choice([Resolution.FOUND_NOT_COMMITTED] * 4 + [Resolution.FOUND_COMMITTED, Resolution.ABANDONED])
            journal.resolve_ambiguous(latest.record_id, resolution=resolution, resolved_by="a", reason="r")
    assert created > 3 and refused > 50, "the walk has to reach both answers"
    assert journal.check_integrity().ok


# ================================================================ AC-3: starting an attempt

@pytest.mark.parametrize("bad, error", [(None, TypeError), (True, TypeError), (1.0, TypeError), ("1", TypeError), (0, ValueError), (-1, ValueError), (2 ** 63, ValueError)])
def test_starting_a_correlated_record_needs_a_valid_attempt(journal, path, bad, error):
    record = _first(journal)
    before = _state_of(journal, path)
    with pytest.raises(error):
        journal.mark_in_flight(record.record_id, ledger_attempt=bad)
    assert _state_of(journal, path) == before


def test_a_retry_of_a_correlated_record_needs_a_higher_attempt(journal, path):
    record = _flying(journal, attempt=3)
    journal.mark_not_committed(record.record_id, expected_ledger_attempt=3)
    before = _state_of(journal, path)
    with pytest.raises(TypeError):
        journal.authorize_retry(record.record_id, authorized_by="o", reason="r")
    for lower in (1, 2, 3):
        with pytest.raises(StaleAttemptError):
            journal.authorize_retry(record.record_id, authorized_by="o", reason="r", ledger_attempt=lower)
    assert _state_of(journal, path) == before
    retried = journal.authorize_retry(record.record_id, authorized_by="o", reason="r", ledger_attempt=7)
    assert (retried.state, retried.ledger_attempt, retried.attempt_count) == (S.IN_FLIGHT, 7, 2), "gaps are allowed: a failed claim uses a number"


def test_an_attempt_must_rise_across_epochs_too(journal, path):
    previous = _resolved(journal, attempt=5)
    second = _next(journal, previous)
    before = _state_of(journal, path)
    for stale in (1, 4, 5):
        with pytest.raises(StaleAttemptError):
            journal.mark_in_flight(second.record_id, ledger_attempt=stale)
    assert _state_of(journal, path) == before
    assert journal.mark_in_flight(second.record_id, ledger_attempt=6).attempt_count == 1, "a new record's own count restarts"


def test_another_requests_attempts_are_not_compared(journal):
    _flying(journal, KEY, attempt=9)
    assert _flying(journal, OTHER, attempt=1).ledger_attempt == 1


# ================================================================ AC-3: outcomes

OUTCOMES = {
    "committed": lambda j, i, **a: j.mark_committed(i, **a),
    "not committed": lambda j, i, **a: j.mark_not_committed(i, **a),
    "ambiguous": lambda j, i, **a: j.mark_ambiguous(i, **a),
}


@pytest.mark.parametrize("outcome", OUTCOMES.values(), ids=OUTCOMES.keys())
@pytest.mark.parametrize("bad, error", [(None, TypeError), (True, TypeError), ("2", TypeError), (2.0, TypeError), (0, ValueError), (1, StaleAttemptError), (3, StaleAttemptError)])
def test_a_wrong_attempt_changes_nothing(journal, path, outcome, bad, error):
    record = _flying(journal, attempt=2)
    before = _state_of(journal, path)
    with pytest.raises(error):
        outcome(journal, record.record_id, expected_ledger_attempt=bad)
    assert _state_of(journal, path) == before
    assert journal.get_record(record.record_id).state is S.IN_FLIGHT


@pytest.mark.parametrize("outcome, state", list(zip(OUTCOMES.values(), (S.COMMITTED, S.NOT_COMMITTED, S.AMBIGUOUS))), ids=list(OUTCOMES))
def test_the_right_attempt_is_recorded_on_row_and_history(journal, outcome, state):
    record = _flying(journal, attempt=2)
    done = outcome(journal, record.record_id, expected_ledger_attempt=2)
    assert (done.state, done.ledger_attempt) == (state, 2)
    assert journal.get_history(record.record_id)[-1].ledger_attempt == 2
    assert journal.check_integrity().ok


@pytest.mark.parametrize("outcome", OUTCOMES.values(), ids=OUTCOMES.keys())
def test_a_repeated_outcome_changes_nothing(journal, path, outcome):
    record = _flying(journal)
    outcome(journal, record.record_id, expected_ledger_attempt=1)
    before = _state_of(journal, path)
    for repeat in OUTCOMES.values():
        with pytest.raises(InvalidTransitionError):
            repeat(journal, record.record_id, expected_ledger_attempt=1)
    assert _state_of(journal, path) == before


@pytest.mark.parametrize("outcome", OUTCOMES.values(), ids=OUTCOMES.keys())
def test_no_outcome_can_be_reported_before_an_attempt_started(journal, path, outcome):
    """Including PENDING to NOT_COMMITTED, which only an uncorrelated record may take."""
    record = _first(journal)
    before = _state_of(journal, path)
    with pytest.raises(InvalidTransitionError) as caught:
        outcome(journal, record.record_id, expected_ledger_attempt=1)
    assert not isinstance(caught.value, StaleAttemptError)
    with pytest.raises(TypeError):
        outcome(journal, record.record_id)
    assert _state_of(journal, path) == before


@pytest.mark.parametrize("outcome", OUTCOMES.values(), ids=OUTCOMES.keys())
def test_an_old_attempt_cannot_touch_the_retry_that_replaced_it(journal, path, outcome):
    record = _flying(journal, attempt=1)
    journal.mark_not_committed(record.record_id, expected_ledger_attempt=1)
    journal.authorize_retry(record.record_id, authorized_by="o", reason="r", ledger_attempt=2)
    before = _state_of(journal, path)
    with pytest.raises(StaleAttemptError):
        outcome(journal, record.record_id, expected_ledger_attempt=1)
    assert _state_of(journal, path) == before
    assert journal.mark_committed(record.record_id, expected_ledger_attempt=2).state is S.COMMITTED


@pytest.mark.parametrize("outcome", OUTCOMES.values(), ids=OUTCOMES.keys())
def test_a_stale_outcome_after_a_newer_intent_exists_changes_neither_record(journal, path, outcome):
    previous = _resolved(journal, attempt=1)
    second = journal.mark_in_flight(_next(journal, previous).record_id, ledger_attempt=2)
    before = _state_of(journal, path)
    with pytest.raises(InvalidTransitionError):
        outcome(journal, previous.record_id, expected_ledger_attempt=1)
    with pytest.raises(StaleAttemptError):
        outcome(journal, previous.record_id, expected_ledger_attempt=2)
    with pytest.raises(StaleAttemptError):
        outcome(journal, second.record_id, expected_ledger_attempt=1)
    assert _state_of(journal, path) == before


def test_a_recovered_record_refuses_its_own_late_outcome(path):
    with NeoWriteJournal(path) as first:
        record = _flying(first, attempt=4)
    with NeoWriteJournal(path) as second:
        assert second.recover_unfinished().recovered == (record.record_id,)
        recovered = second.get_record(record.record_id)
        assert (recovered.state, recovered.ledger_attempt) == (S.AMBIGUOUS, 4)
        assert recovered.recovery_metadata["ledger_attempt"] == 4 and recovered.recovery_metadata["attempt"] == 1
        for outcome in OUTCOMES.values():
            with pytest.raises(InvalidTransitionError):
                outcome(second, record.record_id, expected_ledger_attempt=4)
        assert second.check_integrity().ok


def test_threads_reporting_different_attempts_let_only_the_current_one_win(journal):
    record = _flying(journal, attempt=1)
    journal.mark_not_committed(record.record_id, expected_ledger_attempt=1)
    journal.authorize_retry(record.record_id, authorized_by="o", reason="r", ledger_attempt=5)
    results = []
    barrier = threading.Barrier(12)

    def report(attempt):
        barrier.wait()
        try:
            journal.mark_committed(record.record_id, expected_ledger_attempt=attempt)
            results.append((attempt, "won"))
        except StaleAttemptError:
            results.append((attempt, "stale"))
        except InvalidTransitionError:
            results.append((attempt, "refused"))

    threads = [threading.Thread(target=report, args=(1 if index % 2 else 5,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert [r for r in results if r[1] == "won"] == [(5, "won")]
    assert all(outcome == "stale" for attempt, outcome in results if attempt == 1)
    assert len(journal.get_history(record.record_id)) == 5


def test_two_instances_reporting_at_once_cannot_bypass_the_fence(path):
    with NeoWriteJournal(path) as a, NeoWriteJournal(path) as b:
        record = _flying(a, attempt=2)
        results = []
        barrier = threading.Barrier(2)

        def report(instance, outcome, attempt):
            barrier.wait()
            try:
                outcome(instance, record.record_id, expected_ledger_attempt=attempt)
                results.append("won")
            except InvalidTransitionError:
                results.append("refused")

        threads = [threading.Thread(target=report, args=(a, OUTCOMES["committed"], 2)), threading.Thread(target=report, args=(b, OUTCOMES["ambiguous"], 2))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert sorted(results) == ["refused", "won"]
        assert a.check_integrity().ok


def test_a_failed_transition_on_a_correlated_record_changes_nothing(path):
    with NeoWriteJournal(path) as journal:
        record = _flying(journal)
        before = journal.get_record(record.record_id), journal.get_history(record.record_id)
        journal._conn.execute("CREATE TEMP TRIGGER fail_history BEFORE INSERT ON main.journal_history BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with pytest.raises(module.JournalWriteError):
            journal.mark_committed(record.record_id, expected_ledger_attempt=1)
        journal._conn.execute("DROP TRIGGER temp.fail_history")
        assert (journal.get_record(record.record_id), journal.get_history(record.record_id)) == before
        assert journal.mark_committed(record.record_id, expected_ledger_attempt=1).state is S.COMMITTED


# ================================================================ D4: the new intent

def test_a_new_intent_after_a_resolved_non_commit_keeps_the_original_exactly(journal, path):
    previous = _resolved(journal, attempt=1)
    old_row = journal.get_record(previous.record_id)
    old_history = journal.get_history(previous.record_id)
    second = _next(journal, previous)
    assert (second.state, second.epoch, second.correlation_key, second.ledger_attempt, second.attempt_count) == (S.PENDING, 2, KEY, None, 0)
    assert second.record_id != previous.record_id and second.intent_sha256 == previous.intent_sha256
    assert _identical(second.statements, previous.statements)
    event = journal.get_history(second.record_id)[0]
    assert (event.cause, event.actor, event.reason, event.epoch) == ("CREATED", "analyst", "retry authorized", 2)
    assert journal.get_record(previous.record_id) == old_row and journal.get_history(previous.record_id) == old_history
    flying = journal.mark_in_flight(second.record_id, ledger_attempt=2)
    assert journal.mark_committed(flying.record_id, expected_ledger_attempt=2).state is S.COMMITTED
    assert journal.get_record(previous.record_id).state is S.RESOLVED, "a resolved intent stays final"
    assert journal.check_integrity().ok


def test_a_confirmed_non_commit_retries_on_the_same_record_and_gets_no_new_intent(journal, path):
    """Path A of the contract: only RESOLVED leads to a new epoch."""
    record = _flying(journal, attempt=1)
    journal.mark_not_committed(record.record_id, expected_ledger_attempt=1, error_type="ServiceUnavailable")
    before = _state_of(journal, path)
    with pytest.raises(EpochChainError) as caught:
        _next(journal, record)
    assert caught.value.code is P.PREVIOUS_NOT_RESOLVED
    assert _state_of(journal, path) == before
    retried = journal.authorize_retry(record.record_id, authorized_by="op", reason="r", ledger_attempt=2)
    assert (retried.record_id, retried.epoch, retried.attempt_count) == (record.record_id, 1, 2)
    assert journal.mark_committed(record.record_id, expected_ledger_attempt=2).state is S.COMMITTED
    events = journal.get_history(record.record_id)
    assert [e.ledger_attempt for e in events] == [None, 1, 1, 2, 2], "one intent, two attempts, both kept"
    assert (events[3].actor, events[3].reason) == ("op", "r")


@pytest.mark.parametrize("arguments", [{}, {"authorized_by": "a"}, {"reason": "r"}, {"authorized_by": " ", "reason": "r"}, {"authorized_by": "a", "reason": ""}, {"authorized_by": 5, "reason": "r"}],
                         ids=["neither", "no reason", "no authorizer", "blank authorizer", "blank reason", "authorizer not text"])
def test_a_new_intent_without_stated_authorization_is_refused(journal, path, arguments):
    previous = _resolved(journal)
    before = _state_of(journal, path)
    with pytest.raises(ValueError):
        journal.create_intent(OP, previous.statements, correlation_key=KEY, epoch=2, **arguments)
    assert _state_of(journal, path) == before


@pytest.mark.parametrize("resolution", [Resolution.FOUND_COMMITTED, Resolution.ABANDONED], ids=["committed", "abandoned"])
def test_a_request_resolved_any_other_way_can_never_get_a_new_intent(journal, path, resolution):
    previous = _resolved(journal, resolution)
    before = _state_of(journal, path)
    with pytest.raises(EpochChainError) as caught:
        _next(journal, previous)
    assert caught.value.code is P.PREVIOUS_NOT_RETRYABLE
    assert _state_of(journal, path) == before


@pytest.mark.parametrize("change", [
    lambda s: [("MATCH (x) SET x.y = 1", None)],
    lambda s: [(s[0][0], {**s[0][1], "tps": 1})],
    lambda s: [(s[0][0], {**s[0][1], "t": s[0][1]["t"] + dt.timedelta(microseconds=1)})],
    lambda s: [(s[0][0], {**s[0][1], "t": s[0][1]["t"].astimezone(dt.timezone.utc)})],
    lambda s: [(s[0][0], {**s[0][1], "tps": 0.0})],
    lambda s: [(s[0][0], {**s[0][1], "pair": list(s[0][1]["pair"])})],
    lambda s: [(s[0][0], dict(reversed(list(s[0][1].items()))))],
    lambda s: [*s, ("EXTRA", None)],
], ids=["other query", "other value", "other instant", "same instant, other zone", "int became float", "tuple became list", "other key order", "extra statement"])
def test_a_new_epoch_cannot_change_the_intent_in_any_way(journal, path, change):
    previous = _resolved(journal)
    before = _state_of(journal, path)
    with pytest.raises(EpochChainError) as caught:
        journal.create_intent(OP, change(previous.statements), correlation_key=KEY, epoch=2, authorized_by="a", reason="r")
    assert caught.value.code is P.INTENT_MISMATCH
    assert _state_of(journal, path) == before


def test_a_new_epoch_cannot_change_the_operation_or_return_rows(journal):
    previous = _resolved(journal)
    arguments = {"correlation_key": KEY, "epoch": 2, "authorized_by": "a", "reason": "r"}
    for call in (lambda: journal.create_intent("other", previous.statements, **arguments),
                 lambda: journal.create_intent(OP, previous.statements, return_rows=True, **arguments)):
        with pytest.raises(EpochChainError):
            call()
    assert len(journal.list_records()) == 1


def test_the_stored_statements_rebuild_an_identical_intent(journal):
    """What a retry has to rely on: the record alone reproduces its hash, for each type a payload may hold."""
    from tests.test_neo4j_write_journal import PRESERVED
    for index, (name, value) in enumerate(PRESERVED.items()):
        key = f"key-{index}"
        original = journal.create_intent(OP, [("Q", {"p": value})], correlation_key=key, epoch=1)
        again = journal.preflight_intent(OP, journal.get_record(original.record_id).statements)
        assert again.intent_sha256 == original.intent_sha256, name


@pytest.mark.parametrize("epoch", [3, 4, 100])
def test_a_skipped_epoch_is_refused(journal, path, epoch):
    previous = _resolved(journal)
    before = _state_of(journal, path)
    with pytest.raises(EpochChainError) as caught:
        _next(journal, previous, epoch)
    assert caught.value.code is P.EPOCH_OUT_OF_ORDER
    assert _state_of(journal, path) == before


def test_a_regressed_or_repeated_epoch_is_refused(journal):
    previous = _resolved(journal)
    second = _next(journal, previous)
    for epoch in (1, 2):
        with pytest.raises(DuplicateCorrelationError):
            journal.create_intent(OP, previous.statements, correlation_key=KEY, epoch=epoch, **({"authorized_by": "a", "reason": "r"} if epoch > 1 else {}))
    assert [r.record_id for r in journal.find_by_correlation_key(KEY)] == [previous.record_id, second.record_id]


def test_a_first_epoch_cannot_be_late_and_a_later_one_cannot_be_first(journal):
    with pytest.raises(EpochChainError) as caught:
        journal.create_intent(OP, INTENT, correlation_key=KEY, epoch=2, authorized_by="a", reason="r")
    assert caught.value.code is P.EPOCH_OUT_OF_ORDER
    assert journal.list_records() == ()


def test_an_intent_identity_can_never_be_supplied_or_reused(journal):
    previous = _resolved(journal)
    with pytest.raises(TypeError):
        journal.create_intent(OP, INTENT, record_id=previous.record_id)
    second = _next(journal, previous)
    assert second.record_id != previous.record_id
    with pytest.raises(InvalidTransitionError):
        journal.mark_in_flight(previous.record_id, ledger_attempt=9)
    with pytest.raises(InvalidTransitionError):
        journal.authorize_retry(previous.record_id, authorized_by="o", reason="r", ledger_attempt=9)
    assert journal.get_record(previous.record_id).state is S.RESOLVED


def test_threads_creating_the_same_next_epoch_produce_one_record(journal):
    previous = _resolved(journal)
    results = []
    barrier = threading.Barrier(12)

    def attempt():
        barrier.wait()
        try:
            results.append(_next(journal, previous).record_id)
        except DuplicateCorrelationError as error:
            results.append(("duplicate", error.record_id))

    threads = [threading.Thread(target=attempt) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    winners = [r for r in results if isinstance(r, str)]
    assert len(winners) == 1 and len(results) == 12
    assert all(r == ("duplicate", winners[0]) for r in results if not isinstance(r, str))
    assert [r.epoch for r in journal.find_by_correlation_key(KEY)] == [1, 2]


def test_two_instances_creating_the_same_next_epoch_produce_one_record(path):
    with NeoWriteJournal(path) as a, NeoWriteJournal(path) as b:
        previous = _resolved(a)
        results = []
        barrier = threading.Barrier(2)

        def attempt(instance):
            barrier.wait()
            try:
                results.append(_next(instance, previous).epoch)
            except DuplicateCorrelationError:
                results.append("duplicate")

        threads = [threading.Thread(target=attempt, args=(i,)) for i in (a, b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert sorted(map(str, results)) == ["2", "duplicate"]


def test_the_journal_never_replays_anything(journal):
    """Nothing here has an execute path, so a new intent stays PENDING until a caller moves it."""
    second = _next(journal, _resolved(journal))
    assert journal.recover_unfinished().pending == (second.record_id,)
    assert journal.get_record(second.record_id).state is S.PENDING


# ================================================================ integrity

def _raw(path):
    conn = sqlite3.connect(path, isolation_level=None)
    for name in ("journal_history_no_update", "journal_history_no_delete", "journal_records_no_delete", "journal_records_intent_fixed",
                 "journal_records_correlation_fixed", "journal_records_attempt_rises"):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    return conn


def _restore_triggers(conn):
    for statement in (*module._SCHEMA, *module._SCHEMA_V2):
        if statement.startswith("CREATE TRIGGER"):
            conn.execute(statement)
    conn.close()


def _rewrite(conn, record_id, row_changes=None, event_changes=None):
    """Forges a record consistently: every checksum is recomputed, so only a check of the contents can notice.
    event_changes maps a transition number to its changed fields."""
    row = conn.execute(f"SELECT {', '.join(_RECORD_COLUMNS)} FROM journal_records WHERE record_id = ?", (record_id,)).fetchone()
    values = dict(zip(_RECORD_COLUMNS, row))
    previous = module._NO_HISTORY
    events = conn.execute(f"SELECT {', '.join(_HISTORY_COLUMNS)} FROM journal_history WHERE record_id = ? ORDER BY transition_no", (record_id,)).fetchall()
    for event_row in events:
        event = dict(zip(_HISTORY_COLUMNS, event_row))
        event.update((event_changes or {}).get(event["transition_no"], {}))
        entry = NeoWriteJournal._event_hash(event, previous)
        assignments = ", ".join(f"{name} = ?" for name in _HISTORY_COLUMNS)
        conn.execute(f"UPDATE journal_history SET {assignments}, prev_sha256 = ?, entry_sha256 = ? WHERE record_id = ? AND transition_no = ?",
                     (*(event[name] for name in _HISTORY_COLUMNS), previous, entry, record_id, event_row[1]))
        previous = entry
    values.update(row_changes or {})
    values["history_head"] = previous
    assignments = ", ".join(f"{name} = ?" for name in _RECORD_COLUMNS)
    conn.execute(f"UPDATE journal_records SET {assignments}, row_sha256 = ? WHERE record_id = ?",
                 (*(values[name] for name in _RECORD_COLUMNS), NeoWriteJournal._row_hash(values), record_id))


def _two_epochs(path):
    with NeoWriteJournal(path) as journal:
        first = _resolved(journal, attempt=1)
        second = _next(journal, first)
        journal.mark_in_flight(second.record_id, ledger_attempt=2)
        journal.mark_committed(second.record_id, expected_ledger_attempt=2)
        assert journal.check_integrity().ok
    return first.record_id, second.record_id


def test_triggers_keep_correlation_fixed_and_attempts_rising(journal):
    record = _flying(journal, attempt=5)
    for statement in ("UPDATE journal_records SET correlation_key = 'x'", "UPDATE journal_records SET correlation_key = NULL", "UPDATE journal_records SET epoch = 2",
                      "UPDATE journal_records SET ledger_attempt = 4", "UPDATE journal_records SET ledger_attempt = 5", "UPDATE journal_records SET ledger_attempt = NULL"):
        with pytest.raises(sqlite3.IntegrityError):
            journal._conn.execute(statement)
    assert journal.get_record(record.record_id) == record


@pytest.mark.parametrize("damage", [
    "UPDATE journal_records SET correlation_key = 'other' WHERE record_id = ?",
    "UPDATE journal_records SET correlation_key = NULL WHERE record_id = ?",
    "UPDATE journal_records SET epoch = 7 WHERE record_id = ?",
    "UPDATE journal_records SET ledger_attempt = 9 WHERE record_id = ?",
    "UPDATE journal_records SET ledger_attempt = NULL WHERE record_id = ?",
    "UPDATE journal_history SET correlation_key = 'other' WHERE record_id = ? AND transition_no = 1",
    "UPDATE journal_history SET epoch = 7 WHERE record_id = ? AND transition_no = 2",
    "UPDATE journal_history SET ledger_attempt = 9 WHERE record_id = ? AND transition_no = 2",
    "UPDATE journal_history SET actor = NULL WHERE record_id = ? AND transition_no = 1",
], ids=["key changed", "key removed", "epoch changed", "attempt changed", "attempt removed", "history key changed", "history epoch changed", "history attempt changed", "authorizer removed"])
def test_changed_correlation_data_is_detected(path, damage):
    _, second = _two_epochs(path)
    conn = _raw(path)
    conn.execute(damage, (second,))
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        report = journal.check_integrity()
        assert not report.ok and second in report.corrupt_record_ids


@pytest.mark.parametrize("name, row_changes, event_changes", [
    ("key removed from the row only", {"correlation_key": None, "epoch": None, "ledger_attempt": None}, None),
    ("epoch changed on the row only", {"epoch": 7}, None),
    ("key differs in one history entry", None, {2: {"correlation_key": "other"}}),
    ("attempt on the row isn't the last started", {"ledger_attempt": 9}, None),
    ("attempt in the history doesn't carry forward", None, {3: {"ledger_attempt": 3}}),
    ("attempt missing at the start of an attempt", None, {2: {"ledger_attempt": None}}),
    ("attempt present before anything started", None, {1: {"ledger_attempt": 1}}),
    ("no authorization on a later epoch", None, {1: {"actor": None, "reason": None}}),
], ids=lambda value: value if isinstance(value, str) else "")
def test_a_forged_record_with_valid_checksums_is_detected_by_its_contents(path, name, row_changes, event_changes):
    _, second = _two_epochs(path)
    conn = _raw(path)
    _rewrite(conn, second, row_changes, event_changes)
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        report = journal.check_integrity()
        assert not report.ok and second in report.corrupt_record_ids


def test_a_forged_attempt_that_does_not_rise_within_one_record_is_detected(path):
    with NeoWriteJournal(path) as journal:
        record = _flying(journal, attempt=1)
        journal.mark_not_committed(record.record_id, expected_ledger_attempt=1)
        journal.authorize_retry(record.record_id, authorized_by="o", reason="r", ledger_attempt=2)
        journal.mark_committed(record.record_id, expected_ledger_attempt=2)
        assert journal.check_integrity().ok
    conn = _raw(path)
    _rewrite(conn, record.record_id, {"ledger_attempt": 1}, {4: {"ledger_attempt": 1}, 5: {"ledger_attempt": 1}})
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        assert record.record_id in journal.check_integrity().corrupt_record_ids


@pytest.mark.parametrize("name, value", [("correlation_key", "other"), ("epoch", 2), ("ledger_attempt", 3)])
def test_the_history_chain_itself_covers_each_correlation_field(name, value):
    """Not only the content checks: a change must also break the chain's hash, whatever else is forged."""
    event = {column: f"value-{index}" for index, column in enumerate(_HISTORY_COLUMNS)}
    event.update(correlation_key=KEY, epoch=1, ledger_attempt=1)
    assert NeoWriteJournal._event_hash({**event, name: value}, "p") != NeoWriteJournal._event_hash(event, "p")


@pytest.mark.parametrize("name, value", [("correlation_key", "other"), ("epoch", 2), ("ledger_attempt", 3)])
def test_the_row_checksum_covers_each_correlation_field(name, value):
    values = {column: f"value-{index}" for index, column in enumerate(_RECORD_COLUMNS)}
    values.update(correlation_key=KEY, epoch=1, ledger_attempt=1)
    assert NeoWriteJournal._row_hash({**values, name: value}) != NeoWriteJournal._row_hash(values)


def test_a_forged_first_epoch_claiming_authorization_is_detected(path):
    first, _ = _two_epochs(path)
    conn = _raw(path)
    _rewrite(conn, first, None, {1: {"actor": "someone", "reason": "r"}})
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        assert first in journal.check_integrity().corrupt_record_ids


def test_a_missing_epoch_is_detected(path):
    first, second = _two_epochs(path)
    conn = _raw(path)
    conn.execute("DELETE FROM journal_history WHERE record_id = ?", (first,))
    conn.execute("DELETE FROM journal_records WHERE record_id = ?", (first,))
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        report = journal.check_integrity()
        assert not report.ok and second in report.corrupt_record_ids
        assert any("epoch chain" in problem for problem in report.problems)


def test_a_forged_later_epoch_with_a_different_intent_is_detected(path):
    _, second = _two_epochs(path)
    conn = _raw(path)
    text = conn.execute("SELECT intent_json FROM journal_records WHERE record_id = ?", (second,)).fetchone()[0].replace("E1", "E2")
    _rewrite(conn, second, {"intent_json": text, "intent_sha256": module._sha256(text)})
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        report = journal.check_integrity()
        assert second in report.corrupt_record_ids and any("epoch chain" in problem for problem in report.problems)


@pytest.mark.parametrize("resolution", ["FOUND_COMMITTED", "ABANDONED"])
def test_a_forged_chain_past_an_unretryable_resolution_is_detected(path, resolution):
    first, second = _two_epochs(path)
    conn = _raw(path)
    _rewrite(conn, first, {"resolution": resolution}, {4: {"resolution": resolution}})
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        report = journal.check_integrity()
        assert second in report.corrupt_record_ids and any("epoch chain" in problem for problem in report.problems)


def test_a_forged_attempt_that_does_not_rise_across_epochs_is_detected(path):
    _, second = _two_epochs(path)
    conn = _raw(path)
    _rewrite(conn, second, {"ledger_attempt": 1}, {2: {"ledger_attempt": 1}, 3: {"ledger_attempt": 1}})
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        report = journal.check_integrity()
        assert second in report.corrupt_record_ids


def test_a_forged_correlation_on_an_uncorrelated_record_is_detected(path):
    with NeoWriteJournal(path) as journal:
        record = journal.create_intent(OP, INTENT)
    conn = _raw(path)
    _rewrite(conn, record.record_id, {"correlation_key": KEY, "epoch": 1})
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        assert record.record_id in journal.check_integrity().corrupt_record_ids


def test_a_damaged_record_is_refused_by_every_correlated_path(path):
    first, second = _two_epochs(path)
    conn = _raw(path)
    conn.execute("UPDATE journal_records SET epoch = 7 WHERE record_id = ?", (second,))
    _restore_triggers(conn)
    with NeoWriteJournal(path) as journal:
        for call in (lambda: journal.find_by_correlation_key(KEY), lambda: journal.get_record(second),
                     lambda: journal.create_intent(OP, INTENT, correlation_key=KEY, epoch=3, authorized_by="a", reason="r")):
            with pytest.raises(JournalCorruptionError):
                call()


# ================================================================ what's exposed

def test_keys_stay_out_of_reprs_errors_and_logs_but_correlated_records_still_work(path, caplog):
    caplog.set_level(logging.DEBUG)
    with NeoWriteJournal(path) as journal:
        record = _first(journal, statements=[("Q", {"token": SECRET})])
        assert KEY not in repr(record) and SECRET not in repr(record)
        flying = journal.mark_in_flight(record.record_id, ledger_attempt=2)
        assert KEY not in repr(flying) and KEY not in repr(journal.get_history(record.record_id)[0])
        for call in (lambda: _first(journal, statements=[("Q", {"token": SECRET})]), lambda: journal.mark_committed(record.record_id, expected_ledger_attempt=1),
                     lambda: journal.create_intent(OP, INTENT, correlation_key=KEY, epoch=5, authorized_by="a", reason="r")):
            with pytest.raises(module.JournalError) as caught:
                call()
            assert KEY not in str(caught.value) and SECRET not in str(caught.value)
        journal.mark_ambiguous(record.record_id, expected_ledger_attempt=2)
    assert KEY not in caplog.text and SECRET not in caplog.text


def test_the_row_checksum_of_a_record_without_correlation_is_the_version_1_formula():
    """Older rows verify unchanged only if this holds. The frozen copy is the previous implementation itself."""
    from tests import _neo4j_journal_v1_frozen as frozen
    values = {name: f"value-{index}" for index, name in enumerate(frozen._RECORD_COLUMNS)}
    event = {name: f"value-{index}" for index, name in enumerate(frozen._HISTORY_COLUMNS)}
    correlation = dict.fromkeys(module._CORRELATION_COLUMNS)
    assert NeoWriteJournal._row_hash({**values, **correlation}) == frozen.NeoWriteJournal._row_hash(values)
    assert NeoWriteJournal._event_hash({**event, **correlation}, "p") == frozen.NeoWriteJournal._event_hash(event, "p")
    assert NeoWriteJournal._row_hash({**values, **correlation, "epoch": 1}) != frozen.NeoWriteJournal._row_hash(values)


def test_counters_are_distinct_after_a_retry_and_a_new_epoch(journal):
    """Logical request, epoch, ledger attempt, intent id and the record's own count: none stands in for another."""
    record = _flying(journal, attempt=4)
    journal.mark_not_committed(record.record_id, expected_ledger_attempt=4)
    journal.authorize_retry(record.record_id, authorized_by="o", reason="r", ledger_attempt=6)
    journal.mark_ambiguous(record.record_id, expected_ledger_attempt=6)
    journal.resolve_ambiguous(record.record_id, resolution=Resolution.FOUND_NOT_COMMITTED, resolved_by="a", reason="r")
    first = journal.get_record(record.record_id)
    assert (first.correlation_key, first.epoch, first.ledger_attempt, first.attempt_count) == (KEY, 1, 6, 2)
    second = journal.mark_in_flight(_next(journal, first).record_id, ledger_attempt=9)
    assert (second.correlation_key, second.epoch, second.ledger_attempt, second.attempt_count) == (KEY, 2, 9, 1)
    assert second.record_id != first.record_id
    assert json.loads(json.dumps([second.epoch, second.ledger_attempt, second.attempt_count])) == [2, 9, 1]

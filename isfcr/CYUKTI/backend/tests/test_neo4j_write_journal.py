"""Unit tests for neo4j_write_journal: storage, exact preservation, the state machine, the API.
Everything lives in pytest's temporary directories. Real process death is in
test_neo4j_write_journal_crash.py, and failing storage in test_neo4j_write_journal_faults.py."""
from __future__ import annotations

import ast
import collections
import datetime as dt
import decimal
import enum
import json
import logging
import os
import random
import stat
import subprocess
import sys
import threading
import zoneinfo
from pathlib import Path

import neo4j.spatial
import neo4j.time as neo4j_time
import pytest
import pytz

import neo4j_write_journal as module
from neo4j_write_journal import (
    InvalidTransitionError,
    JournalClosedError,
    JournalReadOnlyError,
    JournalState as S,
    NeoWriteJournal,
    RecordNotFoundError,
    Resolution,
    UnsupportedParameterError,
    _identical,
)

BACKEND = Path(module.__file__).resolve().parent
SECRET = "s3cret-parameter-value"
ONE = [("CREATE (n:X {v:$v})", {"v": 1})]


@pytest.fixture
def path(tmp_path):
    return tmp_path / "journal.db"


@pytest.fixture
def journal(path):
    with NeoWriteJournal(path) as instance:
        yield instance


def _in_flight(journal, statements=ONE):
    record = journal.create_intent("op", statements)
    return journal.mark_in_flight(record.record_id, worker_id="w1")


def _ambiguous(journal):
    record = _in_flight(journal)
    return journal.mark_ambiguous(record.record_id, error_type="IncompleteCommit", diagnostic="IncompleteCommit: connection lost during commit")


def _not_committed(journal):
    record = _in_flight(journal)
    return journal.mark_not_committed(record.record_id, error_type="ServiceUnavailable")


# ================================================================ initialization and storage

def test_a_new_journal_has_the_schema_version_and_is_empty(journal):
    report = journal.check_integrity()
    assert report.ok and report.schema_version == module.SCHEMA_VERSION == 2
    assert journal.list_records() == ()


def test_the_durable_settings_are_actually_in_force(journal):
    assert journal._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert journal._conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL


def test_a_new_journal_file_is_private_to_its_owner(path, journal):
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_there_is_no_default_path():
    with pytest.raises(TypeError):
        NeoWriteJournal()


def test_a_missing_directory_is_refused_rather_than_created(tmp_path):
    with pytest.raises(module.JournalError):
        NeoWriteJournal(tmp_path / "absent" / "journal.db")
    assert not (tmp_path / "absent").exists()


def test_reopening_keeps_records_and_does_not_touch_them(path):
    with NeoWriteJournal(path) as first:
        record = first.create_intent("op", ONE)
    with NeoWriteJournal(path) as second:
        again = second.get_record(record.record_id)
        assert again == record
        assert second.check_integrity().ok


def test_opening_an_empty_file_initializes_it(path):
    path.touch()
    with NeoWriteJournal(path) as journal:
        assert journal.check_integrity().ok


def test_close_is_safe_to_repeat_and_blocks_further_use(path):
    journal = NeoWriteJournal(path)
    record = journal.create_intent("op", ONE)
    journal.close()
    journal.close()
    for call in (lambda: journal.get_record(record.record_id), lambda: journal.create_intent("op", ONE),
                 lambda: journal.mark_in_flight(record.record_id), journal.recover_unfinished, journal.check_integrity):
        with pytest.raises(JournalClosedError):
            call()


def test_a_read_only_instance_reads_and_refuses_changes(path):
    with NeoWriteJournal(path) as writer:
        record = writer.create_intent("op", ONE)
        with NeoWriteJournal(path, read_only=True) as reader:
            assert reader.get_record(record.record_id).state is S.PENDING
            assert reader.check_integrity().ok
            assert reader.session_id is None
            for call in (lambda: reader.create_intent("op", ONE), lambda: reader.mark_in_flight(record.record_id), reader.recover_unfinished):
                with pytest.raises(JournalReadOnlyError):
                    call()
            writer.mark_in_flight(record.record_id)
            assert reader.get_record(record.record_id).state is S.IN_FLIGHT


def test_read_only_never_creates_a_journal(path):
    with pytest.raises(module.JournalError, match="no journal exists"):
        NeoWriteJournal(path, read_only=True)
    assert not path.exists()


# ================================================================ exact preservation

class _Weird(dt.tzinfo):
    def utcoffset(self, when):
        return dt.timedelta(hours=1)

    def dst(self, when):
        return None

    def tzname(self, when):
        return "weird"


PRESERVED = {
    "unicode": "héllo ☃ 𝒳",
    "nul and controls": "a\x00b\n\t\r",
    "lone surrogate": "\ud800",
    "empty string": "",
    "looks like a tag": {"$map": {"$bytes": "x"}, "$float": "nan"},
    "max int": 2 ** 63 - 1,
    "min int": -(2 ** 63),
    "beyond float precision": 2 ** 53 + 1,
    "float tenth": 0.1,
    "float max": 1.7976931348623157e308,
    "float min subnormal": 5e-324,
    "negative zero": -0.0,
    "whole float stays float": 1.0,
    "nan": float("nan"),
    "infinity": float("inf"),
    "negative infinity": float("-inf"),
    "true": True,
    "false": False,
    "null": None,
    "empty list": [],
    "empty map": {},
    "tuple stays tuple": (1, "a", (None,)),
    "nested": {"a": [1, {"b": [None, True, 2.5, {"c": []}]}], "z": {"y": {"x": "w"}}},
    "bytes": b"\x00\xff\x7f",
    "bytearray": bytearray(b"\x01\x02"),
    "aware utc datetime": dt.datetime(2026, 1, 1, 0, 10, 0, 123456, tzinfo=dt.timezone.utc),
    "naive datetime": dt.datetime(2026, 1, 1, 0, 10),
    "fixed offset datetime": dt.datetime(2026, 3, 4, 12, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))),
    "named offset datetime": dt.datetime(2026, 3, 4, 12, tzinfo=dt.timezone(dt.timedelta(hours=-3), "BRT")),
    "zoneinfo datetime": dt.datetime(2026, 7, 1, 12, tzinfo=zoneinfo.ZoneInfo("Europe/Paris")),
    "zoneinfo second fold": dt.datetime(2026, 10, 25, 2, 30, tzinfo=zoneinfo.ZoneInfo("Europe/Paris"), fold=1),
    "pytz datetime": pytz.timezone("Asia/Kolkata").localize(dt.datetime(2026, 3, 4, 12)),
    "pytz summer": pytz.timezone("Europe/Paris").localize(dt.datetime(2026, 7, 1, 12)),
    "pytz ambiguous hour, second": pytz.timezone("Europe/Paris").localize(dt.datetime(2026, 10, 25, 2, 30), is_dst=False),
    "pytz ambiguous hour, first": pytz.timezone("Europe/Paris").localize(dt.datetime(2026, 10, 25, 2, 30), is_dst=True),
    "pytz utc": dt.datetime(2026, 1, 1, tzinfo=pytz.utc),
    "pytz fixed": dt.datetime(2026, 1, 1, tzinfo=pytz.FixedOffset(330)),
    "date": dt.date(2026, 3, 4),
    "time": dt.time(12, 34, 56, 789),
    "time with offset": dt.time(12, 34, tzinfo=dt.timezone(dt.timedelta(hours=2))),
    "time in a named zone": dt.time(12, tzinfo=zoneinfo.ZoneInfo("Europe/Paris")),
    "explicitly named utc-offset zone": dt.datetime(2026, 1, 1, tzinfo=dt.timezone(dt.timedelta(0), "GMT")),
    "timedelta": dt.timedelta(days=-2, seconds=3, microseconds=4),
    "neo4j date": neo4j_time.Date(2026, 3, 4),
    "neo4j time": neo4j_time.Time(12, 34, 56, 123456789),
    "neo4j time with offset": neo4j_time.Time(12, 34, 56, 123456789, tzinfo=pytz.FixedOffset(330)),
    "neo4j local datetime": neo4j_time.DateTime(2026, 3, 4, 12, 34, 56, 123456789),
    "neo4j datetime, pytz fixed": neo4j_time.DateTime(2026, 3, 4, 12, 34, 56, 123456789, tzinfo=pytz.FixedOffset(330)),
    "neo4j datetime, named zone": pytz.timezone("Asia/Kolkata").localize(neo4j_time.DateTime(2026, 3, 4, 12, 34, 56, 123456789)),
    "neo4j datetime, utc": neo4j_time.DateTime(2026, 3, 4, 12, 34, 56, 0, tzinfo=pytz.utc),
    "neo4j duration": neo4j_time.Duration(months=1, days=2, seconds=3, nanoseconds=4),
    "negative neo4j duration": neo4j_time.Duration(months=-1, days=-2, seconds=-3, nanoseconds=-4),
}


@pytest.mark.parametrize("value", PRESERVED.values(), ids=PRESERVED.keys())
def test_a_value_comes_back_identical_after_closing_and_reopening(path, value):
    with NeoWriteJournal(path) as first:
        record = first.create_intent("op", [("Q", {"p": value})])
    with NeoWriteJournal(path) as second:
        restored = second.get_record(record.record_id).statements[0][1]["p"]
    assert _identical(restored, value)
    assert type(restored) is type(value)


def test_order_and_structure_of_a_batch_are_preserved(path):
    statements = [("A $z $a", {"z": 1, "a": 2, "m": 3}), ("B", None), ("C", {}), ("D", {"records": [{"z": 1, "a": 2}]})]
    with NeoWriteJournal(path) as journal:
        record = journal.create_intent("op", statements)
        restored = journal.get_record(record.record_id).statements
    assert [q for q, _ in restored] == ["A $z $a", "B", "C", "D"]
    assert list(restored[0][1]) == ["z", "a", "m"]
    assert restored[1][1] is None and restored[2][1] == {}
    assert list(restored[3][1]["records"][0]) == ["z", "a"]


def test_the_real_duplicate_batch_shape_round_trips(journal):
    """What batch_update_duplicates passes, and the aware datetime every writer's event_time is."""
    rows = [{"fingerprint": "f" * 64, "event_id": "E", "campaign_id": "CAMP_1", "occurrences": 3, "total_tps": 12.5,
             "latest_payload": '{"a": 1}', "last_seen": "2026-01-01T00:10:00+00:00"} for _ in range(150)]
    statements = [("UNWIND $records AS r RETURN r", {"records": rows, "event_time": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)})]
    assert _identical(journal.create_intent("flush", statements).statements, tuple(map(tuple, statements)))


def test_return_rows_is_part_of_the_intent(journal):
    assert journal.create_intent("op", ONE, return_rows=True).return_rows is True
    assert journal.create_intent("op", ONE).return_rows is False


def test_statements_can_be_any_iterable_and_are_read_once(journal):
    record = journal.create_intent("op", (s for s in ONE))
    assert record.statements[0][1] == {"v": 1}


def test_serialization_of_the_same_input_is_deterministic():
    statements = [("Q", {"a": [1, 2.5, None, {"b": dt.date(2026, 1, 1)}]})]
    assert module._serialize_intent(statements, False)[0] == module._serialize_intent(statements, False)[0]


_cycle = []
_cycle.append(_cycle)
_deep: list = []
for _ in range(module._MAX_NESTING + 1):
    _deep = [_deep]


class _StrSub(str):
    pass


class _IntEnum(enum.IntEnum):
    A = 1


class _FloatSub(float):
    pass


class _ListSub(list):
    pass


REFUSED = {
    "set": {1, 2},
    "frozenset": frozenset({1}),
    "object": object(),
    "complex": 1j,
    "decimal": decimal.Decimal("1.5"),
    "above 64 bits": 2 ** 63,
    "below 64 bits": -(2 ** 63) - 1,
    "str subclass": _StrSub("x"),
    "int enum": _IntEnum.A,
    "float subclass": _FloatSub(1.5),
    "list subclass": _ListSub([1]),
    "dict subclass": collections.OrderedDict(a=1),
    "non-string map key": {1: "a"},
    "reference cycle": _cycle,
    "nested too deeply": _deep,
    "unknown timezone type": dt.datetime(2026, 1, 1, tzinfo=_Weird()),
    "spatial point": neo4j.spatial.WGS84Point((1.0, 2.0)),
    "time with a pytz zone": dt.time(12, tzinfo=pytz.timezone("Asia/Kolkata")),
    "pytz zone attached without localize": dt.datetime(2026, 1, 1, tzinfo=pytz.timezone("Asia/Kolkata")),
    "nested inside a batch": {"records": [{"ok": 1, "bad": {1, 2}}]},
}


@pytest.mark.parametrize("value", REFUSED.values(), ids=REFUSED.keys())
def test_a_value_that_cannot_be_preserved_is_refused_and_creates_nothing(journal, value):
    with pytest.raises(UnsupportedParameterError):
        journal.create_intent("op", [("Q", {"p": value})])
    assert journal.list_records() == ()


@pytest.mark.parametrize("value, words", [
    ({1, 2}, "builtins.set is not supported"),
    (_StrSub("x"), "_StrSub is not supported"),
    (_IntEnum.A, "_IntEnum is not supported"),
    (_FloatSub(1.5), "_FloatSub is not supported"),
    (_ListSub([1]), "_ListSub is not supported"),
    (collections.OrderedDict(a=1), "OrderedDict is not supported"),
    (_cycle, "reference cycle"),
    (_deep, "nested more than"),
], ids=["set", "str subclass", "int enum", "float subclass", "list subclass", "dict subclass", "cycle", "too deep"])
def test_the_first_check_explains_the_refusal_rather_than_the_round_trip_catching_it(journal, value, words):
    """The round trip is the backstop. The reason should reach the caller before it's needed."""
    with pytest.raises(UnsupportedParameterError, match=words):
        journal.create_intent("op", [("Q", {"p": value})])


@pytest.mark.parametrize("statements, error", [
    ([], ValueError), ((), ValueError), (iter([]), ValueError), (None, TypeError), (7, TypeError), ("Q", TypeError),
    ([("Q",)], TypeError), (["Q"], TypeError), ([("", None)], ValueError), ([("  ", None)], ValueError), ([(1, None)], ValueError),
    ([("Q", 5)], TypeError), ([("Q", [1])], TypeError), ([("Q", {1: "a"})], TypeError),
], ids=["empty list", "empty tuple", "empty iterator", "None", "int", "bare string", "short pair", "not a pair", "empty query",
        "blank query", "query not a string", "parameters an int", "parameters a list", "parameter name not a string"])
def test_a_malformed_batch_creates_nothing(journal, statements, error):
    with pytest.raises(error):
        journal.create_intent("op", statements)
    assert journal.list_records() == ()


@pytest.mark.parametrize("operation", ["", "  ", None, 5])
def test_an_operation_name_is_required(journal, operation):
    with pytest.raises((ValueError, TypeError)):
        journal.create_intent(operation, ONE)


def test_a_returned_record_cannot_be_used_to_change_the_journal(journal):
    record = journal.create_intent("op", [("Q", {"a": [1, {"b": 2}]})])
    record.statements[0][1]["a"].append("changed")
    record.statements[0][1]["a"][1]["b"] = "changed"
    assert record.statements[0][1] == {"a": [1, {"b": 2}]}
    assert journal.get_record(record.record_id).statements[0][1] == {"a": [1, {"b": 2}]}
    with pytest.raises(AttributeError):
        record.state = S.COMMITTED


def test_changing_the_callers_objects_afterwards_does_not_change_the_intent(journal):
    parameters = {"a": [1]}
    record = journal.create_intent("op", [("Q", parameters)])
    parameters["a"].append(2)
    assert journal.get_record(record.record_id).statements[0][1] == {"a": [1]}


def test_only_the_intent_columns_of_the_stored_row_hold_parameters(journal):
    record = journal.create_intent("op", [("Q", {"secret": SECRET})])
    journal.mark_in_flight(record.record_id)
    row = journal._conn.execute("SELECT * FROM journal_history").fetchall()
    assert SECRET not in json.dumps(row)


# ================================================================ state machine

def _reach(journal, state):
    if state is S.PENDING:
        return journal.create_intent("op", ONE)
    if state is S.IN_FLIGHT:
        return _in_flight(journal)
    if state is S.COMMITTED:
        return journal.mark_committed(_in_flight(journal).record_id)
    if state is S.NOT_COMMITTED:
        return _not_committed(journal)
    if state is S.AMBIGUOUS:
        return _ambiguous(journal)
    return journal.resolve_ambiguous(_ambiguous(journal).record_id, resolution=Resolution.ABANDONED, resolved_by="op", reason="r")


MOVES = {
    "mark_in_flight": lambda j, i: j.mark_in_flight(i),
    "mark_committed": lambda j, i: j.mark_committed(i),
    "mark_not_committed": lambda j, i: j.mark_not_committed(i),
    "mark_ambiguous": lambda j, i: j.mark_ambiguous(i),
    "authorize_retry": lambda j, i: j.authorize_retry(i, authorized_by="op", reason="r"),
    "resolve_ambiguous": lambda j, i: j.resolve_ambiguous(i, resolution=Resolution.ABANDONED, resolved_by="op", reason="r"),
}
ALLOWED = {
    (S.PENDING, "mark_in_flight"): S.IN_FLIGHT,
    (S.PENDING, "mark_not_committed"): S.NOT_COMMITTED,
    (S.IN_FLIGHT, "mark_committed"): S.COMMITTED,
    (S.IN_FLIGHT, "mark_not_committed"): S.NOT_COMMITTED,
    (S.IN_FLIGHT, "mark_ambiguous"): S.AMBIGUOUS,
    (S.NOT_COMMITTED, "authorize_retry"): S.IN_FLIGHT,
    (S.AMBIGUOUS, "resolve_ambiguous"): S.RESOLVED,
}


@pytest.mark.parametrize("state", list(S), ids=[s.value for s in S])
@pytest.mark.parametrize("move", MOVES, ids=list(MOVES))
def test_every_move_from_every_state_is_exactly_as_specified(journal, state, move):
    record = _reach(journal, state)
    before = journal.get_record(record.record_id)
    history = journal.get_history(record.record_id)
    expected = ALLOWED.get((state, move))
    if expected is not None:
        assert MOVES[move](journal, record.record_id).state is expected
        assert len(journal.get_history(record.record_id)) == len(history) + 1
    else:
        with pytest.raises(InvalidTransitionError):
            MOVES[move](journal, record.record_id)
        assert journal.get_record(record.record_id) == before
        assert journal.get_history(record.record_id) == history
    assert journal.check_integrity().ok


def test_the_table_in_the_module_is_the_specified_lifecycle():
    allowed = set(module._TRANSITIONS)
    assert allowed == {(S.PENDING, S.IN_FLIGHT), (S.PENDING, S.NOT_COMMITTED), (S.IN_FLIGHT, S.COMMITTED), (S.IN_FLIGHT, S.NOT_COMMITTED),
                       (S.IN_FLIGHT, S.AMBIGUOUS), (S.NOT_COMMITTED, S.IN_FLIGHT), (S.AMBIGUOUS, S.RESOLVED)}
    assert not any(source is S.AMBIGUOUS and target is not S.RESOLVED for source, target in allowed)
    assert not any(source in (S.COMMITTED, S.RESOLVED) for source, _ in allowed)
    assert module._TRANSITIONS[(S.NOT_COMMITTED, S.IN_FLIGHT)] == {module._Cause.RETRY}


def test_an_unknown_record_is_reported(journal):
    for call in (lambda: journal.get_record("nope"), lambda: journal.mark_in_flight("nope"), lambda: journal.get_history("nope")):
        with pytest.raises(RecordNotFoundError):
            call()


def test_a_normal_life_is_recorded_in_order(journal):
    record = journal.create_intent("op", ONE)
    assert record.state is S.PENDING and record.attempt_count == 0 and record.version == 1
    flying = journal.mark_in_flight(record.record_id, worker_id="w1")
    assert flying.state is S.IN_FLIGHT and flying.attempt_count == 1 and flying.worker_id == "w1"
    assert flying.owner_session == journal.session_id
    done = journal.mark_committed(record.record_id)
    assert done.state is S.COMMITTED and done.last_outcome == "COMMITTED" and not done.retry_eligible
    assert done.updated_at >= done.created_at and done.created_at == record.created_at
    events = journal.get_history(record.record_id)
    assert [(e.from_state, e.to_state, e.cause) for e in events] == [
        (None, S.PENDING, "CREATED"), (S.PENDING, S.IN_FLIGHT, "EXECUTE"), (S.IN_FLIGHT, S.COMMITTED, "OUTCOME")]
    assert [e.transition_no for e in events] == [1, 2, 3]
    assert [e.attempt for e in events] == [0, 1, 1]


def test_error_details_are_stored_and_kept_in_history_after_they_clear(journal):
    record = _in_flight(journal)
    failed = journal.mark_not_committed(record.record_id, error_type="ServiceUnavailable", error_code="Neo.X", diagnostic="d")
    assert (failed.error_type, failed.error_code, failed.diagnostic, failed.last_outcome) == ("ServiceUnavailable", "Neo.X", "d", "NOT_COMMITTED")
    retried = journal.authorize_retry(record.record_id, authorized_by="op", reason="r")
    assert retried.error_type is None and retried.last_outcome == "NOT_COMMITTED"
    assert journal.get_history(record.record_id)[2].error_type == "ServiceUnavailable"


def test_an_exception_is_refused_as_an_error_detail_because_its_message_can_quote_values(journal):
    record = _in_flight(journal)
    with pytest.raises(TypeError, match="diagnostic must be a string or None"):
        journal.mark_ambiguous(record.record_id, diagnostic=ValueError(SECRET))
    assert journal.get_record(record.record_id).state is S.IN_FLIGHT


def test_long_details_are_cut_to_a_limit(journal):
    record = _in_flight(journal)
    assert len(journal.mark_ambiguous(record.record_id, diagnostic="x" * 5000).diagnostic) == module._MAX_DETAIL_LENGTH


# ================================================================ retry and resolution

def test_retry_keeps_the_intent_and_the_whole_history(journal):
    original = [("CREATE (n:X {v:$v})", {"v": 2 ** 53 + 1})]
    record = journal.create_intent("op", original)
    journal.mark_in_flight(record.record_id)
    journal.mark_not_committed(record.record_id, error_type="ServiceUnavailable")
    retried = journal.authorize_retry(record.record_id, authorized_by="operator", reason="connection restored", worker_id="w2")
    assert retried.state is S.IN_FLIGHT and retried.attempt_count == 2 and retried.worker_id == "w2"
    assert _identical(retried.statements, tuple(map(tuple, original)))
    assert retried.intent_sha256 == record.intent_sha256 and retried.record_id == record.record_id
    event = journal.get_history(record.record_id)[-1]
    assert (event.cause, event.actor, event.reason, event.attempt) == ("RETRY", "operator", "connection restored", 2)
    assert journal.mark_committed(record.record_id).state is S.COMMITTED


@pytest.mark.parametrize("field_name", ["authorized_by", "reason"])
@pytest.mark.parametrize("value", ["", "  ", None])
def test_a_retry_needs_a_named_authorizer_and_a_reason(journal, field_name, value):
    record = _not_committed(journal)
    arguments = {"authorized_by": "op", "reason": "r", field_name: value}
    with pytest.raises(ValueError):
        journal.authorize_retry(record.record_id, **arguments)
    assert journal.get_record(record.record_id).state is S.NOT_COMMITTED


def test_mark_in_flight_cannot_be_used_to_retry(journal):
    record = _not_committed(journal)
    with pytest.raises(InvalidTransitionError):
        journal.mark_in_flight(record.record_id)


@pytest.mark.parametrize("resolution", list(Resolution), ids=[r.value for r in Resolution])
def test_resolution_keeps_the_ambiguity_and_records_who_when_and_why(journal, resolution):
    record = _ambiguous(journal)
    resolved = journal.resolve_ambiguous(record.record_id, resolution=resolution, resolved_by="analyst", reason="checked the database")
    assert resolved.state is S.RESOLVED and resolved.resolution is resolution
    assert resolved.last_outcome == "AMBIGUOUS", "what the write's outcome was known to be doesn't change"
    assert not resolved.retry_eligible
    events = journal.get_history(record.record_id)
    assert [e.to_state for e in events] == [S.PENDING, S.IN_FLIGHT, S.AMBIGUOUS, S.RESOLVED]
    assert events[2].error_type == "IncompleteCommit"
    assert (events[3].actor, events[3].reason, events[3].resolution) == ("analyst", "checked the database", resolution)
    assert events[3].at
    assert resolved.statements == record.statements


@pytest.mark.parametrize("arguments", [
    {"resolution": Resolution.ABANDONED, "resolved_by": "", "reason": "r"},
    {"resolution": Resolution.ABANDONED, "resolved_by": "a", "reason": " "},
    {"resolution": "NOT_A_RESOLUTION", "resolved_by": "a", "reason": "r"},
], ids=["no one", "no reason", "unknown resolution"])
def test_a_resolution_must_be_complete(journal, arguments):
    record = _ambiguous(journal)
    with pytest.raises(ValueError):
        journal.resolve_ambiguous(record.record_id, **arguments)
    assert journal.get_record(record.record_id).state is S.AMBIGUOUS


@pytest.mark.parametrize("resolution", list(Resolution), ids=[r.value for r in Resolution])
def test_no_resolution_makes_a_record_replayable(journal, resolution):
    record = _ambiguous(journal)
    journal.resolve_ambiguous(record.record_id, resolution=resolution, resolved_by="a", reason="r")
    for move in MOVES.values():
        with pytest.raises(InvalidTransitionError):
            move(journal, record.record_id)


def test_random_walks_never_let_an_ambiguous_record_run_again(journal):
    """Every order of every call, many times: once AMBIGUOUS, no path leads to IN_FLIGHT."""
    generator = random.Random(20261105)
    became_ambiguous = 0
    for _ in range(150):
        record = journal.create_intent("op", ONE)
        seen_ambiguous = False
        for _ in range(10):
            name = generator.choice(list(MOVES))
            try:
                state = MOVES[name](journal, record.record_id).state
            except InvalidTransitionError:
                state = journal.get_record(record.record_id).state
            seen_ambiguous = seen_ambiguous or state is S.AMBIGUOUS
            if seen_ambiguous:
                assert state in (S.AMBIGUOUS, S.RESOLVED)
                assert not journal.get_record(record.record_id).retry_eligible
        became_ambiguous += seen_ambiguous
        journal.recover_unfinished()
    assert became_ambiguous > 10, "the walk has to reach the case it is testing"
    assert journal.check_integrity().ok


# ================================================================ listing, recovery in one process

def test_listing_filters_by_state_in_creation_order(journal):
    pending = [journal.create_intent("op", ONE) for _ in range(3)]
    journal.mark_in_flight(pending[1].record_id)
    assert [r.record_id for r in journal.list_records(S.PENDING)] == [pending[0].record_id, pending[2].record_id]
    assert [r.record_id for r in journal.list_records("IN_FLIGHT")] == [pending[1].record_id]
    assert len(journal.list_records()) == 3
    assert journal.list_records(S.COMMITTED) == ()
    with pytest.raises(ValueError):
        journal.list_records("NOT_A_STATE")


def test_an_instance_never_recovers_its_own_running_work(journal):
    """Its worker may still be executing. Only the owner's death allows the conversion."""
    record = _in_flight(journal)
    report = journal.recover_unfinished()
    assert report.still_running == (record.record_id,) and report.recovered == ()
    assert journal.get_record(record.record_id).state is S.IN_FLIGHT


def test_a_live_instance_in_the_same_process_counts_as_alive(path):
    with NeoWriteJournal(path) as owner, NeoWriteJournal(path) as other:
        record = _in_flight(owner)
        report = other.recover_unfinished()
        assert report.still_running == (record.record_id,) and report.recovered == ()
        assert owner.mark_committed(record.record_id).state is S.COMMITTED


def test_recovery_after_a_clean_close_treats_unfinished_work_as_ambiguous(path):
    """close() doesn't stop a database call, so a clean close proves nothing about the outcome."""
    with NeoWriteJournal(path) as first:
        record = _in_flight(first)
    with NeoWriteJournal(path) as second:
        report = second.recover_unfinished()
        assert report.recovered == (record.record_id,) and report.ambiguous == (record.record_id,)
        recovered = second.get_record(record.record_id)
        assert recovered.state is S.AMBIGUOUS and recovered.last_outcome == "AMBIGUOUS"
        assert recovered.error_type == "UnresolvedAfterRestart"
        assert recovered.recovery_metadata["previous_owner_session"] == first.session_id
        assert recovered.recovery_metadata["recovered_by_session"] == second.session_id
        assert recovered.statements == record.statements
        assert second.get_history(record.record_id)[-1].cause == "RECOVERY"


def test_recovery_leaves_every_other_state_alone_and_reports_it(path):
    with NeoWriteJournal(path) as first:
        ids = {state: _reach(first, state).record_id for state in S}
    with NeoWriteJournal(path) as second:
        before = {state: second.get_record(i) for state, i in ids.items()}
        report = second.recover_unfinished()
        assert report.pending == (ids[S.PENDING],) and report.not_committed == (ids[S.NOT_COMMITTED],)
        assert set(report.ambiguous) == {ids[S.AMBIGUOUS], ids[S.IN_FLIGHT]} and report.recovered == (ids[S.IN_FLIGHT],)
        for state, record in before.items():
            if state is not S.IN_FLIGHT:
                assert second.get_record(ids[state]) == record


def test_recovery_is_repeatable_and_adds_nothing_the_second_time(path):
    with NeoWriteJournal(path) as first:
        record = _in_flight(first)
    with NeoWriteJournal(path) as second:
        second.recover_unfinished()
        events = second.get_history(record.record_id)
        again = second.recover_unfinished()
        assert again.recovered == () and again.ambiguous == (record.record_id,)
        assert second.get_history(record.record_id) == events


def test_two_recoveries_at_once_convert_each_record_exactly_once(path):
    with NeoWriteJournal(path) as first:
        ids = [_in_flight(first).record_id for _ in range(20)]
    reports = []

    def recover():
        with NeoWriteJournal(path) as instance:
            reports.append(instance.recover_unfinished())

    threads = [threading.Thread(target=recover) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert len(reports) == 4
    with NeoWriteJournal(path) as final:
        final.recover_unfinished()
        assert all(final.get_record(i).state is S.AMBIGUOUS for i in ids)
        for i in ids:
            assert sum(e.cause == "RECOVERY" for e in final.get_history(i)) == 1
        assert final.check_integrity().ok


def test_a_worker_that_finishes_after_the_caller_stopped_waiting_still_records_its_outcome(journal):
    """Scenario 8: the join times out, nothing is concluded, and no second execution starts."""
    record = _in_flight(journal)
    release = threading.Event()
    executions = []

    def worker():
        executions.append("query")
        release.wait(10)
        journal.mark_committed(record.record_id)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=0.05)
    assert thread.is_alive()
    assert journal.get_record(record.record_id).state is S.IN_FLIGHT, "a caller's timeout concludes nothing"
    assert journal.recover_unfinished().still_running == (record.record_id,)
    release.set()
    thread.join(10)
    assert journal.get_record(record.record_id).state is S.COMMITTED
    assert executions == ["query"]


def test_a_worker_finishing_after_close_gets_an_error_not_a_false_state(path):
    journal = NeoWriteJournal(path)
    record = _in_flight(journal)
    journal.close()
    with pytest.raises(JournalClosedError):
        journal.mark_committed(record.record_id)
    with NeoWriteJournal(path) as reopened:
        assert reopened.get_record(record.record_id).state is S.IN_FLIGHT


def test_a_late_outcome_cannot_overwrite_a_recovered_record(path):
    with NeoWriteJournal(path) as first:
        record = _in_flight(first)
    with NeoWriteJournal(path) as second:
        second.recover_unfinished()
        for move in ("mark_committed", "mark_not_committed"):
            with pytest.raises(InvalidTransitionError):
                MOVES[move](second, record.record_id)
        assert second.get_record(record.record_id).state is S.AMBIGUOUS


# ================================================================ concurrency in one process

def test_threads_racing_for_one_transition_produce_one_winner(journal):
    record = journal.create_intent("op", ONE)
    results = []
    barrier = threading.Barrier(16)

    def attempt():
        barrier.wait()
        try:
            journal.mark_in_flight(record.record_id)
            results.append("won")
        except InvalidTransitionError:
            results.append("refused")

    threads = [threading.Thread(target=attempt) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert sorted(results) == ["refused"] * 15 + ["won"]
    assert journal.get_record(record.record_id).attempt_count == 1
    assert len(journal.get_history(record.record_id)) == 2


def test_two_instances_racing_produce_one_winner(path):
    with NeoWriteJournal(path) as a, NeoWriteJournal(path) as b:
        record = a.create_intent("op", ONE)
        results = []
        barrier = threading.Barrier(2)

        def attempt(instance):
            barrier.wait()
            try:
                instance.mark_in_flight(record.record_id)
                results.append("won")
            except InvalidTransitionError:
                results.append("refused")

        threads = [threading.Thread(target=attempt, args=(i,)) for i in (a, b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert sorted(results) == ["refused", "won"]
        assert a.check_integrity().ok


def test_many_threads_writing_different_records_keep_the_journal_sound(journal):
    def work():
        for _ in range(10):
            record = journal.create_intent("op", ONE)
            journal.mark_in_flight(record.record_id)
            journal.mark_committed(record.record_id)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert len(journal.list_records(S.COMMITTED)) == 60
    report = journal.check_integrity()
    assert report.ok and report.records_checked == 60


# ================================================================ safety of what's exposed

def test_nothing_sensitive_reaches_repr_errors_or_logs(path, caplog):
    caplog.set_level(logging.DEBUG)
    with NeoWriteJournal(path) as journal:
        record = journal.create_intent("op", [("Q", {"token": SECRET})])
        assert SECRET not in repr(record)
        with pytest.raises(InvalidTransitionError) as caught:
            journal.mark_committed(record.record_id)
        assert SECRET not in str(caught.value)
        with pytest.raises(UnsupportedParameterError) as caught:
            journal.create_intent("op", [("Q", {"token": {SECRET}})])
        assert SECRET not in str(caught.value)
        journal.mark_in_flight(record.record_id)
        journal.mark_ambiguous(record.record_id)
        assert record.statements[0][1]["token"] == SECRET, "the intent itself must still be recoverable"
    assert SECRET not in caplog.text


def test_triggers_keep_history_append_only_and_intents_fixed(journal):
    record = journal.create_intent("op", ONE)
    conn = journal._conn
    for statement in ("UPDATE journal_history SET actor = 'x'", "DELETE FROM journal_history", "DELETE FROM journal_records",
                      "UPDATE journal_records SET intent_json = '{}'", "UPDATE journal_records SET operation = 'x'",
                      "UPDATE journal_records SET state = 'NOT_A_STATE'"):
        with pytest.raises(module.sqlite3.IntegrityError):
            conn.execute(statement)
    assert journal.get_record(record.record_id) == record


def test_journal_methods_never_reach_a_driver_and_there_is_no_replay_method():
    public = {name for name in vars(NeoWriteJournal) if not name.startswith("_")}
    assert public == {"create_intent", "get_record", "list_records", "get_history", "mark_in_flight", "mark_committed", "mark_not_committed",
                      "mark_ambiguous", "authorize_retry", "resolve_ambiguous", "recover_unfinished", "check_integrity", "close",
                      "preflight_intent", "find_by_correlation_key"}
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert not any("replay" in name.lower() or "execute" == name.lower() for name in names)


def test_the_module_loads_neither_the_client_nor_configuration():
    code = ("import sys; sys.path.insert(0, %r); from tests._production_safety import install_child_guards; install_child_guards(); "
            "before = set(sys.modules); import neo4j_write_journal; new = set(sys.modules) - before; "
            "print(sorted(new & {'neo4j_client', 'config', 'dedup_engine', 'dotenv'}))" % str(BACKEND))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=dict(os.environ))
    assert result.returncode == 0, result.stderr[-500:]
    assert result.stdout.strip() == "[]"


def test_no_production_module_uses_the_journal_yet():
    """Integration needs separate authorization. Until then nothing is protected by it."""
    users = []
    for source in BACKEND.rglob("*.py"):
        if "tests" in source.parts or source.name == "neo4j_write_journal.py" or "__pycache__" in source.parts:
            continue
        if "neo4j_write_journal" in source.read_text(encoding="utf-8", errors="replace"):
            users.append(str(source))
    # WP-C1: the coordinator is the one permitted user, and test_no_production_module_imports_the_coordinator_yet keeps it uncalled.
    assert users == [str(BACKEND / "neo4j_write_coordinator.py")]


def test_the_helper_is_untouched_by_the_journal():
    source = (BACKEND / "neo4j_client.py").read_text(encoding="utf-8")
    assert "neo4j_write_journal" not in source

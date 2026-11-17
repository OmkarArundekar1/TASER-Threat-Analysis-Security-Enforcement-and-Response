"""Unit tests for neo4j_idempotency_ledger: identity, fingerprints, the state machine, the API.
Everything lives in pytest's temporary directories. Separate files cover real process death
(_crash), races between threads and processes (_concurrency) and failing storage (_faults)."""
from __future__ import annotations

import ast
import collections
import datetime as dt
import decimal
import enum
import hashlib
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

import neo4j_idempotency_ledger as module
from neo4j_idempotency_ledger import (
    CheckStatus,
    FINGERPRINT_VERSION,
    IdempotencyConflictError,
    IdempotencyLedger,
    InvalidTransitionError,
    LedgerClosedError,
    LedgerReadOnlyError,
    LedgerState as S,
    RequestNotFoundError,
    Resolution,
    StaleAttemptError,
    UnsupportedPayloadError,
    fingerprint_payload,
)

BACKEND = Path(module.__file__).resolve().parent
KEY = "campaign-42/flush/7"
SECRET = "s3cret-parameter-value"
ONE = [("CREATE (n:X {v:$v})", {"v": 1})]
FP = fingerprint_payload("op", ONE).value


@pytest.fixture
def path(tmp_path):
    return tmp_path / "ledger.db"


@pytest.fixture
def ledger(path):
    with IdempotencyLedger(path) as instance:
        yield instance


def _fp(statements, operation="op", **kwargs):
    return fingerprint_payload(operation, statements, **kwargs).value


def _begin(ledger, key=KEY):
    ledger.register(key, "op", ONE)
    return ledger.begin_execution(key, FP, worker_id="w1")


def _reach(ledger, state, key=KEY):
    if state is S.REGISTERED:
        return ledger.register(key, "op", ONE).request
    request = _begin(ledger, key)
    if state is S.IN_PROGRESS:
        return request
    if state is S.COMPLETED:
        return ledger.record_completed(key, attempt=1)
    if state is S.NOT_COMPLETED:
        return ledger.record_not_completed(key, attempt=1)
    if state is S.RETRY_AUTHORIZED:
        ledger.record_not_completed(key, attempt=1)
        return ledger.authorize_retry(key, authorized_by="op", reason="r")
    ledger.record_uncertain(key, attempt=1, error_type="IncompleteCommit")
    if state is S.UNCERTAIN:
        return ledger.get_request(key)
    return ledger.resolve_uncertain(key, resolution=Resolution.ABANDONED, resolved_by="op", reason="r")


# ================================================================ 1-4 identity

def test_first_registration_is_new_and_stored(ledger):
    registration = ledger.register(KEY, "op", ONE)
    assert registration.status is CheckStatus.NEW
    request = registration.request
    assert request.idempotency_key == KEY and request.fingerprint == FP and request.fingerprint_version == FINGERPRINT_VERSION
    assert request.state is S.REGISTERED and request.attempt_count == 0 and request.claimable
    assert ledger.get_request(KEY) == request
    assert [e.cause for e in ledger.get_history(KEY)] == ["REGISTERED"]


def test_repeating_a_registration_is_a_duplicate_that_changes_nothing(ledger):
    first = ledger.register(KEY, "op", ONE).request
    history = ledger.get_history(KEY)
    again = ledger.register(KEY, "op", ONE)
    assert again.status is CheckStatus.DUPLICATE and again.request == first
    assert ledger.get_history(KEY) == history
    assert len(ledger.list_requests()) == 1


@pytest.mark.parametrize("state", list(S), ids=[s.value for s in S])
def test_a_duplicate_reports_the_current_state_and_never_resets_it(ledger, state):
    before = _reach(ledger, state)
    again = ledger.register(KEY, "op", ONE)
    assert again.status is CheckStatus.DUPLICATE
    assert again.request.state is state
    assert ledger.get_request(KEY).version == before.version


def test_a_duplicate_grants_no_right_to_run(ledger):
    _begin(ledger)
    ledger.register(KEY, "op", ONE)
    with pytest.raises(InvalidTransitionError):
        ledger.begin_execution(KEY, FP)


def test_the_same_key_with_a_different_payload_conflicts_and_overwrites_nothing(ledger):
    original = ledger.register(KEY, "op", ONE).request
    other = [("CREATE (n:X {v:$v})", {"v": 2})]
    with pytest.raises(IdempotencyConflictError) as caught:
        ledger.register(KEY, "op", other)
    assert caught.value.existing_fingerprint == FP
    assert caught.value.presented_fingerprint == _fp(other)
    assert caught.value.differences == ("statement 0 parameters",)
    assert caught.value.conflict_no == 1
    current = ledger.get_request(KEY)
    assert current.fingerprint == FP and current.state is original.state and current.version == original.version
    assert current.conflict_count == 1


def test_conflicts_stay_conflicts_and_are_each_audited_without_values(ledger):
    ledger.register(KEY, "op", ONE)
    for number in (1, 2, 3):
        with pytest.raises(IdempotencyConflictError) as caught:
            ledger.register(KEY, "op", [("Q", {"token": SECRET})])
        assert caught.value.conflict_no == number
    conflicts = ledger.get_conflicts(KEY)
    assert [c.conflict_no for c in conflicts] == [1, 2, 3]
    assert conflicts[0].differences == ("statement 0 query", "statement 0 parameters")
    assert conflicts[0].session_id == ledger.session_id and conflicts[0].presented_version == FINGERPRINT_VERSION
    assert conflicts[0].presented_fingerprint == _fp([("Q", {"token": SECRET})])
    assert SECRET not in repr(conflicts)


@pytest.mark.parametrize("first, second, words", [
    (("op", ONE, False), ("other", ONE, False), "operation"),
    (("op", ONE, False), ("op", ONE, True), "return_rows"),
    (("op", ONE, False), ("op", ONE * 2, False), "statement count"),
    (("op", [("A", None)], False), ("op", [("B", None)], False), "statement 0 query"),
    (("op", [("A", {"p": 1}), ("B", {"p": 2})], False), ("op", [("A", {"p": 1}), ("B", {"p": 3})], False), "statement 1 parameters"),
], ids=["operation", "return_rows", "statement count", "query", "second statement"])
def test_a_conflict_says_what_differs(ledger, first, second, words):
    ledger.register(KEY, first[0], first[1], return_rows=first[2])
    with pytest.raises(IdempotencyConflictError) as caught:
        ledger.register(KEY, second[0], second[1], return_rows=second[2])
    assert words in caught.value.differences


def test_different_keys_with_identical_payloads_are_separate_requests(ledger):
    a = ledger.register("a", "op", ONE)
    b = ledger.register("b", "op", ONE)
    assert a.status is b.status is CheckStatus.NEW
    assert a.request.fingerprint == b.request.fingerprint
    ledger.begin_execution("a", FP)
    assert ledger.get_request("b").state is S.REGISTERED


@pytest.mark.parametrize("first, second", [
    ("key", "KEY"), ("key", "key2"), ("é", "é"), ("a/b", "a\\b"),
], ids=["case", "suffix", "unicode forms", "separator"])
def test_keys_match_exactly_with_no_normalization(ledger, first, second):
    assert ledger.register(first, "op", ONE).status is CheckStatus.NEW
    assert ledger.register(second, "op", ONE).status is CheckStatus.NEW


@pytest.mark.parametrize("key, error", [
    (None, TypeError), (5, TypeError), (b"k", TypeError), ("", ValueError), (" k", ValueError), ("k ", ValueError), ("\n", ValueError),
    ("a\x00b", ValueError), ("a\nb", ValueError), ("\ud800", ValueError), ("k" * 513, ValueError),
], ids=["None", "int", "bytes", "empty", "leading space", "trailing space", "only whitespace", "NUL", "newline", "lone surrogate", "too long"])
def test_an_unusable_key_is_refused_by_every_call(ledger, key, error):
    for call in (lambda: ledger.register(key, "op", ONE), lambda: ledger.check(key, "op", ONE), lambda: ledger.get_request(key),
                 lambda: ledger.begin_execution(key, FP)):
        with pytest.raises(error):
            call()
    assert ledger.list_requests() == ()


def test_a_key_is_required_and_never_generated():
    for function in (IdempotencyLedger.register, IdempotencyLedger.check):
        with pytest.raises(TypeError):
            function(None, operation="op", statements=ONE)  # no key argument
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    uses = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "uuid4"]
    assert len(uses) == 1, "uuid4 names a session, and nothing else"
    begin_session = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_begin_session")
    assert any(isinstance(n, ast.Attribute) and n.attr == "uuid4" for n in ast.walk(begin_session))


def test_the_longest_allowed_key_works(ledger):
    assert ledger.register("k" * 512, "op", ONE).status is CheckStatus.NEW


# ================================================================ 5-6 fingerprint

DIFFERENT = {
    "statement order": ([("A", None), ("B", None)], [("B", None), ("A", None)]),
    "list order": ([("Q", {"p": [1, 2]})], [("Q", {"p": [2, 1]})]),
    "batch row order": ([("Q", {"r": [{"a": 1}, {"a": 2}]})], [("Q", {"r": [{"a": 2}, {"a": 1}]})]),
    "query whitespace": ([("Q  X", None)], [("Q X", None)]),
    "query case": ([("MATCH", None)], [("match", None)]),
    "parameter name": ([("Q", {"a": 1})], [("Q", {"b": 1})]),
    "extra parameter": ([("Q", {"a": 1})], [("Q", {"a": 1, "b": None})]),
    "int and float": ([("Q", {"p": 1})], [("Q", {"p": 1.0})]),
    "int and bool": ([("Q", {"p": 1})], [("Q", {"p": True})]),
    "zero and false": ([("Q", {"p": 0})], [("Q", {"p": False})]),
    "zero and negative zero": ([("Q", {"p": 0.0})], [("Q", {"p": -0.0})]),
    "near floats": ([("Q", {"p": 0.1})], [("Q", {"p": 0.1 + 2 ** -55})]),
    "beyond float precision": ([("Q", {"p": 2 ** 53})], [("Q", {"p": 2 ** 53 + 1})]),
    "int and string": ([("Q", {"p": 1})], [("Q", {"p": "1"})]),
    "null and empty string": ([("Q", {"p": None})], [("Q", {"p": ""})]),
    "null and missing": ([("Q", {"p": None})], [("Q", {})]),
    "no parameters and empty": ([("Q", None)], [("Q", {})]),
    "list and tuple": ([("Q", {"p": [1]})], [("Q", {"p": (1,)})]),
    "empty list and map": ([("Q", {"p": []})], [("Q", {"p": {}})]),
    "nesting": ([("Q", {"p": [1, 2]})], [("Q", {"p": [[1], 2]})]),
    "bytes and bytearray": ([("Q", {"p": b"a"})], [("Q", {"p": bytearray(b"a")})]),
    "bytes and string": ([("Q", {"p": b"a"})], [("Q", {"p": "a"})]),
    "unicode forms": ([("Q", {"p": "é"})], [("Q", {"p": "é"})]),
    "a map looking like a tag": ([("Q", {"p": {"float": "0x1p+0"}})], [("Q", {"p": 1.0})]),
    "same instant, another offset": ([("Q", {"p": dt.datetime(2026, 1, 1, 12, tzinfo=dt.timezone.utc)})],
                                     [("Q", {"p": dt.datetime(2026, 1, 1, 17, 30, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))})]),
    "aware and naive": ([("Q", {"p": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)})], [("Q", {"p": dt.datetime(2026, 1, 1)})]),
    "named and unnamed offset": ([("Q", {"p": dt.datetime(2026, 1, 1, tzinfo=dt.timezone(dt.timedelta(0), "GMT"))})],
                                 [("Q", {"p": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)})]),
    "fold": ([("Q", {"p": dt.datetime(2026, 10, 25, 2, 30, tzinfo=zoneinfo.ZoneInfo("Europe/Paris"))})],
             [("Q", {"p": dt.datetime(2026, 10, 25, 2, 30, fold=1, tzinfo=zoneinfo.ZoneInfo("Europe/Paris"))})]),
    "pytz ambiguous hour": ([("Q", {"p": pytz.timezone("Europe/Paris").localize(dt.datetime(2026, 10, 25, 2, 30), is_dst=True)})],
                            [("Q", {"p": pytz.timezone("Europe/Paris").localize(dt.datetime(2026, 10, 25, 2, 30), is_dst=False)})]),
    "zoneinfo and pytz": ([("Q", {"p": dt.datetime(2026, 1, 1, tzinfo=zoneinfo.ZoneInfo("Asia/Kolkata"))})],
                          [("Q", {"p": pytz.timezone("Asia/Kolkata").localize(dt.datetime(2026, 1, 1))})]),
    "datetime and date": ([("Q", {"p": dt.datetime(2026, 1, 1)})], [("Q", {"p": dt.date(2026, 1, 1)})]),
    "python and neo4j date": ([("Q", {"p": dt.date(2026, 1, 1)})], [("Q", {"p": neo4j_time.Date(2026, 1, 1)})]),
    "python and neo4j datetime": ([("Q", {"p": dt.datetime(2026, 1, 1)})], [("Q", {"p": neo4j_time.DateTime(2026, 1, 1)})]),
    "nanoseconds": ([("Q", {"p": neo4j_time.DateTime(2026, 1, 1, 0, 0, 0, 1)})], [("Q", {"p": neo4j_time.DateTime(2026, 1, 1, 0, 0, 0, 2)})]),
    "neo4j time": ([("Q", {"p": neo4j_time.Time(1, 2, 3, 4)})], [("Q", {"p": neo4j_time.Time(1, 2, 3, 5)})]),
    "time": ([("Q", {"p": dt.time(1, 2, 3)})], [("Q", {"p": dt.time(1, 2, 4)})]),
    "timedelta": ([("Q", {"p": dt.timedelta(1)})], [("Q", {"p": dt.timedelta(2)})]),
    "duration": ([("Q", {"p": neo4j_time.Duration(months=1)})], [("Q", {"p": neo4j_time.Duration(months=2)})]),
    "date": ([("Q", {"p": dt.date(2026, 1, 1)})], [("Q", {"p": dt.date(2026, 1, 2)})]),
    "pytz fixed": ([("Q", {"p": dt.datetime(2026, 1, 1, tzinfo=pytz.FixedOffset(60))})], [("Q", {"p": dt.datetime(2026, 1, 1, tzinfo=pytz.FixedOffset(120))})]),
    "neo4j date value": ([("Q", {"p": neo4j_time.Date(2026, 1, 1)})], [("Q", {"p": neo4j_time.Date(2026, 1, 2)})]),
}


@pytest.mark.parametrize("first, second", DIFFERENT.values(), ids=DIFFERENT.keys())
def test_meaningful_differences_change_the_fingerprint(ledger, first, second):
    assert _fp(first) != _fp(second)
    ledger.register(KEY, "op", first)
    with pytest.raises(IdempotencyConflictError):
        ledger.register(KEY, "op", second)


SAME = {
    "map key order": ([("Q", {"a": 1, "b": 2})], [("Q", {"b": 2, "a": 1})]),
    "nested map key order": ([("Q", {"p": [{"a": 1, "b": {"x": 1, "y": 2}}]})], [("Q", {"p": [{"b": {"y": 2, "x": 1}, "a": 1}]})]),
    "pair as tuple or list": ([("Q", None)], [["Q", None]]),
    "batch as list or tuple": ([("Q", None)], (("Q", None),)),
    "nan and nan": ([("Q", {"p": float("nan")})], [("Q", {"p": float("nan")})]),
    "rebuilt datetime": ([("Q", {"p": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)})], [("Q", {"p": dt.datetime(2026, 1, 1, tzinfo=dt.timezone(dt.timedelta(0)))})]),
}


@pytest.mark.parametrize("first, second", SAME.values(), ids=SAME.keys())
def test_representations_of_the_same_payload_are_a_duplicate(ledger, first, second):
    assert _fp(first) == _fp(second)
    ledger.register(KEY, "op", first)
    assert ledger.register(KEY, "op", second).status is CheckStatus.DUPLICATE


def test_a_generator_is_read_once_and_fingerprints_the_same():
    assert _fp(s for s in ONE) == FP


def test_identical_values_in_two_places_are_not_a_cycle():
    shared = [1, 2]
    assert _fp([("Q", {"a": shared, "b": shared})]) == _fp([("Q", {"a": [1, 2], "b": [1, 2]})])


def _expected(text):
    return hashlib.sha256(b"cyukti-idempotency-fingerprint\x00" + text.encode("ascii")).hexdigest()


def test_the_canonical_format_is_frozen_simple():
    """Written out by hand, so a change to the format fails here. Changing it needs a new version."""
    text = ('{"canonicalization":1,"operation":"op","return_rows":false,"statements":'
            '[{"parameters":{"map":{"a":{"float":"0x1.8000000000000p+0"},"b":1}},"query":"Q $a"}]}')
    assert _fp([("Q $a", {"b": 1, "a": 1.5})]) == _expected(text)


def test_the_canonical_format_is_frozen_with_every_tag():
    parameters = {
        "t": (1, "a"), "b": b"\x00\xff", "ba": bytearray(b"\x01"), "n": None, "y": True, "s": "é",
        "d": dt.datetime(2026, 1, 2, 3, 4, 5, 6, tzinfo=dt.timezone.utc), "nd": dt.datetime(2026, 1, 1),
        "z": dt.datetime(2026, 7, 1, tzinfo=zoneinfo.ZoneInfo("Europe/Paris")),
        "pf": dt.datetime(2026, 1, 1, tzinfo=pytz.FixedOffset(330)),
        "pz": pytz.timezone("Asia/Kolkata").localize(dt.datetime(2026, 1, 1)),
        "o": dt.date(2026, 1, 2), "m": dt.time(1, 2, 3, 4), "td": dt.timedelta(1, 2, 3),
        "nt": neo4j_time.DateTime(2026, 1, 2, 3, 4, 5, 6), "ne": neo4j_time.Date(2026, 1, 2), "ni": neo4j_time.Time(1, 2, 3, 4),
        "du": neo4j_time.Duration(months=1, days=2, seconds=3, nanoseconds=4), "l": [1, [2.5]], "inf": float("inf"), "nan": float("nan"),
    }
    text = ('{"canonicalization":1,"operation":"flush","return_rows":true,"statements":[{"parameters":null,"query":"W"},'
            '{"parameters":{"map":{'
            '"b":{"bytes":"AP8="},"ba":{"bytearray":"AQ=="},'
            '"d":{"datetime":[2026,1,2,3,4,5,6,0,["fixed",0,0,0,null]]},'
            '"du":{"neo4j_duration":[1,2,3,4]},'
            '"inf":{"float":"inf"},'
            '"l":[1,[{"float":"0x1.4000000000000p+1"}]],'
            '"m":{"time":[1,2,3,4,0,null]},'
            '"n":null,"nan":{"float":"nan"},"nd":{"datetime":[2026,1,1,0,0,0,0,0,null]},'
            '"ne":{"neo4j_date":[2026,1,2]},"ni":{"neo4j_time":[1,2,3,4,null]},'
            '"nt":{"neo4j_datetime":[2026,1,2,3,4,5,6,null]},'
            '"o":{"date":[2026,1,2]},'
            '"pf":{"datetime":[2026,1,1,0,0,0,0,0,["pytz_fixed",330]]},'
            '"pz":{"datetime":[2026,1,1,0,0,0,0,0,["pytz","Asia/Kolkata",19800.0]]},'
            '"s":"\\u00e9","t":{"tuple":[1,"a"]},"td":{"timedelta":[1,2,3]},"y":true,'
            '"z":{"datetime":[2026,7,1,0,0,0,0,0,["zoneinfo","Europe/Paris"]]}'
            '}},"query":"R"}]}')
    assert _fp([("W", None), ("R", parameters)], operation="flush", return_rows=True) == _expected(text)


def test_statement_digests_are_per_statement_and_pinned():
    result = fingerprint_payload("op", [("A", None), ("A", {"p": 1})])
    assert result.version == 1 and result.operation == "op" and result.return_rows is False
    assert result.statement_digests[0][0] == result.statement_digests[1][0], "same query"
    assert result.statement_digests[0][1] != result.statement_digests[1][1], "different parameters"
    assert result.statement_digests[0][0] == hashlib.sha256(b'cyukti-idempotency-statement\x00"A"').hexdigest()


class _StrSub(str):
    pass


class _IntEnum(enum.IntEnum):
    A = 1


class _FloatSub(float):
    pass


class _ListSub(list):
    pass


class _Weird(dt.tzinfo):
    def utcoffset(self, when):
        return dt.timedelta(hours=1)

    def dst(self, when):
        return None

    def tzname(self, when):
        return "weird"


_cycle: list = []
_cycle.append(_cycle)
_deep: list = []
for _ in range(module._MAX_NESTING + 1):
    _deep = [_deep]
_map_cycle: dict = {}
_map_cycle["self"] = _map_cycle

REFUSED = {
    "set": ({1, 2}, "builtins.set is not supported"),
    "frozenset": (frozenset({1}), "frozenset is not supported"),
    "object": (object(), "builtins.object is not supported"),
    "complex": (1j, "complex is not supported"),
    "decimal": (decimal.Decimal("1.5"), "Decimal is not supported"),
    "above 64 bits": (2 ** 63, "64-bit"),
    "below 64 bits": (-(2 ** 63) - 1, "64-bit"),
    "str subclass": (_StrSub("x"), "_StrSub is not supported"),
    "int enum": (_IntEnum.A, "_IntEnum is not supported"),
    "float subclass": (_FloatSub(1.5), "_FloatSub is not supported"),
    "list subclass": (_ListSub([1]), "_ListSub is not supported"),
    "dict subclass": (collections.OrderedDict(a=1), "OrderedDict is not supported"),
    "non-string map key": ({1: "a"}, "map key type int"),
    "reference cycle": (_cycle, "reference cycle"),
    "map cycle": (_map_cycle, "reference cycle"),
    "nested too deeply": (_deep, "nested more than"),
    "unknown timezone type": (dt.datetime(2026, 1, 1, tzinfo=_Weird()), "timezone type _Weird"),
    "spatial point": (neo4j.spatial.WGS84Point((1.0, 2.0)), "WGS84Point is not supported"),
    "nested inside a batch": ({"records": [{"ok": 1, "bad": {1, 2}}]}, "builtins.set is not supported"),
}


@pytest.mark.parametrize("value, words", REFUSED.values(), ids=REFUSED.keys())
def test_a_value_with_no_canonical_form_is_refused_and_registers_nothing(ledger, value, words):
    with pytest.raises(UnsupportedPayloadError, match=words):
        ledger.register(KEY, "op", [("Q", {"p": value})])
    assert ledger.list_requests() == ()


@pytest.mark.parametrize("statements, error", [
    ([], ValueError), ((), ValueError), (iter([]), ValueError), (None, TypeError), (7, TypeError), ("Q", TypeError), (b"Q", TypeError),
    ([("Q",)], TypeError), (["Q"], TypeError), ([("", None)], ValueError), ([("  ", None)], ValueError), ([(1, None)], ValueError),
    ([("Q", 5)], TypeError), ([("Q", [1])], TypeError), ([("Q", {1: "a"})], UnsupportedPayloadError),
], ids=["empty list", "empty tuple", "empty iterator", "None", "int", "string", "bytes", "short pair", "not a pair", "empty query",
        "blank query", "query not a string", "parameters an int", "parameters a list", "parameter name not a string"])
def test_a_malformed_batch_registers_nothing(ledger, statements, error):
    with pytest.raises(error):
        ledger.register(KEY, "op", statements)
    assert ledger.list_requests() == ()


@pytest.mark.parametrize("operation", ["", "  ", None, 5])
def test_an_operation_name_is_required(ledger, operation):
    with pytest.raises((ValueError, TypeError)):
        ledger.register(KEY, operation, ONE)


def test_return_rows_must_be_a_bool(ledger):
    with pytest.raises(TypeError):
        ledger.register(KEY, "op", ONE, return_rows=1)


# ================================================================ check

def test_check_reports_each_status_and_records_nothing(ledger):
    assert ledger.check(KEY, "op", ONE).status is CheckStatus.NEW
    assert ledger.list_requests() == ()
    ledger.register(KEY, "op", ONE)
    assert ledger.check(KEY, "op", ONE).status is CheckStatus.DUPLICATE
    other = [("Q", None)]
    result = ledger.check(KEY, "op", other)
    assert result.status is CheckStatus.CONFLICT
    assert result.presented_fingerprint == _fp(other) and result.request.fingerprint == FP
    assert result.differences == ("statement 0 query", "statement 0 parameters")
    assert ledger.get_conflicts(KEY) == (), "only register() audits a conflict"


def test_check_works_read_only(path):
    with IdempotencyLedger(path) as writer:
        writer.register(KEY, "op", ONE)
        with IdempotencyLedger(path, read_only=True) as reader:
            assert reader.check(KEY, "op", ONE).status is CheckStatus.DUPLICATE
            assert reader.check("new", "op", ONE).status is CheckStatus.NEW


# ================================================================ state machine

MOVES = {
    "begin_execution": lambda l, a: l.begin_execution(KEY, FP),
    "record_completed": lambda l, a: l.record_completed(KEY, attempt=a),
    "record_not_completed": lambda l, a: l.record_not_completed(KEY, attempt=a),
    "record_uncertain": lambda l, a: l.record_uncertain(KEY, attempt=a),
    "authorize_retry": lambda l, a: l.authorize_retry(KEY, authorized_by="op", reason="r"),
    "resolve_found_completed": lambda l, a: l.resolve_uncertain(KEY, resolution=Resolution.FOUND_COMPLETED, resolved_by="op", reason="r"),
    "resolve_found_not_completed": lambda l, a: l.resolve_uncertain(KEY, resolution=Resolution.FOUND_NOT_COMPLETED, resolved_by="op", reason="r"),
    "resolve_abandoned": lambda l, a: l.resolve_uncertain(KEY, resolution=Resolution.ABANDONED, resolved_by="op", reason="r"),
}
ALLOWED = {
    (S.REGISTERED, "begin_execution"): S.IN_PROGRESS,
    (S.RETRY_AUTHORIZED, "begin_execution"): S.IN_PROGRESS,
    (S.IN_PROGRESS, "record_completed"): S.COMPLETED,
    (S.IN_PROGRESS, "record_not_completed"): S.NOT_COMPLETED,
    (S.IN_PROGRESS, "record_uncertain"): S.UNCERTAIN,
    (S.NOT_COMPLETED, "authorize_retry"): S.RETRY_AUTHORIZED,
    (S.UNCERTAIN, "resolve_found_completed"): S.COMPLETED,
    (S.UNCERTAIN, "resolve_found_not_completed"): S.NOT_COMPLETED,
    (S.UNCERTAIN, "resolve_abandoned"): S.ABANDONED,
}


@pytest.mark.parametrize("state", list(S), ids=[s.value for s in S])
@pytest.mark.parametrize("move", MOVES, ids=list(MOVES))
def test_every_move_from_every_state_is_exactly_as_specified(ledger, state, move):
    request = _reach(ledger, state)
    before, history = ledger.get_request(KEY), ledger.get_history(KEY)
    expected = ALLOWED.get((state, move))
    attempt = max(request.attempt_count, 1)
    if expected is not None:
        assert MOVES[move](ledger, attempt).state is expected
        assert len(ledger.get_history(KEY)) == len(history) + 1
    else:
        with pytest.raises(InvalidTransitionError):
            MOVES[move](ledger, attempt)
        assert ledger.get_request(KEY) == before
        assert ledger.get_history(KEY) == history
    assert ledger.check_integrity().ok


def test_the_table_in_the_module_is_the_specified_machine():
    table = module._TRANSITIONS
    assert set(table) == {(S.REGISTERED, S.IN_PROGRESS), (S.RETRY_AUTHORIZED, S.IN_PROGRESS), (S.IN_PROGRESS, S.COMPLETED),
                          (S.IN_PROGRESS, S.NOT_COMPLETED), (S.IN_PROGRESS, S.UNCERTAIN), (S.NOT_COMPLETED, S.RETRY_AUTHORIZED),
                          (S.UNCERTAIN, S.COMPLETED), (S.UNCERTAIN, S.NOT_COMPLETED), (S.UNCERTAIN, S.ABANDONED)}
    assert all(causes == {module._Cause.RESOLUTION} for (source, _), causes in table.items() if source is S.UNCERTAIN)
    assert not any(source in (S.COMPLETED, S.ABANDONED) for source, _ in table)
    assert not any(target is S.IN_PROGRESS and source is not S.REGISTERED and source is not S.RETRY_AUTHORIZED for source, target in table)
    assert table[(S.IN_PROGRESS, S.UNCERTAIN)] == {module._Cause.OUTCOME, module._Cause.RECOVERY}
    assert module._CLAIMABLE == {S.REGISTERED, S.RETRY_AUTHORIZED}


def test_only_claimable_states_report_claimable(ledger):
    for number, state in enumerate(S):
        request = _reach(ledger, state, f"key/{number}")
        assert request.claimable is (state in (S.REGISTERED, S.RETRY_AUTHORIZED))


def test_a_normal_life_is_recorded_in_order(ledger):
    ledger.register(KEY, "op", ONE)
    claimed = ledger.begin_execution(KEY, FP, worker_id="w1")
    assert claimed.state is S.IN_PROGRESS and claimed.attempt_count == 1 and claimed.worker_id == "w1"
    assert claimed.owner_session == ledger.session_id and not claimed.claimable
    done = ledger.record_completed(KEY, attempt=1)
    assert done.state is S.COMPLETED and done.last_outcome == "COMPLETED"
    events = ledger.get_history(KEY)
    assert [(e.from_state, e.to_state, e.cause) for e in events] == [
        (None, S.REGISTERED, "REGISTERED"), (S.REGISTERED, S.IN_PROGRESS, "EXECUTE"), (S.IN_PROGRESS, S.COMPLETED, "OUTCOME")]
    assert [e.transition_no for e in events] == [1, 2, 3] and [e.attempt for e in events] == [0, 1, 1]


def test_begin_execution_needs_the_registered_fingerprint(ledger):
    ledger.register(KEY, "op", ONE)
    wrong = _fp([("Q", None)])
    with pytest.raises(IdempotencyConflictError) as caught:
        ledger.begin_execution(KEY, wrong)
    assert caught.value.presented_fingerprint == wrong
    assert ledger.get_request(KEY).state is S.REGISTERED
    with pytest.raises(TypeError):
        ledger.begin_execution(KEY, None)


def test_an_unregistered_key_cannot_be_claimed_or_reported(ledger):
    for call in (lambda: ledger.begin_execution(KEY, FP), lambda: ledger.record_completed(KEY, attempt=1), lambda: ledger.get_request(KEY),
                 lambda: ledger.get_history(KEY), lambda: ledger.get_conflicts(KEY), lambda: ledger.authorize_retry(KEY, authorized_by="o", reason="r")):
        with pytest.raises(RequestNotFoundError):
            call()
    assert ledger.list_requests() == ()


def test_an_outcome_for_an_old_attempt_cannot_land_on_the_current_one(ledger):
    _begin(ledger)
    ledger.record_not_completed(KEY, attempt=1)
    ledger.authorize_retry(KEY, authorized_by="op", reason="r")
    assert ledger.begin_execution(KEY, FP).attempt_count == 2
    before = ledger.get_request(KEY)
    for call in (lambda: ledger.record_completed(KEY, attempt=1), lambda: ledger.record_not_completed(KEY, attempt=1), lambda: ledger.record_uncertain(KEY, attempt=1),
                 lambda: ledger.record_completed(KEY, attempt=3)):
        with pytest.raises(StaleAttemptError):
            call()
    assert ledger.get_request(KEY) == before
    assert ledger.record_completed(KEY, attempt=2).state is S.COMPLETED


@pytest.mark.parametrize("attempt", [None, 0, -1, "1", True, 1.0])
def test_an_outcome_must_name_a_real_attempt(ledger, attempt):
    _begin(ledger)
    with pytest.raises(ValueError):
        ledger.record_completed(KEY, attempt=attempt)
    assert ledger.get_request(KEY).state is S.IN_PROGRESS


def test_error_details_are_stored_then_cleared_with_history_kept(ledger):
    _begin(ledger)
    failed = ledger.record_not_completed(KEY, attempt=1, error_type="ServiceUnavailable", error_code="Neo.X", diagnostic="d")
    assert (failed.error_type, failed.error_code, failed.diagnostic, failed.last_outcome) == ("ServiceUnavailable", "Neo.X", "d", "NOT_COMPLETED")
    retried = ledger.authorize_retry(KEY, authorized_by="op", reason="r")
    assert retried.error_type is None and retried.last_outcome == "NOT_COMPLETED"
    assert ledger.get_history(KEY)[2].error_type == "ServiceUnavailable"


def test_an_exception_is_refused_as_a_detail_because_its_message_can_quote_values(ledger):
    _begin(ledger)
    with pytest.raises(TypeError, match="diagnostic must be a string or None"):
        ledger.record_uncertain(KEY, attempt=1, diagnostic=ValueError(SECRET))
    assert ledger.get_request(KEY).state is S.IN_PROGRESS


def test_long_details_are_cut_to_a_limit(ledger):
    _begin(ledger)
    assert len(ledger.record_uncertain(KEY, attempt=1, diagnostic="x" * 5000).diagnostic) == module._MAX_DETAIL_LENGTH


# ================================================================ retries

def test_a_not_completed_request_cannot_run_without_authorization(ledger):
    _begin(ledger)
    ledger.record_not_completed(KEY, attempt=1)
    assert not ledger.get_request(KEY).claimable
    with pytest.raises(InvalidTransitionError):
        ledger.begin_execution(KEY, FP)


def test_retry_keeps_the_identity_and_is_audited(ledger):
    original = _begin(ledger)
    ledger.record_not_completed(KEY, attempt=1, error_type="ServiceUnavailable")
    authorized = ledger.authorize_retry(KEY, authorized_by="operator", reason="connection restored")
    assert authorized.state is S.RETRY_AUTHORIZED and authorized.attempt_count == 1, "an attempt counts when it starts"
    event = ledger.get_history(KEY)[-1]
    assert (event.cause, event.actor, event.reason) == ("RETRY", "operator", "connection restored")
    again = ledger.begin_execution(KEY, FP, worker_id="w2")
    assert again.attempt_count == 2 and again.worker_id == "w2"
    assert again.fingerprint == original.fingerprint and again.idempotency_key == original.idempotency_key
    assert ledger.record_completed(KEY, attempt=2).state is S.COMPLETED
    assert len(ledger.list_requests()) == 1


def test_one_authorization_allows_exactly_one_attempt(ledger):
    _begin(ledger)
    ledger.record_not_completed(KEY, attempt=1)
    ledger.authorize_retry(KEY, authorized_by="op", reason="r")
    ledger.begin_execution(KEY, FP)
    with pytest.raises(InvalidTransitionError):
        ledger.begin_execution(KEY, FP)
    ledger.record_not_completed(KEY, attempt=2)
    with pytest.raises(InvalidTransitionError):
        ledger.begin_execution(KEY, FP)
    ledger.authorize_retry(KEY, authorized_by="op", reason="again")
    assert ledger.begin_execution(KEY, FP).attempt_count == 3


def test_authorizing_twice_is_refused(ledger):
    _begin(ledger)
    ledger.record_not_completed(KEY, attempt=1)
    ledger.authorize_retry(KEY, authorized_by="op", reason="r")
    with pytest.raises(InvalidTransitionError):
        ledger.authorize_retry(KEY, authorized_by="op", reason="r")


@pytest.mark.parametrize("field_name", ["authorized_by", "reason"])
@pytest.mark.parametrize("value", ["", "  ", None])
def test_a_retry_needs_a_named_authorizer_and_a_reason(ledger, field_name, value):
    _begin(ledger)
    ledger.record_not_completed(KEY, attempt=1)
    arguments = {"authorized_by": "op", "reason": "r", field_name: value}
    with pytest.raises(ValueError):
        ledger.authorize_retry(KEY, **arguments)
    assert ledger.get_request(KEY).state is S.NOT_COMPLETED


# ================================================================ uncertainty

def test_uncertain_cannot_run_or_be_retried(ledger):
    _begin(ledger)
    ledger.record_uncertain(KEY, attempt=1)
    assert not ledger.get_request(KEY).claimable
    with pytest.raises(InvalidTransitionError):
        ledger.begin_execution(KEY, FP)
    with pytest.raises(InvalidTransitionError):
        ledger.authorize_retry(KEY, authorized_by="op", reason="r")
    assert ledger.register(KEY, "op", ONE).request.state is S.UNCERTAIN, "registering again doesn't reset it"
    assert ledger.get_request(KEY).state is S.UNCERTAIN


@pytest.mark.parametrize("resolution, state", [
    (Resolution.FOUND_COMPLETED, S.COMPLETED), (Resolution.FOUND_NOT_COMPLETED, S.NOT_COMPLETED), (Resolution.ABANDONED, S.ABANDONED),
], ids=[r.value for r in Resolution])
def test_resolution_keeps_the_uncertainty_and_records_who_when_and_why(ledger, resolution, state):
    _begin(ledger)
    ledger.record_uncertain(KEY, attempt=1, error_type="IncompleteCommit")
    resolved = ledger.resolve_uncertain(KEY, resolution=resolution, resolved_by="analyst", reason="checked the database")
    assert resolved.state is state and resolved.resolution is resolution
    assert resolved.last_outcome == "UNCERTAIN", "what execution reported doesn't change"
    assert not resolved.claimable
    events = ledger.get_history(KEY)
    assert [e.to_state for e in events] == [S.REGISTERED, S.IN_PROGRESS, S.UNCERTAIN, state]
    assert events[2].error_type == "IncompleteCommit"
    assert (events[3].cause, events[3].actor, events[3].reason, events[3].resolution) == ("RESOLUTION", "analyst", "checked the database", resolution)
    assert events[3].at


def test_running_an_uncertain_request_again_takes_two_separate_explicit_acts(ledger):
    _begin(ledger)
    ledger.record_uncertain(KEY, attempt=1)
    ledger.resolve_uncertain(KEY, resolution=Resolution.FOUND_NOT_COMPLETED, resolved_by="analyst", reason="not in the database")
    with pytest.raises(InvalidTransitionError):
        ledger.begin_execution(KEY, FP)
    ledger.authorize_retry(KEY, authorized_by="operator", reason="safe now")
    second = ledger.begin_execution(KEY, FP)
    assert second.attempt_count == 2 and second.resolution is None, "the old conclusion doesn't describe this attempt"
    assert ledger.get_history(KEY)[3].resolution is Resolution.FOUND_NOT_COMPLETED, "history keeps it"


@pytest.mark.parametrize("resolution", [Resolution.FOUND_COMPLETED, Resolution.ABANDONED], ids=["found completed", "abandoned"])
def test_other_resolutions_can_never_run_again(ledger, resolution):
    _begin(ledger)
    ledger.record_uncertain(KEY, attempt=1)
    ledger.resolve_uncertain(KEY, resolution=resolution, resolved_by="a", reason="r")
    for move in MOVES.values():
        with pytest.raises(InvalidTransitionError):
            move(ledger, 1)


@pytest.mark.parametrize("arguments", [
    {"resolution": Resolution.ABANDONED, "resolved_by": "", "reason": "r"},
    {"resolution": Resolution.ABANDONED, "resolved_by": "a", "reason": " "},
    {"resolution": "NOT_A_RESOLUTION", "resolved_by": "a", "reason": "r"},
], ids=["no one", "no reason", "unknown resolution"])
def test_a_resolution_must_be_complete(ledger, arguments):
    _begin(ledger)
    ledger.record_uncertain(KEY, attempt=1)
    with pytest.raises(ValueError):
        ledger.resolve_uncertain(KEY, **arguments)
    assert ledger.get_request(KEY).state is S.UNCERTAIN


def test_automatic_calls_alone_can_never_leave_uncertain(path):
    """Every order of the calls a program could make unattended, including recovery."""
    generator = random.Random(20261105)
    automatic = ["begin_execution", "record_completed", "record_not_completed", "record_uncertain"]
    reached = 0
    with IdempotencyLedger(path) as ledger:
        for number in range(120):
            key = f"walk/{number}"
            ledger.register(key, "op", ONE)
            uncertain = False
            for _ in range(8):
                name = generator.choice(automatic)
                request = ledger.get_request(key)
                try:
                    if name == "begin_execution":
                        ledger.begin_execution(key, FP)
                    else:
                        getattr(ledger, name)(key, attempt=max(request.attempt_count, 1))
                except InvalidTransitionError:
                    pass
                ledger.recover_unfinished()
                uncertain = uncertain or ledger.get_request(key).state is S.UNCERTAIN
                if uncertain:
                    assert ledger.get_request(key).state is S.UNCERTAIN
            reached += uncertain
        assert reached > 10, "the walk has to reach the case it is testing"
        assert ledger.check_integrity().ok


def test_with_every_call_available_a_rerun_of_uncertain_always_has_both_explicit_acts(ledger):
    generator = random.Random(20261106)
    names = list(MOVES)
    for number in range(100):
        key = f"walk/{number}"
        ledger.register(key, "op", ONE)
        for _ in range(14):
            name = generator.choice(names)
            attempt = max(ledger.get_request(key).attempt_count, 1)
            try:
                if name == "begin_execution":
                    ledger.begin_execution(key, FP)
                elif name == "authorize_retry":
                    ledger.authorize_retry(key, authorized_by="op", reason="r")
                elif name.startswith("resolve"):
                    resolution = {"resolve_found_completed": Resolution.FOUND_COMPLETED, "resolve_found_not_completed": Resolution.FOUND_NOT_COMPLETED,
                                  "resolve_abandoned": Resolution.ABANDONED}[name]
                    ledger.resolve_uncertain(key, resolution=resolution, resolved_by="op", reason="r")
                else:
                    getattr(ledger, name)(key, attempt=attempt)
            except InvalidTransitionError:
                pass
        events = ledger.get_history(key)
        for index, event in enumerate(events):
            if event.to_state is S.IN_PROGRESS and any(e.to_state is S.UNCERTAIN for e in events[:index]):
                previous_uncertain = max(i for i, e in enumerate(events[:index]) if e.to_state is S.UNCERTAIN)
                causes = [e.cause for e in events[previous_uncertain + 1:index]]
                assert "RESOLUTION" in causes and "RETRY" in causes


# ================================================================ recovery in one process

def test_an_instance_never_recovers_its_own_running_work(ledger):
    _begin(ledger)
    report = ledger.recover_unfinished()
    assert report.still_running == (KEY,) and report.recovered == ()
    assert ledger.get_request(KEY).state is S.IN_PROGRESS


def test_its_own_work_is_safe_even_if_its_lock_file_was_removed(ledger):
    """Its own process is plainly alive, whatever has happened to the file."""
    _begin(ledger)
    os.unlink(ledger._session_path)
    assert ledger.recover_unfinished().still_running == (KEY,)
    assert ledger.get_request(KEY).state is S.IN_PROGRESS


def test_a_live_instance_counts_as_alive(path):
    with IdempotencyLedger(path) as owner, IdempotencyLedger(path) as other:
        _begin(owner)
        report = other.recover_unfinished()
        assert report.still_running == (KEY,) and report.recovered == ()
        assert owner.record_completed(KEY, attempt=1).state is S.COMPLETED


def test_a_session_whose_lock_file_is_gone_counts_as_ended(path):
    """The documented conservative direction: uncertain, never executable."""
    with IdempotencyLedger(path) as owner, IdempotencyLedger(path) as other:
        _begin(owner)
        os.unlink(owner._session_path)
        assert other.recover_unfinished().recovered == (KEY,)
        assert other.get_request(KEY).state is S.UNCERTAIN


def test_recovery_after_a_clean_close_treats_unfinished_work_as_uncertain(path):
    with IdempotencyLedger(path) as first:
        _begin(first)
    with IdempotencyLedger(path) as second:
        report = second.recover_unfinished()
        assert report.recovered == (KEY,) and report.uncertain == (KEY,)
        request = second.get_request(KEY)
        assert request.state is S.UNCERTAIN and request.last_outcome == "UNCERTAIN" and request.error_type == "UnresolvedAfterRestart"
        assert request.recovery_metadata["previous_owner_session"] == first.session_id
        assert request.recovery_metadata["recovered_by_session"] == second.session_id
        assert request.recovery_metadata["attempt"] == 1
        event = second.get_history(KEY)[-1]
        assert (event.cause, event.actor) == ("RECOVERY", "recover_unfinished")


def test_recovery_leaves_every_other_state_alone_and_reports_it(path):
    with IdempotencyLedger(path) as first:
        keys = {state: f"key/{state.value}" for state in S}
        for state, key in keys.items():
            _reach(first, state, key)
    with IdempotencyLedger(path) as second:
        before = {state: second.get_request(key) for state, key in keys.items()}
        report = second.recover_unfinished()
        assert report.registered == (keys[S.REGISTERED],) and report.not_completed == (keys[S.NOT_COMPLETED],)
        assert report.retry_authorized == (keys[S.RETRY_AUTHORIZED],)
        assert set(report.uncertain) == {keys[S.UNCERTAIN], keys[S.IN_PROGRESS]} and report.recovered == (keys[S.IN_PROGRESS],)
        for state, request in before.items():
            if state is not S.IN_PROGRESS:
                assert second.get_request(keys[state]) == request


def test_recovery_runs_nothing_and_grants_nothing(path):
    with IdempotencyLedger(path) as first:
        _reach(first, S.RETRY_AUTHORIZED)
    with IdempotencyLedger(path) as second:
        second.recover_unfinished()
        request = second.get_request(KEY)
        assert request.state is S.RETRY_AUTHORIZED and request.attempt_count == 1, "the attempt is still unused"


def test_recovery_is_repeatable_and_adds_nothing_the_second_time(path):
    with IdempotencyLedger(path) as first:
        _begin(first)
    with IdempotencyLedger(path) as second:
        second.recover_unfinished()
        events = second.get_history(KEY)
        again = second.recover_unfinished()
        assert again.recovered == () and again.uncertain == (KEY,)
        assert second.get_history(KEY) == events


def test_a_late_outcome_cannot_overwrite_a_recovered_request(path):
    with IdempotencyLedger(path) as first:
        _begin(first)
    with IdempotencyLedger(path) as second:
        second.recover_unfinished()
        for call in (lambda: second.record_completed(KEY, attempt=1), lambda: second.record_not_completed(KEY, attempt=1)):
            with pytest.raises(InvalidTransitionError):
                call()
        assert second.get_request(KEY).state is S.UNCERTAIN


def test_a_worker_that_finishes_after_the_caller_stopped_waiting_still_records_its_outcome(ledger):
    _begin(ledger)
    release = threading.Event()
    executions = []

    def worker():
        executions.append("query")
        release.wait(10)
        ledger.record_completed(KEY, attempt=1)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=0.05)
    assert thread.is_alive()
    assert ledger.get_request(KEY).state is S.IN_PROGRESS, "a caller's timeout concludes nothing"
    with pytest.raises(InvalidTransitionError):
        ledger.begin_execution(KEY, FP)
    assert ledger.recover_unfinished().still_running == (KEY,)
    release.set()
    thread.join(10)
    assert ledger.get_request(KEY).state is S.COMPLETED and executions == ["query"]


def test_a_worker_finishing_after_close_gets_an_error_not_a_false_state(path):
    ledger = IdempotencyLedger(path)
    _begin(ledger)
    ledger.close()
    with pytest.raises(LedgerClosedError):
        ledger.record_completed(KEY, attempt=1)
    with IdempotencyLedger(path) as reopened:
        assert reopened.get_request(KEY).state is S.IN_PROGRESS


# ================================================================ storage, API

def test_a_new_ledger_has_the_schema_version_and_is_empty(ledger):
    report = ledger.check_integrity()
    assert report.ok and report.schema_version == module.SCHEMA_VERSION == 1 and report.requests_checked == 0
    assert ledger.list_requests() == ()


def test_the_durable_settings_are_actually_in_force(ledger):
    assert ledger._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert ledger._conn.execute("PRAGMA synchronous").fetchone()[0] == 2


def test_a_new_ledger_file_is_private_to_its_owner(path, ledger):
    ledger.register(KEY, "op", ONE)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(str(path) + ".sessions").st_mode) == 0o700
    assert stat.S_IMODE(os.stat(ledger._session_path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(str(path) + "-wal").st_mode) == 0o600


def test_there_is_no_default_path():
    with pytest.raises(TypeError):
        IdempotencyLedger()


def test_a_missing_directory_is_refused_rather_than_created(tmp_path):
    with pytest.raises(module.LedgerError, match="directory does not exist"):
        IdempotencyLedger(tmp_path / "absent" / "ledger.db")
    assert not (tmp_path / "absent").exists()


def test_reopening_keeps_requests_and_identity(path):
    with IdempotencyLedger(path) as first:
        request = first.register(KEY, "op", ONE).request
    with IdempotencyLedger(path) as second:
        assert second.get_request(KEY) == request
        assert second.register(KEY, "op", ONE).status is CheckStatus.DUPLICATE
        with pytest.raises(IdempotencyConflictError):
            second.register(KEY, "op", [("Q", None)])
        assert second.check_integrity().ok


def test_opening_an_empty_file_initializes_it(path):
    path.touch()
    with IdempotencyLedger(path) as ledger:
        assert ledger.check_integrity().ok


def test_close_is_safe_to_repeat_and_blocks_further_use(path):
    ledger = IdempotencyLedger(path)
    ledger.register(KEY, "op", ONE)
    ledger.close()
    ledger.close()
    for call in (lambda: ledger.get_request(KEY), lambda: ledger.register(KEY, "op", ONE), lambda: ledger.begin_execution(KEY, FP),
                 ledger.recover_unfinished, ledger.check_integrity, ledger.list_requests, lambda: ledger.check(KEY, "op", ONE)):
        with pytest.raises(LedgerClosedError):
            call()


def test_a_read_only_instance_reads_and_refuses_changes(path):
    with IdempotencyLedger(path) as writer:
        writer.register(KEY, "op", ONE)
        with IdempotencyLedger(path, read_only=True) as reader:
            assert reader.get_request(KEY).state is S.REGISTERED
            assert reader.check_integrity().ok and reader.session_id is None
            for call in (lambda: reader.register("k", "op", ONE), lambda: reader.begin_execution(KEY, FP), reader.recover_unfinished,
                         lambda: reader.authorize_retry(KEY, authorized_by="o", reason="r")):
                with pytest.raises(LedgerReadOnlyError):
                    call()
            writer.begin_execution(KEY, FP)
            assert reader.get_request(KEY).state is S.IN_PROGRESS


def test_read_only_never_creates_a_ledger(path):
    with pytest.raises(module.LedgerError, match="no ledger exists"):
        IdempotencyLedger(path, read_only=True)
    assert not path.exists()


def test_listing_filters_by_state_in_registration_order(ledger):
    for key in ("a", "b", "c"):
        ledger.register(key, "op", ONE)
    ledger.begin_execution("b", FP)
    assert [r.idempotency_key for r in ledger.list_requests(S.REGISTERED)] == ["a", "c"]
    assert [r.idempotency_key for r in ledger.list_requests("IN_PROGRESS")] == ["b"]
    assert len(ledger.list_requests()) == 3 and ledger.list_requests(S.COMPLETED) == ()
    with pytest.raises(ValueError):
        ledger.list_requests("NOT_A_STATE")


def test_returned_values_cannot_change_the_ledger(ledger):
    request = ledger.register(KEY, "op", ONE).request
    with pytest.raises(AttributeError):
        request.state = S.COMPLETED
    digests = request.statement_digests
    assert isinstance(digests, tuple) and isinstance(digests[0], tuple)
    assert ledger.get_request(KEY) is not request and ledger.get_request(KEY) == request
    assert request.return_rows is False
    assert request.recovery_metadata is None


def test_recovery_metadata_is_a_fresh_copy_each_time(path):
    with IdempotencyLedger(path) as first:
        _begin(first)
    with IdempotencyLedger(path) as second:
        second.recover_unfinished()
        request = second.get_request(KEY)
        request.recovery_metadata["attempt"] = 99
        assert request.recovery_metadata["attempt"] == 1


def test_changing_the_callers_objects_afterwards_changes_nothing(ledger):
    parameters = {"a": [1]}
    ledger.register(KEY, "op", [("Q", parameters)])
    fingerprint = ledger.get_request(KEY).fingerprint
    parameters["a"].append(2)
    assert ledger.get_request(KEY).fingerprint == fingerprint
    with pytest.raises(IdempotencyConflictError):
        ledger.register(KEY, "op", [("Q", parameters)])


# ================================================================ 19 append-only

@pytest.mark.parametrize("statement", [
    "UPDATE ledger_history SET actor = 'x'", "DELETE FROM ledger_history",
    "UPDATE ledger_conflicts SET presented_fingerprint = 'x'", "DELETE FROM ledger_conflicts",
    "DELETE FROM ledger_requests", "UPDATE ledger_requests SET fingerprint = 'x'", "UPDATE ledger_requests SET idempotency_key = 'x'",
    "UPDATE ledger_requests SET operation = 'x'", "UPDATE ledger_requests SET payload_summary = '{}'", "UPDATE ledger_requests SET fingerprint_version = 2",
    "UPDATE ledger_requests SET created_at = 'x'", "UPDATE ledger_requests SET state = 'NOT_A_STATE'",
], ids=["history update", "history delete", "conflict update", "conflict delete", "request delete", "fingerprint", "key", "operation",
        "summary", "fingerprint version", "created_at", "unknown state"])
def test_the_database_itself_refuses_changes_to_history_and_identity(ledger, statement):
    ledger.register(KEY, "op", ONE)
    with pytest.raises(IdempotencyConflictError):
        ledger.register(KEY, "op", [("Q", None)])
    with pytest.raises(module.sqlite3.IntegrityError):
        ledger._conn.execute(statement)
    assert ledger.check_integrity().ok


def test_the_database_itself_refuses_a_second_row_for_a_key(ledger):
    ledger.register(KEY, "op", ONE)
    columns = ", ".join(module._REQUEST_COLUMNS)
    with pytest.raises(module.sqlite3.IntegrityError, match="UNIQUE"):
        ledger._conn.execute(f"INSERT INTO ledger_requests ({columns}, row_sha256) SELECT {columns}, row_sha256 FROM ledger_requests")


# ================================================================ safety of what's exposed

def test_parameter_values_never_reach_the_database_file(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, "op", [("Q", {"token": SECRET})])
        with pytest.raises(IdempotencyConflictError):
            ledger.register(KEY, "op", [("Q", {"token": SECRET + "2"})])
        for name in (path, Path(str(path) + "-wal")):
            assert SECRET.encode() not in name.read_bytes()
    assert SECRET.encode() not in path.read_bytes()


def test_nothing_sensitive_reaches_repr_errors_or_logs(path, caplog):
    caplog.set_level(logging.DEBUG)
    secret_key = "customer-" + SECRET
    with IdempotencyLedger(path) as ledger:
        request = ledger.register(secret_key, "op", [("Q", {"token": SECRET})]).request
        assert SECRET not in repr(request) and SECRET not in repr(ledger.get_history(secret_key))
        with pytest.raises(IdempotencyConflictError) as caught:
            ledger.register(secret_key, "op", [("Q", {"token": "other-" + SECRET})])
        assert SECRET not in str(caught.value)
        with pytest.raises(InvalidTransitionError) as caught:
            ledger.record_completed(secret_key, attempt=1)
        assert SECRET not in str(caught.value)
        with pytest.raises(RequestNotFoundError) as caught:
            ledger.get_request(secret_key + "x")
        assert SECRET not in str(caught.value)
        with pytest.raises(UnsupportedPayloadError) as caught:
            ledger.register("k", "op", [("Q", {"token": {SECRET}})])
        assert SECRET not in str(caught.value)
        ledger.begin_execution(secret_key, request.fingerprint)
        ledger.record_uncertain(secret_key, attempt=1)
        assert request.idempotency_key == secret_key, "the caller can still read its own key"
        assert request.key_ref.startswith("key:") and SECRET not in request.key_ref
    assert SECRET not in caplog.text


def test_the_ledger_has_no_way_to_run_or_replay_anything():
    public = {name for name in vars(IdempotencyLedger) if not name.startswith("_")}
    assert public == {"register", "check", "get_request", "list_requests", "get_history", "get_conflicts", "begin_execution", "record_completed",
                      "record_not_completed", "record_uncertain", "authorize_retry", "resolve_uncertain", "recover_unfinished", "check_integrity", "close"}
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert imported.isdisjoint({"neo4j_client", "config", "neo4j_write_journal", "dedup_engine", "subprocess", "socket", "requests"})
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert calls.isdisjoint({"execute_write", "session", "run", "begin_transaction"})


def test_the_module_loads_neither_the_client_nor_configuration():
    code = ("import sys; sys.path.insert(0, %r); from tests._production_safety import install_child_guards; install_child_guards(); "
            "before = set(sys.modules); import neo4j_idempotency_ledger; new = set(sys.modules) - before; "
            "print(sorted(new & {'neo4j_client', 'config', 'dedup_engine', 'dotenv', 'neo4j_write_journal'}))" % str(BACKEND))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=dict(os.environ))
    assert result.returncode == 0, result.stderr[-500:]
    assert json.loads(result.stdout.strip().replace("'", '"')) == []


@pytest.mark.parametrize("name", ["neo4j_idempotency_ledger", "neo4j_write_journal"])
def test_no_production_module_uses_the_ledger_or_journal(name):
    users = []
    for source in BACKEND.rglob("*.py"):
        if "tests" in source.parts or source.name == f"{name}.py" or "__pycache__" in source.parts:
            continue
        if name in source.read_text(encoding="utf-8", errors="replace"):
            users.append(str(source))
    # WP-C1: the coordinator is the one permitted user, and test_no_production_module_imports_the_coordinator_yet keeps it uncalled.
    assert users == [str(BACKEND / "neo4j_write_coordinator.py")]

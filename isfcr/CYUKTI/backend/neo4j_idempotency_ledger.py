"""Durable ledger of logical write requests: which exist, and whether one may run.

Standalone. Nothing here runs a query or replays a write, it imports neither neo4j_client nor
config nor the journal, and no production code uses it. Production writes are NOT protected by
it, and production keys don't exist yet: mapping writers to keys is a later decision.

Identity
    A request is an idempotency key, which the caller must supply and which is matched exactly, plus
    a payload fingerprint. Same key and same fingerprint: the same logical request. Same key and a
    different fingerprint: a conflict, which is audited and raised, and never overwrites anything.
    Different keys with the same payload: separate requests. The ledger never generates a key.

Fingerprint
    SHA-256 over a versioned canonical form of the operation, return_rows and the ordered statements.
    The rule is "when in doubt, different": a false conflict stops a write for someone to look at,
    while a false match would silently skip one. So batch and list order, 1 and 1.0 and True, 0.0 and
    -0.0, None and {}, list and tuple, bytes and bytearray, datetime and neo4j.time, the same instant
    in another timezone, and any change to query text all differ. Only map key order doesn't, because
    Cypher maps have none. Exact types only: subclasses and unknown types are refused.

    The format is frozen by version. It's separate from the journal's serialization on purpose, so a
    stored fingerprint can't change because the journal did. A request stored under another version
    can't be compared, and that's an error rather than being treated as new.

    Parameter values are not stored here: only digests, per statement, so a conflict can say which
    statement differs. Digests of low-entropy values can be guessed by trying candidates, so they
    aren't encryption. The operation name and the key are stored in plaintext, so don't put parameter
    values in keys. Logs and errors refer to a key by a short hash.

States
    REGISTERED        known, never started
    IN_PROGRESS       claimed by begin_execution, outcome not yet recorded
    COMPLETED         completion confirmed, or found by a person. Final
    NOT_COMPLETED     established as not completed. Runs again only through authorize_retry()
    RETRY_AUTHORIZED  one more attempt allowed, by a named person or tool. Used up by begin_execution
    UNCERTAIN         unknown. Never executable. Leaves only through resolve_uncertain()
    ABANDONED         a person decided it won't run. Final

    Only REGISTERED and RETRY_AUTHORIZED can be claimed, and begin_execution is atomic, so one caller
    wins. A registration, duplicate or not, grants nothing. After FOUND_NOT_COMPLETED a request still
    needs authorize_retry(), so running an uncertain request again takes two explicit acts.

What the ledger can't do
    It can't commit atomically with Neo4j. If Neo4j commits and the process dies before
    record_completed(), recovery makes the request UNCERTAIN. So this gives at most one automatic
    execution per authorization, on one host: it doesn't make a Neo4j write exactly-once, and
    COMPLETED records what was reported, not what Neo4j holds. The file is local, so processes on
    another machine, or using another file, aren't covered.

Durability
    SQLite in WAL mode with synchronous=FULL, verified on open. Verified by tests: those settings are
    in force, and committed changes survive SIGKILL. Not verifiable here: power loss, which depends
    on the disk and filesystem honouring fsync. Use a local POSIX filesystem: WAL and flock aren't
    reliable on network filesystems or WSL's /mnt/c. If a call raises, treat the change as not recorded.

Liveness
    Each writable instance holds a flock on its own file in <path>.sessions/ while open, and the kernel
    releases it when the process dies. recover_unfinished() converts IN_PROGRESS only when the owner's
    lock is free. A forked child inherits the lock and keeps its owner looking alive until it exits.
    A missing lock file counts as an ended session, which errs towards UNCERTAIN.

Integrity
    SHA-256 over each row, plus hash chains over each request's history and conflicts, detects accidental
    damage. There is no secret key, so it doesn't stop deliberate tampering. A damaged or foreign file is
    refused and left as it is: nothing here recreates or truncates a ledger.

Retention
    Nothing is ever deleted, and triggers enforce it, because removing a key would make that request
    executable again. The file grows, including by one row per conflicting registration.
"""

from __future__ import annotations

import base64
import datetime as dt
import fcntl
import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import unicodedata
import uuid
import zoneinfo
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional
from urllib.parse import quote

import neo4j.time as neo4j_time
import pytz

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
FINGERPRINT_VERSION = 1
_FINGERPRINT_DOMAIN = b"cyukti-idempotency-fingerprint\x00"
_STATEMENT_DOMAIN = b"cyukti-idempotency-statement\x00"
_APPLICATION_ID = 0x4359494C  # "CYIL"
_SQLITE_MAGIC = b"SQLite format 3\x00"
_MAX_NESTING = 64
_MAX_KEY_LENGTH = 512
_MAX_DETAIL_LENGTH = 500
_MAX_ACTOR_LENGTH = 200


class LedgerState(str, Enum):
    REGISTERED = "REGISTERED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    NOT_COMPLETED = "NOT_COMPLETED"
    RETRY_AUTHORIZED = "RETRY_AUTHORIZED"
    UNCERTAIN = "UNCERTAIN"
    ABANDONED = "ABANDONED"


class Resolution(str, Enum):
    """What someone concluded about an uncertain request."""
    FOUND_COMPLETED = "FOUND_COMPLETED"
    FOUND_NOT_COMPLETED = "FOUND_NOT_COMPLETED"
    ABANDONED = "ABANDONED"


class CheckStatus(str, Enum):
    NEW = "NEW"
    DUPLICATE = "DUPLICATE"
    CONFLICT = "CONFLICT"


class _Cause(str, Enum):
    EXECUTE = "EXECUTE"
    OUTCOME = "OUTCOME"
    RETRY = "RETRY"
    RECOVERY = "RECOVERY"
    RESOLUTION = "RESOLUTION"


_S = LedgerState
_TRANSITIONS: dict[tuple[LedgerState, LedgerState], frozenset[_Cause]] = {
    (_S.REGISTERED, _S.IN_PROGRESS): frozenset({_Cause.EXECUTE}),
    (_S.RETRY_AUTHORIZED, _S.IN_PROGRESS): frozenset({_Cause.EXECUTE}),
    (_S.IN_PROGRESS, _S.COMPLETED): frozenset({_Cause.OUTCOME}),
    (_S.IN_PROGRESS, _S.NOT_COMPLETED): frozenset({_Cause.OUTCOME}),
    (_S.IN_PROGRESS, _S.UNCERTAIN): frozenset({_Cause.OUTCOME, _Cause.RECOVERY}),
    (_S.NOT_COMPLETED, _S.RETRY_AUTHORIZED): frozenset({_Cause.RETRY}),
    (_S.UNCERTAIN, _S.COMPLETED): frozenset({_Cause.RESOLUTION}),
    (_S.UNCERTAIN, _S.NOT_COMPLETED): frozenset({_Cause.RESOLUTION}),
    (_S.UNCERTAIN, _S.ABANDONED): frozenset({_Cause.RESOLUTION}),
}
_RESOLUTION_TARGET = {
    Resolution.FOUND_COMPLETED: _S.COMPLETED,
    Resolution.FOUND_NOT_COMPLETED: _S.NOT_COMPLETED,
    Resolution.ABANDONED: _S.ABANDONED,
}
_CLAIMABLE = frozenset({_S.REGISTERED, _S.RETRY_AUTHORIZED})


class LedgerError(Exception):
    """Base class. Messages carry short key hashes and type names, never parameter values."""


class LedgerWriteError(LedgerError):
    """A write to the ledger failed, so the change wasn't recorded."""


class LedgerBusyError(LedgerWriteError):
    """Another connection held the database for longer than busy_timeout."""


class LedgerCorruptionError(LedgerError):
    """The ledger or a request in it is damaged. Nothing was changed."""


class LedgerSchemaError(LedgerError):
    """The file isn't a ledger this version can use. Nothing was changed."""


class LedgerClosedError(LedgerError):
    """The ledger has been closed, or this process isn't the one that opened it."""


class LedgerReadOnlyError(LedgerError):
    """A change was attempted through a read-only instance."""


class RequestNotFoundError(LedgerError):
    pass


class InvalidTransitionError(LedgerError):
    """The state machine refused the change. The request wasn't modified."""


class StaleAttemptError(InvalidTransitionError):
    """An outcome named an attempt other than the current one. The request wasn't modified."""


class UnsupportedPayloadError(LedgerError, TypeError):
    """A payload value has no canonical form, so nothing was registered."""


class FingerprintVersionError(LedgerError):
    """The stored request uses a fingerprint format this code can't compare. It's not treated as
    new, because that would allow a second execution."""


class IdempotencyConflictError(LedgerError):
    """The key is registered with a different payload. The existing request wasn't changed."""

    def __init__(self, message: str, *, existing_fingerprint: str, presented_fingerprint: str,
                 differences: tuple[str, ...], conflict_no: Optional[int] = None) -> None:
        super().__init__(message)
        self.existing_fingerprint = existing_fingerprint
        self.presented_fingerprint = presented_fingerprint
        self.differences = differences
        self.conflict_no = conflict_no


# ---------------------------------------------------------------- canonical form, frozen by version

_FIXED_PYTZ_TYPE = type(pytz.FixedOffset(60))


def _canonical_timezone(value: Any, where: str) -> Any:
    tz = value.tzinfo
    if tz is None:
        return None
    if type(tz) is dt.timezone:
        offset = tz.utcoffset(None)
        name = tz.tzname(None)
        return ["fixed", offset.days, offset.seconds, offset.microseconds,
                None if name == dt.timezone(offset).tzname(None) else name]
    if type(tz) is zoneinfo.ZoneInfo and tz.key:
        return ["zoneinfo", tz.key]
    if type(tz) is _FIXED_PYTZ_TYPE:
        return ["pytz_fixed", int(tz.utcoffset(None).total_seconds() // 60)]
    if isinstance(tz, pytz.tzinfo.BaseTzInfo) and tz.zone:
        offset = value.utcoffset()
        return ["pytz", tz.zone, None if offset is None else offset.total_seconds()]
    raise UnsupportedPayloadError(f"{where}: timezone type {type(tz).__name__} is not supported")


def _canonical_value(value: Any, where: str, depth: int = 0, ancestors: frozenset = frozenset()) -> Any:
    """Every JSON object in the result is a tagged value with one key, and a JSON number is always
    an integer, so nothing in a caller's data can be mistaken for something else."""
    kind = type(value)
    if value is None or kind is bool or kind is str:
        return value
    if kind is int:
        if not -(2 ** 63) <= value < 2 ** 63:
            raise UnsupportedPayloadError(f"{where}: integer is outside the 64-bit range")
        return value
    if kind is float:
        return {"float": value.hex()}
    if kind in (list, tuple, dict):
        if depth >= _MAX_NESTING:
            raise UnsupportedPayloadError(f"{where}: nested more than {_MAX_NESTING} levels")
        if id(value) in ancestors:
            raise UnsupportedPayloadError(f"{where}: contains a reference cycle")
        inner = ancestors | {id(value)}
        if kind is dict:
            for key in value:
                if type(key) is not str:
                    raise UnsupportedPayloadError(f"{where}: map key type {type(key).__name__} is not supported")
            return {"map": {k: _canonical_value(v, where, depth + 1, inner) for k, v in value.items()}}
        items = [_canonical_value(v, where, depth + 1, inner) for v in value]
        return items if kind is list else {"tuple": items}
    if kind is bytes:
        return {"bytes": base64.b64encode(value).decode("ascii")}
    if kind is bytearray:
        return {"bytearray": base64.b64encode(bytes(value)).decode("ascii")}
    if kind is dt.datetime:
        return {"datetime": [value.year, value.month, value.day, value.hour, value.minute, value.second,
                             value.microsecond, value.fold, _canonical_timezone(value, where)]}
    if kind is dt.date:
        return {"date": [value.year, value.month, value.day]}
    if kind is dt.time:
        return {"time": [value.hour, value.minute, value.second, value.microsecond, value.fold,
                         _canonical_timezone(value, where)]}
    if kind is dt.timedelta:
        return {"timedelta": [value.days, value.seconds, value.microseconds]}
    if kind is neo4j_time.DateTime:
        return {"neo4j_datetime": [value.year, value.month, value.day, value.hour, value.minute, value.second,
                                   value.nanosecond, _canonical_timezone(value, where)]}
    if kind is neo4j_time.Date:
        return {"neo4j_date": [value.year, value.month, value.day]}
    if kind is neo4j_time.Time:
        return {"neo4j_time": [value.hour, value.minute, value.second, value.nanosecond,
                               _canonical_timezone(value, where)]}
    if kind is neo4j_time.Duration:
        return {"neo4j_duration": [value.months, value.days, value.seconds, value.nanoseconds]}
    raise UnsupportedPayloadError(f"{where}: type {kind.__module__}.{kind.__qualname__} is not supported")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _digest(domain: bytes, text: str) -> str:
    return hashlib.sha256(domain + text.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class PayloadFingerprint:
    """statement_digests holds (query digest, parameters digest) per statement, in order."""
    version: int
    value: str
    operation: str
    return_rows: bool
    statement_digests: tuple[tuple[str, str], ...]

    def _summary_json(self) -> str:
        return _canonical_json({"operation": self.operation, "return_rows": self.return_rows,
                                "statements": [list(pair) for pair in self.statement_digests]})


def fingerprint_payload(operation: str, statements: Any, *, return_rows: bool = False) -> PayloadFingerprint:
    """Computes the fingerprint and stores nothing. Raises if the payload has no canonical form."""
    if type(operation) is not str or not operation.strip():
        raise ValueError("operation must be a non-empty string")
    if type(return_rows) is not bool:
        raise TypeError("return_rows must be a bool")
    if isinstance(statements, (str, bytes)):
        raise TypeError("statements must be a sequence of (query, parameters) pairs")
    try:
        materialized = list(statements)
    except TypeError:
        raise TypeError("statements must be a sequence of (query, parameters) pairs") from None
    if not materialized:
        raise ValueError("a write needs at least one statement")
    canonical = []
    digests = []
    for index, statement in enumerate(materialized):
        if type(statement) not in (tuple, list) or len(statement) != 2:
            raise TypeError(f"statement {index} must be a (query, parameters) pair")
        query, parameters = statement
        if type(query) is not str or not query.strip():
            raise ValueError(f"statement {index}: the query must be a non-empty string")
        if parameters is None:
            body = None
        elif type(parameters) is not dict:
            raise TypeError(f"statement {index}: parameters must be a dict or None")
        else:
            for name in parameters:
                if type(name) is not str:
                    raise UnsupportedPayloadError(f"statement {index}: parameter name type {type(name).__name__} is not supported")
            body = {"map": {name: _canonical_value(value, f"statement {index}, parameter {name!r}")
                            for name, value in parameters.items()}}
        canonical.append({"query": query, "parameters": body})
        digests.append((_digest(_STATEMENT_DOMAIN, _canonical_json(query)), _digest(_STATEMENT_DOMAIN, _canonical_json(body))))
    text = _canonical_json({"canonicalization": FINGERPRINT_VERSION, "operation": operation,
                            "return_rows": return_rows, "statements": canonical})
    return PayloadFingerprint(FINGERPRINT_VERSION, _digest(_FINGERPRINT_DOMAIN, text), operation, return_rows, tuple(digests))


def _differences(existing_summary: str, presented_summary: str) -> tuple[str, ...]:
    """Where two payloads differ, in terms that reveal no values."""
    existing, presented = json.loads(existing_summary), json.loads(presented_summary)
    found = []
    if existing["operation"] != presented["operation"]:
        found.append("operation")
    if existing["return_rows"] != presented["return_rows"]:
        found.append("return_rows")
    if len(existing["statements"]) != len(presented["statements"]):
        found.append("statement count")
    for index, (old, new) in enumerate(zip(existing["statements"], presented["statements"])):
        if old[0] != new[0]:
            found.append(f"statement {index} query")
        if old[1] != new[1]:
            found.append(f"statement {index} parameters")
    return tuple(found)


def _checked_key(key: Any) -> str:
    """Matched exactly from here on, so anything likely to be an accident is refused rather than
    normalized. A generated key can't be detected here: not generating one is the caller's contract."""
    if type(key) is not str:
        raise TypeError("the idempotency key must be a string")
    if not key or len(key) > _MAX_KEY_LENGTH:
        raise ValueError(f"the idempotency key must be 1 to {_MAX_KEY_LENGTH} characters")
    if key != key.strip():
        raise ValueError("the idempotency key must not have surrounding whitespace")
    if any(unicodedata.category(ch) in ("Cc", "Cs") for ch in key):
        raise ValueError("the idempotency key must not contain control characters or lone surrogates")
    return key


def _key_ref(key: str) -> str:
    """How logs and errors name a key."""
    return "key:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- returned values

@dataclass(frozen=True)
class LedgerRequest:
    """A copy of a request at the time it was read. Changing it can't change the ledger."""
    idempotency_key: str = field(repr=False)
    key_ref: str
    fingerprint_version: int
    fingerprint: str
    operation: str
    created_at: str
    updated_at: str
    state: LedgerState
    attempt_count: int
    worker_id: Optional[str]
    owner_session: Optional[str]
    last_outcome: Optional[str]
    error_type: Optional[str]
    error_code: Optional[str]
    diagnostic: Optional[str]
    resolution: Optional[Resolution]
    conflict_count: int
    version: int
    _summary_json: str = field(repr=False)
    _recovery_json: Optional[str] = field(repr=False)

    @property
    def claimable(self) -> bool:
        """Whether begin_execution could succeed now. Advisory: only that call decides."""
        return self.state in _CLAIMABLE

    @property
    def statement_digests(self) -> tuple[tuple[str, str], ...]:
        return tuple(tuple(pair) for pair in json.loads(self._summary_json)["statements"])

    @property
    def return_rows(self) -> bool:
        return json.loads(self._summary_json)["return_rows"]

    @property
    def recovery_metadata(self) -> Optional[dict]:
        return None if self._recovery_json is None else json.loads(self._recovery_json)


@dataclass(frozen=True)
class Registration:
    """status is NEW or DUPLICATE; a conflict raises. Neither status grants the right to run."""
    status: CheckStatus
    request: LedgerRequest


@dataclass(frozen=True)
class LedgerCheck:
    """What register() would find, without recording anything."""
    status: CheckStatus
    request: Optional[LedgerRequest]
    presented_fingerprint: str
    differences: tuple[str, ...] = ()


@dataclass(frozen=True)
class LedgerEvent:
    """One entry in a request's append-only history."""
    key_ref: str
    transition_no: int
    from_state: Optional[LedgerState]
    to_state: LedgerState
    cause: str
    at: str
    actor: Optional[str]
    reason: Optional[str]
    session_id: Optional[str]
    worker_id: Optional[str]
    attempt: int
    error_type: Optional[str]
    error_code: Optional[str]
    diagnostic: Optional[str]
    resolution: Optional[Resolution]


@dataclass(frozen=True)
class LedgerConflict:
    """One audited attempt to register a key with a different payload. Holds no values."""
    key_ref: str
    conflict_no: int
    at: str
    session_id: Optional[str]
    presented_version: int
    presented_fingerprint: str
    differences: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryReport:
    """recovered: IN_PROGRESS requests whose owner is gone, now UNCERTAIN. still_running: owner alive,
    left alone. corrupt: damaged, left alone. The rest are listed and untouched. Nothing is executed."""
    recovered: tuple[str, ...]
    still_running: tuple[str, ...]
    registered: tuple[str, ...]
    not_completed: tuple[str, ...]
    retry_authorized: tuple[str, ...]
    uncertain: tuple[str, ...]
    corrupt: tuple[str, ...]


@dataclass(frozen=True)
class IntegrityReport:
    """corrupt_keys are key refs."""
    ok: bool
    schema_version: Optional[int]
    requests_checked: int
    corrupt_keys: tuple[str, ...]
    problems: tuple[str, ...]


# ---------------------------------------------------------------- storage

_SCHEMA = (
    """CREATE TABLE ledger_requests (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        idempotency_key TEXT NOT NULL UNIQUE,
        fingerprint_version INTEGER NOT NULL,
        fingerprint TEXT NOT NULL,
        operation TEXT NOT NULL,
        payload_summary TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN
            ('REGISTERED','IN_PROGRESS','COMPLETED','NOT_COMPLETED','RETRY_AUTHORIZED','UNCERTAIN','ABANDONED')),
        attempt_count INTEGER NOT NULL,
        worker_id TEXT,
        owner_session TEXT,
        last_outcome TEXT,
        error_type TEXT,
        error_code TEXT,
        diagnostic TEXT,
        recovery_json TEXT,
        resolution TEXT,
        version INTEGER NOT NULL,
        history_head TEXT NOT NULL,
        conflict_count INTEGER NOT NULL,
        conflict_head TEXT NOT NULL,
        row_sha256 TEXT NOT NULL
    )""",
    "CREATE INDEX ledger_requests_state ON ledger_requests(state, seq)",
    """CREATE TABLE ledger_history (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        idempotency_key TEXT NOT NULL,
        transition_no INTEGER NOT NULL,
        from_state TEXT,
        to_state TEXT NOT NULL,
        cause TEXT NOT NULL,
        at TEXT NOT NULL,
        actor TEXT,
        reason TEXT,
        session_id TEXT,
        worker_id TEXT,
        attempt INTEGER NOT NULL,
        error_type TEXT,
        error_code TEXT,
        diagnostic TEXT,
        resolution TEXT,
        prev_sha256 TEXT NOT NULL,
        entry_sha256 TEXT NOT NULL,
        UNIQUE (idempotency_key, transition_no)
    )""",
    """CREATE TABLE ledger_conflicts (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        idempotency_key TEXT NOT NULL,
        conflict_no INTEGER NOT NULL,
        at TEXT NOT NULL,
        session_id TEXT,
        presented_version INTEGER NOT NULL,
        presented_fingerprint TEXT NOT NULL,
        presented_summary TEXT NOT NULL,
        prev_sha256 TEXT NOT NULL,
        entry_sha256 TEXT NOT NULL,
        UNIQUE (idempotency_key, conflict_no)
    )""",
    """CREATE TRIGGER ledger_history_no_update BEFORE UPDATE ON ledger_history
        BEGIN SELECT RAISE(ABORT, 'ledger history is append-only'); END""",
    """CREATE TRIGGER ledger_history_no_delete BEFORE DELETE ON ledger_history
        BEGIN SELECT RAISE(ABORT, 'ledger history is append-only'); END""",
    """CREATE TRIGGER ledger_conflicts_no_update BEFORE UPDATE ON ledger_conflicts
        BEGIN SELECT RAISE(ABORT, 'ledger conflicts are append-only'); END""",
    """CREATE TRIGGER ledger_conflicts_no_delete BEFORE DELETE ON ledger_conflicts
        BEGIN SELECT RAISE(ABORT, 'ledger conflicts are append-only'); END""",
    """CREATE TRIGGER ledger_requests_no_delete BEFORE DELETE ON ledger_requests
        BEGIN SELECT RAISE(ABORT, 'ledger requests are never deleted'); END""",
    """CREATE TRIGGER ledger_requests_identity_fixed
        BEFORE UPDATE OF idempotency_key, fingerprint_version, fingerprint, operation, payload_summary, created_at
        ON ledger_requests
        BEGIN SELECT RAISE(ABORT, 'a registered identity cannot change'); END""",
)
_REQUIRED_OBJECTS = frozenset({
    "ledger_requests", "ledger_history", "ledger_conflicts", "ledger_requests_state", "ledger_history_no_update",
    "ledger_history_no_delete", "ledger_conflicts_no_update", "ledger_conflicts_no_delete",
    "ledger_requests_no_delete", "ledger_requests_identity_fixed",
})
_REQUEST_COLUMNS = (
    "idempotency_key", "fingerprint_version", "fingerprint", "operation", "payload_summary", "created_at", "updated_at",
    "state", "attempt_count", "worker_id", "owner_session", "last_outcome", "error_type", "error_code", "diagnostic",
    "recovery_json", "resolution", "version", "history_head", "conflict_count", "conflict_head",
)
_HISTORY_COLUMNS = (
    "idempotency_key", "transition_no", "from_state", "to_state", "cause", "at", "actor", "reason", "session_id",
    "worker_id", "attempt", "error_type", "error_code", "diagnostic", "resolution",
)
_CONFLICT_COLUMNS = (
    "idempotency_key", "conflict_no", "at", "session_id", "presented_version", "presented_fingerprint", "presented_summary",
)
_CHAIN_START = "0" * 64
_SELECT_REQUEST = f"SELECT {', '.join(_REQUEST_COLUMNS)}, row_sha256 FROM ledger_requests"
_KEEP = object()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _sha256(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(list(parts), ensure_ascii=True, separators=(",", ":")).encode("ascii")).hexdigest()


def _clean_detail(value: Any, name: str, limit: int = _MAX_DETAIL_LENGTH) -> Optional[str]:
    """Takes safe summaries such as WriteResult.diagnostic. An exception is refused, because its
    message can quote parameter values."""
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{name} must be a string or None, not {type(value).__name__}")
    return value[:limit]


def _required_text(value: Any, name: str, limit: int = _MAX_ACTOR_LENGTH) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value[:limit]


def _checked_attempt(attempt: Any) -> int:
    if type(attempt) is not int or attempt < 1:
        raise ValueError("attempt must be the positive attempt number begin_execution returned")
    return attempt


def _translate(error: sqlite3.Error) -> LedgerError:
    if "could not decode" in str(error).lower():
        return LedgerCorruptionError("a value in the ledger file is damaged")
    code = (getattr(error, "sqlite_errorcode", 0) or 0) & 0xFF
    if code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        return LedgerBusyError("the ledger is locked by another connection")
    if code in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB):
        return LedgerCorruptionError("SQLite reports the ledger file is damaged")
    name = getattr(error, "sqlite_errorname", None) or type(error).__name__
    return LedgerWriteError(f"ledger storage failed: {name}")


def _inspect_existing_file(path: str) -> bool:
    """True if a ledger exists at path, False if it's absent or empty. Reads only, before SQLite
    opens anything: opening a foreign database in WAL mode could change it."""
    try:
        size = os.stat(path).st_size
    except FileNotFoundError:
        return False
    if size == 0:
        return False
    with open(path, "rb") as handle:
        header = handle.read(100)
    if len(header) < 100 or not header.startswith(_SQLITE_MAGIC):
        raise LedgerCorruptionError("not a ledger: the file has no valid SQLite header. It was left unchanged")
    if int.from_bytes(header[68:72], "big") != _APPLICATION_ID:
        raise LedgerSchemaError("not a ledger: this SQLite database belongs to something else. It was left unchanged")
    return True


class IdempotencyLedger:
    """Thread-safe: one connection, serialized by a lock. Several instances or processes can share a
    file: the key's UNIQUE constraint and BEGIN IMMEDIATE decide every race, and each change commits
    its request row and audit entry together.

    path must be given explicitly, and its directory must already exist."""

    def __init__(self, path: Any, *, busy_timeout: float = 5.0, read_only: bool = False) -> None:
        self._path = os.path.abspath(os.fspath(path))
        self._read_only = read_only
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._conn: Optional[sqlite3.Connection] = None
        self._session_fd: Optional[int] = None
        self._session_path: Optional[str] = None
        self.session_id: Optional[str] = None

        if not os.path.isdir(os.path.dirname(self._path)):
            raise LedgerError("the ledger's directory does not exist")
        exists = _inspect_existing_file(self._path)
        if read_only and not exists:
            raise LedgerError("no ledger exists at this path to open read-only")
        try:
            if read_only:
                self._conn = sqlite3.connect(f"file:{quote(self._path)}?mode=ro", uri=True, timeout=busy_timeout,
                                             isolation_level=None, check_same_thread=False)
            else:
                if not exists:
                    os.close(os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600))
                self._conn = sqlite3.connect(self._path, timeout=busy_timeout, isolation_level=None,
                                             check_same_thread=False)
                if not exists:
                    self._initialize_if_new()
            self._validate_schema()
            if not read_only:
                self._configure_durability()
                self._begin_session()
            if not exists:
                self._sync_directory(os.path.dirname(self._path))
        except sqlite3.Error as error:
            self._abandon()
            raise _translate(error) from error
        except BaseException:
            self._abandon()
            raise

    def __enter__(self) -> "IdempotencyLedger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _abandon(self) -> None:
        """A failed open. Leaves the file exactly as it is."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None
        self._release_session()

    @staticmethod
    def _sync_directory(directory: str) -> None:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _initialize_if_new(self) -> None:
        """Header and schema commit together, in rollback-journal mode, so a crash here leaves an
        empty file rather than one this class would later refuse."""
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            count = conn.execute("SELECT count(*) FROM sqlite_schema").fetchone()[0]
            if version == 0 and count == 0 and conn.execute("PRAGMA application_id").fetchone()[0] == 0:
                conn.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                for statement in _SCHEMA:
                    conn.execute(statement)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def _validate_schema(self) -> None:
        conn = self._conn
        if conn.execute("PRAGMA application_id").fetchone()[0] != _APPLICATION_ID:
            raise LedgerSchemaError("not a ledger: unexpected application id. It was left unchanged")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise LedgerSchemaError(f"ledger schema version {version} is not supported; this code uses {SCHEMA_VERSION}")
        present = {row[0] for row in conn.execute("SELECT name FROM sqlite_schema")}
        missing = _REQUIRED_OBJECTS - present
        if missing:
            raise LedgerCorruptionError(f"the ledger is missing {len(missing)} of its schema objects: {sorted(missing)}")
        if conn.execute("PRAGMA quick_check(1)").fetchone()[0] != "ok":
            raise LedgerCorruptionError("SQLite's quick check reports the ledger file is damaged")

    def _configure_durability(self) -> None:
        conn = self._conn
        mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        conn.execute("PRAGMA synchronous = FULL")
        synchronous = conn.execute("PRAGMA synchronous").fetchone()[0]
        if mode.lower() != "wal" or synchronous != 2:
            raise LedgerWriteError(f"could not establish durable settings (journal_mode={mode}, synchronous={synchronous})")

    def _begin_session(self) -> None:
        directory = self._path + ".sessions"
        os.makedirs(directory, mode=0o700, exist_ok=True)
        self.session_id = uuid.uuid4().hex
        self._session_path = os.path.join(directory, self.session_id + ".lock")
        self._session_fd = os.open(self._session_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
        fcntl.flock(self._session_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release_session(self) -> None:
        if self._session_fd is not None:
            try:
                os.unlink(self._session_path)
            except OSError:
                pass
            os.close(self._session_fd)
            self._session_fd = None

    def _session_is_alive(self, session_id: str) -> tuple[bool, Optional[int]]:
        """Whether that session's owner still holds its lock. If not, returns the lock's fd so the
        caller can hold it during recovery. A missing lock file means the session ended."""
        path = os.path.join(self._path + ".sessions", session_id + ".lock")
        try:
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        except FileNotFoundError:
            return False, None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return True, None
        return False, fd

    # ------------------------------------------------------------ plumbing

    def _usable(self, write: bool) -> sqlite3.Connection:
        if self._conn is None or os.getpid() != self._pid:
            raise LedgerClosedError("the ledger is closed, or was opened by another process")
        if write and self._read_only:
            raise LedgerReadOnlyError("this ledger instance is read-only")
        return self._conn

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._usable(write=True)
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except sqlite3.Error as error:
                self._rollback(conn)
                raise _translate(error) from error
            except BaseException:
                self._rollback(conn)
                raise

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._usable(write=False)
            try:
                yield conn
            except sqlite3.Error as error:
                raise _translate(error) from error

    @staticmethod
    def _rollback(conn: sqlite3.Connection) -> None:
        if conn.in_transaction:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error as error:
                logger.error("Ledger rollback failed: %s", type(error).__name__)

    @staticmethod
    def _row_hash(values: dict) -> str:
        return _sha256(*(values[name] for name in _REQUEST_COLUMNS))

    @staticmethod
    def _event_hash(values: dict, previous: str) -> str:
        return _sha256(previous, *(values[name] for name in _HISTORY_COLUMNS))

    @staticmethod
    def _conflict_hash(values: dict, previous: str) -> str:
        return _sha256(previous, *(values[name] for name in _CONFLICT_COLUMNS))

    def _verified(self, row: tuple) -> dict:
        values = dict(zip(_REQUEST_COLUMNS, row))
        if row[len(_REQUEST_COLUMNS)] != self._row_hash(values):
            raise LedgerCorruptionError(f"ledger request {_key_ref(values['idempotency_key'])} failed its integrity check")
        return values

    @staticmethod
    def _request(values: dict) -> LedgerRequest:
        return LedgerRequest(
            idempotency_key=values["idempotency_key"], key_ref=_key_ref(values["idempotency_key"]),
            fingerprint_version=values["fingerprint_version"], fingerprint=values["fingerprint"],
            operation=values["operation"], created_at=values["created_at"], updated_at=values["updated_at"],
            state=LedgerState(values["state"]), attempt_count=values["attempt_count"], worker_id=values["worker_id"],
            owner_session=values["owner_session"], last_outcome=values["last_outcome"], error_type=values["error_type"],
            error_code=values["error_code"], diagnostic=values["diagnostic"],
            resolution=Resolution(values["resolution"]) if values["resolution"] else None,
            conflict_count=values["conflict_count"], version=values["version"],
            _summary_json=values["payload_summary"], _recovery_json=values["recovery_json"],
        )

    def _load(self, conn: sqlite3.Connection, key: str) -> Optional[dict]:
        row = conn.execute(f"{_SELECT_REQUEST} WHERE idempotency_key = ?", (key,)).fetchone()
        return None if row is None else self._verified(row)

    def _require(self, conn: sqlite3.Connection, key: str) -> dict:
        values = self._load(conn, key)
        if values is None:
            raise RequestNotFoundError(f"no ledger request for {_key_ref(key)}")
        return values

    @staticmethod
    def _require_comparable(values: dict) -> None:
        if values["fingerprint_version"] != FINGERPRINT_VERSION:
            raise FingerprintVersionError(
                f"{_key_ref(values['idempotency_key'])} uses fingerprint format {values['fingerprint_version']}; "
                f"this code uses {FINGERPRINT_VERSION}, so the payloads can't be compared")

    def _append_event(self, conn: sqlite3.Connection, event: dict, previous: str) -> str:
        entry = self._event_hash(event, previous)
        columns = ", ".join(_HISTORY_COLUMNS)
        marks = ", ".join("?" * len(_HISTORY_COLUMNS))
        conn.execute(f"INSERT INTO ledger_history ({columns}, prev_sha256, entry_sha256) VALUES ({marks}, ?, ?)",
                     (*(event[name] for name in _HISTORY_COLUMNS), previous, entry))
        return entry

    def _update(self, conn: sqlite3.Connection, values: dict, updated: dict) -> None:
        changed = [name for name in _REQUEST_COLUMNS if updated[name] != values[name]]
        assignments = ", ".join(f"{name} = ?" for name in changed)
        conn.execute(f"UPDATE ledger_requests SET {assignments}, row_sha256 = ? WHERE idempotency_key = ?",
                     (*(updated[name] for name in changed), self._row_hash(updated), values["idempotency_key"]))

    def _transition(self, key: str, target: LedgerState, cause: _Cause, *, attempt: Optional[int] = None,
                    fingerprint: Optional[str] = None, actor: Optional[str] = None, reason: Optional[str] = None,
                    worker_id: Any = _KEEP, last_outcome: Any = _KEEP, error_type: Optional[str] = None,
                    error_code: Optional[str] = None, diagnostic: Optional[str] = None,
                    resolution: Optional[Resolution] = None, recovery_json: Any = _KEEP) -> LedgerRequest:
        """The only code that changes a request's state. The checks, the update and the history entry
        are one transaction, so a refused or failed change leaves everything as it was."""
        now = _now()
        with self._write() as conn:
            values = self._require(conn, key)
            current = LedgerState(values["state"])
            if fingerprint is not None:
                self._require_comparable(values)
                if fingerprint != values["fingerprint"]:
                    raise IdempotencyConflictError(
                        f"{_key_ref(key)}: the fingerprint presented isn't the one registered",
                        existing_fingerprint=values["fingerprint"], presented_fingerprint=fingerprint, differences=("fingerprint",))
            if cause not in _TRANSITIONS.get((current, target), frozenset()):
                raise InvalidTransitionError(f"{_key_ref(key)}: {current.value} to {target.value} is not allowed ({cause.value})")
            if attempt is not None and attempt != values["attempt_count"]:
                raise StaleAttemptError(f"{_key_ref(key)}: attempt {attempt} reported, but the current attempt is {values['attempt_count']}")
            entering = target is LedgerState.IN_PROGRESS
            updated = dict(values)
            updated.update(
                updated_at=now, state=target.value, version=values["version"] + 1,
                attempt_count=values["attempt_count"] + (1 if entering else 0),
                error_type=error_type, error_code=error_code, diagnostic=diagnostic,
            )
            if resolution is not None:
                updated["resolution"] = resolution.value
            if entering:
                updated["owner_session"] = self.session_id
                updated["resolution"] = None
            if worker_id is not _KEEP:
                updated["worker_id"] = worker_id
            if last_outcome is not _KEEP:
                updated["last_outcome"] = last_outcome
            if recovery_json is not _KEEP:
                updated["recovery_json"] = recovery_json
            event = {
                "idempotency_key": key, "transition_no": updated["version"], "from_state": current.value,
                "to_state": target.value, "cause": cause.value, "at": now, "actor": actor, "reason": reason,
                "session_id": self.session_id, "worker_id": updated["worker_id"], "attempt": updated["attempt_count"],
                "error_type": error_type, "error_code": error_code, "diagnostic": diagnostic,
                "resolution": resolution.value if resolution else None,
            }
            updated["history_head"] = self._append_event(conn, event, values["history_head"])
            self._update(conn, values, updated)
        logger.debug("Ledger %s: %s to %s", _key_ref(key), current.value, target.value)
        return self._request(updated)

    # ------------------------------------------------------------ public API

    def register(self, key: str, operation: str, statements: Any, *, return_rows: bool = False) -> Registration:
        """Registers the request, or finds it. NEW and DUPLICATE both grant nothing: only
        begin_execution claims the right to run. A different payload raises IdempotencyConflictError
        after durably recording the attempt, and leaves the existing request as it was."""
        key = _checked_key(key)
        presented = fingerprint_payload(operation, statements, return_rows=return_rows)
        summary = presented._summary_json()
        now = _now()
        conflict: Optional[IdempotencyConflictError] = None
        with self._write() as conn:
            values = self._load(conn, key)
            if values is None:
                event = {
                    "idempotency_key": key, "transition_no": 1, "from_state": None, "to_state": LedgerState.REGISTERED.value,
                    "cause": "REGISTERED", "at": now, "actor": None, "reason": None, "session_id": self.session_id,
                    "worker_id": None, "attempt": 0, "error_type": None, "error_code": None, "diagnostic": None, "resolution": None,
                }
                values = {
                    "idempotency_key": key, "fingerprint_version": presented.version, "fingerprint": presented.value,
                    "operation": operation, "payload_summary": summary, "created_at": now, "updated_at": now,
                    "state": LedgerState.REGISTERED.value, "attempt_count": 0, "worker_id": None, "owner_session": None,
                    "last_outcome": None, "error_type": None, "error_code": None, "diagnostic": None, "recovery_json": None,
                    "resolution": None, "version": 1, "conflict_count": 0, "conflict_head": _CHAIN_START,
                }
                values["history_head"] = self._append_event(conn, event, _CHAIN_START)
                columns = ", ".join(_REQUEST_COLUMNS)
                marks = ", ".join("?" * len(_REQUEST_COLUMNS))
                conn.execute(f"INSERT INTO ledger_requests ({columns}, row_sha256) VALUES ({marks}, ?)",
                             (*(values[name] for name in _REQUEST_COLUMNS), self._row_hash(values)))
                registration = Registration(CheckStatus.NEW, self._request(values))
            else:
                self._require_comparable(values)
                if values["fingerprint"] == presented.value:
                    registration = Registration(CheckStatus.DUPLICATE, self._request(values))
                else:
                    conflict = self._record_conflict(conn, values, presented, summary, now)
        if conflict is not None:
            logger.warning("Ledger %s: conflicting registration %d", _key_ref(key), conflict.conflict_no)
            raise conflict
        logger.debug("Ledger %s: %s", _key_ref(key), registration.status.value)
        return registration

    def _record_conflict(self, conn: sqlite3.Connection, values: dict, presented: PayloadFingerprint, summary: str,
                         now: str) -> IdempotencyConflictError:
        key = values["idempotency_key"]
        number = values["conflict_count"] + 1
        entry = {"idempotency_key": key, "conflict_no": number, "at": now, "session_id": self.session_id,
                 "presented_version": presented.version, "presented_fingerprint": presented.value, "presented_summary": summary}
        digest = self._conflict_hash(entry, values["conflict_head"])
        columns = ", ".join(_CONFLICT_COLUMNS)
        marks = ", ".join("?" * len(_CONFLICT_COLUMNS))
        conn.execute(f"INSERT INTO ledger_conflicts ({columns}, prev_sha256, entry_sha256) VALUES ({marks}, ?, ?)",
                     (*(entry[name] for name in _CONFLICT_COLUMNS), values["conflict_head"], digest))
        updated = dict(values)
        updated.update(conflict_count=number, conflict_head=digest)
        self._update(conn, values, updated)
        differences = _differences(values["payload_summary"], summary)
        return IdempotencyConflictError(
            f"{_key_ref(key)} is registered with a different payload; differs in: {', '.join(differences)}",
            existing_fingerprint=values["fingerprint"], presented_fingerprint=presented.value,
            differences=differences, conflict_no=number)

    def check(self, key: str, operation: str, statements: Any, *, return_rows: bool = False) -> LedgerCheck:
        """What register() would find, recording nothing. Can be out of date by the time it returns,
        so it decides nothing: register() and begin_execution() do."""
        key = _checked_key(key)
        presented = fingerprint_payload(operation, statements, return_rows=return_rows)
        with self._read() as conn:
            values = self._load(conn, key)
        if values is None:
            return LedgerCheck(CheckStatus.NEW, None, presented.value)
        self._require_comparable(values)
        if values["fingerprint"] == presented.value:
            return LedgerCheck(CheckStatus.DUPLICATE, self._request(values), presented.value)
        return LedgerCheck(CheckStatus.CONFLICT, self._request(values), presented.value,
                           _differences(values["payload_summary"], presented._summary_json()))

    def get_request(self, key: str) -> LedgerRequest:
        key = _checked_key(key)
        with self._read() as conn:
            return self._request(self._require(conn, key))

    def list_requests(self, state: Optional[LedgerState] = None) -> tuple[LedgerRequest, ...]:
        """In registration order. Raises LedgerCorruptionError on a damaged request; check_integrity()
        reports which ones."""
        with self._read() as conn:
            if state is None:
                rows = conn.execute(f"{_SELECT_REQUEST} ORDER BY seq").fetchall()
            else:
                rows = conn.execute(f"{_SELECT_REQUEST} WHERE state = ? ORDER BY seq", (LedgerState(state).value,)).fetchall()
            return tuple(self._request(self._verified(row)) for row in rows)

    def get_history(self, key: str) -> tuple[LedgerEvent, ...]:
        key = _checked_key(key)
        with self._read() as conn:
            self._require(conn, key)
            rows = conn.execute(
                f"SELECT {', '.join(_HISTORY_COLUMNS)} FROM ledger_history WHERE idempotency_key = ? ORDER BY transition_no",
                (key,)).fetchall()
        events = []
        for row in rows:
            values = dict(zip(_HISTORY_COLUMNS, row))
            values["key_ref"] = _key_ref(values.pop("idempotency_key"))
            values["from_state"] = LedgerState(values["from_state"]) if values["from_state"] else None
            values["to_state"] = LedgerState(values["to_state"])
            values["resolution"] = Resolution(values["resolution"]) if values["resolution"] else None
            events.append(LedgerEvent(**values))
        return tuple(events)

    def get_conflicts(self, key: str) -> tuple[LedgerConflict, ...]:
        key = _checked_key(key)
        with self._read() as conn:
            values = self._require(conn, key)
            rows = conn.execute(
                f"SELECT {', '.join(_CONFLICT_COLUMNS)} FROM ledger_conflicts WHERE idempotency_key = ? ORDER BY conflict_no",
                (key,)).fetchall()
        conflicts = []
        for row in rows:
            entry = dict(zip(_CONFLICT_COLUMNS, row))
            conflicts.append(LedgerConflict(_key_ref(key), entry["conflict_no"], entry["at"], entry["session_id"],
                                            entry["presented_version"], entry["presented_fingerprint"],
                                            _differences(values["payload_summary"], entry["presented_summary"])))
        return tuple(conflicts)

    def begin_execution(self, key: str, fingerprint: str, *, worker_id: Optional[str] = None) -> LedgerRequest:
        """Claims the right to run: REGISTERED or RETRY_AUTHORIZED to IN_PROGRESS. Atomic, so one caller
        wins and the rest get InvalidTransitionError. Call before sending the query, then report the
        outcome with the returned attempt_count. Runs nothing."""
        key = _checked_key(key)
        if type(fingerprint) is not str:
            raise TypeError("fingerprint must be the string register() returned")
        return self._transition(key, LedgerState.IN_PROGRESS, _Cause.EXECUTE, fingerprint=fingerprint,
                                worker_id=_clean_detail(worker_id, "worker_id", _MAX_ACTOR_LENGTH))

    def record_completed(self, key: str, *, attempt: int) -> LedgerRequest:
        """Only when completion was confirmed, for example WriteOutcome.COMMITTED."""
        return self._transition(_checked_key(key), LedgerState.COMPLETED, _Cause.OUTCOME, attempt=_checked_attempt(attempt),
                                last_outcome="COMPLETED")

    def record_not_completed(self, key: str, *, attempt: int, error_type: Optional[str] = None, error_code: Optional[str] = None,
                             diagnostic: Optional[str] = None) -> LedgerRequest:
        """Only when non-completion is established. A caller's timeout doesn't establish it: the
        worker may still be running, so leave the request IN_PROGRESS."""
        return self._transition(_checked_key(key), LedgerState.NOT_COMPLETED, _Cause.OUTCOME, attempt=_checked_attempt(attempt),
                                last_outcome="NOT_COMPLETED", error_type=_clean_detail(error_type, "error_type"),
                                error_code=_clean_detail(error_code, "error_code"), diagnostic=_clean_detail(diagnostic, "diagnostic"))

    def record_uncertain(self, key: str, *, attempt: int, error_type: Optional[str] = None, error_code: Optional[str] = None,
                         diagnostic: Optional[str] = None) -> LedgerRequest:
        return self._transition(_checked_key(key), LedgerState.UNCERTAIN, _Cause.OUTCOME, attempt=_checked_attempt(attempt),
                                last_outcome="UNCERTAIN", error_type=_clean_detail(error_type, "error_type"),
                                error_code=_clean_detail(error_code, "error_code"), diagnostic=_clean_detail(diagnostic, "diagnostic"))

    def authorize_retry(self, key: str, *, authorized_by: str, reason: str) -> LedgerRequest:
        """NOT_COMPLETED to RETRY_AUTHORIZED: allows one more attempt under the same key, which
        begin_execution uses up. Runs nothing."""
        return self._transition(_checked_key(key), LedgerState.RETRY_AUTHORIZED, _Cause.RETRY,
                                actor=_required_text(authorized_by, "authorized_by"),
                                reason=_required_text(reason, "reason", _MAX_DETAIL_LENGTH))

    def resolve_uncertain(self, key: str, *, resolution: Resolution, resolved_by: str, reason: str) -> LedgerRequest:
        """Records what someone concluded, and who, when and why. History keeps the uncertainty.
        FOUND_NOT_COMPLETED doesn't make the request claimable: it still needs authorize_retry()."""
        resolution = Resolution(resolution)
        return self._transition(_checked_key(key), _RESOLUTION_TARGET[resolution], _Cause.RESOLUTION, resolution=resolution,
                                actor=_required_text(resolved_by, "resolved_by"),
                                reason=_required_text(reason, "reason", _MAX_DETAIL_LENGTH))

    def recover_unfinished(self) -> RecoveryReport:
        """Call after a restart. IN_PROGRESS requests whose owner has gone become UNCERTAIN. Safe to
        repeat, and each conversion is its own transaction. Runs nothing and changes nothing else.
        This instance's own requests are left alone: a worker of its own may still be running."""
        self._usable(write=True)
        with self._read() as conn:
            rows = conn.execute(f"{_SELECT_REQUEST} ORDER BY seq").fetchall()
        recovered, running, registered, not_completed, authorized, uncertain, corrupt = [], [], [], [], [], [], []
        in_progress = []
        for row in rows:
            try:
                values = self._verified(row)
            except LedgerCorruptionError:
                corrupt.append(_key_ref(row[0]))
                continue
            state = LedgerState(values["state"])
            key = values["idempotency_key"]
            if state is LedgerState.IN_PROGRESS:
                in_progress.append(values)
            elif state is LedgerState.REGISTERED:
                registered.append(key)
            elif state is LedgerState.NOT_COMPLETED:
                not_completed.append(key)
            elif state is LedgerState.RETRY_AUTHORIZED:
                authorized.append(key)
            elif state is LedgerState.UNCERTAIN:
                uncertain.append(key)

        liveness: dict[str, bool] = {}
        held: dict[str, Optional[int]] = {}
        try:
            for values in in_progress:
                key, owner = values["idempotency_key"], values["owner_session"]
                if owner == self.session_id:
                    running.append(key)
                    continue
                if owner not in liveness:
                    liveness[owner], held[owner] = self._session_is_alive(owner)
                if liveness[owner]:
                    running.append(key)
                    continue
                metadata = {"recovered_at": _now(), "recovered_by_session": self.session_id, "previous_owner_session": owner,
                            "attempt": values["attempt_count"], "reason": "the owning process ended with no recorded outcome"}
                try:
                    self._transition(key, LedgerState.UNCERTAIN, _Cause.RECOVERY, actor="recover_unfinished",
                                     reason=metadata["reason"], last_outcome="UNCERTAIN", error_type="UnresolvedAfterRestart",
                                     diagnostic=metadata["reason"], recovery_json=json.dumps(metadata, sort_keys=True))
                except InvalidTransitionError:
                    # Another recovery got there first. Its state is reported, not assumed.
                    if self.get_request(key).state is LedgerState.UNCERTAIN:
                        uncertain.append(key)
                    continue
                recovered.append(key)
                uncertain.append(key)
        finally:
            for owner, fd in held.items():
                if fd is not None:
                    try:
                        os.unlink(os.path.join(self._path + ".sessions", owner + ".lock"))
                    except OSError:
                        pass
                    os.close(fd)
        if recovered or corrupt:
            logger.warning("Ledger recovery: %d requests became UNCERTAIN, %d damaged", len(recovered), len(corrupt))
        return RecoveryReport(tuple(recovered), tuple(running), tuple(registered), tuple(not_completed), tuple(authorized),
                              tuple(uncertain), tuple(corrupt))

    def check_integrity(self) -> IntegrityReport:
        """Checks the file, schema, every row checksum and every chain. Reports what it finds rather
        than raising, and changes nothing."""
        problems: list[str] = []
        corrupt: list[str] = []
        version = None
        with self._lock:
            conn = self._usable(write=False)
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version != SCHEMA_VERSION:
                    problems.append(f"schema version is {version}, expected {SCHEMA_VERSION}")
                if conn.execute("PRAGMA application_id").fetchone()[0] != _APPLICATION_ID:
                    problems.append("unexpected application id")
                result = [row[0] for row in conn.execute("PRAGMA integrity_check")]
                if result != ["ok"]:
                    problems.append(f"SQLite integrity check reported {len(result)} problem(s)")
                present = {row[0] for row in conn.execute("SELECT name FROM sqlite_schema")}
                if _REQUIRED_OBJECTS - present:
                    problems.append(f"schema objects missing: {sorted(_REQUIRED_OBJECTS - present)}")
                rows = conn.execute(f"{_SELECT_REQUEST} ORDER BY seq").fetchall()
                events: dict[str, list[tuple]] = {}
                for row in conn.execute(f"SELECT {', '.join(_HISTORY_COLUMNS)}, prev_sha256, entry_sha256 FROM ledger_history "
                                        "ORDER BY idempotency_key, transition_no"):
                    events.setdefault(row[0], []).append(row)
                conflicts: dict[str, list[tuple]] = {}
                for row in conn.execute(f"SELECT {', '.join(_CONFLICT_COLUMNS)}, prev_sha256, entry_sha256 FROM ledger_conflicts "
                                        "ORDER BY idempotency_key, conflict_no"):
                    conflicts.setdefault(row[0], []).append(row)
                highest = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'ledger_requests'").fetchone()
                if highest is not None and highest[0] != len(rows):
                    problems.append(f"{highest[0]} requests were registered but {len(rows)} are present")
            except sqlite3.Error as error:
                problems.append(f"SQLite failed while checking: {getattr(error, 'sqlite_errorname', type(error).__name__)}")
                return IntegrityReport(False, version, 0, (), tuple(problems))
        known = set()
        for row in rows:
            known.add(row[0])
            if not self._request_is_sound(row, events.get(row[0], []), conflicts.get(row[0], [])):
                corrupt.append(_key_ref(row[0]))
        orphans = (set(events) | set(conflicts)) - known
        if orphans:
            problems.append(f"audit entries exist for {len(orphans)} request(s) that are not present")
        return IntegrityReport(not problems and not corrupt, version, len(rows), tuple(corrupt), tuple(problems))

    def _request_is_sound(self, row: tuple, events: list[tuple], conflicts: list[tuple]) -> bool:
        try:
            values = self._verified(row)
            LedgerState(values["state"])
            json.loads(values["payload_summary"])
        except (LedgerCorruptionError, ValueError, TypeError):
            return False
        previous = _CHAIN_START
        event = None
        for number, event_row in enumerate(events, start=1):
            event = dict(zip(_HISTORY_COLUMNS, event_row))
            prev_sha, entry_sha = event_row[len(_HISTORY_COLUMNS):]
            if event["transition_no"] != number or prev_sha != previous or entry_sha != self._event_hash(event, previous):
                return False
            previous = entry_sha
        if event is None or previous != values["history_head"] or len(events) != values["version"] or event["to_state"] != values["state"]:
            return False
        previous = _CHAIN_START
        for number, conflict_row in enumerate(conflicts, start=1):
            entry = dict(zip(_CONFLICT_COLUMNS, conflict_row))
            prev_sha, entry_sha = conflict_row[len(_CONFLICT_COLUMNS):]
            if entry["conflict_no"] != number or prev_sha != previous or entry_sha != self._conflict_hash(entry, previous):
                return False
            previous = entry_sha
        return previous == values["conflict_head"] and len(conflicts) == values["conflict_count"]

    def close(self) -> None:
        """Safe to repeat. Requests still IN_PROGRESS stay that way, and the next recovery by another
        instance makes them UNCERTAIN, because closing doesn't stop a worker's database call."""
        with self._lock:
            if os.getpid() != self._pid:
                self._conn, self._session_fd = None, None
                return
            conn, self._conn = self._conn, None
            if conn is None:
                return
            try:
                if self.session_id is not None:
                    unfinished = conn.execute(
                        "SELECT count(*) FROM ledger_requests WHERE state = 'IN_PROGRESS' AND owner_session = ?",
                        (self.session_id,)).fetchone()[0]
                    if unfinished:
                        logger.warning("Ledger closed with %d of its requests IN_PROGRESS", unfinished)
                conn.close()
            except sqlite3.Error as error:
                raise _translate(error) from error
            finally:
                self._release_session()

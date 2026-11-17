"""Durable journal of intended Neo4j writes and what is known about each one's outcome.

Stores the original statements and parameters, and tracks each record through a validated
state machine, so that a write whose outcome is unknown stays visible and is never replayed
automatically. This module only records: nothing here runs a query, and it imports neither
neo4j_client nor config.

No production code uses it yet, so production writes are NOT protected by it. That needs a
later integration in which the journal transition happens before the write executes.

States
    PENDING        intent recorded, execution not started
    IN_FLIGHT      execution started, outcome not yet recorded
    COMMITTED      the helper confirmed a commit. Never replayable
    NOT_COMMITTED  established as not committed. Retried only through authorize_retry()
    AMBIGUOUS      unknown. Never replayable. Leaves only through resolve_ambiguous()
    RESOLVED       a person or tool recorded a conclusion about an ambiguous record. Final:
                   to write again, create a new intent

What the journal can't do
    It can't commit atomically with Neo4j. If Neo4j commits and the process dies before
    mark_committed(), the record is IN_FLIGHT, and recovery makes it AMBIGUOUS. So AMBIGUOUS
    can describe a write that committed, one that didn't, or one that never started.

    It provides no idempotency. Records have unique ids, but nothing relates them to data in
    Neo4j, so running a record twice can apply it twice.

Durability
    SQLite in WAL mode with synchronous=FULL, so a transition has been fsynced before the method
    returns. Verified here: that those settings are in force, and that committed transitions
    survive SIGKILL. Not verifiable here: power loss, which depends on the disk and filesystem
    honouring fsync. Use a local POSIX filesystem. WAL and flock aren't reliable on network
    filesystems or WSL's /mnt/c, so don't put the journal there. If a transition raises, treat the
    original write as not recorded: don't acknowledge or discard it.

Liveness
    Each writable instance holds a flock on its own file in <path>.sessions/ for as long as it
    is open, and the kernel releases it when the process dies. recover_unfinished() converts
    IN_FLIGHT records only when their owner's lock is free, so it never mislabels a live
    worker. A forked child inherits the lock and keeps its owner looking alive until it exits.

Integrity
    SHA-256 over each record's intent and row, plus a hash chain over its history, detects
    accidental damage. There is no secret key, so it doesn't stop deliberate tampering. A damaged
    or foreign file is refused and left as it is: nothing here recreates or truncates a journal.

Sensitive data
    Parameters are stored in plaintext, because they must be recoverable exactly. New files
    are created 0600. Errors and logs from this module carry ids and type names, never parameter
    values, and JournalRecord's repr leaves out the intent.

Growth
    Nothing here deletes records, so the file grows until a later phase defines retention.
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
INTENT_FORMAT_VERSION = 1
_APPLICATION_ID = 0x43594A4C  # "CYJL"
_SQLITE_MAGIC = b"SQLite format 3\x00"
_MAX_NESTING = 64
_MAX_DETAIL_LENGTH = 500
_MAX_ACTOR_LENGTH = 200


class JournalState(str, Enum):
    PENDING = "PENDING"
    IN_FLIGHT = "IN_FLIGHT"
    COMMITTED = "COMMITTED"
    NOT_COMMITTED = "NOT_COMMITTED"
    AMBIGUOUS = "AMBIGUOUS"
    RESOLVED = "RESOLVED"


class Resolution(str, Enum):
    """What was concluded about an ambiguous record. None of these make it replayable."""
    FOUND_COMMITTED = "FOUND_COMMITTED"
    FOUND_NOT_COMMITTED = "FOUND_NOT_COMMITTED"
    ABANDONED = "ABANDONED"


class _Cause(str, Enum):
    EXECUTE = "EXECUTE"
    OUTCOME = "OUTCOME"
    RETRY = "RETRY"
    RECOVERY = "RECOVERY"
    RESOLUTION = "RESOLUTION"


_S = JournalState
_TRANSITIONS: dict[tuple[JournalState, JournalState], frozenset[_Cause]] = {
    (_S.PENDING, _S.IN_FLIGHT): frozenset({_Cause.EXECUTE}),
    (_S.PENDING, _S.NOT_COMMITTED): frozenset({_Cause.OUTCOME}),
    (_S.IN_FLIGHT, _S.COMMITTED): frozenset({_Cause.OUTCOME}),
    (_S.IN_FLIGHT, _S.NOT_COMMITTED): frozenset({_Cause.OUTCOME}),
    (_S.IN_FLIGHT, _S.AMBIGUOUS): frozenset({_Cause.OUTCOME, _Cause.RECOVERY}),
    (_S.NOT_COMMITTED, _S.IN_FLIGHT): frozenset({_Cause.RETRY}),
    (_S.AMBIGUOUS, _S.RESOLVED): frozenset({_Cause.RESOLUTION}),
}


class JournalError(Exception):
    """Base class. Messages carry ids and type names, never parameter values."""


class JournalWriteError(JournalError):
    """A write to the journal failed, so the change wasn't recorded. Don't acknowledge or
    discard the original write because of it."""


class JournalBusyError(JournalWriteError):
    """Another connection held the database for longer than busy_timeout."""


class JournalCorruptionError(JournalError):
    """The journal or a record in it is damaged. Nothing was changed."""


class JournalSchemaError(JournalError):
    """The file isn't a journal this version can use. Nothing was changed."""


class JournalClosedError(JournalError):
    """The journal has been closed, or this process isn't the one that opened it."""


class JournalReadOnlyError(JournalError):
    """A change was attempted through a read-only instance."""


class RecordNotFoundError(JournalError):
    pass


class InvalidTransitionError(JournalError):
    """The state machine refused the change. The record wasn't modified."""


class UnsupportedParameterError(JournalError, TypeError):
    """A parameter can't be stored and restored exactly, so no record was created."""


# ---------------------------------------------------------------- intent serialization

_FIXED_PYTZ_TYPE = type(pytz.FixedOffset(60))


def _encode_timezone(tz: Any, where: str) -> Any:
    if tz is None:
        return None
    if type(tz) is dt.timezone:
        offset = tz.utcoffset(None)
        # Only an explicit name: timezone(offset) generates its own, and differs from a named one.
        name = tz.tzname(None)
        return {"fixed": [offset.days, offset.seconds, offset.microseconds],
                "name": None if name == dt.timezone(offset).tzname(None) else name}
    if type(tz) is zoneinfo.ZoneInfo and tz.key:
        return {"zoneinfo": tz.key}
    if type(tz) is _FIXED_PYTZ_TYPE:
        return {"pytz_fixed": int(tz.utcoffset(None).total_seconds() // 60)}
    if isinstance(tz, pytz.tzinfo.BaseTzInfo) and tz.zone:
        return {"pytz": tz.zone}
    raise UnsupportedParameterError(f"{where}: timezone type {type(tz).__name__} is not supported")


def _apply_timezone(naive: Any, spec: Any, is_dst: bool) -> Any:
    if spec is None:
        return naive
    if "fixed" in spec:
        days, seconds, micros = spec["fixed"]
        offset = dt.timedelta(days=days, seconds=seconds, microseconds=micros)
        tz = dt.timezone(offset) if spec["name"] is None else dt.timezone(offset, spec["name"])
        return naive.replace(tzinfo=tz)
    if "zoneinfo" in spec:
        return naive.replace(tzinfo=zoneinfo.ZoneInfo(spec["zoneinfo"]))
    if "pytz_fixed" in spec:
        return naive.replace(tzinfo=pytz.FixedOffset(spec["pytz_fixed"]))
    return pytz.timezone(spec["pytz"]).localize(naive, is_dst=is_dst)


def _pytz_is_dst(value: Any) -> bool:
    """Tells pytz which side of a clock change a local time is on. Nothing else needs this."""
    return isinstance(value.tzinfo, pytz.tzinfo.BaseTzInfo) and bool(value.dst())


def _encode_value(value: Any, where: str, depth: int = 0, ancestors: frozenset = frozenset()) -> Any:
    """Every JSON object in the result is a tagged value with exactly one key, so a caller's map
    can't be mistaken for a tag. Exact types only: a subclass could carry behaviour or state
    that wouldn't come back, so it's refused rather than stored as its base type."""
    kind = type(value)
    if value is None or kind is bool or kind is str:
        return value
    if kind is int:
        if not -(2 ** 63) <= value < 2 ** 63:
            raise UnsupportedParameterError(f"{where}: integer is outside the 64-bit range")
        return value
    if kind is float:
        return value if math.isfinite(value) else {"$float": repr(value)}
    if kind in (list, tuple, dict):
        if depth >= _MAX_NESTING:
            raise UnsupportedParameterError(f"{where}: nested more than {_MAX_NESTING} levels")
        if id(value) in ancestors:
            raise UnsupportedParameterError(f"{where}: contains a reference cycle")
        inner = ancestors | {id(value)}
        if kind is dict:
            for key in value:
                if type(key) is not str:
                    raise UnsupportedParameterError(f"{where}: map key type {type(key).__name__} is not supported")
            return {"$map": {k: _encode_value(v, where, depth + 1, inner) for k, v in value.items()}}
        items = [_encode_value(v, where, depth + 1, inner) for v in value]
        return items if kind is list else {"$tuple": items}
    if kind is bytes:
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if kind is bytearray:
        return {"$bytearray": base64.b64encode(bytes(value)).decode("ascii")}
    if kind is dt.datetime:
        fields = [value.year, value.month, value.day, value.hour, value.minute, value.second, value.microsecond]
        return {"$datetime": {"f": fields, "fold": value.fold, "dst": _pytz_is_dst(value),
                              "tz": _encode_timezone(value.tzinfo, where)}}
    if kind is dt.date:
        return {"$date": [value.year, value.month, value.day]}
    if kind is dt.time:
        fields = [value.hour, value.minute, value.second, value.microsecond]
        return {"$time": {"f": fields, "fold": value.fold, "tz": _encode_timezone(value.tzinfo, where)}}
    if kind is dt.timedelta:
        return {"$timedelta": [value.days, value.seconds, value.microseconds]}
    if kind is neo4j_time.DateTime:
        hour, minute, second, nano = value.hour, value.minute, value.second, value.nanosecond
        return {"$neo4j_datetime": {"f": [value.year, value.month, value.day, hour, minute, second, nano],
                                    "dst": _pytz_is_dst(value),
                                    "tz": _encode_timezone(value.tzinfo, where)}}
    if kind is neo4j_time.Date:
        return {"$neo4j_date": [value.year, value.month, value.day]}
    if kind is neo4j_time.Time:
        return {"$neo4j_time": {"f": [value.hour, value.minute, value.second, value.nanosecond],
                                "tz": _encode_timezone(value.tzinfo, where)}}
    if kind is neo4j_time.Duration:
        return {"$neo4j_duration": [value.months, value.days, value.seconds, value.nanoseconds]}
    raise UnsupportedParameterError(f"{where}: type {kind.__module__}.{kind.__qualname__} is not supported")


def _decode_value(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is list:
        return [_decode_value(v) for v in value]
    if type(value) is not dict or len(value) != 1:
        raise ValueError("not a tagged value")
    (tag, body), = value.items()
    if tag == "$map":
        return {k: _decode_value(v) for k, v in body.items()}
    if tag == "$tuple":
        return tuple(_decode_value(v) for v in body)
    if tag == "$float":
        return float(body)
    if tag == "$bytes":
        return base64.b64decode(body, validate=True)
    if tag == "$bytearray":
        return bytearray(base64.b64decode(body, validate=True))
    if tag == "$datetime":
        naive = dt.datetime(*body["f"], fold=body["fold"])
        return _apply_timezone(naive, body["tz"], body["dst"])
    if tag == "$date":
        return dt.date(*body)
    if tag == "$time":
        return _apply_timezone(dt.time(*body["f"], fold=body["fold"]), body["tz"], False)
    if tag == "$timedelta":
        return dt.timedelta(days=body[0], seconds=body[1], microseconds=body[2])
    if tag == "$neo4j_datetime":
        return _apply_timezone(neo4j_time.DateTime(*body["f"]), body["tz"], body["dst"])
    if tag == "$neo4j_date":
        return neo4j_time.Date(*body)
    if tag == "$neo4j_time":
        return _apply_timezone(neo4j_time.Time(*body["f"]), body["tz"], False)
    if tag == "$neo4j_duration":
        return neo4j_time.Duration(months=body[0], days=body[1], seconds=body[2], nanoseconds=body[3])
    raise ValueError(f"unknown tag {tag}")


def _identical(a: Any, b: Any) -> bool:
    """Stricter than ==: same types throughout, same floats including NaN and -0.0, same map
    order, and the same timezone rather than only the same instant."""
    if type(a) is not type(b):
        return False
    if type(a) is float:
        return repr(a) == repr(b)
    if type(a) is dict:
        return list(a) == list(b) and all(_identical(a[k], b[k]) for k in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b))
    if hasattr(a, "tzinfo"):
        return (a == b and str(a) == str(b) and repr(a.tzinfo) == repr(b.tzinfo)
                and getattr(a, "fold", 0) == getattr(b, "fold", 0))
    return a == b


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _serialize_intent(statements: Any, return_rows: bool) -> tuple[str, list]:
    """Returns (json, the statements as given). Checks the shape the write helper accepts, then
    proves the round trip, so a value that can't come back exactly creates no record."""
    if isinstance(statements, (str, bytes)):
        raise TypeError("statements must be a sequence of (query, parameters) pairs")
    try:
        materialized = list(statements)
    except TypeError:
        raise TypeError("statements must be a sequence of (query, parameters) pairs") from None
    if not materialized:
        raise ValueError("a write needs at least one statement")
    encoded = []
    for index, statement in enumerate(materialized):
        if type(statement) not in (tuple, list) or len(statement) != 2:
            raise TypeError(f"statement {index} must be a (query, parameters) pair")
        query, parameters = statement
        if type(query) is not str or not query.strip():
            raise ValueError(f"statement {index}: the query must be a non-empty string")
        if parameters is None:
            encoded.append([query, None])
            continue
        if type(parameters) is not dict:
            raise TypeError(f"statement {index}: parameters must be a dict or None")
        for key in parameters:
            if type(key) is not str:
                raise UnsupportedParameterError(f"statement {index}: parameter name type {type(key).__name__} is not supported")
        body = {key: _encode_value(value, f"statement {index}, parameter {key!r}") for key, value in parameters.items()}
        encoded.append([query, {"$map": body}])
    text = _to_json({"statements": encoded, "return_rows": return_rows})
    try:
        restored, restored_rows = _deserialize_intent(text)
        wanted = tuple((q, p) for q, p in materialized)
        exact = _identical(restored, wanted) and restored_rows is return_rows
    except Exception as error:
        raise UnsupportedParameterError(f"a parameter value could not be restored: {type(error).__name__}") from None
    if not exact:
        raise UnsupportedParameterError("a parameter value could not be restored exactly")
    return text, materialized


def _deserialize_intent(text: str) -> tuple[tuple, bool]:
    document = json.loads(text)
    return (tuple((query, _decode_value(params)) for query, params in document["statements"]),
            document["return_rows"])


def _sha256(*parts: Any) -> str:
    return hashlib.sha256(_to_json(list(parts)).encode("ascii")).hexdigest()


# ---------------------------------------------------------------- returned values

@dataclass(frozen=True)
class JournalRecord:
    """A copy of a record at the time it was read. Changing it can't change the journal, and
    each access to `statements` decodes a new copy of the intent."""
    record_id: str
    format_version: int
    operation: str
    created_at: str
    updated_at: str
    state: JournalState
    attempt_count: int
    worker_id: Optional[str]
    owner_session: Optional[str]
    last_outcome: Optional[str]
    error_type: Optional[str]
    error_code: Optional[str]
    diagnostic: Optional[str]
    resolution: Optional[Resolution]
    intent_sha256: str
    version: int
    _intent_json: str = field(repr=False)
    _recovery_json: Optional[str] = field(repr=False)

    @property
    def statements(self) -> tuple:
        """The original ordered ((query, parameters), ...), exactly as recorded."""
        return _deserialize_intent(self._intent_json)[0]

    @property
    def return_rows(self) -> bool:
        return _deserialize_intent(self._intent_json)[1]

    @property
    def recovery_metadata(self) -> Optional[dict]:
        return None if self._recovery_json is None else json.loads(self._recovery_json)

    @property
    def retry_eligible(self) -> bool:
        return self.state is JournalState.NOT_COMMITTED


@dataclass(frozen=True)
class JournalEvent:
    """One entry in a record's append-only history."""
    record_id: str
    transition_no: int
    from_state: Optional[JournalState]
    to_state: JournalState
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
class RecoveryReport:
    """recovered: IN_FLIGHT records whose owner is gone, now AMBIGUOUS. still_running: IN_FLIGHT
    records whose owner is alive, left alone. corrupt: damaged records, left alone. pending,
    not_committed and ambiguous are listed for the caller and were not touched. Nothing is executed."""
    recovered: tuple[str, ...]
    still_running: tuple[str, ...]
    pending: tuple[str, ...]
    not_committed: tuple[str, ...]
    ambiguous: tuple[str, ...]
    corrupt: tuple[str, ...]


@dataclass(frozen=True)
class IntegrityReport:
    ok: bool
    schema_version: Optional[int]
    records_checked: int
    corrupt_record_ids: tuple[str, ...]
    problems: tuple[str, ...]


# ---------------------------------------------------------------- storage

_SCHEMA = (
    """CREATE TABLE journal_records (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id TEXT NOT NULL UNIQUE,
        format_version INTEGER NOT NULL,
        operation TEXT NOT NULL,
        intent_json TEXT NOT NULL,
        intent_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN
            ('PENDING','IN_FLIGHT','COMMITTED','NOT_COMMITTED','AMBIGUOUS','RESOLVED')),
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
        row_sha256 TEXT NOT NULL
    )""",
    "CREATE INDEX journal_records_state ON journal_records(state, seq)",
    """CREATE TABLE journal_history (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id TEXT NOT NULL,
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
        UNIQUE (record_id, transition_no)
    )""",
    """CREATE TRIGGER journal_history_no_update BEFORE UPDATE ON journal_history
        BEGIN SELECT RAISE(ABORT, 'journal history is append-only'); END""",
    """CREATE TRIGGER journal_history_no_delete BEFORE DELETE ON journal_history
        BEGIN SELECT RAISE(ABORT, 'journal history is append-only'); END""",
    """CREATE TRIGGER journal_records_no_delete BEFORE DELETE ON journal_records
        BEGIN SELECT RAISE(ABORT, 'journal records are never deleted'); END""",
    """CREATE TRIGGER journal_records_intent_fixed
        BEFORE UPDATE OF record_id, format_version, operation, intent_json, intent_sha256, created_at
        ON journal_records
        BEGIN SELECT RAISE(ABORT, 'a recorded intent cannot change'); END""",
)
_REQUIRED_OBJECTS = frozenset({
    "journal_records", "journal_history", "journal_records_state", "journal_history_no_update",
    "journal_history_no_delete", "journal_records_no_delete", "journal_records_intent_fixed",
})
_RECORD_COLUMNS = (
    "record_id", "format_version", "operation", "intent_json", "intent_sha256", "created_at", "updated_at",
    "state", "attempt_count", "worker_id", "owner_session", "last_outcome", "error_type", "error_code",
    "diagnostic", "recovery_json", "resolution", "version", "history_head",
)
_HISTORY_COLUMNS = (
    "record_id", "transition_no", "from_state", "to_state", "cause", "at", "actor", "reason", "session_id",
    "worker_id", "attempt", "error_type", "error_code", "diagnostic", "resolution",
)
_NO_HISTORY = "0" * 64
_SELECT_RECORD = f"SELECT {', '.join(_RECORD_COLUMNS)}, row_sha256 FROM journal_records"
_KEEP = object()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _clean_detail(value: Any, name: str, limit: int = _MAX_DETAIL_LENGTH) -> Optional[str]:
    """Takes safe summaries such as WriteResult.diagnostic. An exception is refused, because its
    message can quote parameter values."""
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{name} must be a string or None, not {type(value).__name__}")
    return value[:limit]


def _required_text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value[:_MAX_ACTOR_LENGTH] if name != "reason" else value[:_MAX_DETAIL_LENGTH]


def _translate(error: sqlite3.Error) -> JournalError:
    if "could not decode" in str(error).lower():
        return JournalCorruptionError("a value in the journal file is damaged")
    code = (getattr(error, "sqlite_errorcode", 0) or 0) & 0xFF
    if code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        return JournalBusyError("the journal is locked by another connection")
    if code in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB):
        return JournalCorruptionError("SQLite reports the journal file is damaged")
    name = getattr(error, "sqlite_errorname", None) or type(error).__name__
    return JournalWriteError(f"journal storage failed: {name}")


def _inspect_existing_file(path: str) -> bool:
    """True if a journal exists at path, False if it's absent or empty. Reads only, before SQLite
    opens anything: opening a foreign database in WAL mode would change it."""
    try:
        size = os.stat(path).st_size
    except FileNotFoundError:
        return False
    if size == 0:
        return False
    with open(path, "rb") as handle:
        header = handle.read(100)
    if len(header) < 100 or not header.startswith(_SQLITE_MAGIC):
        raise JournalCorruptionError("not a journal: the file has no valid SQLite header. It was left unchanged")
    application_id = int.from_bytes(header[68:72], "big")
    if application_id != _APPLICATION_ID:
        raise JournalSchemaError("not a journal: this SQLite database belongs to something else. It was left unchanged")
    return True


class NeoWriteJournal:
    """Thread-safe: one connection, serialized by a lock. Several instances or processes can
    share a file, with SQLite's locks ordering their transactions. Each change runs in one
    BEGIN IMMEDIATE transaction, so a record and its history always change together.

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
            raise JournalError("the journal's directory does not exist")
        exists = _inspect_existing_file(self._path)
        if read_only and not exists:
            raise JournalError("no journal exists at this path to open read-only")
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

    def __enter__(self) -> "NeoWriteJournal":
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
        """Header and schema commit together, while still in rollback-journal mode, so a crash
        here leaves an empty file rather than one this class would later refuse."""
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
            raise JournalSchemaError("not a journal: unexpected application id. It was left unchanged")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise JournalSchemaError(f"journal schema version {version} is not supported; this code uses {SCHEMA_VERSION}")
        present = {row[0] for row in conn.execute("SELECT name FROM sqlite_schema")}
        missing = _REQUIRED_OBJECTS - present
        if missing:
            raise JournalCorruptionError(f"the journal is missing {len(missing)} of its schema objects: {sorted(missing)}")
        check = conn.execute("PRAGMA quick_check(1)").fetchone()[0]
        if check != "ok":
            raise JournalCorruptionError("SQLite's quick check reports the journal file is damaged")

    def _configure_durability(self) -> None:
        conn = self._conn
        mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        conn.execute("PRAGMA synchronous = FULL")
        synchronous = conn.execute("PRAGMA synchronous").fetchone()[0]
        if mode.lower() != "wal" or synchronous != 2:
            raise JournalWriteError(f"could not establish durable settings (journal_mode={mode}, synchronous={synchronous})")

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
            raise JournalClosedError("the journal is closed, or was opened by another process")
        if write and self._read_only:
            raise JournalReadOnlyError("this journal instance is read-only")
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
                logger.error("Journal rollback failed: %s", type(error).__name__)

    @staticmethod
    def _row_hash(values: dict) -> str:
        return _sha256(*(values[name] for name in _RECORD_COLUMNS))

    @staticmethod
    def _event_hash(values: dict, previous: str) -> str:
        return _sha256(previous, *(values[name] for name in _HISTORY_COLUMNS))

    def _verified(self, row: tuple) -> dict:
        """The row as a dict, once its checksums match."""
        values = dict(zip(_RECORD_COLUMNS, row))
        if row[len(_RECORD_COLUMNS)] != self._row_hash(values) or values["intent_sha256"] != _sha256(values["intent_json"]):
            raise JournalCorruptionError(f"journal record {values['record_id']} failed its integrity check")
        return values

    @staticmethod
    def _record(values: dict) -> JournalRecord:
        return JournalRecord(
            record_id=values["record_id"], format_version=values["format_version"], operation=values["operation"],
            created_at=values["created_at"], updated_at=values["updated_at"], state=JournalState(values["state"]),
            attempt_count=values["attempt_count"], worker_id=values["worker_id"], owner_session=values["owner_session"],
            last_outcome=values["last_outcome"], error_type=values["error_type"], error_code=values["error_code"],
            diagnostic=values["diagnostic"], resolution=Resolution(values["resolution"]) if values["resolution"] else None,
            intent_sha256=values["intent_sha256"], version=values["version"], _intent_json=values["intent_json"],
            _recovery_json=values["recovery_json"],
        )

    def _load(self, conn: sqlite3.Connection, record_id: str) -> dict:
        row = conn.execute(f"{_SELECT_RECORD} WHERE record_id = ?", (record_id,)).fetchone()
        if row is None:
            raise RecordNotFoundError(f"no journal record {record_id}")
        return self._verified(row)

    def _append_event(self, conn: sqlite3.Connection, event: dict, previous: str) -> str:
        entry = self._event_hash(event, previous)
        columns = ", ".join(_HISTORY_COLUMNS)
        marks = ", ".join("?" * len(_HISTORY_COLUMNS))
        conn.execute(f"INSERT INTO journal_history ({columns}, prev_sha256, entry_sha256) VALUES ({marks}, ?, ?)",
                     (*(event[name] for name in _HISTORY_COLUMNS), previous, entry))
        return entry

    def _transition(self, record_id: str, target: JournalState, cause: _Cause, *, actor: Optional[str] = None,
                    reason: Optional[str] = None, worker_id: Any = _KEEP, last_outcome: Any = _KEEP,
                    error_type: Optional[str] = None, error_code: Optional[str] = None,
                    diagnostic: Optional[str] = None, resolution: Optional[Resolution] = None,
                    recovery_json: Any = _KEEP) -> JournalRecord:
        """The only code that changes an existing record. The check, the update and the history
        entry are one transaction, so a refused or failed change leaves everything as it was."""
        now = _now()
        with self._write() as conn:
            values = self._load(conn, record_id)
            current = JournalState(values["state"])
            if cause not in _TRANSITIONS.get((current, target), frozenset()):
                raise InvalidTransitionError(f"record {record_id}: {current.value} to {target.value} is not allowed ({cause.value})")
            entering = target is JournalState.IN_FLIGHT
            updated = dict(values)
            updated.update(
                updated_at=now, state=target.value, version=values["version"] + 1,
                attempt_count=values["attempt_count"] + (1 if entering else 0),
                error_type=error_type, error_code=error_code, diagnostic=diagnostic,
                resolution=resolution.value if resolution else values["resolution"],
            )
            if entering:
                updated["owner_session"] = self.session_id
            if worker_id is not _KEEP:
                updated["worker_id"] = worker_id
            if last_outcome is not _KEEP:
                updated["last_outcome"] = last_outcome
            if recovery_json is not _KEEP:
                updated["recovery_json"] = recovery_json
            event = {
                "record_id": record_id, "transition_no": updated["version"], "from_state": current.value,
                "to_state": target.value, "cause": cause.value, "at": now, "actor": actor, "reason": reason,
                "session_id": self.session_id, "worker_id": updated["worker_id"], "attempt": updated["attempt_count"],
                "error_type": error_type, "error_code": error_code, "diagnostic": diagnostic,
                "resolution": resolution.value if resolution else None,
            }
            updated["history_head"] = self._append_event(conn, event, values["history_head"])
            mutable = [name for name in _RECORD_COLUMNS if updated[name] != values[name]]
            assignments = ", ".join(f"{name} = ?" for name in mutable)
            conn.execute(f"UPDATE journal_records SET {assignments}, row_sha256 = ? WHERE record_id = ?",
                         (*(updated[name] for name in mutable), self._row_hash(updated), record_id))
        logger.debug("Journal record %s: %s to %s", record_id, current.value, target.value)
        return self._record(updated)

    # ------------------------------------------------------------ public API

    def create_intent(self, operation: str, statements: Any, *, return_rows: bool = False) -> JournalRecord:
        """Records the write as PENDING. When this returns, the intent has been committed and
        fsynced. Raises and creates nothing if it can't be stored or restored exactly."""
        if type(operation) is not str or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        if type(return_rows) is not bool:
            raise TypeError("return_rows must be a bool")
        text, _ = _serialize_intent(statements, return_rows)
        record_id = uuid.uuid4().hex
        now = _now()
        event = {
            "record_id": record_id, "transition_no": 1, "from_state": None, "to_state": JournalState.PENDING.value,
            "cause": "CREATED", "at": now, "actor": None, "reason": None, "session_id": self.session_id,
            "worker_id": None, "attempt": 0, "error_type": None, "error_code": None, "diagnostic": None, "resolution": None,
        }
        values = {
            "record_id": record_id, "format_version": INTENT_FORMAT_VERSION, "operation": operation, "intent_json": text,
            "intent_sha256": _sha256(text), "created_at": now, "updated_at": now, "state": JournalState.PENDING.value,
            "attempt_count": 0, "worker_id": None, "owner_session": None, "last_outcome": None, "error_type": None,
            "error_code": None, "diagnostic": None, "recovery_json": None, "resolution": None, "version": 1,
        }
        with self._write() as conn:
            values["history_head"] = self._append_event(conn, event, _NO_HISTORY)
            columns = ", ".join(_RECORD_COLUMNS)
            marks = ", ".join("?" * len(_RECORD_COLUMNS))
            conn.execute(f"INSERT INTO journal_records ({columns}, row_sha256) VALUES ({marks}, ?)",
                         (*(values[name] for name in _RECORD_COLUMNS), self._row_hash(values)))
        return self._record(values)

    def get_record(self, record_id: str) -> JournalRecord:
        with self._read() as conn:
            return self._record(self._load(conn, record_id))

    def list_records(self, state: Optional[JournalState] = None) -> tuple[JournalRecord, ...]:
        """In creation order. Raises JournalCorruptionError on a damaged record; check_integrity()
        reports which ones."""
        with self._read() as conn:
            if state is None:
                rows = conn.execute(f"{_SELECT_RECORD} ORDER BY seq").fetchall()
            else:
                rows = conn.execute(f"{_SELECT_RECORD} WHERE state = ? ORDER BY seq", (JournalState(state).value,)).fetchall()
            return tuple(self._record(self._verified(row)) for row in rows)

    def get_history(self, record_id: str) -> tuple[JournalEvent, ...]:
        with self._read() as conn:
            self._load(conn, record_id)
            rows = conn.execute(
                f"SELECT {', '.join(_HISTORY_COLUMNS)} FROM journal_history WHERE record_id = ? ORDER BY transition_no",
                (record_id,)).fetchall()
        events = []
        for row in rows:
            values = dict(zip(_HISTORY_COLUMNS, row))
            values["from_state"] = JournalState(values["from_state"]) if values["from_state"] else None
            values["to_state"] = JournalState(values["to_state"])
            values["resolution"] = Resolution(values["resolution"]) if values["resolution"] else None
            events.append(JournalEvent(**values))
        return tuple(events)

    def mark_in_flight(self, record_id: str, *, worker_id: Optional[str] = None) -> JournalRecord:
        """PENDING to IN_FLIGHT. Call before sending the query. A NOT_COMMITTED record needs
        authorize_retry() instead."""
        return self._transition(record_id, JournalState.IN_FLIGHT, _Cause.EXECUTE,
                                worker_id=_clean_detail(worker_id, "worker_id", _MAX_ACTOR_LENGTH))

    def mark_committed(self, record_id: str) -> JournalRecord:
        """Only when the commit was confirmed, for example WriteOutcome.COMMITTED. Sending a
        query or COMMIT isn't confirmation."""
        return self._transition(record_id, JournalState.COMMITTED, _Cause.OUTCOME, last_outcome="COMMITTED")

    def mark_not_committed(self, record_id: str, *, error_type: Optional[str] = None, error_code: Optional[str] = None,
                           diagnostic: Optional[str] = None) -> JournalRecord:
        """Only when non-commit is established. A caller's timeout doesn't establish it: the
        worker may still be running, so leave the record IN_FLIGHT."""
        return self._transition(record_id, JournalState.NOT_COMMITTED, _Cause.OUTCOME, last_outcome="NOT_COMMITTED",
                                error_type=_clean_detail(error_type, "error_type"),
                                error_code=_clean_detail(error_code, "error_code"),
                                diagnostic=_clean_detail(diagnostic, "diagnostic"))

    def mark_ambiguous(self, record_id: str, *, error_type: Optional[str] = None, error_code: Optional[str] = None,
                       diagnostic: Optional[str] = None) -> JournalRecord:
        return self._transition(record_id, JournalState.AMBIGUOUS, _Cause.OUTCOME, last_outcome="AMBIGUOUS",
                                error_type=_clean_detail(error_type, "error_type"),
                                error_code=_clean_detail(error_code, "error_code"),
                                diagnostic=_clean_detail(diagnostic, "diagnostic"))

    def authorize_retry(self, record_id: str, *, authorized_by: str, reason: str, worker_id: Optional[str] = None) -> JournalRecord:
        """NOT_COMMITTED to IN_FLIGHT, the only way to run a record again. It runs nothing: the
        caller executes record.statements and reports the outcome as usual."""
        return self._transition(record_id, JournalState.IN_FLIGHT, _Cause.RETRY,
                                actor=_required_text(authorized_by, "authorized_by"), reason=_required_text(reason, "reason"),
                                worker_id=_clean_detail(worker_id, "worker_id", _MAX_ACTOR_LENGTH))

    def resolve_ambiguous(self, record_id: str, *, resolution: Resolution, resolved_by: str, reason: str) -> JournalRecord:
        """AMBIGUOUS to RESOLVED, recording who, when and why. History keeps the ambiguity. The
        record never becomes replayable: to write again, create a new intent."""
        return self._transition(record_id, JournalState.RESOLVED, _Cause.RESOLUTION, resolution=Resolution(resolution),
                                actor=_required_text(resolved_by, "resolved_by"), reason=_required_text(reason, "reason"),
                                error_type=None, error_code=None, diagnostic=None)

    def recover_unfinished(self) -> RecoveryReport:
        """Call after a restart. IN_FLIGHT records whose owner has gone become AMBIGUOUS. Safe to
        repeat, and each conversion is its own transaction, so a crash here loses nothing. Runs
        no query, and changes nothing else. This instance's own records are left alone, because a
        worker of its own may still be running."""
        self._usable(write=True)
        with self._read() as conn:
            rows = conn.execute(f"{_SELECT_RECORD} ORDER BY seq").fetchall()
        recovered, running, pending, not_committed, ambiguous, corrupt = [], [], [], [], [], []
        in_flight = []
        for row in rows:
            try:
                values = self._verified(row)
            except JournalCorruptionError:
                corrupt.append(row[0])
                continue
            state = JournalState(values["state"])
            if state is JournalState.IN_FLIGHT:
                in_flight.append(values)
            elif state is JournalState.PENDING:
                pending.append(values["record_id"])
            elif state is JournalState.NOT_COMMITTED:
                not_committed.append(values["record_id"])
            elif state is JournalState.AMBIGUOUS:
                ambiguous.append(values["record_id"])

        liveness: dict[str, bool] = {}
        held: dict[str, Optional[int]] = {}
        try:
            for values in in_flight:
                owner = values["owner_session"]
                if owner == self.session_id:
                    running.append(values["record_id"])
                    continue
                if owner not in liveness:
                    liveness[owner], held[owner] = self._session_is_alive(owner)
                if liveness[owner]:
                    running.append(values["record_id"])
                    continue
                metadata = {"recovered_at": _now(), "recovered_by_session": self.session_id, "previous_owner_session": owner,
                            "attempt": values["attempt_count"], "reason": "the owning process ended with no recorded outcome"}
                try:
                    self._transition(values["record_id"], JournalState.AMBIGUOUS, _Cause.RECOVERY,
                                     actor="recover_unfinished", reason=metadata["reason"], last_outcome="AMBIGUOUS",
                                     error_type="UnresolvedAfterRestart", diagnostic=metadata["reason"],
                                     recovery_json=json.dumps(metadata, sort_keys=True))
                except InvalidTransitionError:
                    # Another recovery got there first. Its state is reported, not assumed.
                    if self.get_record(values["record_id"]).state is JournalState.AMBIGUOUS:
                        ambiguous.append(values["record_id"])
                    continue
                recovered.append(values["record_id"])
                ambiguous.append(values["record_id"])
        finally:
            for owner, fd in held.items():
                if fd is not None:
                    try:
                        os.unlink(os.path.join(self._path + ".sessions", owner + ".lock"))
                    except OSError:
                        pass
                    os.close(fd)
        if recovered or corrupt:
            logger.warning("Journal recovery: %d records became AMBIGUOUS, %d damaged", len(recovered), len(corrupt))
        return RecoveryReport(tuple(recovered), tuple(running), tuple(pending), tuple(not_committed), tuple(ambiguous), tuple(corrupt))

    def check_integrity(self) -> IntegrityReport:
        """Checks the file, schema, every record's checksums and every history chain. Reports what
        it finds rather than raising, and changes nothing."""
        problems: list[str] = []
        corrupt: list[str] = []
        version = None
        checked = 0
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
                rows = conn.execute(f"{_SELECT_RECORD} ORDER BY seq").fetchall()
                events: dict[str, list[tuple]] = {}
                select = f"SELECT {', '.join(_HISTORY_COLUMNS)}, prev_sha256, entry_sha256 FROM journal_history ORDER BY record_id, transition_no"
                for row in conn.execute(select):
                    events.setdefault(row[0], []).append(row)
                highest = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'journal_records'").fetchone()
                if highest is not None and highest[0] != len(rows):
                    problems.append(f"{highest[0]} records were created but {len(rows)} are present")
            except sqlite3.Error as error:
                problems.append(f"SQLite failed while checking: {getattr(error, 'sqlite_errorname', type(error).__name__)}")
                return IntegrityReport(False, version, checked, tuple(corrupt), tuple(problems))
        known = set()
        for row in rows:
            checked += 1
            record_id = row[0]
            known.add(record_id)
            if not self._record_is_sound(row, events.get(record_id, [])):
                corrupt.append(record_id)
        orphans = set(events) - known
        if orphans:
            problems.append(f"history exists for {len(orphans)} record(s) that are not present")
        return IntegrityReport(not problems and not corrupt, version, checked, tuple(corrupt), tuple(problems))

    def _record_is_sound(self, row: tuple, events: list[tuple]) -> bool:
        try:
            values = self._verified(row)
            JournalState(values["state"])
            _deserialize_intent(values["intent_json"])
        except (JournalCorruptionError, ValueError, KeyError, TypeError):
            return False
        previous = _NO_HISTORY
        for number, event_row in enumerate(events, start=1):
            event = dict(zip(_HISTORY_COLUMNS, event_row))
            prev_sha, entry_sha = event_row[len(_HISTORY_COLUMNS):]
            if event["transition_no"] != number or prev_sha != previous or entry_sha != self._event_hash(event, previous):
                return False
            previous = entry_sha
        return bool(events) and previous == values["history_head"] and len(events) == values["version"] \
            and event["to_state"] == values["state"]

    def close(self) -> None:
        """Safe to repeat. Records still IN_FLIGHT stay that way, and the next recovery by another
        instance makes them AMBIGUOUS, because closing doesn't stop a worker's database call."""
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
                        "SELECT count(*) FROM journal_records WHERE state = 'IN_FLIGHT' AND owner_session = ?",
                        (self.session_id,)).fetchone()[0]
                    if unfinished:
                        logger.warning("Journal closed with %d of its records IN_FLIGHT", unfinished)
                conn.close()
            except sqlite3.Error as error:
                raise _translate(error) from error
            finally:
                self._release_session()

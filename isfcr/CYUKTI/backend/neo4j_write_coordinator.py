"""Coordinates one logical Neo4j write across the idempotency ledger, the write journal and a writer.

Phase 2.7, WP-C1. Nothing in production calls this yet, so production writes are NOT protected by
it. It knows no real operation and imports no Neo4j code: the writer is passed in, and tests pass a
fake. A fake says nothing about how Neo4j behaves.

It stores nothing. Every state it answers from is read from the two stores:

    Does this request exist, with this payload? May it run?    the ledger
    What exactly is sent, and what did the attempt report?     the journal
    Who concludes an uncertain outcome, or authorizes a retry? a named person

Order, so that a failure always leaves a request less runnable, never more:

    in   preflight, ledger register, journal intent, ledger claim, journal IN_FLIGHT, then the writer
    out  journal outcome, then ledger outcome, each naming the ledger's attempt

The writer is called only after both the claim and IN_FLIGHT have committed, and it's given the
statements the journal stored. Nothing here retries a write: an ambiguous outcome waits for a person,
and a confirmed non-commit waits for authorize_retry(). A bounded retry of a local store write isn't
a retry of the Neo4j write, and it never re-runs anything.

What a result means
    CoordinatedResult.applied is true only once the outcome is durable in both stores.
    CoordinatedResult.must_hold means a dependency is down or an outcome isn't durable, so the caller
    mustn't acknowledge or discard what it holds. Submitting it again later can't repeat a write.
    There is no fallback to an unprotected write on any path, including a refused identity.

What this can't do
    SQLite and Neo4j can't commit together, so this isn't exactly-once. If the process dies between a
    commit and the journal's record of it, recovery makes the request uncertain, and a person decides.

    Without markers (WP-M1 and WP-M2), a retry a person wrongly authorized can apply twice, and nothing
    can show whether an uncertain write committed. ConstraintValidationFailed is reported as the
    non-commit it is, and never read as "already applied".

The caller owns the stores: this opens and closes neither. Logs and errors carry key references and
type names, never keys or parameter values.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Protocol, Sequence
from urllib.parse import unquote

from listener.utils import SOURCE_KIND, extract_source_identity, request_key
from neo4j_idempotency_ledger import (
    IdempotencyConflictError,
    IdempotencyLedger,
    InvalidTransitionError as LedgerTransitionError,
    LedgerClosedError,
    LedgerError,
    LedgerRequest,
    LedgerState,
    LedgerWriteError,
    RequestNotFoundError,
    Resolution as LedgerResolution,
    UnsupportedPayloadError,
    _key_ref,
    fingerprint_payload,
)
from neo4j_write_journal import (
    DuplicateCorrelationError,
    InvalidTransitionError as JournalTransitionError,
    JournalClosedError,
    JournalError,
    JournalRecord,
    JournalState,
    JournalWriteError,
    NeoWriteJournal,
    PreflightCode,
    Resolution as JournalResolution,
)

logger = logging.getLogger(__name__)

_OUTCOME_COMMITTED = "COMMITTED"
_OUTCOME_NOT_COMMITTED = "NOT_COMMITTED"
_OUTCOME_AMBIGUOUS = "AMBIGUOUS"


class CoordinatorError(Exception):
    """Base class. Messages carry key references and type names, never keys or values."""


class InvalidRequest(CoordinatorError, ValueError):
    """Malformed arguments, or an operation that isn't registered. Nothing was stored or run."""


class PayloadNotJournalable(CoordinatorError):
    """A value can't be fingerprinted or stored exactly. Nothing was registered or run."""


class JournalUnavailable(CoordinatorError):
    """An operator-side call couldn't reach the journal. Nothing was concluded."""


class LedgerUnavailable(CoordinatorError):
    """An operator-side call couldn't reach the ledger. Nothing was concluded."""


class CoordinationInconsistency(CoordinatorError):
    """The stores show something that shouldn't arise, or one is damaged. Nothing ran. A person looks."""


class NotAuthorized(CoordinatorError):
    """A retry or authorization the request's state doesn't allow. Nothing changed."""


class PayloadMismatch(CoordinationInconsistency):
    """The journal's stored payload doesn't match the ledger's fingerprint. Treated as damage."""


class ResolutionConflict(CoordinatorError):
    """A resolution contradicting an earlier one, or what the records show. Nothing changed."""


class Disposition(str, Enum):
    """What happened to a request in this call."""
    EXECUTED_COMMITTED = "EXECUTED_COMMITTED"              # ran here, confirmed, durable in both stores
    EXECUTED_NOT_COMMITTED = "EXECUTED_NOT_COMMITTED"      # ran here, established as not committed
    NOT_EXECUTED_CONFIRMED = "NOT_EXECUTED_CONFIRMED"      # the writer established it sent nothing
    EXECUTED_UNCERTAIN = "EXECUTED_UNCERTAIN"              # ran here, outcome unknown. A person decides
    COMMITTED_UNRECORDED = "COMMITTED_UNRECORDED"          # commit reported, but not durable in both stores
    ALREADY_COMPLETED = "ALREADY_COMPLETED"                # finished earlier. Nothing ran
    IN_PROGRESS_ELSEWHERE = "IN_PROGRESS_ELSEWHERE"        # someone holds the claim. Nothing ran
    AWAITING_AUTHORIZATION = "AWAITING_AUTHORIZATION"      # not completed. Nothing ran
    BLOCKED_UNCERTAIN = "BLOCKED_UNCERTAIN"                # uncertain. Nothing ran
    BLOCKED_ABANDONED = "BLOCKED_ABANDONED"                # a person gave it up. Nothing ran
    CONFLICT = "CONFLICT"                                  # same key, different payload. Nothing ran
    REFUSED_IDENTITY = "REFUSED_IDENTITY"                  # no usable identity. Nothing stored or run
    NOT_DISPATCHED_UNAVAILABLE = "NOT_DISPATCHED_UNAVAILABLE"  # a dependency is down. The writer wasn't called


_APPLIED = frozenset({Disposition.EXECUTED_COMMITTED, Disposition.ALREADY_COMPLETED})
_RAN_A_WRITE = frozenset({Disposition.EXECUTED_COMMITTED, Disposition.EXECUTED_NOT_COMMITTED,
                          Disposition.EXECUTED_UNCERTAIN, Disposition.COMMITTED_UNRECORDED})
_ANSWER_BY_STATE = {
    LedgerState.IN_PROGRESS: Disposition.IN_PROGRESS_ELSEWHERE,
    LedgerState.COMPLETED: Disposition.ALREADY_COMPLETED,
    LedgerState.NOT_COMPLETED: Disposition.AWAITING_AUTHORIZATION,
    # A re-delivery doesn't spend an authorization: that's for retry(), which sends the stored payload.
    LedgerState.RETRY_AUTHORIZED: Disposition.AWAITING_AUTHORIZATION,
    LedgerState.UNCERTAIN: Disposition.BLOCKED_UNCERTAIN,
    LedgerState.ABANDONED: Disposition.BLOCKED_ABANDONED,
}


class IdentityRefusal(str, Enum):
    """Why a request has no usable identity (D14). None of these is ever replaced by a made-up key."""
    MISSING = "MISSING"            # None or empty
    NOT_A_STRING = "NOT_A_STRING"
    MALFORMED = "MALFORMED"        # not of the form wazuh:<manager>/<id>
    INVALID = "INVALID"            # that form, but the listener's identity rule doesn't accept it


def check_source_key(source_key: Any) -> Optional[IdentityRefusal]:
    """None if this is a source key the listener's rule would produce. The rule stays in one place:
    the parts go back through extract_source_identity(), and the result must match exactly."""
    if source_key is None or source_key == "":
        return IdentityRefusal.MISSING
    if type(source_key) is not str:
        return IdentityRefusal.NOT_A_STRING
    prefix = f"{SOURCE_KIND}:"
    if not source_key.startswith(prefix):
        return IdentityRefusal.MALFORMED
    manager_part, separator, source_id = source_key[len(prefix):].rpartition("/")
    if not separator or not manager_part or not source_id:
        return IdentityRefusal.MALFORMED
    try:
        manager = unquote(manager_part, errors="strict")
    except UnicodeDecodeError:
        return IdentityRefusal.MALFORMED
    identity = extract_source_identity({"id": source_id, "manager": {"name": manager}})
    if not identity.valid or identity.source_key != source_key:
        return IdentityRefusal.INVALID
    return None


@dataclass(frozen=True)
class RegisteredOperation:
    """One write this coordinator may run. The statement is fixed here, so a caller supplies only values."""
    name: str
    query: str = field(repr=False)
    parameter_names: frozenset

    def __post_init__(self) -> None:
        request_key(f"{SOURCE_KIND}:registry/0.0", self.name)  # the listener's rule for operation names
        if type(self.query) is not str or not self.query.strip():
            raise ValueError("a registered operation needs a query")
        names = frozenset(self.parameter_names)
        if not all(type(name) is str for name in names):
            raise TypeError("parameter names must be strings")
        object.__setattr__(self, "parameter_names", names)

    def statements(self, parameters: Any) -> list:
        if type(parameters) is not dict:
            raise InvalidRequest(f"{self.name}: parameters must be a dict")
        if frozenset(parameters) != self.parameter_names:
            raise InvalidRequest(f"{self.name}: parameters don't match the registered names")
        return [(self.query, parameters)]


class OperationRegistry:
    """The operations one coordinator may run. Fixed once built, with no global default."""

    def __init__(self, operations: Iterable[RegisteredOperation]) -> None:
        self._operations: dict[str, RegisteredOperation] = {}
        for operation in operations:
            if type(operation) is not RegisteredOperation:
                raise TypeError("a registry holds RegisteredOperation values")
            if operation.name in self._operations:
                raise ValueError(f"operation {operation.name} is registered twice")
            self._operations[operation.name] = operation

    def get(self, name: Any) -> RegisteredOperation:
        operation = self._operations.get(name) if type(name) is str else None
        if operation is None:
            raise InvalidRequest("the operation isn't registered")
        return operation


@dataclass(frozen=True)
class WriteRequest:
    """source_key is alert["source_identity"]["source_key"], which may be None. operation is a registered name."""
    source_key: Optional[str] = field(repr=False)
    operation: str
    parameters: dict = field(repr=False)


class Writer(Protocol):
    """The whole boundary. neo4j_client.write_statements() has this shape. It returns an object with
    `outcome` (`.value` of COMMITTED, NOT_COMMITTED or AMBIGUOUS), `attempts`, `exception_type`,
    `exception_code` and `diagnostic`. Any other outcome, or an exception, counts as AMBIGUOUS."""

    def __call__(self, operation: str, statements: Sequence, *, limits: Any) -> Any: ...


class HealthGate(Protocol):
    """Whether writes may be dispatched now. Only a boolean True allows it, and an exception doesn't."""

    def allow_dispatch(self) -> bool: ...


class MarkerReader(Protocol):
    """The marker reader boundary (e.g. read_applied_write)."""

    def __call__(self, request_key: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class CoordinatedResult:
    """What the caller learns. writer_invoked says whether this call reached the writer. finalized says
    whether what this call established is durable in both stores: if not, recovery finishes or
    downgrades it. journal_record_id is safe to log, and nothing here holds a key or a value."""
    key_ref: Optional[str]
    disposition: Disposition
    ledger_attempt: Optional[int] = None
    journal_record_id: Optional[str] = None
    writer_invoked: bool = False
    writer_attempts: Optional[int] = None
    finalized: bool = True
    diagnostic: str = ""
    refusal: Optional[IdentityRefusal] = None
    unavailable: Optional[str] = None
    differences: tuple = ()

    @property
    def applied(self) -> bool:
        return self.disposition in _APPLIED and self.finalized

    @property
    def must_hold(self) -> bool:
        return self.disposition is Disposition.NOT_DISPATCHED_UNAVAILABLE or not self.finalized

    @property
    def executed(self) -> bool:
        return self.disposition in _RAN_A_WRITE


class Decision(str, Enum):
    """A person's conclusion about an uncertain request."""
    FOUND_COMMITTED = "FOUND_COMMITTED"
    FOUND_NOT_COMMITTED = "FOUND_NOT_COMMITTED"
    ABANDONED = "ABANDONED"


_DECISIONS = {
    Decision.FOUND_COMMITTED: (JournalResolution.FOUND_COMMITTED, LedgerResolution.FOUND_COMPLETED, LedgerState.COMPLETED),
    Decision.FOUND_NOT_COMMITTED: (JournalResolution.FOUND_NOT_COMMITTED, LedgerResolution.FOUND_NOT_COMPLETED, LedgerState.NOT_COMPLETED),
    Decision.ABANDONED: (JournalResolution.ABANDONED, LedgerResolution.ABANDONED, LedgerState.ABANDONED),
}


class CombinedState(str, Enum):
    """Computed from the two stores each time, and never stored."""
    NOT_FOUND = "NOT_FOUND"
    REGISTERED_NO_INTENT = "REGISTERED_NO_INTENT"      # can't run: no payload is stored
    REGISTERED = "REGISTERED"
    CLAIMED = "CLAIMED"
    DISPATCHED = "DISPATCHED"
    OUTCOME_NOT_MIRRORED = "OUTCOME_NOT_MIRRORED"      # the journal has it, and recovery finishes the ledger
    COMPLETED = "COMPLETED"
    AWAITING_AUTHORIZATION = "AWAITING_AUTHORIZATION"
    AUTHORIZED = "AUTHORIZED"
    UNCERTAIN = "UNCERTAIN"
    CLAIMED_NOT_DISPATCHED = "CLAIMED_NOT_DISPATCHED"  # uncertain to the ledger, but the writer can't have run
    RESOLUTION_INCOMPLETE = "RESOLUTION_INCOMPLETE"    # journal resolved, ledger not yet. Repeat the resolve
    ABANDONED = "ABANDONED"
    INCONSISTENT = "INCONSISTENT"


def combined_state(request: Optional[LedgerRequest], chain: Sequence[JournalRecord]) -> CombinedState:
    """The pairing of the ledger's state with the latest journal epoch. Anything not listed as legal is
    INCONSISTENT, which is reported and never resolved towards running."""
    C, J = CombinedState, JournalState
    if request is None:
        return C.INCONSISTENT if chain else C.NOT_FOUND
    latest = chain[-1] if chain else None
    state = request.state
    if latest is None:
        return C.REGISTERED_NO_INTENT if state is LedgerState.REGISTERED else C.INCONSISTENT
    journal = latest.state
    tag = latest.ledger_attempt
    attempt = request.attempt_count
    if tag is not None and tag > attempt:
        return C.INCONSISTENT
    never_started = journal is J.PENDING and tag is None  # a PENDING record has never held an attempt
    earlier_non_commit = journal is J.NOT_COMMITTED and tag is not None and tag < attempt  # a claim that never reached it
    at_this_attempt = tag == attempt
    if state is LedgerState.REGISTERED:
        return C.REGISTERED if never_started and attempt == 0 else C.INCONSISTENT
    if state is LedgerState.IN_PROGRESS:
        if never_started or earlier_non_commit:
            return C.CLAIMED
        if not at_this_attempt:
            return C.INCONSISTENT
        if journal is J.IN_FLIGHT:
            return C.DISPATCHED
        return C.OUTCOME_NOT_MIRRORED if journal in (J.COMMITTED, J.NOT_COMMITTED, J.AMBIGUOUS) else C.INCONSISTENT
    if state is LedgerState.COMPLETED:
        found_committed = journal is J.RESOLVED and latest.resolution is JournalResolution.FOUND_COMMITTED
        return C.COMPLETED if at_this_attempt and (journal is J.COMMITTED or found_committed) else C.INCONSISTENT
    if state in (LedgerState.NOT_COMPLETED, LedgerState.RETRY_AUTHORIZED):
        found_not_committed = journal is J.RESOLVED and latest.resolution is JournalResolution.FOUND_NOT_COMMITTED
        if never_started or earlier_non_commit or (at_this_attempt and (journal is J.NOT_COMMITTED or found_not_committed)):
            return C.AWAITING_AUTHORIZATION if state is LedgerState.NOT_COMPLETED else C.AUTHORIZED
        return C.INCONSISTENT
    if state is LedgerState.UNCERTAIN:
        if never_started or earlier_non_commit:
            return C.CLAIMED_NOT_DISPATCHED
        if at_this_attempt and journal is J.AMBIGUOUS:
            return C.UNCERTAIN
        return C.RESOLUTION_INCOMPLETE if at_this_attempt and journal is J.RESOLVED else C.INCONSISTENT
    # ABANDONED
    if never_started or earlier_non_commit:
        return C.ABANDONED
    return C.ABANDONED if at_this_attempt and journal is J.RESOLVED and latest.resolution is JournalResolution.ABANDONED else C.INCONSISTENT


@dataclass(frozen=True)
class JournalEpochView:
    epoch: Optional[int]
    record_id: str
    state: JournalState
    ledger_attempt: Optional[int]
    resolution: Optional[JournalResolution]


@dataclass(frozen=True)
class RequestView:
    key_ref: str
    combined: CombinedState
    ledger_state: Optional[LedgerState]
    ledger_attempt: int
    journal: tuple

    @property
    def consistent(self) -> bool:
        return self.combined is not CombinedState.INCONSISTENT


class RecoveryCategory(str, Enum):
    RECONCILED = "RECONCILED"                          # ledger finished from a confirmed journal outcome
    NEWLY_UNCERTAIN = "NEWLY_UNCERTAIN"
    UNCERTAIN = "UNCERTAIN"
    CLAIMED_NOT_DISPATCHED = "CLAIMED_NOT_DISPATCHED"
    RESOLUTION_INCOMPLETE = "RESOLUTION_INCOMPLETE"
    REGISTERED_NO_INTENT = "REGISTERED_NO_INTENT"
    PENDING = "PENDING"
    AWAITING_AUTHORIZATION = "AWAITING_AUTHORIZATION"
    RETRY_AUTHORIZED = "RETRY_AUTHORIZED"
    STILL_RUNNING = "STILL_RUNNING"
    CORRUPT = "CORRUPT"
    INCONSISTENT = "INCONSISTENT"


@dataclass(frozen=True)
class RecoveryResult:
    """Keys per category, for an operator. The repr shows only counts. Nothing was executed, and nothing
    but the ledger's and journal's own recovery transitions was changed."""
    keys: Mapping = field(repr=False)
    integrity_ok: bool = True
    problems: tuple = ()

    def of(self, category: RecoveryCategory) -> tuple:
        return tuple(self.keys.get(category, ()))

    def counts(self) -> dict:
        return {category.value: len(keys) for category, keys in self.keys.items() if keys}

    def __repr__(self) -> str:
        return f"RecoveryResult(counts={self.counts()}, integrity_ok={self.integrity_ok}, problems={len(self.problems)})"


_COMBINED_TO_CATEGORY = {
    CombinedState.REGISTERED_NO_INTENT: RecoveryCategory.REGISTERED_NO_INTENT,
    CombinedState.REGISTERED: RecoveryCategory.PENDING,
    CombinedState.CLAIMED: RecoveryCategory.STILL_RUNNING,
    CombinedState.DISPATCHED: RecoveryCategory.STILL_RUNNING,
    CombinedState.OUTCOME_NOT_MIRRORED: RecoveryCategory.STILL_RUNNING,
    CombinedState.AWAITING_AUTHORIZATION: RecoveryCategory.AWAITING_AUTHORIZATION,
    CombinedState.AUTHORIZED: RecoveryCategory.RETRY_AUTHORIZED,
    CombinedState.UNCERTAIN: RecoveryCategory.UNCERTAIN,
    CombinedState.CLAIMED_NOT_DISPATCHED: RecoveryCategory.CLAIMED_NOT_DISPATCHED,
    CombinedState.RESOLUTION_INCOMPLETE: RecoveryCategory.RESOLUTION_INCOMPLETE,
    CombinedState.INCONSISTENT: RecoveryCategory.INCONSISTENT,
}


@dataclass(frozen=True)
class _Reported:
    """What one writer call established, in terms the stores accept."""
    outcome: str
    attempts: Optional[int] = None
    error_type: Optional[str] = None
    error_code: Optional[str] = None
    diagnostic: Optional[str] = None


def _text_or_none(value: Any) -> Optional[str]:
    return value if type(value) is str else None


def _read_report(result: Any) -> _Reported:
    """Reads only the fields that are safe to keep, never an exception object. An outcome this code
    doesn't recognise isn't evidence of anything, so it's unknown."""
    outcome = getattr(getattr(result, "outcome", None), "value", None)
    attempts = getattr(result, "attempts", None)
    attempts = attempts if type(attempts) is int else None
    if outcome not in (_OUTCOME_COMMITTED, _OUTCOME_NOT_COMMITTED, _OUTCOME_AMBIGUOUS):
        return _Reported(_OUTCOME_AMBIGUOUS, attempts, error_type="UnrecognisedWriterOutcome",
                         diagnostic="the writer reported something other than a known outcome")
    return _Reported(outcome, attempts, _text_or_none(getattr(result, "exception_type", None)),
                     _text_or_none(getattr(result, "exception_code", None)), _text_or_none(getattr(result, "diagnostic", None)))


class WriteCoordinator:
    """Thread-safe as far as the stores are: each is one connection serialized by a lock, and its own
    counters take a lock held only to count. No lock is held across a writer call. Claims across
    instances and processes are decided by the ledger, on one host, sharing one pair of files.

    store_write_attempts bounds retries of a local store write after a failure while recording. Those
    never repeat the write itself."""

    def __init__(self, journal: NeoWriteJournal, ledger: IdempotencyLedger, *, writer: Writer, registry: OperationRegistry,
                 marker_reader: Optional[Any] = None,
                 limits: Any = None, health: Optional[HealthGate] = None, worker_id: Optional[str] = None,
                 store_write_attempts: int = 3, store_retry_delay: float = 0.05) -> None:
        if not isinstance(journal, NeoWriteJournal):
            raise TypeError("journal must be a NeoWriteJournal")
        if not isinstance(ledger, IdempotencyLedger):
            raise TypeError("ledger must be an IdempotencyLedger")
        if not callable(writer):
            raise TypeError("writer must be callable: there's no default, so nothing can run unprotected by accident")
        if not isinstance(registry, OperationRegistry):
            raise TypeError("registry must be an OperationRegistry")
        if type(store_write_attempts) is not int or store_write_attempts < 1:
            raise ValueError("store_write_attempts must be at least 1")
        self._journal = journal
        self._ledger = ledger
        self._writer = writer
        self._registry = registry
        self._marker_reader = marker_reader
        self._limits = limits
        self._health = health
        self._worker_id = worker_id
        self._store_write_attempts = store_write_attempts
        self._store_retry_delay = store_retry_delay
        self._counts: Counter = Counter()
        self._counts_lock = threading.Lock()

    def verify_startup(self, **kwargs: Any) -> Any:
        """Verifies startup through the configured marker reader or startup verifier."""
        if self._marker_reader is None:
            raise CoordinatorError("no marker_reader configured")
        verifier = getattr(self._marker_reader, "verify_startup", None)
        if callable(verifier):
            return verifier(**kwargs)
        if callable(self._marker_reader):
            # caller may pass a direct verify_startup callable or module
            return self._marker_reader(**kwargs)
        raise CoordinatorError("configured marker_reader does not support verify_startup")

    # ------------------------------------------------------------ results and counters

    def counters(self) -> dict:
        """Counts for this instance since it was created, by disposition and identity refusal. In memory
        only: a durable count belongs to WP-O1."""
        with self._counts_lock:
            return dict(self._counts)

    def _count(self, name: str) -> None:
        with self._counts_lock:
            self._counts[name] += 1

    def _result(self, disposition: Disposition, key_ref: Optional[str], **details: Any) -> CoordinatedResult:
        result = CoordinatedResult(key_ref, disposition, **details)
        self._count(f"disposition.{disposition.value}")
        if disposition in (Disposition.EXECUTED_UNCERTAIN, Disposition.COMMITTED_UNRECORDED, Disposition.CONFLICT) or not result.finalized:
            level = logging.ERROR
        elif disposition in (Disposition.EXECUTED_COMMITTED, Disposition.ALREADY_COMPLETED):
            level = logging.DEBUG
        else:
            level = logging.WARNING
        logger.log(level, "Coordinated write %s: %s attempt=%s finalized=%s %s", key_ref, disposition.value,
                   result.ledger_attempt, result.finalized, result.diagnostic)
        return result

    def _answer_by_state(self, request: LedgerRequest, key_ref: str) -> CoordinatedResult:
        disposition = _ANSWER_BY_STATE.get(request.state)
        if disposition is None:
            raise CoordinationInconsistency(f"{key_ref}: lost a claim to a request still {request.state.value}")
        return self._result(disposition, key_ref, ledger_attempt=request.attempt_count)

    def _unavailable(self, key_ref: str, which: str, cause: Exception, **details: Any) -> CoordinatedResult:
        return self._result(Disposition.NOT_DISPATCHED_UNAVAILABLE, key_ref, unavailable=which,
                            diagnostic=f"{which} unavailable ({type(cause).__name__}): the writer wasn't called", **details)

    # ------------------------------------------------------------ store errors

    @contextmanager
    def _journal_call(self) -> Iterator[None]:
        try:
            yield
        except (JournalWriteError, JournalClosedError) as error:
            raise JournalUnavailable(f"the journal is unavailable: {type(error).__name__}") from error
        except (JournalTransitionError, DuplicateCorrelationError):
            raise
        except JournalError as error:
            raise CoordinationInconsistency(f"journal: {type(error).__name__}") from error

    @contextmanager
    def _ledger_call(self) -> Iterator[None]:
        try:
            yield
        except (LedgerWriteError, LedgerClosedError) as error:
            raise LedgerUnavailable(f"the ledger is unavailable: {type(error).__name__}") from error
        except (LedgerTransitionError, IdempotencyConflictError, RequestNotFoundError, UnsupportedPayloadError):
            raise
        except LedgerError as error:
            raise CoordinationInconsistency(f"ledger: {type(error).__name__}") from error

    def _gate_is_open(self) -> bool:
        if self._health is None:
            return True
        try:
            return self._health.allow_dispatch() is True
        except Exception as error:
            logger.error("The health gate failed (%s), so nothing is dispatched", type(error).__name__)
            return False

    # ------------------------------------------------------------ admission

    def submit(self, request: WriteRequest) -> CoordinatedResult:
        """Registers the request if it's new, and runs it if its state allows and this call wins the
        claim. The only call that can start a first attempt.

        Returns a disposition for everything about the write itself, including a conflict, a refused
        identity and an unavailable store. Raises InvalidRequest and PayloadNotJournalable for a caller's
        mistake, and CoordinationInconsistency for damage, always before the writer is called."""
        if type(request) is not WriteRequest:
            raise InvalidRequest("submit takes a WriteRequest")
        operation = self._registry.get(request.operation)
        refusal = check_source_key(request.source_key)
        if refusal is not None:
            return self._refuse_identity(refusal)
        statements = operation.statements(request.parameters)
        key = request_key(request.source_key, operation.name)
        key_ref = _key_ref(key)
        if not self._gate_is_open():
            return self._result(Disposition.NOT_DISPATCHED_UNAVAILABLE, key_ref, unavailable="neo4j",
                                diagnostic="the health gate is closed: nothing was stored or dispatched")
        try:
            return self._admit(key, key_ref, operation, statements)
        except JournalUnavailable as error:
            return self._unavailable(key_ref, "journal", error)
        except LedgerUnavailable as error:
            return self._unavailable(key_ref, "ledger", error)

    def _refuse_identity(self, refusal: IdentityRefusal) -> CoordinatedResult:
        self._count(f"identity_refusal.{refusal.value}")
        return self._result(Disposition.REFUSED_IDENTITY, None, refusal=refusal, finalized=True,
                            diagnostic=f"no usable source identity ({refusal.value}): nothing was stored, and nothing runs unprotected")

    def _admit(self, key: str, key_ref: str, operation: RegisteredOperation, statements: list) -> CoordinatedResult:
        self._preflight(key, key_ref, operation.name, statements)
        try:
            with self._ledger_call():
                registration = self._ledger.register(key, operation.name, statements)
        except IdempotencyConflictError as error:
            return self._result(Disposition.CONFLICT, key_ref, differences=error.differences,
                                diagnostic="the key is registered with a different payload, which was left as it was")
        except UnsupportedPayloadError as error:
            raise PayloadNotJournalable(f"{key_ref}: {type(error).__name__}") from error
        request = registration.request
        if request.state is not LedgerState.REGISTERED:
            return self._answer_by_state(request, key_ref)
        record = self._first_intent(key, key_ref, operation.name, statements, request)
        if record is None:
            return self._answer_by_state(self._read_request(key), key_ref)
        return self._claim_and_run(key, key_ref, request.fingerprint, record)

    def _preflight(self, key: str, key_ref: str, operation: str, statements: list) -> None:
        with self._journal_call():
            found = self._journal.preflight_intent(operation, statements, correlation_key=key, epoch=1)
        if found.accepted or found.code is PreflightCode.DUPLICATE_CORRELATION:
            return
        if found.code in (PreflightCode.UNSUPPORTED_PARAMETER, PreflightCode.INVALID_ARGUMENT):
            raise PayloadNotJournalable(f"{key_ref}: the journal can't store this payload ({found.code.value})")
        raise CoordinationInconsistency(f"{key_ref}: the journal refused the intent ({found.code.value})")

    def _read_request(self, key: str) -> LedgerRequest:
        with self._ledger_call():
            return self._ledger.get_request(key)

    def _read_chain(self, key: str) -> tuple:
        with self._journal_call():
            return self._journal.find_by_correlation_key(key)

    def _verify_stored_payload(self, record: JournalRecord, request: LedgerRequest, key_ref: str) -> None:
        try:
            stored = fingerprint_payload(record.operation, record.statements, return_rows=record.return_rows).value
        except (TypeError, ValueError) as error:
            raise PayloadMismatch(f"{key_ref}: journal record {record.record_id} has no fingerprint: {type(error).__name__}") from error
        if stored != request.fingerprint:
            raise PayloadMismatch(f"{key_ref}: journal record {record.record_id} doesn't match the ledger's fingerprint")

    def _first_intent(self, key: str, key_ref: str, operation: str, statements: list, request: LedgerRequest) -> Optional[JournalRecord]:
        """The PENDING intent to run from: an existing one, as after a crash, or a new one. None if the
        request has moved on since it was read, so the caller answers from its state instead."""
        chain = self._read_chain(key)
        if not chain:
            try:
                with self._journal_call():
                    return self._journal.create_intent(operation, statements, correlation_key=key, epoch=1)
            except DuplicateCorrelationError as error:  # another caller won the race to create it
                with self._journal_call():
                    chain = (self._journal.get_record(error.record_id),)
        record = chain[-1]
        if record.state is JournalState.PENDING and record.epoch == 1:
            self._verify_stored_payload(record, request, key_ref)
            return record
        if self._read_request(key).state is LedgerState.REGISTERED:
            raise CoordinationInconsistency(f"{key_ref}: registered, but journal record {record.record_id} is {record.state.value}")
        return None

    # ------------------------------------------------------------ claim, dispatch, finalize

    def _claim_and_run(self, key: str, key_ref: str, fingerprint: str, record: JournalRecord,
                       authorization: Optional[tuple] = None) -> CoordinatedResult:
        try:
            with self._ledger_call():
                claim = self._ledger.begin_execution(key, fingerprint, worker_id=self._worker_id)
        except LedgerTransitionError:
            return self._answer_by_state(self._read_request(key), key_ref)
        except IdempotencyConflictError as error:
            raise PayloadMismatch(f"{key_ref}: the claim's fingerprint isn't the one registered") from error
        attempt = claim.attempt_count
        try:
            with self._journal_call():
                if authorization is None:
                    started = self._journal.mark_in_flight(record.record_id, worker_id=self._worker_id, ledger_attempt=attempt)
                else:
                    started = self._journal.authorize_retry(record.record_id, authorized_by=authorization[0], reason=authorization[1],
                                                             worker_id=self._worker_id, ledger_attempt=attempt)
        except JournalUnavailable as error:
            released = self._release_claim(key, attempt)
            return self._unavailable(key_ref, "journal", error, ledger_attempt=attempt, journal_record_id=record.record_id, finalized=released)
        except JournalTransitionError as error:
            raise CoordinationInconsistency(f"{key_ref}: journal record {record.record_id} refused attempt {attempt}: {type(error).__name__}") from error
        return self._run_writer(key, key_ref, started, attempt)

    def _release_claim(self, key: str, attempt: int) -> bool:
        """The journal refused IN_FLIGHT, so the writer certainly wasn't called. That's established, so it's
        NOT_COMPLETED rather than uncertain. If this fails too, recovery reports CLAIMED_NOT_DISPATCHED."""
        return self._record_in_ledger(key, LedgerState.NOT_COMPLETED, lambda: self._ledger.record_not_completed(
            key, attempt=attempt, error_type="JournalUnavailable", diagnostic="the journal refused IN_FLIGHT, so the writer wasn't called"))

    def _run_writer(self, key: str, key_ref: str, record: JournalRecord, attempt: int) -> CoordinatedResult:
        """Only reached once the claim and IN_FLIGHT have both committed."""
        try:
            report = _read_report(self._writer(record.operation, record.statements, limits=self._limits))
        except Exception as error:
            report = _Reported(_OUTCOME_AMBIGUOUS, error_type=type(error).__name__,
                               diagnostic="the writer raised instead of reporting an outcome")
        except BaseException as error:
            self._record_uncertain_best_effort(key, record.record_id, attempt, type(error).__name__)
            raise
        return self._finalize(key, key_ref, record.record_id, attempt, report)

    def _record_uncertain_best_effort(self, key: str, record_id: str, attempt: int, error_type: str) -> None:
        detail = "the call ended while the writer was running"
        try:
            self._journal.mark_ambiguous(record_id, error_type=error_type, diagnostic=detail, expected_ledger_attempt=attempt)
            self._ledger.record_uncertain(key, attempt=attempt, error_type=error_type, diagnostic=detail)
        except (JournalError, LedgerError) as error:
            logger.error("Coordinated write %s: not recorded as uncertain (%s). Recovery will do it", _key_ref(key), type(error).__name__)

    def _finalize(self, key: str, key_ref: str, record_id: str, attempt: int, report: _Reported) -> CoordinatedResult:
        """Journal first, then ledger. A store failure here never re-runs anything: it leaves the result
        unfinalized, and recovery finishes or downgrades it. Success isn't reported until both are durable."""
        common = {"ledger_attempt": attempt, "journal_record_id": record_id, "writer_invoked": True, "writer_attempts": report.attempts}
        details = {"error_type": report.error_type, "error_code": report.error_code, "diagnostic": report.diagnostic}
        if report.outcome == _OUTCOME_COMMITTED:
            done = (self._record_in_journal(lambda: self._journal.mark_committed(record_id, expected_ledger_attempt=attempt))
                    and self._record_in_ledger(key, LedgerState.COMPLETED, lambda: self._ledger.record_completed(key, attempt=attempt)))
            if done:
                return self._result(Disposition.EXECUTED_COMMITTED, key_ref, **common)
            return self._result(Disposition.COMMITTED_UNRECORDED, key_ref, finalized=False,
                                diagnostic="the writer reported a commit, but it isn't durable in both stores. Never run this again", **common)
        if report.outcome == _OUTCOME_NOT_COMMITTED:
            if not self._record_in_journal(lambda: self._journal.mark_not_committed(record_id, expected_ledger_attempt=attempt, **details)):
                return self._result(Disposition.EXECUTED_UNCERTAIN, key_ref, finalized=False,
                                    diagnostic="a non-commit couldn't be recorded, so recovery will treat it as uncertain", **common)
            finalized = self._record_in_ledger(key, LedgerState.NOT_COMPLETED, lambda: self._ledger.record_not_completed(key, attempt=attempt, **details))
            disposition = Disposition.NOT_EXECUTED_CONFIRMED if report.attempts == 0 else Disposition.EXECUTED_NOT_COMMITTED
            return self._result(disposition, key_ref, finalized=finalized, diagnostic=report.diagnostic or "", **common)
        finalized = (self._record_in_journal(lambda: self._journal.mark_ambiguous(record_id, expected_ledger_attempt=attempt, **details))
                     and self._record_in_ledger(key, LedgerState.UNCERTAIN, lambda: self._ledger.record_uncertain(key, attempt=attempt, **details)))
        return self._result(Disposition.EXECUTED_UNCERTAIN, key_ref, finalized=finalized, diagnostic=report.diagnostic or "", **common)

    def _with_bounded_retries(self, store: str, retryable: tuple, call: Callable[[], Any]) -> bool:
        for attempt in range(1, self._store_write_attempts + 1):
            try:
                call()
                return True
            except retryable as error:
                logger.warning("%s write failed (%s), try %d of %d", store, type(error).__name__, attempt, self._store_write_attempts)
                if attempt < self._store_write_attempts:
                    time.sleep(self._store_retry_delay)
        return False

    def _record_in_journal(self, call: Callable[[], Any]) -> bool:
        """False if the outcome isn't recorded. A refused transition, such as a stale attempt, changed
        nothing and must never overwrite, so it's logged and left."""
        try:
            return self._with_bounded_retries("Journal", (JournalWriteError,), call)
        except JournalError as error:
            logger.error("Journal outcome not recorded: %s", type(error).__name__)
            return False

    def _record_in_ledger(self, key: str, target: LedgerState, call: Callable[[], Any]) -> bool:
        """A refused transition is fine only if the ledger already holds this outcome, which means a
        concurrent recovery finished it from the journal."""
        try:
            return self._with_bounded_retries("Ledger", (LedgerWriteError,), call)
        except LedgerTransitionError as error:
            try:
                reached = self._ledger.get_request(key).state is target
            except LedgerError:
                reached = False
            if not reached:
                logger.error("Ledger outcome not recorded: %s", type(error).__name__)
            return reached
        except LedgerError as error:
            logger.error("Ledger outcome not recorded: %s", type(error).__name__)
            return False

    # ------------------------------------------------------------ people: authorize, resolve, retry

    @staticmethod
    def _checked_person_and_reason(person: Any, reason: Any, name: str) -> None:
        if type(person) is not str or not person.strip():
            raise InvalidRequest(f"{name} must be a non-empty string")
        if type(reason) is not str or not reason.strip():
            raise InvalidRequest("reason must be a non-empty string")

    @staticmethod
    def _checked_key(key: Any) -> str:
        if type(key) is not str or not key:
            raise InvalidRequest("a request key must be a non-empty string")
        return key

    def _existing_request(self, key: str) -> LedgerRequest:
        try:
            return self._read_request(key)
        except RequestNotFoundError as error:
            raise InvalidRequest(f"no request for {_key_ref(key)}") from error
        except (TypeError, ValueError) as error:
            raise InvalidRequest(f"not a usable request key: {type(error).__name__}") from error

    def authorize_retry(self, key: str, *, authorized_by: str, reason: str) -> RequestView:
        """A person allows one more attempt. Touches only the ledger and runs nothing. Only a request
        that is NOT_COMPLETED can be authorized: an uncertain one has to be resolved first."""
        key = self._checked_key(key)
        self._checked_person_and_reason(authorized_by, reason, "authorized_by")
        request = self._existing_request(key)
        try:
            with self._ledger_call():
                self._ledger.authorize_retry(key, authorized_by=authorized_by, reason=reason)
        except LedgerTransitionError as error:
            raise NotAuthorized(f"{request.key_ref}: a request that's {request.state.value} can't be authorized") from error
        logger.warning("Coordinated write %s: retry authorized", request.key_ref)
        return self.inspect(key)

    def resolve(self, key: str, *, decision: Decision, resolved_by: str, reason: str) -> RequestView:
        """A person's conclusion about an uncertain request. Journal, then ledger. Runs nothing, and
        FOUND_NOT_COMMITTED still needs authorize_retry(). Repeating the same decision finishes a
        half-done resolution. A different one raises ResolutionConflict."""
        key = self._checked_key(key)
        self._checked_person_and_reason(resolved_by, reason, "resolved_by")
        try:
            decision = Decision(decision)
        except ValueError as error:
            raise InvalidRequest("decision must be a Decision") from error
        journal_resolution, ledger_resolution, target = _DECISIONS[decision]
        request = self._existing_request(key)
        chain = self._read_chain(key)
        combined = combined_state(request, chain)
        latest = chain[-1] if chain else None
        key_ref = request.key_ref
        if combined is CombinedState.UNCERTAIN:
            try:
                with self._journal_call():
                    self._journal.resolve_ambiguous(latest.record_id, resolution=journal_resolution, resolved_by=resolved_by, reason=reason)
            except JournalTransitionError as error:
                raise ResolutionConflict(f"{key_ref}: resolved by someone else just now. Inspect it, and repeat if still needed") from error
        elif combined is CombinedState.CLAIMED_NOT_DISPATCHED:
            # The journal has nothing to resolve. Its records show the writer wasn't called, so "committed" contradicts them.
            if decision is Decision.FOUND_COMMITTED:
                raise ResolutionConflict(f"{key_ref}: the writer was never dispatched, so this can't be found committed")
        elif latest is not None and latest.state is JournalState.RESOLVED:
            if latest.resolution is not journal_resolution:
                raise ResolutionConflict(f"{key_ref}: already resolved as {latest.resolution.value}")
            if combined is not CombinedState.RESOLUTION_INCOMPLETE:
                return self.inspect(key)  # a repeat of something already finished
        else:
            raise NotAuthorized(f"{key_ref}: {combined.value} isn't uncertain, so there's nothing to resolve")
        try:
            with self._ledger_call():
                self._ledger.resolve_uncertain(key, resolution=ledger_resolution, resolved_by=resolved_by, reason=reason)
        except LedgerTransitionError as error:
            raise ResolutionConflict(f"{key_ref}: resolved by someone else just now. Inspect it, and repeat if still needed") from error
        logger.warning("Coordinated write %s: resolved as %s", key_ref, decision.value)
        return self.inspect(key)

    def retry(self, key: str) -> CoordinatedResult:
        """Runs one authorized retry from the journal's stored payload. Takes no statements, so nothing can
        be substituted. A confirmed non-commit reuses its record. A request a person found not committed
        gets the next epoch, linked by key and carrying the authorization.

        Raises NotAuthorized unless the ledger is RETRY_AUTHORIZED. An unavailable store is returned as
        a disposition, with the authorization still unused unless the claim had already been made."""
        key = self._checked_key(key)
        request = self._existing_request(key)
        key_ref = request.key_ref
        if request.state is not LedgerState.RETRY_AUTHORIZED:
            raise NotAuthorized(f"{key_ref}: {request.state.value} isn't authorized to retry")
        if not self._gate_is_open():
            return self._result(Disposition.NOT_DISPATCHED_UNAVAILABLE, key_ref, unavailable="neo4j", ledger_attempt=request.attempt_count,
                                diagnostic="the health gate is closed: the authorization is still unused")
        try:
            return self._authorized_retry(key, key_ref, request)
        except JournalUnavailable as error:
            return self._unavailable(key_ref, "journal", error, ledger_attempt=request.attempt_count)
        except LedgerUnavailable as error:
            return self._unavailable(key_ref, "ledger", error, ledger_attempt=request.attempt_count)

    def _authorized_retry(self, key: str, key_ref: str, request: LedgerRequest) -> CoordinatedResult:
        chain = self._read_chain(key)
        if not chain:
            raise CoordinationInconsistency(f"{key_ref}: authorized, but the journal holds no record of it")
        latest = chain[-1]
        self._verify_stored_payload(latest, request, key_ref)
        authorized_by, reason = self._authorization_of(key, key_ref)
        if latest.state is JournalState.NOT_COMMITTED:
            return self._claim_and_run(key, key_ref, request.fingerprint, latest, authorization=(authorized_by, reason))
        if latest.state is JournalState.PENDING:
            return self._claim_and_run(key, key_ref, request.fingerprint, latest)
        if latest.state is JournalState.RESOLVED and latest.resolution is JournalResolution.FOUND_NOT_COMMITTED:
            return self._claim_and_run(key, key_ref, request.fingerprint, self._next_epoch(key, key_ref, latest, authorized_by, reason))
        current = self._read_request(key)  # another retry may have taken it since the ledger was read
        if current.state is not LedgerState.RETRY_AUTHORIZED:
            return self._answer_by_state(current, key_ref)
        raise CoordinationInconsistency(f"{key_ref}: authorized, but journal record {latest.record_id} is {latest.state.value}")

    def _authorization_of(self, key: str, key_ref: str) -> tuple:
        with self._ledger_call():
            history = self._ledger.get_history(key)
        for event in reversed(history):
            if event.to_state is LedgerState.RETRY_AUTHORIZED:
                return event.actor, event.reason
        raise CoordinationInconsistency(f"{key_ref}: authorized, but the ledger's history doesn't say by whom")

    def _next_epoch(self, key: str, key_ref: str, previous: JournalRecord, authorized_by: str, reason: str) -> JournalRecord:
        try:
            with self._journal_call():
                return self._journal.create_intent(previous.operation, previous.statements, return_rows=previous.return_rows,
                                                    correlation_key=key, epoch=previous.epoch + 1,
                                                    authorized_by=authorized_by, reason=reason)
        except DuplicateCorrelationError as error:  # a concurrent retry created it, and the claim decides who runs it
            with self._journal_call():
                return self._journal.get_record(error.record_id)
        except JournalTransitionError as error:
            raise CoordinationInconsistency(f"{key_ref}: the journal refused the next epoch: {type(error).__name__}") from error

    # ------------------------------------------------------------ reading and recovery

    def inspect(self, key: str) -> RequestView:
        """Read only, and computed each time."""
        key = self._checked_key(key)
        try:
            request: Optional[LedgerRequest] = self._read_request(key)
        except RequestNotFoundError:
            request = None
        except (TypeError, ValueError) as error:
            raise InvalidRequest(f"not a usable request key: {type(error).__name__}") from error
        chain = self._read_chain(key)
        views = tuple(JournalEpochView(r.epoch, r.record_id, r.state, r.ledger_attempt, r.resolution) for r in chain)
        return RequestView(_key_ref(key), combined_state(request, chain), request.state if request else None,
                           request.attempt_count if request else 0, views)

    def _mirror_journal_outcomes(self) -> tuple[list, list, bool]:
        """Finishes the ledger from outcomes the journal already holds, at the same attempt. Nothing runs,
        and it comes before the ledger's own recovery, so a confirmed outcome isn't reduced to uncertain.
        Returns the keys reconciled from a confirmed outcome, and those that became uncertain and so need
        a person. The bool is False if a damaged row stopped the ledger being listed: those stay for the
        ledger's own recovery, which only makes them less runnable."""
        try:
            with self._ledger_call():
                claimed = self._ledger.list_requests(LedgerState.IN_PROGRESS)
        except CoordinationInconsistency:
            return [], [], False
        reconciled, uncertain = [], []
        for request in claimed:
            key = request.idempotency_key
            attempt = request.attempt_count
            try:
                chain = self._read_chain(key)
            except CoordinationInconsistency:
                continue
            latest = chain[-1] if chain else None
            if latest is None or latest.ledger_attempt != attempt:
                continue
            mirror: Optional[Callable[[], Any]] = None
            detail = {"error_type": latest.error_type, "error_code": latest.error_code, "diagnostic": latest.diagnostic}
            if latest.state is JournalState.COMMITTED:
                target, mirror = LedgerState.COMPLETED, lambda k=key, a=attempt: self._ledger.record_completed(k, attempt=a)
            elif latest.state is JournalState.NOT_COMMITTED:
                target, mirror = LedgerState.NOT_COMPLETED, lambda k=key, a=attempt, d=detail: self._ledger.record_not_completed(k, attempt=a, **d)
            elif latest.state is JournalState.AMBIGUOUS:
                target, mirror = LedgerState.UNCERTAIN, lambda k=key, a=attempt, d=detail: self._ledger.record_uncertain(k, attempt=a, **d)
            if mirror is not None and self._record_in_ledger(key, target, mirror):
                (uncertain if target is LedgerState.UNCERTAIN else reconciled).append(key)
        return reconciled, uncertain, True

    def recover(self) -> RecoveryResult:
        """Run at startup, before workers. Never calls the writer or Neo4j, and repeating it changes
        nothing further. Damaged rows are excluded and reported, and never repaired. A store that can't
        be read raises JournalUnavailable or LedgerUnavailable, and nothing is concluded."""
        with self._journal_call():
            journal_integrity = self._journal.check_integrity()
        with self._ledger_call():
            ledger_integrity = self._ledger.check_integrity()
        problems = (*journal_integrity.problems, *ledger_integrity.problems)

        reconciled, newly_uncertain, listed = self._mirror_journal_outcomes()
        with self._journal_call():
            self._journal.recover_unfinished()
        more_reconciled, more_uncertain, listed_again = self._mirror_journal_outcomes()
        reconciled.extend(more_reconciled)
        newly_uncertain.extend(more_uncertain)
        with self._ledger_call():
            ledger_report = self._ledger.recover_unfinished()
        newly_uncertain.extend(ledger_report.recovered)
        if not (listed and listed_again):
            problems = (*problems, "a damaged ledger row stopped reconciling from the journal")

        keys: dict[RecoveryCategory, list] = {category: [] for category in RecoveryCategory}
        keys[RecoveryCategory.RECONCILED] = reconciled
        keys[RecoveryCategory.NEWLY_UNCERTAIN] = newly_uncertain
        keys[RecoveryCategory.CORRUPT] = [*journal_integrity.corrupt_record_ids, *ledger_integrity.corrupt_keys]
        known: set = set()
        for key in self._keys_to_classify(ledger_report):
            known.add(key)
            try:
                view = self.inspect(key)
            except CoordinationInconsistency:
                continue  # already in CORRUPT
            category = _COMBINED_TO_CATEGORY.get(view.combined)
            if category is not None and key not in keys[RecoveryCategory.NEWLY_UNCERTAIN]:
                keys[category].append(key)
        if not ledger_integrity.corrupt_keys:
            keys[RecoveryCategory.INCONSISTENT].extend(self._journal_keys_without_a_request(known))
        result = RecoveryResult(keys, journal_integrity.ok and ledger_integrity.ok, tuple(problems))
        logger.log(logging.WARNING if result.counts() or problems else logging.INFO, "Coordinator recovery: %s", result)
        return result

    def _keys_to_classify(self, report: Any) -> list:
        try:
            with self._ledger_call():
                return [request.idempotency_key for request in self._ledger.list_requests()]
        except CoordinationInconsistency:
            # A damaged row stops the listing. The ledger's own report still names the sound unfinished ones.
            return [*report.recovered, *report.still_running, *report.registered, *report.not_completed,
                    *report.retry_authorized, *report.uncertain]

    def _journal_keys_without_a_request(self, known: set) -> list:
        try:
            with self._journal_call():
                records = self._journal.list_records()
        except CoordinationInconsistency:
            return []  # the damage is already reported
        return sorted({record.correlation_key for record in records
                       if record.correlation_key is not None and record.correlation_key not in known})

# Coordinator interface proposal

Phase 2.5, design only. **Everything here is [P] proposed. None of it exists.** The code blocks specify an interface for a later task, and they aren't executable production code. Backend paths are relative to `isfcr/CYUKTI/backend/`.

## Required API changes

These change completed phases, so each needs your approval (D3, D4). **None is implemented.**

| ID | Component | Change | Why | Status |
|---|---|---|---|---|
| AC-1 | Journal | `create_intent(..., correlation_key, epoch)` with `UNIQUE(correlation_key, epoch)`. `find_by_correlation_key(key)`. `mark_in_flight` and `authorize_retry` take `ledger_attempt`, kept in history | [V] Neither store can find the other's record, and `create_intent` makes a new record every call (`neo4j_write_journal.py:NeoWriteJournal.create_intent` L861-888) | **Required** |
| AC-2 | Journal | Public `check_statements(statements, return_rows)` running the existing check without storing | [V] The ledger accepts values the journal refuses. Only private `_serialize_intent` can test that today (`neo4j_write_journal.py:_serialize_intent` L325-362) | **Required** |
| AC-3 | Journal | `mark_committed`, `mark_not_committed`, `mark_ambiguous` take `expected_ledger_attempt`, checked in the same transaction | [V] They have no attempt argument (`neo4j_write_journal.py:NeoWriteJournal.mark_committed` L925-928). A check beforehand leaves a small race | **Required** |
| AC-4 | Both | `evidence` argument on resolve | Your rule asks for evidence. Without it, evidence is a reference inside `reason`, limited to 500 characters | Optional |
| AC-5 | Ledger | `abandon_registered(key, abandoned_by, reason)`: `REGISTERED` to `ABANDONED` | [V] `REGISTERED` can only go to `IN_PROGRESS`, so a request that will never run stays open forever (`neo4j_idempotency_ledger.py` L142-152) | Optional, harmless without it |
| AC-6 | Writers | Carry a source identity from the listener into each writer | [V] `process_alert` has none today | **Required to migrate** |
| AC-7 | Writers | Derive `event_id`, `campaign_id` and `operation_id` from the key | [V] They're random, inside the writers. Needed for W01, W05, W06, W16 (D7) | Required for those |

[V] The helper needs no change: it already returns everything the coordinator needs (`neo4j_client.py:WriteResult` L183-208).

[V] The journal's schema version would change to 2. It refuses other versions, so a migration step is needed (`neo4j_write_journal.py:NeoWriteJournal._validate_schema` L678-691). [A] No journal file exists in production, because nothing uses it yet, so the migration should be trivial.

## Interface

### Values

```python
@dataclass(frozen=True)
class WriteRequest:
    key: str                 # from the key registry, never built by hand
    operation: str           # from the operation registry
    statements: Sequence     # ordered (query, parameters) pairs
    return_rows: bool = False
    limits: WriteLimits | None = None
    worker_id: str | None = None

class Disposition(Enum):
    EXECUTED_COMMITTED      # this call ran it, and commit was confirmed
    EXECUTED_NOT_COMMITTED  # this call ran it, and non-commit was established
    EXECUTED_UNCERTAIN      # this call ran it, and the outcome is unknown
    COMMITTED_UNRECORDED    # commit confirmed, but the journal write failed even after retrying
    ALREADY_COMPLETED       # finished earlier; nothing ran
    IN_PROGRESS_ELSEWHERE   # someone holds the claim; nothing ran
    AWAITING_AUTHORIZATION  # NOT_COMPLETED; nothing ran
    BLOCKED_UNCERTAIN       # UNCERTAIN; nothing ran
    BLOCKED_ABANDONED       # ABANDONED; nothing ran

@dataclass(frozen=True)
class CoordinatedResult:
    key_ref: str             # safe to log; never the raw key
    disposition: Disposition
    ledger_attempt: int
    journal_record_id: str | None
    write_result: WriteResult | None   # None unless this call ran the helper
    records: list | None     # rows, only from EXECUTED_COMMITTED with return_rows
    diagnostic: str          # safe to log

class Decision(Enum):
    FOUND_COMMITTED, FOUND_NOT_COMMITTED, ABANDONED

@dataclass(frozen=True)
class RequestView:
    key_ref: str
    ledger_state: LedgerState | None
    journal_records: tuple   # every epoch, oldest first
    ledger_attempt: int
    combined: str            # computed each time, never stored
    consistent: bool
```

### Coordinator

```python
class WriteCoordinator:
    def __init__(self, journal: NeoWriteJournal, ledger: IdempotencyLedger, writer=write_statements): ...
```
Takes open stores, so it opens no file and has no default path. `writer` is injectable so tests can use fakes.

---

```python
def submit(self, request: WriteRequest) -> CoordinatedResult
```

Registers the request if new, and runs it if allowed. The only call that can start a first attempt.

| | |
|---|---|
| **Steps** | Fingerprint, preflight, `register`, find or create the intent, `begin_execution`, `IN_FLIGHT`, helper, journal outcome, ledger outcome |
| **Runs Neo4j** | Only if the state allows it and this call wins the claim |
| **Returns** | A disposition, for every outcome of the write itself |
| **Raises** | `RequestConflict`, `PayloadNotJournalable`, `InvalidRequest`, `JournalUnavailable`, `LedgerUnavailable`, `CoordinationInconsistency` |
| **Guarantee** | If it raises before dispatch, Neo4j wasn't called. Once dispatched, it reports an outcome and doesn't raise for the write itself |

---

```python
def retry(self, key: str, *, worker_id: str | None = None) -> CoordinatedResult
```

Runs an authorized retry from the journal's stored payload. Takes no statements, so a caller can't substitute anything.

| | |
|---|---|
| **Precondition** | Ledger is `RETRY_AUTHORIZED` |
| **Steps** | Load the latest journal record. Check its statements against the ledger's fingerprint. If it's `RESOLVED`, create the next epoch. Then claim, and move the journal to `IN_FLIGHT`: `mark_in_flight` from `PENDING`, journal `authorize_retry` from `NOT_COMMITTED`. Then as `submit` |
| **Raises** | `NotAuthorized`, `PayloadMismatch`, plus the store errors |

---

```python
def authorize_retry(self, key: str, *, principal: str, reason: str) -> RequestView
```

Grants one more attempt. **Touches only the ledger and runs nothing.** [V] The ledger refuses unless it's `NOT_COMPLETED`, and it requires both arguments (`neo4j_idempotency_ledger.py:IdempotencyLedger.authorize_retry` L1125-1130).

---

```python
def resolve(self, key: str, *, decision: Decision, principal: str, reason: str, evidence: str) -> RequestView
```

Records an operator's conclusion about an uncertain request.

| | |
|---|---|
| **Steps** | Journal `resolve_ambiguous`, then ledger `resolve_uncertain`, per the mapping in `CROSS_COMPONENT_STATE_TRANSITIONS.md` |
| **Journal `PENDING`** (`CLAIMED_NOT_DISPATCHED`) | The journal has nothing to resolve, so only the ledger changes. A later retry uses `mark_in_flight` |
| **Repeat** | Safe: a repeat with the same decision finishes a half-done resolution. A **different** decision raises `ResolutionConflict` |
| **Runs Neo4j** | Never. `FOUND_NOT_COMMITTED` still needs `authorize_retry` |
| **Requires** | `principal`, `reason` and `evidence`, all non-empty |

---

```python
def recover(self) -> RecoveryResult
```

The algorithm in `FAILURE_WINDOW_AND_RECOVERY_ANALYSIS.md`. **Never calls the helper.** Safe to repeat. Returns the report categories listed there.

---

```python
def inspect(self, key: str) -> RequestView
def list_attention(self) -> tuple[RequestView, ...]
def check_consistency(self) -> ConsistencyReport
```

Read only. `list_attention` is what an operator works from. `check_consistency` runs both integrity checks and every cross-store comparison.

### Key registry

```python
def request_key(operation: Operation, source: SourceId, *discriminators: str) -> str
```

The only place keys are built. It raises if `source` is missing, and it validates against the ledger's rules. `SourceId` comes from the D1 rule, and `Operation` is a fixed enum.

## Error taxonomy

| Error | Meaning | Neo4j called? | Caller should |
|---|---|---|---|
| `InvalidRequest` | Malformed key, statements or arguments | No | Fix the code |
| `PayloadNotJournalable` | A value can't be stored exactly | No | Fix the value. Nothing was registered |
| `RequestConflict` | Same key, different payload. Carries differences, no values | No | Stop. A person investigates |
| `JournalUnavailable` | Journal failed before dispatch | No | Don't acknowledge or discard (D15) |
| `LedgerUnavailable` | Ledger failed before dispatch | No | Don't acknowledge or discard (D15) |
| `CoordinationInconsistency` | A combination that shouldn't arise | No | Stop. A person investigates |
| `NotAuthorized` | `retry` without `RETRY_AUTHORIZED` | No | Authorize first |
| `PayloadMismatch` | The journal's stored payload doesn't match the ledger's fingerprint | No | Stop. Treat as damage |
| `ResolutionConflict` | A different decision on a half-resolved request | No | A person investigates |

### Mapping store errors

| Raised by a store | Before dispatch | After dispatch |
|---|---|---|
| `*BusyError`, `*WriteError` | `JournalUnavailable` or `LedgerUnavailable` | Retry the local write, bounded. Then `COMMITTED_UNRECORDED`, or leave for recovery. **Never re-execute** |
| `*CorruptionError` | `CoordinationInconsistency` | The same |
| `*SchemaError` | At startup: refuse to start | n/a |
| `InvalidTransitionError` | Re-read the state, and answer with its disposition | Re-read. Usually a recovery or another caller finished it |
| `StaleAttemptError` | n/a | Log and drop that outcome. **It must never overwrite** |
| `IdempotencyConflictError` | `RequestConflict` | n/a |

## Rules for implementers

1. Never hold a lock across a store call and a Neo4j call.
2. Never pass an exception object to a store. Only `exception_type`, `exception_code` and `diagnostic`.
3. Never log `key`, `statements`, `parameters`, or `WriteResult.exception` as text. Use `key_ref`.
4. Never call the helper unless the ledger claim and the journal `IN_FLIGHT` both committed.
5. Never call the journal's `authorize_retry` from an operator path. It's a dispatch step.
6. Never store a combined state.
7. A `BaseException` escaping the helper gets a best-effort `mark_ambiguous` and `record_uncertain`, then propagates.

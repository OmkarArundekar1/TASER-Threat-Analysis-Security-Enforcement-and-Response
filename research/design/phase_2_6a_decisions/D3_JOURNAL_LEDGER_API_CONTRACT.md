# D3: journal and ledger API contract

Phase 2.6A, design only. **Nothing here is implemented: the journal and ledger are unchanged.** Backend paths are relative to `isfcr/CYUKTI/backend/`.

**Labels:** **[V]** verified current behaviour, **[M]** measured evidence, **[P]** proposed behaviour, **[A]** assumption, **[D]** open decision, **[R]** requires production validation.

## Verdict

**AC-1, AC-2 and AC-3 are sufficient to link and fence the two stores, and the ledger needs no change for them.** They aren't sufficient for the whole coordinator, which also needs AC-6 and, with D6, a marker read (AC-8). This phase tightens Phase 2.5's versions with four rules:

1. Correlated records must always pass the attempt arguments. Omitting one is an error.
2. A record's ledger attempt can only rise.
3. A later epoch is allowed only after the earlier one was resolved as not committed.
4. A later epoch must carry byte-identical intent.

## AC-1: correlation

### Existing limitation

[V] `neo4j_write_journal.py:NeoWriteJournal.create_intent` L861-888 makes a new random `record_id` on every call and stores nothing that names the request. Nothing can find a record by business key, or stop a second record for one request. `neo4j_write_journal.py:NeoWriteJournal.mark_in_flight` L919-923 and `neo4j_write_journal.py:NeoWriteJournal.authorize_retry` L946-951 have no attempt argument.

### Method and fields

```python
def create_intent(self, operation, statements, *, return_rows=False,
                  correlation_key=None, epoch=None) -> JournalRecord
def find_by_correlation_key(self, key) -> tuple[JournalRecord, ...]   # oldest epoch first; () if none
def mark_in_flight(self, record_id, *, worker_id=None, ledger_attempt=None) -> JournalRecord
def authorize_retry(self, record_id, *, authorized_by, reason, worker_id=None, ledger_attempt=None) -> JournalRecord
```

| Item | Contract |
|---|---|
| New record columns | `correlation_key TEXT NULL`, `epoch INTEGER NULL`, `ledger_attempt INTEGER NULL` |
| New history column | `ledger_attempt INTEGER NULL`, on every event |
| `JournalRecord`, `JournalEvent` | Expose the new fields |
| Key format | The ledger key, unchanged. Exact `str`, 1 to 512 characters, no surrounding whitespace, no control characters or lone surrogates, the same rule as `neo4j_idempotency_ledger.py:_checked_key` L377-388. The journal doesn't import the ledger, so a test pins the two rules as equal |
| Epoch | Exact `int`, at least 1. **Not an attempt number:** see below |
| Both or neither | `correlation_key` and `epoch` come together, or `create_intent` raises `TypeError` |
| Standalone use | Both `None` keeps today's behaviour. Then `ledger_attempt` must also be `None` |
| Correlated use | `mark_in_flight` and `authorize_retry` **must** pass `ledger_attempt`, an exact `int` of at least 1. Omitting it is a `TypeError`, so forgetting can't pass silently |

### Epoch semantics

[P] An epoch is **one journal record's life for a request.**

- The first record is epoch 1.
- A new epoch exists only because the previous record ended `RESOLVED` with `FOUND_NOT_COMMITTED`, and a retry was authorized.
- **Enforced in the store,** in the same transaction as the insert: for epoch `n > 1`, epoch `n - 1` must exist, be `RESOLVED`, have resolution `FOUND_NOT_COMMITTED`, and have the same `operation` and `intent_sha256`. Otherwise it raises `InvalidTransitionError` and creates nothing.
- For `n = 1`, no other record may hold that key.
- **Why it's in the store:** the store can't see the ledger, but this check needs only the journal's own rows. It stops a coordinator bug from creating an epoch after a commit, or swapping in a different payload.

### Ledger attempt association

| Fact | Rule |
|---|---|
| A ledger attempt | Stored on the record when it enters `IN_FLIGHT`, and written to that history event |
| One intent, many attempts | Allowed. After a confirmed `NOT_COMMITTED`, the same record re-enters `IN_FLIGHT` by `authorize_retry` with a higher attempt. History keeps each one |
| One attempt, many intents | Never, because the attempt can rise only. The coordinator checks across epochs |
| Attempts with no journal entry | Expected. A claim whose `mark_in_flight` failed consumed a number, so **gaps are normal** |
| Rise-only | `ledger_attempt` must exceed the record's current value, or `InvalidTransitionError`, and nothing changes |
| Attempt count | `attempt_count` stays per record, and restarts at each epoch. `ledger_attempt` never restarts |

### Lookup and duplicates

| Case | Result |
|---|---|
| Lookup | `find_by_correlation_key(key)`, through a partial index, verified against the row hash like every read. An unknown key returns `()`, which is not an error |
| Damaged row | `JournalCorruptionError`, the same as other reads |
| Duplicate `(key, epoch)` | New `DuplicateCorrelationError(JournalError)`, raised with no change. It says the record exists, and contains no values |
| Coordinator's response | Re-reads and compares `intent_sha256`. Same: it reuses the record. Different: `CoordinationInconsistency`, and a person looks |
| Reuse of a key | Under the epoch rule only. There's no other way to use a key again |

### Other required properties

| Property | Contract |
|---|---|
| Uniqueness | `CREATE UNIQUE INDEX ... ON journal_records(correlation_key, epoch) WHERE correlation_key IS NOT NULL` |
| Persistence | In `_RECORD_COLUMNS`, so inside the row hash. Fsynced by the existing write path |
| Immutability | `correlation_key` and `epoch` never change, enforced by trigger |
| Integrity | `check_integrity` also checks both-or-neither, epoch at least 1, and the epoch chain |
| Compatibility | Schema version 2. [V] The journal refuses other versions (`neo4j_write_journal.py:NeoWriteJournal._validate_schema` L678-691). [A] No journal file exists in production, because nothing imports it; **confirm with a file search at implementation.** Then refuse v1 rather than write a migrator. The archived Phase 2.3 files stay as they are |
| Recovery | `recover_unfinished` keeps `ledger_attempt` on the `AMBIGUOUS` record and its history. It doesn't change otherwise |
| Concurrency | The existing `BEGIN IMMEDIATE` write path, plus the index. Losers get `DuplicateCorrelationError` |
| Logging | Never the key. A hash reference, as the ledger does (`neo4j_idempotency_ledger.py:_key_ref` L391-393) |

### Tests

These would change, and it's why this needs your approval: `tests/test_neo4j_write_journal.py` L75 (schema version), L424-429 (transitions, unchanged but re-read), L799 (exact public method set), and `tests/test_neo4j_write_journal_faults.py` L24-26, L390-391 (`_RECORD_COLUMNS`), L482-487 (version).

**New tests:**

| Area | Case |
|---|---|
| Round trip | Key, epoch and attempt survive reopen and appear in history |
| Both or neither | Each half alone raises and creates nothing |
| Duplicate | Same `(key, epoch)` raises and changes nothing, including under two real processes |
| Epoch chain | Each refused case: gap, previous committed, previous abandoned, previous unresolved, different intent |
| Attempt | Missing, lower, equal and wrong type all refused, and the record is unchanged |
| Standalone | All current journal tests pass with both `None` |
| Integrity | A tampered key or epoch is detected |
| Crash | A child killed around each new write leaves one consistent state. Reuse the Phase 2.3 harness and its no-op control |
| Key rule parity | The same list of bad keys is refused by both stores |

## AC-2: preflight

### Existing limitation

[V] The ledger accepts values the journal refuses, such as a `pytz` zone attached without `localize`. Only the private `neo4j_write_journal.py:_serialize_intent` L325-362 can test this. With the ledger writing first, a refused value could strand a registered request.

### Function

```python
def check_statements(operation, statements, *, return_rows=False) -> None
```

| Item | Contract |
|---|---|
| Placement | **A module function,** so it needs no open journal |
| Behaviour | Runs the validation `create_intent` runs, including the round-trip proof, and stores nothing |
| Errors | Exactly those `create_intent` raises for the same input: `ValueError`, `TypeError`, `UnsupportedParameterError` |
| Side effects | None: no file, no clock, no randomness, no lock |
| **What it doesn't guarantee** | **That `create_intent` will succeed.** The disk can be full, the database busy, or the journal closed. It says only that this value can be represented. The coordinator must still handle a later failure |
| Uniqueness, persistence, recovery | Not applicable, because nothing is stored |
| Compatibility | Additive. `create_intent` calls the same code, so they can't drift |
| Concurrency | Pure, so safe from any thread |

[M] **All 7,552 W07U payloads built from the sample pass the real serializer,** identically across two separate parses (`evidence/d3_d13_payload_analysis_output.txt`). Their `event_time` values are stdlib `timezone` objects, so the `pytz` gap doesn't arise on this path.

**Tests:** same accept and refuse decisions as `create_intent` over the Phase 2.3 value table; a directory-listing check that no file appears; the module's public names updated; a real value from each migrated writer.

## AC-3: attempt fencing on outcomes

### Existing limitation

[V] `neo4j_write_journal.py:NeoWriteJournal.mark_committed` L925-928, `neo4j_write_journal.py:NeoWriteJournal.mark_not_committed` L930-937 and `neo4j_write_journal.py:NeoWriteJournal.mark_ambiguous` L939-944 take no attempt. The ledger's outcomes do (`neo4j_idempotency_ledger.py:IdempotencyLedger._transition` L919-968). Since the journal hears first, a stale outcome could reach it first.

### Method and fields

```python
def mark_committed(self, record_id, *, expected_ledger_attempt=None) -> JournalRecord
def mark_not_committed(self, record_id, *, error_type=None, error_code=None, diagnostic=None,
                       expected_ledger_attempt=None) -> JournalRecord
def mark_ambiguous(self, record_id, *, error_type=None, error_code=None, diagnostic=None,
                   expected_ledger_attempt=None) -> JournalRecord
class StaleAttemptError(InvalidTransitionError): ...
```

| Item | Contract |
|---|---|
| Type | Exact `int`, at least 1 |
| Correlated record | **Required.** Omitting it is a `TypeError` |
| Standalone record | Must be `None`, or `TypeError` |
| Check | Inside `BEGIN IMMEDIATE`, against the row, before the state check, so there's no race |
| Mismatch | `StaleAttemptError`, with no change to row or history |
| `PENDING` to `NOT_COMMITTED` | [V] Allowed today. On a correlated record the row has no attempt yet, so **refused,** with `InvalidTransitionError`. A claim that never reached `IN_FLIGHT` touches only the ledger |
| Persistence | No new column. The attempt that committed is already in the history event |
| Coordinator | `StaleAttemptError`: log it and drop that outcome. It must never overwrite |

### Interaction with `RESOLVED`

| Situation | Result |
|---|---|
| Stale outcome, wrong attempt, record `RESOLVED` | `StaleAttemptError`. Unchanged |
| Outcome with the matching attempt, record `RESOLVED` | [V] `InvalidTransitionError`, since `RESOLVED` is final. Unchanged |
| Outcome to an old epoch after a new one exists | Refused the same way. The old record is `RESOLVED` |
| After `resolve_ambiguous` | The row's attempt remains, as history |

### How often this fires

[V] **Rarely.** `listener/wazuh_listener.py:queue_worker` L150-167 is one thread, and journal recovery leaves live sessions alone. Fencing is a guard against bugs and against a slow worker outliving a caller. It's still required, because without it such an outcome would land silently in the journal.

### Tests

Each outcome with wrong, lower, higher, missing and mistyped attempts; a stale outcome on `RESOLVED`; on an old epoch; two real processes racing on one record; all leaving row and history byte-identical; and all current standalone tests unchanged.

## Other API changes

| ID | Change | Why | Status |
|---|---|---|---|
| AC-4 | `evidence` on resolve | With a marker, evidence becomes a real thing to record. `reason` holds 500 characters, which is enough for a marker result | Optional. **Leave out** |
| AC-5 | `abandon_registered` on the ledger | [V] A `REGISTERED` request can never be closed. It's harmless | Optional. **Leave out** |
| AC-6 | Source identity reaches writers | Required for any writer. **For W07U it's only local:** the alert is already an argument of `realtime_socgraph.py:_process_unattributed_alert` L170-286 | Required |
| AC-7 | Derived ids | Required for W01, W05, W06, W16. **Not W07U** | Required for those |
| **AC-8, new** | `read_applied_write(request_key)` in `neo4j_client.py` | Reads a marker, and returns `APPLIED`, `ABSENT` or `UNKNOWN`. See `D6_APPLIED_WRITE_MARKER_DESIGN.md` | Required if D6 is approved |
| **AC-9, new, optional** | Transaction metadata on the helper's `unit_of_work` | [V] It sets only a timeout (`neo4j_client.py:_execute_write` L289-360), so `SHOW TRANSACTIONS` can't identify a request's open transaction. Needed only to make absence stronger evidence | Open. **Not for the pilot** |

[V] **The ledger needs no change** for AC-1 to AC-3, and neither does the helper's result, which already carries `exception_code` and counters.

## If rejected

Crash recovery can't link the stores, so a coordinator can't be built on these two files. The alternative is a third database, which adds a store and more failure windows.

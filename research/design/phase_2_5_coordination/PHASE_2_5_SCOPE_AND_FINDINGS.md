# Phase 2.5 scope and findings

Investigation and design only. Inspected at HEAD `3afe62c` plus the uncommitted working tree. Backend paths are relative to `isfcr/CYUKTI/backend/`.

**Nothing in this folder is implemented.** No production code, test, journal, ledger or helper changed.

**Labels:** **[V]** verified current behaviour, **[P]** proposed behaviour, **[D]** open decision, **[A]** assumption requiring verification, **[L]** known limitation.

## Reading order

| Document | Purpose |
|---|---|
| This one | What exists, what was found, and the decisions needed |
| `PRODUCTION_NEO4J_WRITER_INVENTORY.md` | All 20 production writers and 14 exclusions |
| `REQUEST_IDENTITY_MAPPING.md` | What one logical request is |
| `JOURNAL_LEDGER_COORDINATION_CONTRACT.md` | The central design: ownership, mapping, ordering, retry policy, diagrams |
| `CROSS_COMPONENT_STATE_TRANSITIONS.md` | The 21-scenario matrix |
| `FAILURE_WINDOW_AND_RECOVERY_ANALYSIS.md` | Every crash window, and the recovery algorithm |
| `COORDINATOR_INTERFACE_PROPOSAL.md` | Signatures, errors, required API changes |
| `PHASE_2_6_IMPLEMENTATION_AND_VALIDATION_PLAN.md` | Build order, test matrix, acceptance criteria |
| `EVIDENCE_INDEX.md` | Where every claim comes from |

## Main findings

1. **[V] There are 20 production writers, none protected, and none using the helper.** Almost all run from `process_alert`, with up to 13 separate writes per alert and no shared transaction.
2. **[V] The obvious identity anchor doesn't work.** `alert_id` never reaches the writers, and it collides: 70 of 9,015 sample alerts share one while their Wazuh `id` and content differ. Wazuh's own `id` was unique for all 9,015.
3. **[V] That same collision looks like an existing alert-loss issue,** independent of this design: the listener would discard those alerts as repeats (D2).
4. **[V] Random ids inside writers break fingerprints.** `event_id`, `campaign_id` and `operation_id` are made inside the writers, so re-running one looks like a conflicting payload. Those writers can't migrate until ids are derived from the key (D7).
5. **[V] The journal and ledger aren't interchangeable,** and three differences force coordinator rules:
   - The journal's `RESOLVED` is final, while the ledger can run again after `FOUND_NOT_COMPLETED`. So one request can need **more than one journal intent.**
   - The journal has no attempt fence on outcomes, and the ledger does. So a stale outcome could reach the journal first unless the coordinator stops it (AC-3).
   - Neither store can find the other's record. **A correlation field is required** (AC-1).
6. **[V] The ledger accepts payloads the journal refuses,** for example a `pytz` zone attached without `localize`. With the ledger going first, that could strand a request with no way to end it, so a preflight check is required (AC-2).
7. **[P] One rule resolves the ordering: close the gate first, and open it last.** The ledger claims before anything runs, and it records outcomes only after the journal has. So a crash at any point can only leave a request less runnable, never more.
8. **[P] An applied-write marker in the same Neo4j transaction would be the only real evidence of commit.** Without it, every crash after dispatch needs a person. It's optional and needs your approval (D6).

## Component behaviour

### Write helper

Verified in `neo4j_client.py`. No production writer calls it yet.

| Question | Verified behaviour | Source |
|---|---|---|
| Submitting a write | `write_statements`, `write_query` or `write_query_rows`, with statements, operation name and optional `WriteLimits` | `neo4j_client.py:write_statements` L382-393, `neo4j_client.py:write_query` L396-400, `neo4j_client.py:write_query_rows` L403-409 |
| One attempt | One run of the unit of work inside `session.execute_write`. `WriteResult.attempts` counts them | `neo4j_client.py:_execute_write` L289-360 |
| Commit | Only when `execute_write` returns normally | `neo4j_client.py:_execute_write` L289-360 |
| Which failures are `NOT_COMMITTED` | Invalid arguments (0 attempts); any driver or server error before COMMIT was sent; a retry budget stop; `Transaction.Terminated` in reply to COMMIT; an unknown error with 0 attempts | `neo4j_client.py:_classify_write_failure` L221-241 |
| Which are `AMBIGUOUS` | `IncompleteCommit`; any other failure after a commit attempt, including `TransactionTimedOutClientConfiguration`; a driver re-run refused after a commit attempt; an unknown error after any attempt | `neo4j_client.py:_classify_write_failure` L221-241 |
| Server-confirmed or client-side? | `exception_code` is set for server errors. The outcome doesn't separate them beyond the rules above | `neo4j_client.py:WriteResult` L183-208 |
| Automatic retries | Before COMMIT: yes, inside the driver's loop, within `max_retry_time`. After COMMIT is sent: refused | `neo4j_client.py:_execute_write.work` L318-339 |
| What callers get | `WriteResult`: operation, outcome, attempts, duration, counters, exception type and code, safe diagnostic, the original exception (not safe to log), records | `neo4j_client.py:WriteResult` L183-208 |

[L] A silent stalled read has no client-side timeout (`neo4j_client.py:WriteLimits` L111-176).
[V] An empty batch raises `ValueError`. Everything else below `BaseException` returns a result. `BaseException` propagates.

### Durable recovery journal

Verified in `neo4j_write_journal.py`.

| Topic | Verified behaviour | Source |
|---|---|---|
| Intent creation | Generates a new random `record_id` each call. **Nothing makes two calls for the same request the same record** | `neo4j_write_journal.py:NeoWriteJournal.create_intent` L861-888 |
| Serialization | Exact tagged JSON with a proved round trip. Exact types only | `neo4j_write_journal.py:_serialize_intent` L325-362 |
| Checksums | SHA-256 over the intent and the row, plus a history chain | `neo4j_write_journal.py:NeoWriteJournal._verified` L782-787 |
| Transitions | PENDING to IN_FLIGHT or NOT_COMMITTED. IN_FLIGHT to COMMITTED, NOT_COMMITTED or AMBIGUOUS. NOT_COMMITTED to IN_FLIGHT by retry only. AMBIGUOUS to RESOLVED | `neo4j_write_journal.py` L115-123 |
| Retry | `authorize_retry` moves straight to IN_FLIGHT and counts an attempt | `neo4j_write_journal.py:NeoWriteJournal.authorize_retry` L946-951 |
| Recovery | IN_FLIGHT becomes AMBIGUOUS only when the owner's file lock is free. A record of its own session is left alone | `neo4j_write_journal.py:NeoWriteJournal.recover_unfinished` L960-1023 |
| Integrity | File, schema, row and chain checks; append-only through triggers | `neo4j_write_journal.py:NeoWriteJournal.check_integrity` L1025-1067 |
| Can establish | What was intended, exactly, and what was reported | |
| Can't establish | Whether Neo4j committed. `AMBIGUOUS` can mean committed, not committed, or never started | Module docstring |

### Idempotency ledger

Verified in `neo4j_idempotency_ledger.py`.

| Topic | Verified behaviour | Source |
|---|---|---|
| Key | Caller-supplied, exact match, never generated | `neo4j_idempotency_ledger.py:_checked_key` L377-388 |
| Fingerprint | Versioned canonical SHA-256, per-statement digests, no values stored | `neo4j_idempotency_ledger.py:fingerprint_payload` L320-356 |
| Duplicate | `DUPLICATE`, state unchanged, grants nothing | `neo4j_idempotency_ledger.py:IdempotencyLedger.register` L972-1012 |
| Conflict | Audited, then raises. Existing request untouched | `neo4j_idempotency_ledger.py:IdempotencyLedger._record_conflict` L1014-1032 |
| Attempts | Incremented only when `begin_execution` claims | `neo4j_idempotency_ledger.py:IdempotencyLedger._transition` L919-968 |
| Transitions | REGISTERED or RETRY_AUTHORIZED to IN_PROGRESS. IN_PROGRESS to COMPLETED, NOT_COMPLETED or UNCERTAIN. NOT_COMPLETED to RETRY_AUTHORIZED. UNCERTAIN to COMPLETED, NOT_COMPLETED or ABANDONED by resolution only | `neo4j_idempotency_ledger.py` L142-152 |
| Retry | `authorize_retry` grants one attempt and runs nothing | `neo4j_idempotency_ledger.py:IdempotencyLedger.authorize_retry` L1125-1130 |
| Stale outcomes | `attempt` is required, and a mismatch raises `StaleAttemptError` | `neo4j_idempotency_ledger.py:IdempotencyLedger._transition` L919-968 |
| Recovery | IN_PROGRESS becomes UNCERTAIN when the owner's lock is free | `neo4j_idempotency_ledger.py:IdempotencyLedger.recover_unfinished` L1140-1205 |
| Can establish | Which requests exist, whether a payload matches, and who may run | |
| Can't establish | Whether Neo4j committed. `COMPLETED` records what was reported | Module docstring |

### API compatibility matrix

Similar names don't mean the same thing.

| Purpose | Journal | Ledger | Equivalent? |
|---|---|---|---|
| Record a request | `create_intent`: always a new record, stores the payload | `register`: finds or creates by key, stores no values | **No** |
| Check without writing | None | `check` | Ledger only |
| Start | `mark_in_flight(record_id)`: no fingerprint check | `begin_execution(key, fingerprint)`: atomic claim with fingerprint check | **No:** only the ledger's is a gate |
| Confirmed commit | `mark_committed(record_id)`: **no attempt argument** | `record_completed(key, attempt=)`: fenced | **No** |
| Confirmed non-commit | `mark_not_committed`: also allowed from PENDING | `record_not_completed(key, attempt=)`: IN_PROGRESS only | **No** |
| Unknown | `mark_ambiguous` | `record_uncertain(key, attempt=)` | Same meaning, different fencing |
| Retry | `authorize_retry`: **goes to IN_FLIGHT, counts an attempt** | `authorize_retry`: **only grants permission** | **No, despite the same name** |
| Resolve unknown | `resolve_ambiguous`: to final `RESOLVED` | `resolve_uncertain`: to `COMPLETED`, `NOT_COMPLETED` or `ABANDONED` | **No** |
| Resolution values | `FOUND_COMMITTED`, `FOUND_NOT_COMMITTED`, `ABANDONED` | `FOUND_COMPLETED`, `FOUND_NOT_COMPLETED`, `ABANDONED` | Map one to one; different effects |
| Recover | `recover_unfinished`: `AMBIGUOUS` | `recover_unfinished`: `UNCERTAIN` | Same meaning |
| Integrity | `check_integrity` | `check_integrity` | Equivalent, separately |
| Attempt count starts | At `mark_in_flight` or `authorize_retry` | At `begin_execution` | Drift after a new journal intent |
| Find by business key | **Not possible** | `get_request(key)` | **No: gap (AC-1)** |
| Payload types refused | Includes a `pytz` zone attached without `localize`, and `time` with a `pytz` zone | Accepts those | **No: gap (AC-2)** |

## Required design decisions

Each one changes identity, retry semantics, recovery authority or production behaviour, so **none is closed here.**

### D1: What is a request's source identity?

- **Options:** (a) `alert_id`. (b) Wazuh `alert["id"]`, with a raw-content hash as fallback. (c) Raw-content hash only.
- **Trade-offs:** (a) exists but collides. (b) is unique in the sample but its production behaviour is an assumption. (c) is stable but merges identical repeated alerts.
- **Recommended:** (b).
- **Why:** 9,015 of 9,015 unique in the sample, against 70 collisions for (a).
- **If deferred:** no writer can migrate.
- **Your approval required:** yes. Please also confirm on production-shaped data.

### D2: Should the `alert_id` collision be fixed?

- **Options:** (a) Leave it. (b) Include Wazuh `id` in `generate_alert_id`. (c) Investigate separately.
- **Trade-offs:** (a) keeps a possible silent loss. (b) is small but changes listener behaviour.
- **Recommended:** (c), then (b). It's a separate defect from this design.
- **If deferred:** the listener may keep dropping distinct alerts, and the coordinator would never see them.
- **Your approval required:** yes. Listener changes aren't authorized.

### D3: How are journal and ledger records correlated?

- **Options:** (a) Add `correlation_key` and attempt fields to the journal, as a schema change. (b) A third coordinator database. (c) Encode them in `operation` or `worker_id`.
- **Trade-offs:** (a) changes Phase 2.3 and needs a migration. (b) leaves those files alone but adds a third store and more failure windows. (c) changes nothing but is fragile, requires scanning, and mixes meanings.
- **Recommended:** (a), as AC-1 and AC-3.
- **Why:** fewest stores and windows, and a real uniqueness constraint.
- **If deferred:** crash recovery can't link the records, so implementation can't start.
- **Your approval required:** yes. It changes a completed phase.

### D4: How does a retry work after a journal record is `RESOLVED`?

- **Options:** (a) A new journal intent, linked by key. (b) Change the journal to allow `RESOLVED` to retry.
- **Trade-offs:** (a) keeps the journal's documented rule and needs an epoch counter. (b) is simpler to track but reverses a Phase 2.3 decision.
- **Recommended:** (a).
- **Your approval required:** yes.

### D5: Who authorizes a retry after a confirmed `NOT_COMMITTED`?

- **Options:** (a) Always a person. (b) A named policy principal for clearly safe cases, such as a connection failure before dispatch.
- **Trade-offs:** (a) is safest, but queues build up during an outage. (b) is more available, with a real risk if classification is wrong.
- **Recommended:** (a) first, then (b) after evidence.
- **If deferred:** (a) by default.
- **Your approval required:** yes.

### D6: Add Neo4j-side evidence of commit?

- **Options:** (a) None: every post-dispatch crash needs a person. (b) An applied-write marker in the same transaction, for reconciliation only. (c) (b) plus a uniqueness constraint, so a second execution fails in Neo4j.
- **Trade-offs:** (a) changes nothing in the graph, with a heavy operator burden. (b) adds nodes and a statement to every migrated write. (c) gives real server-side duplicate protection, but it needs a constraint and a decision on whether marker evidence may resolve automatically.
- **Recommended:** (c) for Tier 1 writers, with resolution still manual at first.
- **Why:** it's the only design here that gives direct evidence, because a marker commits atomically with its write.
- **If deferred:** recovery stays conservative and correct but manual.
- **Your approval required:** yes. It changes the production graph schema.

### D7: Derive ids from the key?

- **Options:** (a) Derive `event_id`, `campaign_id` and `operation_id` from the key. (b) Allocate once and carry them. (c) Keep random ids and don't migrate those writers.
- **Trade-offs:** (a) makes re-delivery produce an identical payload, and it changes the id format. (b) survives a journaled retry but not re-delivery. (c) leaves the highest-risk writers unprotected.
- **Recommended:** (a).
- **Your approval required:** yes. Dashboard and evidence consumers see ids.

### D8: What identifies a duplicate batch (W08)?

- **Options:** (a) Keep contributing source ids in the buffer, with one request per alert. (b) A batch id allocated at snapshot and persisted before dispatch. (c) Remove the buffer for migrated paths.
- **Trade-offs:** (a) has strong identity but more memory and writes. (b) protects a batch against its own retry but not against re-delivered alerts, and restore-and-merge must stop for that batch. (c) is simplest but loses the throughput the buffer provides.
- **Recommended:** (b) to begin, with restore replaced by the journal for those batches. Your earlier ambiguous-flush decision applies here.
- **If deferred:** W08, the reproduced double count, stays unprotected.
- **Your approval required:** yes.

### D9: Are maintenance sweeps (W19, W20) ledgered?

- **Options:** (a) No: helper only. (b) Yes, with a key per watermark.
- **Trade-offs:** (a) is simple, and the writes are convergent. (b) would add a request every 5 s, for no gain.
- **Recommended:** (a).
- **Your approval required:** yes, because it leaves two writers outside the contract.

### D10: Does a crashed alert resume?

- **Options:** (a) No: each request stands alone, and later steps of that alert are lost, as today. (b) An alert-level workflow record. (c) Offset integration, so the alert is re-read.
- **Trade-offs:** (a) is honest but leaves gaps. (b) is a large design. (c) needs (b)-like care, because re-running decisions over mutable state may route differently.
- **Recommended:** (a) for Phase 2.6, then design (c) separately.
- **If deferred:** alerts in the queue at a crash stay lost. **This coordinator doesn't fix that.**
- **Your approval required:** yes.

### D11: Which tiers go through the full coordinator?

- **Options:** (a) All. (b) Tier 1 and 2, with Tier 3 through the helper only. (c) Decide after measuring.
- **Trade-offs:** (a) adds several fsyncs to up to 13 writes per alert. (b) leaves convergent writes without a durable record, though they repair themselves.
- **Recommended:** (c), starting from (b).
- **[A]** The cost is unmeasured. Each request needs several durable SQLite commits.
- **Your approval required:** yes.

### D12: Retention and backup

- **Options:** (a) Keep everything. (b) Archive completed requests after a period, never deleting. (c) Allow deletion.
- **Trade-offs:** (a) grows without limit. (b) needs archive design. (c) makes old requests executable again, so it undermines the ledger.
- **Recommended:** (a) until growth is measured, then (b). Never (c).
- **Your approval required:** yes.

### D13: First migration candidate

- **Recommended:** W07 on the unattributed path. Justified in `PRODUCTION_NEO4J_WRITER_INVENTORY.md`.
- **Your approval required:** yes.

### D14: What happens if a write can't get a key during migration?

- **Options:** (a) Refuse the write. (b) Fall back to the current unprotected path, with a logged warning.
- **Trade-offs:** (a) is strict, but can lose detection data. (b) keeps data flowing, unprotected.
- **Recommended:** (b) only during shadow and rollout stages, then (a).
- **Your approval required:** yes.

### D15: What happens when the coordinator's stores are unavailable?

- **Options:** (a) Stop the write and don't acknowledge. (b) Fall back to an unprotected write.
- **Trade-offs:** (a) keeps the safety rule, but a full disk stops ingestion for migrated writers. (b) keeps ingestion, and silently removes the protection.
- **Recommended:** (a). Your Phase 2.3 rule says a journal failure must never lead to acknowledging or discarding the write.
- **Your approval required:** yes, because it has an availability cost.

## Blockers to implementation

- D1, D3, D4 and D13 must be decided before Phase 2.6 can start.
- D7 must be decided before any id-creating writer migrates.
- D8 must be decided before W08 migrates.

## Scope notes

- [L] This design protects individual write requests. It doesn't make `process_alert` atomic, and it doesn't cover MISP, Shuffle or in-memory caches.
- [L] It gives no exactly-once guarantee. See `FAILURE_WINDOW_AND_RECOVERY_ANALYSIS.md`.
- [L] No test was run for this phase, and no server was started. One read-only measurement ran over a repository sample file.

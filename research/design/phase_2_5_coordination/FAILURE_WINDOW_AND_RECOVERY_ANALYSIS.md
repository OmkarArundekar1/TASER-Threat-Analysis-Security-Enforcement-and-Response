# Failure windows and recovery

Phase 2.5, design only. **Nothing here is implemented.** Everything is **[P] proposed** unless labelled. Backend paths are relative to `isfcr/CYUKTI/backend/`.

## Why SQLite and Neo4j can't commit together

An atomic commit across two systems needs both to support a shared protocol, such as two-phase commit: each promises it can commit, and then a coordinator tells both to do so. That isn't available here:

- **[V] Bolt has no prepare step.** A transaction is begun, run and committed. Nothing can hold it "ready" while SQLite commits.
- **[V] SQLite has no such protocol with anything outside itself.** A commit is local.
- **[A] Even the two SQLite files can't commit together.** SQLite documents that a transaction across attached databases isn't atomic when they use WAL mode, and both files use WAL. [V] So journal and ledger writes are separate commits too.

So there's always an instant between "Neo4j committed" and "the journal knows". A crash there can't be told apart, from the records alone, from a crash just before the commit. **That's why this design can't offer exactly-once execution,** and why uncertain outcomes go to a person.

## Failure windows

Each window is the gap between two durable steps.

| # | Window | What's known | What's unknown | Safe recovery |
|---|---|---|---|---|
| 1 | Before ledger registration | Nothing recorded | Whether the alert existed at all | [L] None possible here. The alert is lost if the process dies, since the offset has moved (D10) |
| 2 | Between registration and journal intent | Identity and fingerprint. No payload stored | Nothing about Neo4j: it can't have run | Reported as `REGISTERED_NO_INTENT`. **It can't run, because there's no payload.** A resubmit from the source continues it |
| 3 | Between intent and claim | Identity and exact payload. Not started | Nothing: it can't have run | Reported as registered and pending. [D] Whether to resume is D10; the default is not to |
| 4 | Between claim and journal `IN_FLIGHT` | Claimed. Journal still `PENDING` | Nothing about Neo4j: by the ordering rule the helper can't have run | Ledger goes `UNCERTAIN` by recovery, journal stays `PENDING`. Reported as `CLAIMED_NOT_DISPATCHED`. Needs a person, whose evidence is that combination |
| 5 | Between journal `IN_FLIGHT` and invoking Neo4j | Both say running | **Whether it reached Neo4j** | Both uncertain. Records alone can't separate this from window 6 or 7 |
| 6 | During the transaction | Both say running | **Whether it committed** | Both uncertain |
| 7 | After Neo4j commit, before the journal records it | Both say running | **That it committed.** It did, but nothing local shows it | Both uncertain. Only a query of Neo4j can tell |
| 8 | Between journal and ledger outcome writes | Journal holds the confirmed outcome | Nothing: the journal has it | **Automatic:** recovery finishes the ledger with `record_*` at the matching attempt. It's not a replay, and nothing runs |
| 9 | During recovery | Whatever each finished transaction recorded | Nothing new | Run recovery again. Each step is its own transaction and idempotent |
| 10 | Between resolution writes | Journal `RESOLVED`, ledger still `UNCERTAIN` | Nothing | Reported as `RESOLUTION_INCOMPLETE`. Still not runnable. Repeating the same resolve finishes it |
| 11 | Between authorization and dispatch | `RETRY_AUTHORIZED`, journal unchanged | Nothing: it hasn't run | The authorization stays valid and unused. [V] Phase 2.4 tests this survives a crash |
| 12 | Between a new-epoch intent and its claim | New journal record `PENDING`, ledger `RETRY_AUTHORIZED` | Nothing | The next dispatch finds the record by key and epoch and reuses it (AC-1) |

### Evidence strength

| Records show | Can run again? | Reasoning |
|---|---|---|
| Windows 1 to 3, 11, 12 | Yes, through the normal path | There was no claim, or the claim was explicitly authorized |
| Window 4 | Only after resolution | The ordering rule implies it never ran, but that relies on the code being correct, so a person confirms |
| Windows 5 to 7 | Only after resolution and authorization | Genuinely unknown |
| Window 8 | Never: it's finished | Confirmed outcome |

## Recovery algorithm

Runs at startup, before workers begin. It can repeat at any time.

**It never calls the helper, and never runs a query against Neo4j.**

1. **Open both stores.** If either refuses, stop and report. Nothing else proceeds.
2. **Check integrity of both.** Exclude damaged records from every step below and report them.
3. **Finish the ledger from confirmed journal outcomes.** For each journal record whose latest `IN_FLIGHT` was tagged with ledger attempt n, where the ledger is `IN_PROGRESS` at attempt n:
   - `COMMITTED` becomes `record_completed(attempt=n)`.
   - `NOT_COMMITTED` becomes `record_not_completed(attempt=n)`.
   - `AMBIGUOUS` becomes `record_uncertain(attempt=n)`.
4. **Convert dead owners' journal records.** Call `journal.recover_unfinished()`. [V] It converts `IN_FLIGHT` to `AMBIGUOUS` only when the owner's lock is free, and leaves live owners alone.
5. **Mirror those into the ledger.** Repeat step 3 for what step 4 just produced.
6. **Convert what's left in the ledger.** Call `ledger.recover_unfinished()`. [V] Same lock rule. This catches window 4.
7. **Classify and report everything else,** including `INCONSISTENT` combinations. Change nothing for those.

**Why step 3 comes before step 6:** the ledger's `recover_unfinished` would turn a request whose commit the journal has already confirmed into `UNCERTAIN`, and then a person would have to resolve something already known.

**Racing a live worker:** a live worker may be between its journal and ledger writes while step 3 runs. One of the two writes then gets `InvalidTransitionError`, which the coordinator treats as already done after re-reading the state.

**Repeating recovery changes nothing further.** [V] Both stores' recovery is idempotent, and step 3 only acts on `IN_PROGRESS`.

### Recovery report

| Category | Meaning | Action |
|---|---|---|
| `RECONCILED` | Ledger finished from a confirmed journal outcome | None |
| `NEWLY_UNCERTAIN` | Became uncertain in this run | Resolve |
| `UNCERTAIN` | Already uncertain | Resolve |
| `CLAIMED_NOT_DISPATCHED` | Window 4 | Resolve, using that evidence |
| `RESOLUTION_INCOMPLETE` | Window 10 | Repeat the resolve |
| `REGISTERED_NO_INTENT` | Window 2 | Resubmit from the source, or leave |
| `PENDING` | Window 3 | [D] D10 |
| `AWAITING_AUTHORIZATION` | `NOT_COMPLETED` | Authorize, or leave |
| `RETRY_AUTHORIZED` | Unused authorization | Dispatch when wanted |
| `STILL_RUNNING` | Live owner | None |
| `CORRUPT` | Failed integrity | Investigate. Never repaired |
| `INCONSISTENT` | A combination that shouldn't arise | Investigate |

## Failures that aren't crashes

| Failure | What happens | Why |
|---|---|---|
| Journal full or unwritable before dispatch | The write isn't sent, and the caller gets an error (D15) | [V] The Phase 2.3 rule: never acknowledge or discard because of a journal failure |
| Journal write fails after a confirmed commit | **Retry that journal write,** bounded. If it still fails, the caller is told `COMMITTED_UNRECORDED` and must not treat the record as settled. Recovery later marks it uncertain | Retrying a local write isn't a Neo4j retry. Reporting it is honest. **Nothing re-executes** |
| Ledger write fails after the journal recorded | Retry, bounded. Otherwise it's left for recovery step 3 | The journal holds the truth |
| Helper never returns (silent stalled read) | Request stays `IN_PROGRESS` and `IN_FLIGHT`, and claims are refused | [L] No client-side timeout exists. Only process death frees it |
| Worker thread outlives its caller | Nothing changes. It records its own outcome when it finishes | A timeout concludes nothing |
| Store locked for too long | [V] `*BusyError`, no change. Treated as a store failure | No state changed |
| Damaged record | Excluded and reported | [V] Both refuse to transition it |

## What recovery can't do

- [L] Tell windows 5, 6 and 7 apart without a query of Neo4j.
- [L] Recover an alert that never reached `register`.
- [L] Recover the later steps of a partly processed alert (D10).
- [L] Repair damaged stores.
- [L] Survive power loss with certainty: that rests on fsync being honoured, which hasn't been tested directly.
- [L] Detect that a store was restored from an old backup, except through `INCONSISTENT` combinations. A restored pair can look consistent while being out of date.

## Reconciling against Neo4j, if D6 is adopted

[P] Without D6, an operator has to judge from the graph: a deterministic `event_id` exists or it doesn't (W05), or a counter "looks right" (W07), which is often unprovable.

[P] With D6, each migrated transaction also writes `(:AppliedWrite {request_key, attempt})`. **It commits atomically with the write,** so its presence is direct evidence.

| Marker query, after the owner died and all its transactions ended | Means |
|---|---|
| Found | The write committed |
| Not found | [A] The write didn't commit, provided no transaction of that attempt could still be open |

- [A] "Could still be open" needs care. A dead client's transaction can outlive it for a while. Phase 2.2.1R saw one gone within about 1 s after a killed connection, and `transaction_timeout` bounds one that's running, but that's limited evidence. The waiting rule needs real-server testing.
- [P] A uniqueness constraint on `request_key` would also make a wrongly authorized second execution **fail in Neo4j,** which classifies as `NOT_COMMITTED`.
- [D] Whether marker evidence may resolve anything automatically is part of D6. The recommended start is that it only informs the operator.

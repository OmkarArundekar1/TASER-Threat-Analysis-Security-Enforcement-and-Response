# Phase 2.7 planning report

**Nothing was implemented.** No production file, service or database was touched. Backend paths are relative to `isfcr/CYUKTI/backend/`.

**Labels:** **[F]** fact from source, tests or Phase 2.6 evidence, **[M]** model output, **[R]** recommendation, **[H]** hypothesis, **[U]** unverified.

## Executive summary

**The coordinator's design is ready to implement, starting with a milestone that touches nothing in production. The throughput design is implementable, but its capacity numbers are placeholders until `process_alert` is measured.**

| Workstream | Conclusion | Status |
|---|---|---|
| **A. Lifecycle** | Fully specified. No state in the coordinator, so no second source of truth | **Ready** |
| **B. Identity and payload** | Each case handled. D13 and stale payload have treatments, each with an experiment or test outstanding | **Ready, with X2 and T-16 outstanding** |
| **C. Throughput** | **Approach A: bounded durable backlog.** It also closes 2.6C's worst gap | **Design ready. Numbers aren't** |
| **D. Timeouts and locks** | 8 s per transaction, on the helper. No production setting changes | **Ready, except X1** |
| **E. Marker rollout** | Steps, refusals, inconsistency table and rehearsal defined | **Ready** |
| **F. Security** | Specified | **Ready** |
| **G. Work packages** | 11 packages, about 36 days, each with acceptance tests | **Ready** |

**The most important finding:** [F] traffic isn't high on average. The largest day averages 4.4 alerts a second, against 11,566 in one event-time second, and the median day holds 198 alerts. So this is a problem of **absorbing rare bursts safely,** which is what a durable backlog does, and batching doesn't.

**The most important limit:** nobody has measured `process_alert`. [H] A rough sum suggests 20 to 40 a second if every alert were coordinated. **That could change which writers are worth coordinating.**

**Errors I caught in my own drafts,** by checking them against source before finishing:

| Draft said | Actually | Effect |
|---|---|---|
| A person can abandon a conflicted request | [F] The ledger reaches `ABANDONED` only from `UNCERTAIN` | New OD-12 |
| The CAS template stood alone | [F] The real statement also has Technique and Campaign clauses | Template noted |
| 96.8% applied to the largest day | [F] That's 26 September, for the old key | Sentence corrected |

## Existing implementation reused

| Component | Status | Hash prefix |
|---|---|---|
| `neo4j_write_journal.py` | Reused unchanged | e6f6ed1bdaf04948 |
| `neo4j_idempotency_ledger.py` | Reused unchanged, except OD-12 if approved | 8f524667d4a4e34e |
| `neo4j_client.py` write helper | Reused unchanged. Reads are additive | 80032046c634c815 |
| `listener/utils.py` keys | Reused. OD-13 is separate | 2657321029162810 |
| Phase 2.6D tooling and `marker_lib.py` | A basis for tests | |

## Confirmed design decisions

These are consequences of what's already settled, with no new choice involved:

1. Ledger then journal, on the way in. Journal then ledger, on the way out.
2. The ledger assigns attempts, and the journal stores and fences them.
3. `NOT_COMMITTED` retries use the same record. `FOUND_NOT_COMMITTED` makes the next epoch.
4. Marker and mutation share one statement. Absence resolves nothing. `ConstraintValidationFailed` alone proves nothing.
5. Marked writes stay on the helper, with explicit timeouts. No production settings change.
6. Marker mode is refused without the verified constraint.
7. No automatic retry and no automatic resolution.

## Proposed design decisions

New in this phase, and **yours to approve:**

| Proposal | Why |
|---|---|
| Registered templates, with statements out of `WriteRequest` | A caller can't forget the marker |
| Two fingerprints: mutation and request | The marker can't hash itself |
| `ALREADY_APPLIED` as journal `NOT_COMMITTED`, ledger `COMPLETED` | Both are true. It adds one legal pair |
| Re-delivery doesn't spend `RETRY_AUTHORIZED` | An authorization is for `retry`, with the stored payload |
| A durable backlog, with `UNIQUE(source_key)` | Throughput, D13, and 2.6C's crash and unusable-offset gaps, from one mechanism. The rotation and UTF-8 gaps need the reader changes beside it |
| Interrupted alerts are held, and never re-run automatically | Most writers aren't idempotent |
| Compare-and-set on `payload_event_time` | Stale payload |

## Unresolved decisions

Full list in `OPEN_DECISIONS_AND_BLOCKERS.md`. **The ones that matter most:**

| ID | Question | Recommendation |
|---|---|---|
| D5, D14, D15 | Needed to start WP-C1 | Person only; refuse and count; hold |
| OD-1 | Approach A, reversing D10's "not for the pilot" | **Yes** |
| OD-3 | Alerts interrupted by a crash | Hold for a person |
| OD-4 | During an outage, hold everything? | **Yes,** and detection is delayed for that time |
| OD-5 | Rotation-aware reader | **Yes** |
| OD-8 | Stale-payload CAS | Yes, after X2 |

**Blockers,** none of which stops WP-C1: BL-1 `process_alert` unmeasured, BL-2 arrival rate unknown, BL-3 lock-wait limit unproven, BL-4 CAS order untested, BL-5 and BL-6 production-scale behaviour, BL-7 possible shortfall in drain rate.

## Production compatibility implications

[F] From 2.6E. **Matching the tested version doesn't prove production behaviour,** so:

| Fact | Implication |
|---|---|
| Neo4j 5.26.27 Community, single server | Same as 2.6D. Startup checks version and edition, and refuses otherwise |
| Ten uniqueness constraints | `ConstraintValidationFailed` is never read alone |
| `AppliedWrite` doesn't exist | Creating it is a separate, approved operator step. No existing node conflicts |
| Transaction and lock timeouts both `0s` | Explicit timeouts on every marked write, and X1 for locks |
| `manager.name` stable, `id` form holds in 681,259 | Keys hold. OD-13 keeps tolerance |
| 110 rotations, some gzip | Rotation-aware reader |
| 11.8% more alerts under the fixed key | Part of what the drain rate must meet |

## Throughput recommendation, and its evidence limits

**[R] Approach A.** Comparison in `THROUGHPUT_AND_BACKPRESSURE_DESIGN.md`.

| Evidence | Limit |
|---|---|
| [M] Backlog model: at 50 a second, the busiest sample day peaks at 4,893 and clears in 99 s | **One day.** Event time stands in for arrival |
| [F] 11,566 in one second | **Event time,** not confirmed arrival |
| [M] 421 MB of text for the largest day | Mean size from a sample. SQLite overhead unmeasured |
| [F] Coordinated parts: 6.5 to 7 ms and 9.2 ms | Disposable server, small graph. **Ledger on this path unmeasured** |
| **Nothing on `process_alert`, or on B** | **I make no claim that batching is faster or slower** |

**Placeholders:** 1,000,000 rows, 2 GiB, 50 a second, 30 minutes. Each is labelled, and B1 replaces them.

**The accepted cost:** in a burst, detection is delayed rather than lost. You decide how much delay is tolerable, under OD-2.

## Exact implementation sequence

| Step | Package | Effort, days | Stops for |
|---|---|---|---|
| 1 | **WP-C1** Coordinator core, fakes only | 5 | D5, D14, D15 first. Your review after |
| 2 | WP-M1 Marker reads, startup checks | 2 | Disposable server authorization |
| 3 | WP-M2 Templates, X1, X2 | 3 | OD-8 |
| 4 | WP-R1 Recovery and operator tool | 4 | OD-12 |
| 5 | WP-T1 Backlog store | 3 | OD-1, OD-3, OD-10, OD-11 |
| 6 | WP-T2 Reader and offset | 3 | **Listener change authorization,** OD-5, OD-9, OD-13 |
| 7 | WP-T3 Rotation-aware reader | 3 | X9 for live confirmation |
| 8 | WP-T4 Worker, pause, routes | 4 | OD-4 |
| 9 | WP-T5 Benchmark | 4 | X3, X6 authorization. **Then OD-2 and D11** |
| 10 | WP-O1 Observability | 2 | |
| 11 | WP-M3 Rehearsal | 3 | **A production copy** |

## Estimated effort

**About 36 focused days, ±50%,** excluding decisions and getting a production copy. These are my estimates, not measurements. WP-C1 alone is 5, and it's useful by itself.

## Prerequisites for Phase 2.8

**Phase 2.8 is the first production writer, W07U, behind a switch that's off by default.** All of these are required:

| # | Prerequisite | Package or source |
|---|---|---|
| 1 | WP-C1, M1, M2, R1 accepted | Packages |
| 2 | WP-T1, T2, T3, T4 and O1 accepted, or an explicit decision to run W07U without the backlog, accepting D10's gap | Packages, OD-1 |
| 3 | X1 and X2 answered | Experiments |
| 4 | Benchmark reported. OD-2 and D11 decided | WP-T5 |
| 5 | Rehearsal passed at production scale | WP-M3 |
| 6 | Production `AppliedWrite` constraint **approved separately,** and created by the runbook | D6 |
| 7 | D5, D13, D14, D15, N1, N2, N3 decided | Register |
| 8 | The 2.6C listener fix, with OD-13, validated live, **with your approval** | 2.6C, 2.6E |
| 9 | A rollback rehearsed: switch off, markers inert | WP-M3 |
| 10 | A named person to resolve uncertain requests | D5 |
| 11 | Your go-ahead, for this writer | You |

## Risks that could invalidate the design

| Risk | Would mean | How it'd show |
|---|---|---|
| **`process_alert` is far slower than assumed** | Backlogs last hours or days. A/B and tier choices change | B1 |
| **Real arrival is far above what the reader or spool can take** | Reader lags into rotation. Rotation logic carries the risk | X3, X7 |
| **Lock waits don't end at the transaction timeout** | One worker can stall indefinitely | X1 |
| **The CAS doesn't evaluate as assumed** | Stale payload stays open | X2 |
| **Marker creation or reads behave differently at size** | Rollout plan changes | X4 |
| **Interrupted alerts are common** | Too much for a person to review, and OD-3 must change | B5, operation |
| **Uncertain requests are common** | The same, and OD-7 becomes pressing | B5 |
| **`id` isn't unique across a future upgrade or second manager** | Door check holds valid alerts. Visible, not lost | `CONFLICT_HELD` count |
| **The campaign-activity decisions go another way** | Writers' semantics change, and templates and fingerprints with them. Drain first | CA-D list |
| **A backup restore** | Stores and graph can disagree silently | Runbook |

## How this package was checked

| Check | Result |
|---|---|
| Preflight reads, per your list | Done. Manifests of six earlier packages verified |
| Citations and figures | `evidence/check_report.py`, output in `evidence/check_report_output.txt` |
| Preservation | `evidence/preservation_audit.txt` |
| Model | `evidence/backlog_model.py`, read only, on a repo sample and 2.6E's evidence |

**Not done, by your instruction:** no test suite, no server, no production contact, no experiment. **Every number from an experiment here comes from an earlier phase.**

**One note:** this package's directory existed before my pre-check listing, because I'd written the model script first. It contains only this phase's files.

# Validation criteria: campaign activity contract

**Status:** acceptance criteria for a future implementation. No tests are written or run by this document.

**Rules for using these criteria**
- Criteria marked `[Dn]` are parameterised: the expected value comes from decision Dn in `OPEN_POLICY_DECISIONS.md`. They can't be finalised until it is decided.
- Passing shows conformance to the contract. **It doesn't show campaign accuracy.** That needs labelled data and separate metrics.
- A change in campaign count is not a pass or fail by itself.
- Run all database-backed tests on a disposable Neo4j, never `neo4j-soc`. Keep MISP and outbound HTTP blocked.
- Keep frozen B3 outputs as a comparison baseline. Differences from them must be explained, not treated as failures by default.

**Test levels:** **U** = unit, no database. **I** = integration, disposable Neo4j, synthetic alerts. **R** = full-archive replay, each run needing approval.

## V1. Deterministic activity updates

| ID | Criterion | Level |
|---|---|---|
| V1.1 | Each event class in the matrix produces exactly the `[D1,D2,D3]` result for memory and Neo4j `last_seen`, one test per row | U, I |
| V1.2 | The same input replayed twice on fresh databases gives an identical view hash | R |
| V1.3 | The same input at different replay speeds gives an identical view hash, including flush behaviour under `[D6]` | R |
| V1.4 | No lifecycle field depends on wall-clock time, other than marked fallback `[D9]`. Test by shifting the system clock in a unit test | U |

## V2. Consistent in-memory and persisted `last_seen`

| ID | Criterion | Level |
|---|---|---|
| V2.1 | After every processed alert, for a cached campaign, memory and Neo4j activity values agree, allowing only pending state defined by `[D6]` | I |
| V2.2 | After a flush with nothing pending, memory and Neo4j agree for every cached campaign | I |
| V2.3 | Expiry, reopen scoring and reload read the value chosen in `[D11]`. Test by giving memory and Neo4j different values and asserting which is used | U, I |
| V2.4 | A reloaded context has the same activity value as the context before reset, given a clean flush | I |

## V3. Correct attribution for buffered duplicates

| ID | Criterion | Level |
|---|---|---|
| V3.1 | **Regression scenario:** fingerprint F is registered to event E1 in campaign C1, a duplicate is buffered, C1 closes, F lapses over 90 s and re-registers to E2 in C2, then another duplicate arrives with no flush in between. The result matches `[D5]`. Under D5b or c, `C1.last_seen` does not take the second duplicate's time | I |
| V3.2 | For every duplicate, the campaign changed by the flush equals its intended campaign per `[D5]`. Measure as a count of `retained ≠ classified`; the frozen CUR figure is 144 of 1,201 under the harness pattern | R |
| V3.3 | `occurrences`, `total_tps` and `risk_score` follow `[D12]` | I |
| V3.4 | The buffer-full path gives the same attribution as the buffered path | U, I |
| V3.5 | Attribution doesn't change if flush points are moved. Test with at least two flush schedules | I, R |

## V4. No timestamp regression

| ID | Criterion | Level |
|---|---|---|
| V4.1 | `Campaign.last_seen` never decreases through any writer, checked with a late event against each writer | I |
| V4.2 | Within a pending record, an earlier-arriving-later duplicate doesn't reduce `last_seen`, if `[D4]` selects maximum | U |
| V4.3 | `AttackEvent.last_seen` never decreases | I |
| V4.4 | Late events behave per `[D8]`, including their effect on `closed_at` in a score-based close | U, I |
| V4.5 | Fallback or future-skewed alerts affect the watermark and `last_seen` only per `[D9]` | U |

## V5. Restart and recovery consistency

| ID | Criterion | Level |
|---|---|---|
| V5.1 | A simulated graceful restart gives the same campaign assignments, operations and transitions as no restart | R |
| V5.2 | The same, with persisted fields identical, or each difference listed and justified by the contract | R |
| V5.3 | Holds at several restart points, including at least one mid-burst and one with a campaign active | R |
| V5.4 | Holds across more than one restart in a run | R |
| V5.5 | Restart through the **real** shutdown path, not only the harness simulation | I |
| V5.6 | Dedup decisions after a restart match no-restart, for fingerprints inside the window, given a clean flush | I |
| V5.7 | Unclean crash matches `[D7]`: simulated kill with records pending, with the loss or recovery asserted | I |
| V5.8 | The watermark after restart behaves as decided; lateness across a restart is handled or documented | U, I |

## V6. Idempotent duplicate flushing

Needed only if the design keeps a buffer. If a chosen alternative doesn't need it, record that.

| ID | Criterion | Level |
|---|---|---|
| V6.1 | Applying the same flush record twice leaves counters and timestamps as after one, if durable replay or retry exists (`[D7]`) | I |
| V6.2 | A batch that fails partway and is retried gives the same final state as one clean success | I |
| V6.3 | A failed flush doesn't lose records | U, I |
| V6.4 | Restoring records after a failure doesn't overwrite newer data for the same fingerprint | U |

## V7. UNKNOWN events

| ID | Criterion | Level |
|---|---|---|
| V7.1 | A new UNKNOWN alert updates memory and Neo4j per `[D2]`, and the two agree | I |
| V7.2 | An UNKNOWN duplicate updates the campaign per `[D3]`. The cause of today's silent no-update (no `MATCHES` edge) must be either preserved deliberately or removed, not accidental | I |
| V7.3 | No technique is ever fabricated, no `MATCHES` edge created, no transition learned, for an UNKNOWN event | U, I |
| V7.4 | An UNKNOWN event doesn't reopen a closed campaign, unless the owner decides otherwise | I |
| V7.5 | Counts of early closure are reported, split by cause. Frozen CUR: 30 of 101, of which 5 flush and 25 UNKNOWN new. Expected change follows `[D2,D3]` | R |

## V8. Graceful shutdown with pending duplicates

| ID | Criterion | Level |
|---|---|---|
| V8.1 | `DuplicateFlushWorker.stop()` with pending records and a working writer: records written once, no exception, buffer empty | U |
| V8.2 | With a failing writer: no `NameError`, records handled per decision, failure logged | U |
| V8.3 | With an empty buffer: returns cleanly | U |
| V8.4 | SIGTERM with queued alerts and pending duplicates: queue drained, then flush, then clean exit, and every accepted alert accounted for | I |
| V8.5 | Offset commit behaviour matches the decision taken for alert loss | I |

## V9. No unexpected `closed_at` and `last_seen` inconsistencies

| ID | Criterion | Level |
|---|---|---|
| V9.1 | Every `closed_at < last_seen` in a replay is listed, classified to a contract case, and permitted by `[D3,D8,D9,D10]`. Unclassified count: 0 | R |
| V9.2 | No inversion comes from another campaign's alert. Frozen CUR has 2 (C0002, C0045), plus C0034 with restart | R |
| V9.3 | A zero count isn't accepted alone: the test also asserts that activity the contract says to record was recorded. This prevents passing by omission, as B does | R |
| V9.4 | An ACTIVE campaign's `closed_at` matches `[D10]` | I |
| V9.5 | `closed_at` has the same value whichever path closes the campaign: alert, cache sweep or DB sweep, in either order | U, I |
| V9.6 | Reopen ordering and candidate selection don't change with flush timing, or the dependence is documented and accepted under `[D11]` | I, R |

## Required reporting for any future implementation

- Source revision, including uncommitted state, and hashes of inputs.
- Parity or an explained difference against frozen `CUR_norestart`, full hash `70c7b12b30ddb0d51e0ebe531fbb519cc72f5b9560fdeee5ef27cb3f0d429e40`.
- Which decisions it implements, and each criterion's result, with skipped ones named.
- Results by evidence class. A prediction isn't reported as an observation.
- Live-flush results kept separate from harness results.

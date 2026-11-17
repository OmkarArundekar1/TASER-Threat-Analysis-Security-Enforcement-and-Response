# Validation criteria: campaign activity contract (review copy)

**Status:** acceptance criteria for a future implementation. **No criterion has been run, and none has passed.** No tests are written or run by this document.

**Rules for using these criteria**
- `[Dn]` marks a parameterised criterion: the expected result comes from decision Dn in `OPEN_POLICY_DECISIONS_FINAL.md`. It can't be finalised until that decision is made. All D1 to D12 are open.
- Passing shows conformance to the contract. **It doesn't show campaign accuracy.** That needs labelled data and separate metrics.
- A change in campaign count is not a pass or fail by itself.
- Run all database-backed tests on a disposable Neo4j, never `neo4j-soc`. Keep MISP and outbound HTTP blocked.
- Keep frozen B3 outputs as a comparison baseline. Differences from them must be explained, not treated as failures by default.
- If a chosen alternative makes a criterion inapplicable, record that and why. Don't delete the criterion.

**Test levels:** **U** = unit, no database. **I** = integration, disposable Neo4j, synthetic alerts. **R** = full-archive replay, each run needing approval.

**Terms:** retained, classified, mutated and intended campaign, and activity vs evidence time, are defined in the decision register.

**Symbols used in V2.** `M` = in-memory `context.last_seen`. `P` = persisted `Campaign.last_seen`. `B` = the latest time in pending records whose intended campaign is this one.

## V1. Deterministic activity updates

| ID | Criterion | Depends on | Level |
|---|---|---|---|
| V1.1 | Each event class in the matrix produces exactly the decided result for `M` and `P`, one test per row | `[D1,D2,D3]` | U, I |
| V1.2 | The same input replayed twice on fresh databases, under the same flush schedule, gives an identical view hash | `[D6]` for the schedule | R |
| V1.3 | The same input at different replay speeds behaves as the table below requires | `[D6,D6-R,D5,D11]` | R |
| V1.4 | No lifecycle field depends on wall-clock time, other than marked fallback. **Test with an injected or fake clock**, never by changing the system clock. Where a value comes from Neo4j's `datetime()`, assert that the event-time parameter is passed and non-null, so the `coalesce` default isn't used | `[D9]` | U |

**V1.3 expected result**

| Owner's answer | Selected flush policy | Expected |
|---|---|---|
| D6-R required | D6b, c, d, or a combination with no wall-clock part | Identical view hash across speeds. Whether each policy satisfies this is what the test finds out; it isn't assumed |
| D6-R required | D6a, or any wall-clock part | Contradiction. Must go back to the owner before implementation |
| D6-R not required | Any | Grouping results and persisted-field differences are listed per speed, and must fall within what the owner accepted. Unlisted differences fail |

A faithful test of D6a needs the real flush thread running, because the harness has no wall-clock flush.

## V2. Consistent in-memory and persisted `last_seen`

**What must agree depends on the decisions.** Pending state allowed by D6 is not an inconsistency.

| Decisions | Expected relationship for a cached campaign |
|---|---|
| Direct writes (D6d) | `M = P` for classes that write both |
| Buffer kept, duplicates extend (D3), memory updated at arrival | `M = max(P, B)` |
| Buffer kept, duplicates extend (D3), memory updated only at flush | `M = P` until flush, with `B` allowed to lead |
| Duplicates don't extend (D3b) | `M = P`; `B` is irrelevant to activity |
| CUR split kept (D1d, D2c, D11a) | `M ≤ P` is allowed. Each difference is reported by event class, not counted as a pass |
| Two fields (D11c) | The relationships apply to activity time. Evidence time may be later |

| ID | Criterion | Depends on | Level |
|---|---|---|---|
| V2.1 | After every processed alert, `M`, `P` and `B` satisfy the row selected above. Any other difference fails | `[D1,D2,D3,D6,D11]` | I |
| V2.2 | After a flush with nothing pending, `B` is empty and `M`, `P` satisfy the selected row for every cached campaign | `[D6,D11]` | I |
| V2.3 | For each operation below, give memory and Neo4j different values and assert the operation reads the value D11 selects | `[D11]` | U, I |
| V2.4 | A reloaded context has the same activity value as the context before reset, given a clean flush. Under D11a, differences are listed instead | `[D11]` | I |

**V2.3 operations:** cached expiry on the alert path; cache sweep; reopen scoring, fast rule; reopen scoring, score; candidate ordering, including the tie-break; database sweep; context reload; continuation scoring. The authority for each is the D11 table.

## V3. Correct attribution for buffered duplicates

| ID | Criterion | Depends on | Level |
|---|---|---|---|
| V3.1 | The regression scenario below ends in the state D5 selects | `[D3,D4,D5,D12]` | I |
| V3.2 | Replay reports **two separate counts**: (1) `retained ≠ classified`, (2) `mutated ≠ intended`. Count 2 must be 0. Count 1 is judged per the table below | `[D5]` | R |
| V3.3 | `Campaign.occurrences`, `total_tps` and `risk_score`, and `AttackEvent.last_seen`, `occurrences` and `tps`, follow the decided attribution, asserted field by field | `[D12,D12-F,D3]` | I |
| V3.4 | The direct path (buffer full or disabled) gives the attribution D5 selects. Differences from the buffered path are listed, including later repeats of the same fingerprint | `[D5]` | U, I |
| V3.5 | Behaviour when flush points move, per the table below. Test with at least two flush schedules | `[D5,D6]` | I, R |
| V3.6 | `AttackEvent.investigation_payload` after a flush matches the decision, including when the last merged alert was classified to another event | `[D12,D12-F]` | I |

**V3.2 count 1**

| D5 | `retained ≠ classified` means | Expected |
|---|---|---|
| a | Retention working as selected | Reported. Not a defect. Frozen CUR: 144 of 1,201, harness pattern |
| b, c | The new rule not working | 0 on the buffered path |
| d | Timing-dependent result the owner accepted | Reported, per flush schedule |

**V3.5 expected result**

| D5 | D6 | Expected |
|---|---|---|
| b | Any | Attribution doesn't change with flush points |
| c | Any with re-registration trigger | Attribution doesn't change with flush points |
| a or d | Input-only flush points | Same result for the same schedule. Different schedules may differ, and the dependence is documented |
| a or d | Wall clock | Dependence on timing is documented and accepted by the owner, not asserted away |

### V3.1 regression scenario

Times are explicit event times, with millisecond precision, from an injected source. No wall clock. No flush except where stated. One attacker, victim, technique, rule and agent, so one fingerprint F.

| Step | Event time | Alert | Precondition asserted |
|---|---|---|---|
| 1 | `T0` | New, creates E1 in C1 | |
| 2 | `T0 + 10.000 s` | Duplicate, record created with retained E1, C1 | |
| 3 | `T0 + 3600.000 s` | Same fingerprint. The gap from step 2 exceeds the window, so it's new: E2 in C2 | C1 closed. **C2 ≠ C1**, meaning no reopen fired. If reopen fires, the scenario is invalid, not failed |
| 4 | `T0 + 3610.000 s` | Duplicate, classified E2, C2 | Pending record still exists |
| 5 | Flush | | |

**Expected after step 5**, with duplicates extending under D3:

| D5 | `C1.last_seen` | `C2.last_seen` |
|---|---|---|
| a | `T0 + 3610.000` | `T0 + 3600.000` |
| b, c | `T0 + 10.000` | `T0 + 3610.000` |
| d | As a, because no flush occurred between steps 2 and 4 | As a |

Under D3b, neither campaign takes a duplicate's time. D4 and D12 adjust the time and counters, and those fields are asserted separately.

**Window boundary cases**, run as separate fixtures. The comparisons are `(now - last_seen) <= 90` in cache (`dedup_engine.py:241`) and `e.last_seen >= reference - 90 s` in the database fallback (`neo4j_client.py:1046`):

| Gap since last sighting | Today's expected class |
|---|---|
| 89.999 s | Duplicate |
| 90.000 s | Duplicate |
| 90.001 s | Not a duplicate, provided `AttackEvent.last_seen` is also outside the window |

The fallback reads `AttackEvent.last_seen`, which a pending record hasn't updated. So the 90.001 s case also needs a variant with a flush before it, and the fixture must state which it is using. If a decision changes the window behaviour, these expected classes change with it.

## V4. No timestamp regression

| ID | Criterion | Depends on | Level |
|---|---|---|---|
| V4.1 | `Campaign.last_seen` never decreases through any writer, checked with a late event against each writer | `[D8]` for routing only | I |
| V4.2 | A later-arriving duplicate with an **earlier event timestamp** behaves as below | `[D4]` | U |
| V4.3 | `AttackEvent.last_seen` never decreases | | I |
| V4.4 | Late events behave as decided, including their effect on `closed_at` in a score-based close | `[D8]` | U, I |
| V4.5 | Fallback or future-skewed alerts affect evidence retention, the watermark, `last_seen` and lifecycle expiry only as decided, asserted separately for each | `[D9,D9-S]` | U |

**V4.2 expected value of the pending record**

| D4 | Result |
|---|---|
| a | Unchanged, because the maximum is kept. Today's buffer would lower it |
| b | Unchanged, because the first time is kept |
| c | No time is carried |

Neo4j's monotone write protects `Campaign.last_seen` under all of these. V4.2 concerns the record itself. `Technique.last_seen` isn't covered: it isn't monotone at `neo4j_client.py:431`, a finding outside this contract's scope unless D12-F says otherwise.

## V5. Restart and recovery consistency

**Four distinct modes. Results from one say nothing about another.**

| Mode | Label | What happens |
|---|---|---|
| Harness-simulated graceful restart | **H** | Study harness flushes, then resets manager, dedup and clock. No listener, worker or `stop()` |
| Real listener shutdown | **L** | Signal to the actual listener process, through its `finally` block and `DuplicateFlushWorker.stop()`, then a new process |
| Unclean termination | **C** | Process killed with records pending. No recovery mechanism assumed |
| Durable recovery | **D** | Only if selected under D7. Restart that replays durable state after C |

**B3 status.** B3 ran mode H once, before alert 4,500. Grouping was identical for CUR, A and B. Persisted fields differed for CUR (2 `last_seen`, 1 score) and A (2 `last_seen`, 1 `closed_at`), and not for B. `OBSERVED`. That is reference data at one point. **It doesn't pass any criterion here, and is no evidence for V5.5, V5.7 or V5.9.**

| ID | Mode | Criterion | Depends on | Level |
|---|---|---|---|---|
| V5.1 | H | **Grouping:** same campaign assignments, operations and transitions as no restart | `[D5,D6,D7,D10,D11]` | R |
| V5.2 | H | **Persisted fields:** identical to no restart, or each difference listed and permitted by the contract. Asserted separately from V5.1 | `[D5,D6,D7,D10,D11]` | R |
| V5.3 | H | V5.1 and V5.2 at several restart points, including at least one mid-burst and one with a campaign active | `[D5,D6,D7,D10,D11]` | R |
| V5.4 | H | V5.1 and V5.2 across more than one restart in a run | `[D5,D6,D7,D10,D11]` | R |
| V5.5 | L | Grouping and persisted fields, asserted separately, through the **real** shutdown path. Requires V8 first | `[D5,D6,D11]` | I |
| V5.6 | H, L | Dedup decisions after a restart match no-restart, for fingerprints inside the window, given a clean flush | `[D6]` | I |
| V5.7 | C | Simulated kill with records pending. The loss is exactly what the decision allows, and is asserted, not inferred | `[D7,D7-O]` | I |
| V5.8 | H, L | The watermark after restart behaves as decided; lateness across a restart is handled or documented | `[D8,D9]` | U, I |
| V5.9 | D | Only under D7b or d. After C, recovery restores what was pending, without double application | `[D7]`, V6 | I |

V5.1 to V5.4 can't have a fixed expected result until D5, D6, D7, D10 and D11 are decided. For example, under D10a and D11a some persisted-field differences are expected and must be listed.

## V6. Idempotent duplicate flushing

**Applicability**

| Selected architecture | V6 status |
|---|---|
| Direct writes, no buffer (D6d, D7c) | V6.1 to V6.6 not applicable. Record that, and test failure handling of the direct write instead |
| Buffer, no retry or durable replay | V6.2 and V6.3 test the failure rule only. V6.1 applies only if a record can be applied twice |
| Buffer with retry or `restore()` | All apply |
| Buffer with durable recovery (D7b) | All apply, and V5.9 |

**How failure is simulated.** Name the mode in each result. The two are not the same test.

| Mode | Meaning | Method |
|---|---|---|
| **T: transactional batch** | One statement in one transaction, all or nothing. This is today's writer per batch | Disposable Neo4j. Put a record with an unparseable `last_seen` last in a batch, so `datetime(r.last_seen)` fails the statement. Assert no row of that batch persisted. Confirms the assumed atomicity, which hasn't been exercised |
| **M: multiple batches** | Earlier batches commit, a later one fails. Possible today, since `MAX_BATCH_SIZE` is 200 | Fake writer that succeeds for `k` batches, then raises. Use more than 200 records |
| **S: sequence of individual writes** | Only if the selected design persists records one at a time | Fake writer that applies `i` records, then raises |

| ID | Criterion | Failure mode | Depends on | Level |
|---|---|---|---|---|
| V6.1 | Applying the same flush record twice leaves counters and timestamps as after one, if retry or durable replay exists | | `[D6,D7]` | I |
| V6.2 | A failed flush, then a retry, gives the same final state as one clean success | T, M, and S if applicable | `[D6,D7]` | I |
| V6.3 | A batch failing before persistence doesn't lose records | T | `[D6,D7]` | U, I |
| V6.4 | **Restoration collision:** add a newer record for fingerprint F while a flush of F is failing, then restore. The asserted values are the newer record's `event_id`, campaign, counters, `last_seen` and payload, per decision | | `[D4,D5,D12]` | U |
| V6.5 | All writes succeed, then logging fails. Records aren't re-applied, and the exception doesn't reach the shutdown path unhandled | | `[D7]` | U |
| V6.6 | The commit outcome is unknown: the client sees an error after the server committed. The behaviour is defined and no counters are doubled silently | | `[D7]` | I |

**V6.4 today, by reading the code, not run:** `restore()` adds the failed record's counters to the newer record, keeps the newer record's `event_id` and `campaign_id`, and overwrites its `last_seen` and payload with the older values. See contract F.2.

## V7. UNKNOWN events

Results are reported separately for UNKNOWN new alerts and UNKNOWN duplicates. They are different event classes under different decisions.

| ID | Criterion | Depends on | Level |
|---|---|---|---|
| V7.1 | **New:** a new UNKNOWN alert updates `M` and `P` as decided, and any disagreement is one the decision allows | `[D1,D2]` | I |
| V7.2 | **Duplicate:** an UNKNOWN duplicate updates the campaign as decided, for open and closed campaigns separately. The cause of today's silent no-update (no `MATCHES` edge) must be either preserved deliberately or removed, not accidental | `[D1,D3,D3-C]` | I |
| V7.3 | **Unconditional.** No technique is ever fabricated, no `MATCHES` edge created, no transition learned, for any UNKNOWN event | None | U, I |
| V7.4 | An UNKNOWN event doesn't reopen a closed campaign, unless the owner selects otherwise | `[D2,D3-C]` | I |
| V7.5 | Counts of early closure are reported, split by cause. Frozen CUR: 30 of 101, of which 5 flush and 25 UNKNOWN new. The expected change follows the decisions. Fewer early closures isn't itself an improvement | `[D2,D3]` | R |

## V8. Graceful shutdown with pending duplicates

**Accepted alert.** A line from `alerts.json` that parsed, wasn't dropped as a raw duplicate, and was successfully placed on `alert_queue`. Each accepted alert must end as one of: processed; rejected for event time; or failed with the failure logged. Lines not accepted (malformed, raw duplicate, queue full) are counted separately.

**Required order, and what the listener does today.** The right-hand column is from reading the code. **It has not been exercised, and this table doesn't assert it works.**

| # | Required step | Current listener |
|---|---|---|
| 1 | Stop accepting new input | The signal sets `stop_event`, and the poll loop ends after its current pass (`wazuh_listener.py:111-116,324`). A pass in progress reads to end of file |
| 2 | Drain or explicitly account for queued alerts | `alert_queue.join()` (`:334`), with no timeout. A failed alert is logged, marked done, and not retried (`:153-156`) |
| 3 | Flush pending duplicates | `duplicate_flush_worker.stop()` (`:338`). It joins the worker for 2 s while the worker sleeps 7 s, so the worker may still be alive. See contract F |
| 4 | Complete persistence | `stop()` writes synchronously. An in-flight write on the daemon worker thread isn't waited for beyond those 2 s |
| 5 | Commit offsets according to the selected policy | **Not in this sequence.** The offset is written during polling, when alerts are queued (`:272-288`) |
| 6 | Exit cleanly | Undefined `logger` in `stop()` is expected to raise when records are pending. Static finding |

| ID | Criterion | Depends on | Level |
|---|---|---|---|
| V8.1 | `DuplicateFlushWorker.stop()` with pending records and a working writer: records written once, no exception, buffer empty | | U |
| V8.2 | With a failing writer: no `NameError`, records handled as decided, failure logged | `[D7]` | U |
| V8.3 | With an empty buffer: returns cleanly | | U |
| V8.4 | SIGTERM with queued alerts and pending duplicates: steps 1 to 6 occur in order, and every accepted alert is accounted for | `[D7-O]` | I |
| V8.5 | The offset boundary and alert loss match the table below | `[D7,D7-O]` | I |
| V8.6 | A queue-full line is neither reported lost while later processed, nor silently skipped on re-read. See contract F.3 | | U, I |

**V8.5 expected result** (reasoning, to be confirmed by the test)

| D7-O | Offset committed | Kill after commit, before processing | Kill after processing, before commit |
|---|---|---|---|
| O1 | When queued | Alert lost | Doesn't apply: commit came first |
| O2 | After `process_alert` returns | Doesn't apply | Alert reprocessed. Reprocessing must be defined, since it isn't idempotent today |
| O3 | After related duplicates persist | Doesn't apply | Alert reprocessed, over a wider window. Needs V6 |

V8 can't pass until it is actually run. Any listener run needs separate approval.

## V9. No unexpected `closed_at` and `last_seen` inconsistencies

| ID | Criterion | Depends on | Level |
|---|---|---|---|
| V9.1 | Every `closed_at < last_seen` in a replay is listed, classified to a contract case, and permitted by the decisions. Unclassified count: 0 | `[D2,D3,D3-C,D6,D8,D9,D9-S,D10]` | R |
| V9.2 | Inversions caused by another campaign's alert are counted. Under D5b or c, expected 0. Under D5a or d, they are expected, listed and accepted. Frozen CUR has 2 (C0002, C0045), plus C0034 with restart | `[D5]` | R |
| V9.3 | **A zero count is never accepted alone.** The test also asserts that activity the contract says to record was recorded. This prevents passing by omission, as B does | `[D1,D2,D3]` | R |
| V9.4 | An ACTIVE campaign's `closed_at` matches the decision: stale value allowed under D10a, absent under D10b or c | `[D10]` | I |
| V9.5 | The expiry paths (alert path, cache sweep, DB sweep) give the same `closed_at`, in either order, **given identical event-time and clock inputs and the same `last_seen` input**. Under D11a the DB sweep reads a different value, so a difference there is reported, not failed. A score-based close follows a different rule and isn't part of this comparison; it belongs to V4.4 | `[D11]` | U, I |
| V9.6 | Reopen ordering and candidate selection, classified as below | `[D6,D11]` | I, R |

**V9.6 classification**

| Result | Class |
|---|---|
| Order differs from what the D11 authority prescribes for the values it read | **Defect** |
| Same rule applied, but a flush moved the value it read, and D6 and D11 document that dependency | **Documented dependency**, reported with its count |
| Same rule applied, a flush moved the value, and no decision accepts that | **Unresolved**, goes back to the owner |

## Required reporting for any future implementation

- Source revision, including uncommitted state, and hashes of inputs.
- Parity or an explained difference against frozen `CUR_norestart`, full hash `70c7b12b30ddb0d51e0ebe531fbb519cc72f5b9560fdeee5ef27cb3f0d429e40`.
- Which decisions it implements, and each criterion's result, with skipped ones named.
- Results by evidence class. A prediction isn't reported as an observation.
- Live-flush results kept separate from harness results.
- Restart results labelled H, L, C or D.
- Flush-failure results labelled T, M or S.
- `retained ≠ classified` and `mutated ≠ intended` reported separately.

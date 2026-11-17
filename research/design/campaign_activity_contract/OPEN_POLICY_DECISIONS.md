# Open policy decisions: campaign activity contract

**Status:** all open. No decision is taken, and alternatives are listed in no order of preference. The owner decides.

**Limits on all evidence below.** One archive with no ground truth, so no alternative can be called more accurate. Harness flush pattern unless stated.

**Decide together:** D1 sets the frame for D2 and D3. D3, D4, D5 and D12 all concern duplicates. D10 and D11 determine what restart equivalence can mean.

| # | Decision | Status |
|---|---|---|
| D1 | Which event classes count as activity | Open |
| D2 | UNKNOWN new alerts | Open |
| D3 | Duplicates, including after closure | Open |
| D4 | Duplicate's time | Open |
| D5 | Duplicate's campaign | Open |
| D6 | Flush trigger | Open |
| D7 | Unclean crash | Open |
| D8 | Late events | Open |
| D9 | No trustworthy timestamp | Open |
| D10 | `closed_at` across reopen | Open, added |
| D11 | Which `last_seen` governs | Open, added |
| D12 | Counters and risk | Open, added |

---

## D1. Should all campaign-associated alerts extend activity, or only selected classes?

| Alternative | Consequence | Evidence |
|---|---|---|
| **a. All associated alerts** (study policy A) | Noise keeps campaigns open. 97 vs 116 campaigns, more score-based closures (35 vs 21), more fast reopens (22 vs 10) | `OBSERVED`, A. 17 of the 19 fewer campaigns are self-loop identities |
| **b. Only resolved new alerts** (study policy B) | Simple and restart-consistent, but UNKNOWN evidence never keeps a campaign alive, and activity is not recorded on the campaign | `OBSERVED`: same grouping as CUR, byte-identical across restart. That isn't proof of correctness |
| **c. Selected classes, set by D2 and D3** | Most explicit; more cases to specify and test | No run tested a mix other than CUR |
| **d. Keep CUR's split** (memory: resolved; Neo4j: resolved, UNKNOWN new, flushed duplicates) | No change, and invariant I3 stays false | `OBSERVED`: 30 of 101 early closures; 2 to 3 inversions |

## D2. Should UNKNOWN new alerts extend activity?

The reviewer feedback cites about 82% UNKNOWN. This archive has 152 UNKNOWN new alerts vs 110 resolved new, and 7,704 of 9,015 alerts (85%) on the UNKNOWN path including duplicates.

| Alternative | Consequence | Evidence |
|---|---|---|
| **a. Yes, in memory and Neo4j** | An attack with unmapped steps stays one campaign longer. Scanning noise can too. Would reduce the 25 early closures caused by the memory/Neo4j split | Part of A; not isolated. `OBSERVED` for the 25 |
| **b. No, in either** | Consistent. An active attack with only unmapped alerts can expire | Part of B; not isolated |
| **c. Keep CUR: Neo4j only** | Behaviour differs depending on whether a campaign is cached or reloaded | `OBSERVED` 25 cases; `SOURCE` |

UNKNOWN alerts never reopen a closed campaign under any current code path (`SOURCE`). This decision is separate from whether UNKNOWN campaigns reach operation correlation.

## D3. Should duplicates extend activity?

Needs separate answers for four sub-cases:

| Sub-case | Count in `CUR_norestart` | Today |
|---|---|---|
| Resolved duplicate, campaign open | 579 same-campaign, 83 cross | Neo4j at flush |
| Resolved duplicate, campaign closed | 478 same-campaign (all C0028), 61 cross | Neo4j at flush |
| UNKNOWN duplicate, campaign open | 2,946 | Neither |
| UNKNOWN duplicate, campaign closed | 4,606 | Neither |

`OBSERVED`.

| Alternative | Consequence | Evidence |
|---|---|---|
| **a. All extend** | One long repeating alert keeps a campaign open indefinitely, since the dedup window slides. Timeout means less | Part of A: 6 to 7 inversions |
| **b. None extend** | Campaigns close on 120 s without a new alert while the same alert is still firing. Repeats stay recorded on the event | Part of B: 0 inversions, by omission |
| **c. Only while campaign is open** | Closure becomes final with respect to duplicates. Needs a defined rule on the duplicate path. Could remove inversion case 3 by construction | Not tested |
| **d. Resolved only** | Keeps a mapped signal. UNKNOWN repeats, the large majority, don't extend | Closest to CUR's Neo4j behaviour, not its memory behaviour |

**Sub-question: what should a duplicate on a closed campaign do?** Ignore it for activity, record it on the event only, attach it to the current campaign, or trigger reopen evaluation. The last changes grouping, and no run tested it.

## D4. Own event time, or the first buffered record's time?

**Today's behaviour is neither "first" nor strictly "latest".** The record stores the last arrival's time, and Neo4j applies it monotonically. `SOURCE`, `OBSERVED`.

| Alternative | Consequence | Evidence |
|---|---|---|
| **a. Each duplicate's own time, keeping the maximum** | Faithful to when repeats happened. Needs a monotone merge in the buffer if out-of-order arrivals matter | CUR is close to this, once attribution is right |
| **b. First buffered time** | Bounds the extension to the first repeat; understates long repeats | Not tested |
| **c. No time** (with D3 "none") | Only counters change | B's behaviour |

## D5. Should the first duplicate's association stay authoritative after the campaign changes?

| Alternative | Consequence | Evidence |
|---|---|---|
| **a. Keep first association** (today) | Simple. Credits later alerts to an earlier campaign when no flush intervenes | `OBSERVED` 144 of 1,201, harness pattern; C0002, C0045 |
| **b. Current association at classification** (one pending record per fingerprint **and** event) | Each repeat is credited to its classified event. Slightly more records. Removes the C0002 and C0045 mechanism; the buffer no longer merges across events | Not tested; prediction only that those inversions wouldn't arise |
| **c. Flush the old record when a fingerprint re-registers** | Keeps one record per fingerprint. Adds a write on the new-alert path, and ties flush to classification | Not tested |
| **d. Rely on periodic flush** | No change to logic, only live cadence. Attribution stays timing-dependent; breaks I5 | **Live rate unmeasured.** The 12% must not be used here |

## D6. Periodic, event-triggered, shutdown-triggered, or a combination?

| Alternative | Consequence | Evidence |
|---|---|---|
| **a. Periodic on wall clock + shutdown** (live today) | Low write load. Flush points depend on processing speed, so replay isn't deterministic | `SOURCE`; effect unmeasured |
| **b. Periodic on event time** | Deterministic across speeds. Idle streams need a separate trigger, since no alerts means no advancing time | Not tested |
| **c. Event-triggered** (for example, on re-registration, see D5c) | Closes the attribution window. Adds work on the hot path | Not tested |
| **d. No buffer: write each duplicate** | Removes the whole class. 1,201 extra writes on this archive and an unmeasured live cost | `ENABLE_DUPLICATE_BUFFER` exists (`config.py:35`), and that path is not exercised in B3 |
| **e. Combination** | Flexible; more to test | Not tested |

The harness's restart-and-end pattern is an artefact of replay, not a candidate. It is why the buffer effect is large in B3.

## D7. Pending duplicates during an unclean crash

| Alternative | Consequence | Evidence |
|---|---|---|
| **a. Accept loss, document it** | No new machinery. Lost occurrences, `last_seen` and risk increments; dedup may treat a repeat as new after restart. A short flush interval bounds the loss | `SOURCE` |
| **b. Durable buffer** (local journal or write-ahead) | Survives crashes, needs replay and idempotency (V6) | Would need new persistence |
| **c. Write duplicates directly** | No pending state; see D6d | Not tested |
| **d. Reconstruct from `alerts.json`** | Needs an offset committed after processing. Today's offset advances at queue time (`wazuh_listener:272-288`) and processing is not idempotent | `SOURCE`; bigger change |

**Crashes are entirely untested.** No evidence exists on what recovery looks like in practice.

## D8. Late events: `last_seen` and closure

`OBSERVED`: 0 late alerts in this archive, so all of this rests on code and unit tests.

| Alternative | Consequence |
|---|---|
| **a. Keep today:** accepted, monotone, joins current campaign in arrival order | Simple. A late alert can join a campaign it didn't occur in, and can set `closed_at` before `last_seen` on a score-based close |
| **b. Route to the campaign whose activity span contains its event time** | Historically truer. Needs a lookup, and grouping becomes order-sensitive differently |
| **c. Keep as evidence, no lifecycle effect beyond a tolerance** | Prevents lifecycle changes from stale data. A long ingestion delay would silently stop activity |
| **d. Score close uses `max(event_time, last_seen)`** | Removes that inversion only. Can combine with the others |

## D9. Events with no trustworthy timestamp

The live default is fallback (`config.py:22`), and replay is strict. So **replay evidence doesn't cover the live default.** `SOURCE`; 0 such alerts in this archive.

| Alternative | Consequence |
|---|---|
| **a. Keep:** strict in replay, marked fallback in live | Keeps evidence live. Processing time enters lifecycle fields and advances the watermark (`event_time.py:162`), so a backlog could mis-expire campaigns |
| **b. Fallback, but don't advance the watermark or extend `last_seen`** | Keeps the event as evidence with no lifecycle effect. May miss real activity |
| **c. Reject in both, keep a dead-letter record** | Deterministic. Risks losing evidence in live operation |
| **d. Also apply to future-skewed alerts** | Today they are only flagged, and still advance the watermark. Needs a tolerance choice, currently 5 minutes as a flag only |

## D10. `closed_at` across reopen (added)

| Alternative | Consequence | Evidence |
|---|---|---|
| **a. Keep:** not cleared, overwritten at next close | An ACTIVE campaign can hold a stale `closed_at`. Earlier closures are lost | `SOURCE`; `OBSERVED` C0028 |
| **b. Clear on reopen** | Matches how operations behave (`neo4j_client:835`). Earlier closures still lost. Consumers must accept null | Not tested |
| **c. Clear on reopen and keep closure history** | Full lifecycle record. New schema | Not tested |

## D11. Which `last_seen` governs expiry, reopen and reload? (added)

| Alternative | Consequence | Evidence |
|---|---|---|
| **a. Keep:** memory for cached expiry, Neo4j for reopen scoring, candidate order, DB sweep, reload | Restart and flush timing change decision inputs | `PREDICTED` for C0028 and C0025; `SOURCE` |
| **b. One value, kept consistent in both** | Needs D1 to D5 settled. Reload then can't differ from memory, apart from pending state | Not tested |
| **c. Activity value separate from "evidence last seen"** (two fields) | Separates what keeps a campaign alive from what was last sighted. Could keep duplicates visible without extending activity. Schema change, and every reader must choose | Not tested |

## D12. Should counters and risk follow the same attribution as `last_seen`? (added)

The buffer credits `occurrences`, `total_tps` and `risk_score` to the retained campaign, along with `last_seen` (`neo4j_client:441-443`). `SOURCE`. Size not measured.

| Alternative | Consequence |
|---|---|
| **a. Always the same campaign as `last_seen`** | Consistent. Fixing D5 fixes all four |
| **b. Independent** | Lets a duplicate count without extending activity, for example under D3 "none" (B does this) |
| **c. Stop adding duplicate TPS to campaign risk** | Changes risk semantics. Separate from this contract, and needs its own evidence |

---

## Decisions this register does not cover

Scoring changes (unseen transitions, `chain_similarity` above 1, prediction and chain overlap), operation correlation for UNKNOWN-only campaigns, self-loop identity fallback, and ATT&CK mapping. These are separate workstreams.

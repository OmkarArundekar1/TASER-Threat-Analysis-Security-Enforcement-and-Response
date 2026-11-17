# Open policy decisions: campaign activity contract (review copy)

**Status:** all open. No decision is taken and nothing here is approved. Alternatives are listed in no order of preference, and their letters are unchanged from the draft. The owner decides.

**Limits on all evidence below.** One archive with no ground truth, so no alternative can be called more accurate. A change in campaign count is not an accuracy result. Harness flush pattern unless stated. Live flush attribution rates have not been measured.

**Tags** are defined in `CAMPAIGN_ACTIVITY_CONTRACT_FINAL_DRAFT.md`. "Not tested" means no run exercised that alternative.

## Terms used throughout

| Term | Meaning |
|---|---|
| **Classified campaign** | The campaign an alert is assigned to when processed. For a duplicate, the dedup entry's `campaign_id` at that moment |
| **Retained campaign** | The campaign stored in the pending buffer record, set when the record was created and not updated by later merges |
| **Mutated campaign** | The campaign whose persisted fields a write actually changes. Today the flush reaches it through the retained `event_id`, then `AttackEvent.campaign_id` (`neo4j_client.py:418,437`). The writer doesn't read the record's `campaign_id`. They matched in all 18 flushed records. `OBSERVED` |
| **Intended campaign** | The campaign the selected D5 alternative says should receive the update. Undefined until D5 is decided |
| **Campaign activity time** | The latest time of an event that counts as activity under D1 to D3. What lifecycle decisions would read |
| **Evidence observation time** | The latest time any associated alert was sighted, whether or not it counts as activity. Used as a distinct concept only under D11c |

## Register

| # | Decision | Sub-decisions | Status |
|---|---|---|---|
| D1 | Which event classes count as activity | | Open |
| D2 | UNKNOWN new alerts | | Open |
| D3 | Duplicates, including after closure | **D3-C:** closed-campaign duplicates | Open |
| D4 | Duplicate's time | | Open |
| D5 | Duplicate's campaign | | Open |
| D6 | Flush policy | **D6-R:** is replay determinism required | Open |
| D7 | Unclean crash | **D7-O:** offset-commit boundary | Open |
| D8 | Late events | | Open |
| D9 | No trustworthy timestamp | **D9-S:** future-skewed events | Open |
| D10 | `closed_at` across reopen | | Open, added in draft |
| D11 | Which `last_seen` governs each operation | | Open, added in draft |
| D12 | Field attribution for duplicate writes | **D12-F:** per field group | Open, added in draft |

**Decide together.** D1 sets the frame for D2 and D3. D3, D4, D5 and D12 all concern duplicates. D5 and D6 together determine whether attribution depends on timing. D10 and D11 determine what restart equivalence can mean. Section "Dependencies" lists which validation criteria each one parameterises.

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
| Resolved duplicate, campaign open | 579 same-campaign, 83 cross | Retained campaign's Neo4j value, at flush |
| Resolved duplicate, campaign closed | 478 same-campaign (all C0028), 61 cross | Retained campaign's Neo4j value, at flush |
| UNKNOWN duplicate, campaign open | 2,946 | `AttackEvent` only; no campaign activity |
| UNKNOWN duplicate, campaign closed | 4,606 | `AttackEvent` only; no campaign activity |

`OBSERVED`. "Cross" means retained ≠ classified, with the status taken from the classified campaign.

| Alternative | Consequence | Evidence |
|---|---|---|
| **a. All extend** | One long repeating alert keeps a campaign open indefinitely, since the dedup window slides. Timeout means less | Part of A: 6 to 7 inversions |
| **b. None extend** | Campaigns close on 120 s without a new alert while the same alert is still firing. Repeats stay recorded on the event | Part of B: 0 inversions, by omission |
| **c. Only while campaign is open** | Closure becomes final with respect to duplicates. Needs a defined rule on the duplicate path. Could remove inversion case 3 by construction | Not tested |
| **d. Resolved only** | Keeps a mapped signal. UNKNOWN repeats, the large majority, don't extend | Closest to CUR's Neo4j behaviour, not its memory behaviour |

### D3-C. What should a duplicate do when its classified campaign is closed?

Needed under D3b, c and d. Under D3a, closed-campaign duplicates extend by definition, so only C3 and C4 would change anything.

| Sub-alternative | Effect | Consequence | Evidence |
|---|---|---|---|
| **C1. Ignore for campaign activity** | Activity time not extended. Whether campaign counters change is left to D12 | Closure stands. The repeat may still count towards volume | Not tested as a rule |
| **C2. Record on the `AttackEvent` only** | No `Campaign` property changes at all, counters included | The event keeps the full record. Campaign totals exclude these repeats | Matches today's UNKNOWN duplicate on `Campaign`. `OBSERVED` for UNKNOWN; not tested for resolved |
| **C3. Associate with a currently active campaign** | The repeat's classified campaign changes | Needs a rule for when no campaign is active. Changes what D5 means, because the dedup entry no longer decides | Not tested |
| **C4. Trigger reopen evaluation** | The duplicate path can reopen a campaign | **Changes grouping semantics, and has not been tested.** Repeats of an old alert could revive a campaign. Adds scoring on the duplicate path: up to 5,145 duplicates on this archive (4,606 + 539, arithmetic). UNKNOWN duplicates have no technique, and no UNKNOWN path reopens today, so they would need their own rule | Not tested |

**Today** matches none of these for resolved duplicates: at flush they change `last_seen` and counters on the retained campaign, even if it is closed. `OBSERVED`, `SOURCE`.

## D4. Own event time, or the first buffered record's time?

**Today's behaviour is neither "first" nor strictly "latest".** The record stores the last arrival's time (`dedup_engine.py:58`), and Neo4j applies it monotonically. An earlier-timestamped duplicate that arrives later therefore lowers the pending value. `SOURCE`, `OBSERVED`.

| Alternative | Consequence | Evidence |
|---|---|---|
| **a. Each duplicate's own time, keeping the maximum** | Faithful to when repeats happened. Needs a monotone merge in the buffer if out-of-order arrivals matter | CUR is close to this, once attribution is right |
| **b. First buffered time** | Bounds the extension to the first repeat; understates long repeats | Not tested |
| **c. No time** (with D3 "none") | Only counters change | B's behaviour |

## D5. Should the first duplicate's association stay authoritative after the campaign changes?

Today a flush mutates the **retained** campaign. It doesn't write to "the duplicate's campaign" when the two differ. D5 decides whether that is what is intended.

| Alternative | Intended campaign | What `retained ≠ classified` means | Consequence | Evidence |
|---|---|---|---|---|
| **a. Keep first association** (today) | Retained | Expected, reported, not a defect by itself | Simple. Later alerts are credited to an earlier campaign when no flush intervenes, and their classified campaign gets no credit | `OBSERVED` 144 of 1,201, harness pattern; C0002, C0045 |
| **b. Current association at classification** (one pending record per fingerprint **and** event) | Classified | Should not occur on the buffered path | Each repeat is credited to its classified event. Slightly more records. The buffer no longer merges across events | Not tested. `PREDICTED` only that the C0002 and C0045 inversions wouldn't arise |
| **c. Flush the old record when a fingerprint re-registers** | Classified | Should not occur on the buffered path | Keeps one record per fingerprint. Adds a write on the new-alert path, and ties flush to classification | Not tested |
| **d. Rely on periodic flush** | Retained, with the owner accepting timing-dependent results | Expected; the rate depends on flush timing | No change to logic, only live cadence. Attribution stays timing-dependent; I5 doesn't hold | **Live rate unmeasured.** The 12% must not be used here |

**D5 doesn't answer:** whether the update extends activity at all (D3), what time it carries (D4), or which fields follow it (D12).

The direct path (buffer full, or `ENABLE_DUPLICATE_BUFFER` false) already mutates the classified campaign. It runs only when no pending record exists for that fingerprint, so its first write hits the same campaign a new record would retain. Later repeats differ, because each one resolves afresh. `SOURCE` (`realtime_socgraph.py:461-475`), not exercised.

## D6. Flush policy

| Alternative | Flush points depend on | Replay determinism of persisted state | Consequence | Evidence |
|---|---|---|---|---|
| **a. Wall-clock periodic + shutdown** (live today) | Processing speed and thread timing | **Not provided.** The same input at a different speed can give different persisted results | Low write load. How far results vary is unmeasured | `SOURCE`; effect unmeasured |
| **b. Event-time periodic** | The input, in principle | Could satisfy it; not tested. Needs rules for idle streams, end of input and late events | No alerts means no advancing time, so idle streams need a separate trigger | Not tested |
| **c. Event-triggered** (for example, on re-registration, see D5c) | The input order, in principle | Could satisfy it; not tested. If it is the only trigger, records can wait indefinitely | Closes the attribution window. Adds work on the hot path | Not tested |
| **d. Direct writes, no buffer** | No flush exists | Removes flush timing as a factor; not tested | Removes the whole class. 1,201 extra writes on this archive and an unmeasured live cost | `ENABLE_DUPLICATE_BUFFER` exists (`config.py:35`), and that path is not exercised in B3 |
| **e. A combination** | Its parts | No better than its weakest part: any wall-clock component brings the dependence back | Flexible; more to test | Not tested |

A shutdown flush can accompany any buffered alternative. The harness's restart-and-end pattern is an artefact of replay, not a candidate. It is why the buffer effect is large in B3.

**A flush policy alone doesn't guarantee determinism.** D5 and D11 decide whether a moved flush point changes attribution or a later decision. The listener's maintenance loop also sleeps 5 s on the wall clock, against an event-time watermark (`wazuh_listener.py:164-180`). `SOURCE`.

### D6-R. Is identical persisted state across replay speeds a requirement?

The owner must say so explicitly, because it constrains D6 and sets V1.3's expected result.

| Answer | Effect |
|---|---|
| **Required** | D6a can't be selected as specified. V1.3 asserts identical hashes |
| **Not required** | A timing-dependent policy is allowed. V1.3 then requires the differences to be listed and limited to what the owner accepted |

## D7. Pending duplicates during an unclean crash

| Alternative | Consequence | Evidence |
|---|---|---|
| **a. Accept loss, document it** | No new machinery. Lost occurrences, `last_seen` and risk increments; dedup may treat a repeat as new after restart. A short flush interval bounds the loss | `SOURCE` |
| **b. Durable buffer** (local journal or write-ahead) | Survives crashes, needs replay and idempotency (V6) | Would need new persistence |
| **c. Write duplicates directly** | No pending state; see D6d | Not tested |
| **d. Reconstruct from `alerts.json`** | Needs an offset committed after processing. Today's offset advances at queue time (`wazuh_listener.py:272-288`) and processing is not idempotent | `SOURCE`; bigger change |

**Crashes are entirely untested.** No evidence exists on what recovery looks like in practice. The B3 graceful-restart results say nothing about this.

### D7-O. When should the alert offset be committed? (added in reconciliation)

V8.5 can't state an expected result without this. It is listed here because D1 to D12 had no entry for it. It isn't an expansion of scope: D7d already depends on it.

| Sub-alternative | Crash outcome, by reasoning | Evidence |
|---|---|---|
| **O1. When queued** (today) | Queued and in-flight alerts are lost. Committed alerts aren't reprocessed | `SOURCE`; not exercised |
| **O2. After `process_alert` returns** | Queued alerts are re-read. The in-flight alert can be reprocessed, and processing isn't idempotent. Pending duplicates still follow D7 | Not tested |
| **O3. After related pending duplicates persist** | As O2, with a wider reprocessing window. Needs idempotency | Not tested |

`seen_alerts` is memory only, so it can't prevent reprocessing across a restart (`wazuh_listener.py:35,125-140`). `SOURCE`.

## D8. Late events: `last_seen` and closure

`OBSERVED`: 0 late alerts in this archive, so all of this rests on code and unit tests.

| Alternative | Consequence |
|---|---|
| **a. Keep today:** accepted, monotone, joins current campaign in arrival order | Simple. A late alert can join a campaign it didn't occur in, and can set `closed_at` before `last_seen` on a score-based close |
| **b. Route to the campaign whose activity span contains its event time** | Historically truer. Needs a lookup, and grouping becomes order-sensitive differently |
| **c. Keep as evidence, no lifecycle effect beyond a tolerance** | Prevents lifecycle changes from stale data. A long ingestion delay would silently stop activity. Any tolerance would be a further owner choice |
| **d. Score close uses `max(event_time, last_seen)`** | Removes that inversion only. Can combine with the others |

## D9. Events with no trustworthy timestamp

The live default is fallback (`config.py:22`), and replay is strict. So **replay evidence doesn't cover the live default.** `SOURCE`; 0 such alerts in this archive.

| Alternative | Consequence |
|---|---|
| **a. Keep:** strict in replay, marked fallback in live | Keeps evidence live. Processing time enters lifecycle fields and advances the watermark (`event_time.py:162`), so a backlog could mis-expire campaigns (`HYPOTHESIS`) |
| **b. Fallback, but don't advance the watermark or extend `last_seen`** | Keeps the event as evidence with no lifecycle effect. May miss real activity |
| **c. Reject in both, keep a dead-letter record** | Deterministic. Risks losing evidence in live operation |
| **d. Also apply to future-skewed alerts** | See D9-S |

### D9-S. Future-skewed events

**What exists today.** `FUTURE_SKEW_TOLERANCE` is five minutes (`event_time.py:65`). An alert more than that ahead of processing time gets `future_skew=True` (`:140`), with its value unchanged. **The flag doesn't stop the alert advancing the watermark** (`:158-163`). Outside the event-time module, the only use a search found is the replay trace (`realtime_socgraph.py:325`), which the listener doesn't pass. So in live operation the flag is neither persisted nor logged. `SOURCE`; not exercised, and replay input is historical so it wouldn't set the flag.

**The policy question is whether a future-skewed event should affect each of these.** Each row needs its own answer.

| Effect | Today |
|---|---|
| Evidence retention | Kept |
| Watermark advancement | Advances |
| Campaign `last_seen` | Extends, under the same rules as any alert |
| Lifecycle expiry | Can trigger it: maintenance expires against the watermark (`wazuh_listener.py:170-172`) |

This register proposes no threshold. Five minutes is today's flag value, and using it for anything beyond flagging would be an owner decision.

## D10. `closed_at` across reopen

| Alternative | ACTIVE campaigns | Historical closure information | Consumers | Evidence |
|---|---|---|---|---|
| **a. Preserve across reopen, overwrite at next closure** (today) | An ACTIVE campaign can hold a stale `closed_at` | Only the latest closure survives | No change. Status must be checked alongside `closed_at`. `closed_at < last_seen` can occur during activity | `SOURCE`; `OBSERVED` C0028 (06:19:05.224 replaced by 06:26:53.353) |
| **b. Clear on reopen** | No `closed_at` while ACTIVE, matching operations (`neo4j_client.py:835`) | Only the latest closure survives, and it is gone while the campaign is active | Must accept null | Not tested |
| **c. Clear on reopen, keep closure history separately** | No `closed_at` while ACTIVE | Every closure is kept | Must accept null. New schema that readers must opt into | Not tested |

**Known consumers.** The only reader of `Campaign.closed_at` a search found is the replay view (`replay/harness.py:256`), so the B3 comparison depends on it. The dashboard's `closed_at` and the frontend's `Operation` type are operation fields (`dashboard_api.py:403`, `frontend/src/types/index.ts:81`). `SOURCE`. A generic read of all node properties wouldn't show in that search, so this needs confirming before any change.

## D11. Which value governs each operation?

| Operation | a. Keep today | b. One value, kept consistent | c. Two fields |
|---|---|---|---|
| **Cached campaign expiry** (alert path and cache sweep) | Memory `context.last_seen` | The single value's memory copy | Memory copy of activity time |
| **Reopen scoring** (fast rule and score) | Neo4j `Campaign.last_seen`, via `load_inactive_campaigns` | The single value, persisted | Persisted activity time |
| **Candidate ordering** (`ORDER BY`, `LIMIT 5`, tie-break) | Neo4j `Campaign.last_seen` | The single value, persisted | Persisted activity time, unless the owner specifies evidence time for this one operation |
| **Database sweep** | Neo4j `Campaign.last_seen`, for ACTIVE nodes the cache sweep didn't close | The single value, persisted | Persisted activity time |
| **Context reload** | Neo4j `Campaign.last_seen` replaces memory | The single value, persisted | Activity time. Evidence time is carried only if a reader needs it |
| Continuation scoring of an active campaign (added for completeness) | Memory | Memory copy | Memory copy of activity time |

| Alternative | Consequence | Evidence |
|---|---|---|
| **a. Keep** | Restart and flush timing change decision inputs. A flush-moved Neo4j value also enters memory at reopen (`campaign_manager.py:643,708`) | `PREDICTED` for C0028 and C0025; `SOURCE` |
| **b. One value** | Needs D1 to D5 settled. Reload then can't differ from memory, apart from pending state. **If duplicates extend and a buffer remains, the owner must say whether decisions see pending activity** | Not tested |
| **c. Two fields** | Could keep duplicates visible without extending activity. Schema change, and every reader must choose | Not tested |

Under D11c, **campaign activity time** comes from events qualifying under D1 to D3 and governs lifecycle. **Evidence observation time** records the latest sighting and never drives lifecycle. Which campaign evidence time attaches to follows D5 and D12. Property names aren't chosen.

## D12. Which attribution do duplicate writes follow, field by field?

D12 covers **every field the retained record changes**, not only `last_seen`. These fields live on three different nodes. `investigation_payload` belongs to `AttackEvent`, not `Campaign`.

| Field | Node | Buffered flush | Direct write (buffer full or disabled) | UNKNOWN duplicate |
|---|---|---|---|---|
| `last_seen` | `Campaign` | Monotone, `r.last_seen` | Monotone | Not reached |
| `occurrences` | `Campaign` | `+ r.occurrences` | `+ 1` | Not reached |
| `total_tps` | `Campaign` | `+ r.total_tps` | `+ tps` | Not reached |
| `risk_score` | `Campaign` | `+ r.total_tps` | `+ tps` | Not reached |
| `last_seen` | `AttackEvent` | Monotone | Monotone | Monotone |
| `occurrences` | `AttackEvent` | `+ r.occurrences` | `+ 1` | `+ 1` |
| `tps` | `AttackEvent` | `+ r.total_tps` | `+ tps` | `+ 0` |
| **`investigation_payload`** | `AttackEvent` | **Replaced** by the last merged payload | Replaced | Replaced |
| `last_seen` | `Technique` | Event time, **not monotone** (`:431`) | **Wall clock** (`:391`) | Not reached |
| `occurrences`, `total_tps` | `Technique` | Add | Add | Not reached |

`SOURCE` (`neo4j_client.py:373-446`). Size not measured.

| Alternative | Consequence |
|---|---|
| **a. All fields follow the same attribution as `last_seen`** | Consistent. Under D3 "none" it would need a rule, because `last_seen` isn't written at all |
| **b. Independent, field group by field group** | Lets a duplicate count without extending activity, for example under D3 "none" (B does this). See D12-F |
| **c. Stop adding duplicate TPS to campaign risk** | Changes risk semantics. Separate from this contract, and needs its own evidence |

### D12-F. Field groups needing an answer

| Group | Fields | Semantics | Question |
|---|---|---|---|
| F1 | `Campaign.last_seen` | Activity | Follows D3, D4, D5 and D11 |
| F2 | `Campaign.occurrences`, `total_tps` | Volume | Same campaign as F1, or the classified campaign regardless of F1? |
| F3 | `Campaign.risk_score` | **Risk, not activity** | Same as F2? |
| F4 | `AttackEvent.last_seen`, `occurrences`, `tps` | Event record | Stay on the retained event, or move to the classified event? |
| F5 | `AttackEvent.investigation_payload` | Evidence shown to analysts (`dashboard_api.py:265,319`) | May a later alert classified to another event replace this event's payload? |
| F6 | `Technique` fields | Outside lifecycle scope | Confirm out of scope, though the same write changes them |

**Activity semantics and risk semantics stay separate.** This register introduces no risk formula and doesn't change what any counter means.

**Fixing D5 doesn't by itself settle D12:**
- It doesn't say whether counters move when activity doesn't.
- `investigation_payload` is overwritten, not added, so choosing a campaign doesn't say which alert's payload wins.
- `restore()` can move counters between records when a newer record exists for the fingerprint (contract F.2, case 5).
- `risk_score` has two meanings. The persisted value is a running sum that includes duplicates. The memory value sums the chain and excludes them (`campaign_manager.py:285,315`). `risk_recalculation.py` can also recompute persisted risk from `AttackEvent.occurrences`. `SOURCE`.

Consumers a search found include `soar/generator.py:39`, `soar/response_plan.py:104`, `active_response/integration.py:65` and `dashboard_api.py:1371,1929`. Which value each one receives wasn't traced.

---

## Dependencies

| Decision | Criteria it parameterises |
|---|---|
| D1 | V1.1, V2.1, V7.1, V7.2, V9.3 |
| D2 | V1.1, V2.1, V7.1, V7.4, V7.5, V9.1, V9.3 |
| D3 | V1.1, V2.1, V3.1, V3.3, V7.2, V7.5, V9.1, V9.3 |
| D3-C | V7.2, V7.4, V9.1 |
| D4 | V3.1, V4.2, V6.4 |
| D5 | V1.3, V3.1, V3.2, V3.4, V3.5, V5.1 to V5.5, V6.4, V9.2 |
| D6 | V1.2, V1.3, V2.1, V2.2, V3.5, V5.1 to V5.6, V6.1 to V6.3, V9.1, V9.6 |
| D6-R | V1.3 |
| D7 | V5.1 to V5.4, V5.7, V5.9, V6.1 to V6.3, V6.5, V6.6, V8.2, V8.5 |
| D7-O | V5.7, V8.4, V8.5 |
| D8 | V4.1, V4.4, V5.8, V9.1 |
| D9 | V1.4, V4.5, V5.8, V9.1 |
| D9-S | V4.5, V9.1 |
| D10 | V5.1 to V5.4, V9.1, V9.4 |
| D11 | V1.3, V2.1 to V2.4, V5.1 to V5.5, V9.5, V9.6 |
| D12 | V3.1, V3.3, V3.6, V6.4 |
| D12-F | V3.3, V3.6 |

Whether V6 applies at all also depends on D6 and D7, as its applicability table sets out.

## Decisions this register does not cover

Scoring changes (unseen transitions, `chain_similarity` above 1, prediction and chain overlap), operation correlation for UNKNOWN-only campaigns, self-loop identity fallback, and ATT&CK mapping. These are separate workstreams.

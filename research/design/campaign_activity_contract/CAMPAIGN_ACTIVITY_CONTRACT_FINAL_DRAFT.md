# Campaign Activity and Lifecycle Contract (review copy)

**Status:** review-ready final draft. **Not an approved specification.** Nothing here is implemented, and it selects no open policy. All of D1 to D12 await the owner.
**Date:** 2026-11-04.
**Companions:** `EVIDENCE_TRACEABILITY_FINAL.md` (sources, by statement ID), `OPEN_POLICY_DECISIONS_FINAL.md` (D1 to D12), `VALIDATION_CRITERIA_FINAL.md` (V1 to V9), `DOCUMENTATION_RECONCILIATION_REPORT.md` (what changed from the draft). `CAMPAIGN_ACTIVITY_CONTRACT_DRAFT.md` is kept unchanged, and this copy supersedes it for review.

**Limits.** The B3 archive has no ground truth, so nothing here measures campaign accuracy, and a change in campaign count is not an accuracy result. All measurements come from one 9,015-alert archive, in which CUR forms 116 campaigns. Replay databases have no ATT&CK corpus. Measured rates don't transfer to live operation, and live flush attribution rates have not been measured.

## How to read this document

| Tag | Meaning |
|---|---|
| `OBSERVED` | Measured in a completed experiment or trace |
| `SOURCE` | Established by reading the current implementation, not exercised for this claim |
| `PREDICTED` | Reproduced by a timing model or calculation, not observed in the relevant execution |
| `HYPOTHESIS` | Plausible, needs more evidence |
| `POLICY` | Must be chosen by the project owner; the D number points to the register |

### Source version
Line numbers refer to the working tree of `isfcr/CYUKTI/backend/` at HEAD `3afe62c` **plus uncommitted Steps 1 to 2A**. HEAD alone does not contain this code. The backend diff against HEAD has sha256 `804bdeb0…`.

| Code | Relation to the working tree | Note |
|---|---|---|
| `b3_diag_v1/variants/CUR` (diagnostic trace) | Identical except `replay/harness.py` and the added `diag_trace.py` | Passed parity against `70c7b12b…` |
| `b3_runs_v2/variants/CUR` | Identical except `replay/harness.py` | Produced the baseline |
| `b3_runs_v2/variants/A` | Adds `touch()` and one `last_seen` update | **Not the repository** |
| `b3_runs_v2/variants/B` | Removes three `c.last_seen` writes | **Not the repository** |
| Repository `replay/harness.py` | Discards the buffer at start (`:319`) and **never flushes it** | Only the study harness flushes (`variants/CUR/replay/harness.py:382-386`) |

`touch()` exists **only in variant A**. It isn't in the repository or in CUR, so its absence from the CUR trace means nothing about its behaviour.

---

## A. Terminology and invariants

### A.1 Terms

| Term | Definition in this contract | Current code |
|---|---|---|
| **Campaign activity** | The events that count as the campaign being alive, for expiry and reopen. **Which events these are is the core open decision (D1).** | Not defined; the writers disagree (see A.2, A.3) |
| **New alert/event** | An alert classified non-duplicate. Creates one `AttackEvent` | `is_duplicate` returns `False` (`dedup_engine.py:258`) |
| **Resolved ATT&CK event** | New alert with a defensible technique | `resolve_mitre` provenance not UNKNOWN or AMBIGUOUS; `process_alert` path `RESOLVED` |
| **UNKNOWN/unattributed event** | Alert without a defensible technique, kept as evidence. Includes AMBIGUOUS | `realtime_socgraph._process_unattributed_alert:170` |
| **Duplicate alert** | Alert whose fingerprint (attacker, victim, technique, rule, agent) is within `DEDUP_WINDOW` (90 s) of that fingerprint's last sighting. The window slides: each duplicate extends it | `dedup_engine.py:164-180`, `:241-243` |
| **Buffered duplicate** | A resolved duplicate merged into the pending record for its fingerprint. UNKNOWN duplicates are not buffered | `DuplicateBuffer.add_duplicate:23-60` |
| **Duplicate flush** | Writing pending records to Neo4j and clearing the buffer | `snapshot:62-66`, `neo4j_client.batch_update_duplicates:410-446` |
| **Classified campaign** | The campaign an alert is assigned to when processed. For a duplicate, the dedup entry's `campaign_id` at that moment | `realtime_socgraph.py:435-441` |
| **Retained campaign** | The campaign stored in the pending buffer record, set when the record is created and not updated by later merges | `dedup_engine.py:42-51` |
| **Mutated campaign** | The campaign whose persisted fields a write actually changes | Flush: retained `event_id`, then `AttackEvent.campaign_id` (`neo4j_client.py:418,437`). The writer doesn't read the record's `campaign_id`. The two matched in 18 of 18 flushed records. `OBSERVED` |
| **Intended campaign** | The campaign the selected D5 alternative says should receive the update | **Undefined until D5 is decided** |
| **Campaign activity time** | The latest time of an event that counts as activity under D1 to D3. What lifecycle decisions would read | Not separated today; `Campaign.last_seen` mixes it with evidence time |
| **Evidence observation time** | The latest time any associated alert was sighted, whether or not it counts as activity | Exists as a separate field only under alternative D11c |
| **Event time** | The alert's own timestamp, in UTC. Lifecycle decisions use it | `event_time.py` |
| **Processing time** | When CYUKTI handled the alert. Recorded separately as `AttackEvent.ingested_at`. Under fallback it is also used as event time, and marked | `event_time.py:128-137` |
| **`last_seen`** | Latest recorded activity. **Exists as four separate values** (see A.2) | |
| **`closed_at`** | Event time at which the campaign became inactive. **Holds only the latest closure** | Four writers (D.2) |
| **Campaign reopen** | An INACTIVE campaign returning to ACTIVE for a new resolved alert, through the fast rule or score at least 0.70 | `campaign_manager.resolve_campaign:610-724` |
| **Restart/recovery** | A new process starting with empty memory, keeping Neo4j. **Four distinct modes, defined in section E.** The B3 "restart" is the first, a **simulation**: flush, then reset of manager, dedup and clock. It doesn't run the listener, the worker or `stop()` | `variants/CUR/replay/harness.py:388-400` |

### A.2 The four `last_seen` values

| Value | Location | Advanced by |
|---|---|---|
| In-memory campaign | `CampaignContext.last_seen` | Resolved alerts, reopen, creation; replaced on reload |
| Persisted campaign | `Campaign.last_seen` | Resolved alerts, UNKNOWN new alerts, reopen, creation, flush |
| Persisted event | `AttackEvent.last_seen` | Its own creation, duplicates; **dedup fallback reads this** |
| Dedup cache | `cache[fingerprint]["last_seen"]` | Every sighting of the fingerprint; drives the 90 s window |

`Technique.last_seen`, `Attacker.last_seen` and `Host.last_seen` also exist. They mostly use processing time (`datetime()`, `neo4j_client.py:120,130,151,156,391`), but `batch_update_duplicates` sets `Technique.last_seen` to event time with no monotonic guard (`:431`). `SOURCE`. Outside this contract's lifecycle scope, recorded as a finding.

### A.3 Three operations that must not be conflated

| Operation | Meaning | Example where it stands alone |
|---|---|---|
| **1. Association** | Which campaign an alert belongs to | UNKNOWN duplicate: recorded against a campaign, which gets no `last_seen` update. `OBSERVED` (`smoke/dup_update_CUR.json`) |
| **2. Activity extension** | Whether the alert keeps the campaign alive | New UNKNOWN alert: extends Neo4j but not memory. `SOURCE`, with `OBSERVED` effect in 25 early closures |
| **3. Timestamp persistence** | When and where a value reaches Neo4j | Buffered duplicate: written at flush, possibly hours later, possibly to the retained campaign rather than the classified one. `OBSERVED` |

The 19:20:11 case shows all three separating. The alerts were **classified** to C0106, **did not extend** C0106, were **retained** against C0002, and **mutated** C0002. `OBSERVED`

### A.4 Proposed invariants

**Correction from the draft.** The draft said these don't depend on any open policy. That was wrong for I3 to I7, because some listed alternatives deliberately don't satisfy them. Only I1 (apart from the marked-fallback exception D9 governs), I2 and I4a are independent of D1 to D12. The others apply only if the named decisions make them apply. Each becomes binding only on owner approval.

| # | Invariant | Applies unless | Holds today? |
|---|---|---|---|
| I1 | Lifecycle decisions use event time; processing time is stored separately | Marked fallback (D9) | Yes, except marked fallback. `SOURCE`, and `OBSERVED` 9,015/9,015 `ALERT_TIMESTAMP` |
| I2 | Campaign `last_seen` never moves backwards | Always applies | Yes in memory and Neo4j. `SOURCE`; not stressed, 0 late alerts in this archive |
| I3 | A defined activity event extends the same campaign in memory and Neo4j, allowing pending state only as D6 defines it | D1d or D2c keep the split deliberately | **No.** `OBSERVED` |
| I4a | **Association.** An alert that reaches a campaign is associated with exactly one classified campaign | Always applies | Yes. `SOURCE` |
| I4b | **Mutation.** A write made on an alert's behalf changes the intended campaign | Applies under every D5 alternative, but what it asserts depends on which | Depends on D5. Under D5a, yes: mutated = retained, with 144 of 1,201 retained ≠ classified. Under D5b or c, no. `OBSERVED`, harness flush pattern |
| I5 | Grouping and persisted activity don't depend on flush timing | D5d or D6a accept the dependence | **No.** Grouping unchanged across one restart; `last_seen` changed. `OBSERVED` |
| I6a | **Grouping.** The same input gives the same grouping across a graceful restart | Depends on D5, D6, D11 | Yes at one tested restart point. `OBSERVED` |
| I6b | **Persisted fields.** The same input gives the same persisted fields across a graceful restart | Depends on D5, D6, D10, D11 | No for CUR and A. Yes for B. `OBSERVED` |
| I7 | Flushing a record twice has the same effect as once | Needed only if retry or durable replay exists (D6, D7) | **No.** Counters add. `SOURCE` |

**The draft's I4** read: "An alert's associated campaign is the campaign any write on its behalf changes." That wording assumed D5b or c, so it is split into I4a and I4b. **Association consistency doesn't guarantee mutation consistency:** I4a holds for the 19:20:11 alerts, classified once to C0106, while their deferred write changed C0002.

**What each invariant applies to**

| Invariant | Event-to-campaign association | In-memory activity | Persisted activity | Deferred duplicate writes | Counters, risk, payload |
|---|---|---|---|---|---|
| I1 | No | Yes | Yes | Yes | No: not time values |
| I2 | No | Yes | Yes | Yes, at the Neo4j write. **Not** inside a pending record (D4) | No |
| I3 | No | Yes | Yes | Only if D3 makes them activity | No |
| I4a | **Yes** | No | No | No: says nothing about where the write lands | No |
| I4b | No | No | Yes | **Yes** | Only as far as D12 says fields follow the same attribution |
| I5 | Yes, for grouping | No | Yes | Yes | Only as far as D12 says |
| I6a | Yes | No | No | No | No |
| I6b | No | No | Yes | Yes | Yes |
| I7 | No | No | Yes | **Yes** | **Yes**: counters are what doubles |

`AttackEvent.last_seen` follows its own monotone rule, and V4.3 covers it.

---

## B. Event-to-activity behaviour matrix

"Current" columns describe CUR. "Contract" gives the open decision, or "yes" where every listed alternative agrees. Rows marked `POLICY` do not select an answer.

| # | Case | Associated with a campaign? | Should extend activity (contract) | Updates in-memory `last_seen` (current) | Updates Neo4j `Campaign.last_seen` (current) | Authoritative timestamp | Class |
|---|---|---|---|---|---|---|---|
| 1 | New resolved alert | Yes: resolved, reopened or created (`realtime_socgraph:493`) | Yes. CUR, A and B all agree | Yes, monotone (`campaign_manager:774,280`) | Yes (`neo4j_client:709,137`) | Its event time | `OBSERVED` 110 alerts; `SOURCE` |
| 2 | New UNKNOWN alert | Yes: active campaign, or new shell (`:240-251`). Never reopens | **Open, D2** | **No**, except at creation (`campaign_manager:465`) | **Yes** (`neo4j_client:312`) | Its event time | `OBSERVED` 152 alerts; 25 early closures match; `POLICY` |
| 3 | Duplicate, resolved | Classified to the dedup entry's campaign, which may be closed | **Open, D3** | **No** | At flush, on the retained campaign, which may differ from the classified one | Latest arrival merged into the record (D4) | `OBSERVED`; `POLICY` |
| 4 | Duplicate, UNKNOWN | Classified to the dedup entry's campaign, which may be closed | **Open, D3, D3-C** | **No** | **No.** Stops at `MATCH (e)-[:MATCHES]->` (`neo4j_client:388`). `AttackEvent.last_seen` does update | Event: its event time. Campaign: none | `OBSERVED` 7,552; `smoke/dup_update_CUR.json`; `POLICY` |
| 5 | Buffered duplicate, retained campaign = classified | Retained = classified = mutated | **Open, D3, D3-C** | **No** | At flush | Latest arrival | `OBSERVED` 1,057 adds: 579 campaign open, 478 closed, all C0028 |
| 6 | Buffered duplicate, retained campaign ≠ classified | **Diverges:** classified to X, retained and mutated Y | **Open, D3 and D5.** D3 says whether it extends. D5 says whether X or Y is intended: Y under D5a or d, X under D5b or c | **No** | Y (mutated) yes at flush; X (classified) **no** | Today Y gets X's alert time | `OBSERVED` 144 adds: 83 with X open, 61 with X closed; `POLICY` |
| 7 | Duplicate buffer flush | No new association | Doesn't itself define activity; carries D3 and D5 | **No** | Yes, monotone, on the retained campaign: `AttackEvent.campaign_id` of the record's retained `event_id` | `r.last_seen` of the record | `OBSERVED` 18 records, 5 campaigns moved |
| 8 | `touch()`, campaign in memory | Variant A only | **Open, D3**. Not in the repository | A: yes | A: no; Neo4j lags until flush | Alert's event time | `SOURCE` for A's code; `PREDICTED` for C0025 06:21:25.702; never traced |
| 9 | `touch()` after restart, campaign absent | Variant A only | **Open, D3 and D11** | A: no-op (`if context is not None`) | A: no | None applied | `SOURCE` for A's code; `PREDICTED` for C0025 06:21:05.795; **no-op not observed** |
| 10 | Late event (earlier than watermark) | Yes, in arrival order, to the current campaign | **Open, D8** | Monotone: unchanged unless later | Monotone: unchanged unless later | Its event time, never moving `last_seen` back | `SOURCE`; `OBSERVED` 0 late alerts here |
| 11a | Bad timestamp, STRICT (replay) | No: rejected | No | No | No | None | `SOURCE`; `OBSERVED` 0 rejected here |
| 11b | Bad timestamp, FALLBACK (**live default**, `config.py:22`) | Yes, marked | **Open, D9** | Yes, with processing time | Yes, with processing time | Processing time, marked | `SOURCE`; not exercised |
| 12 | Campaign reopen | Yes | Yes: all policies do | Yes, monotone (`campaign_manager:643,708`) | Yes, monotone (`neo4j_client:632`) | Triggering alert's event time | `OBSERVED` 13 reopens. `closed_at` is **not cleared** (D10) |
| 13 | Real listener shutdown, pending duplicates | No new association | Handling of pending records follows D3, D6, D12; V8 tests it | Process ends | Intended: final flush. **Runtime unverified**, see F | Record's `last_seen` | `SOURCE`; `HYPOTHESIS` on runtime |
| 14a | Restart after a clean flush (harness simulation only so far) | Existing campaigns kept; contexts reload lazily | What equivalence means follows D5, D6, D10, D11 | Replaced from Neo4j at next load (`campaign_manager:305`) | Unchanged at restart | Neo4j value at reload | Grouping: `OBSERVED`, one point. Values: `PREDICTED` |
| 14b | Unclean crash, pending duplicates | Records lost | **Open, D7** | Lost | **Not written** | None | `SOURCE`; not exercised |
| 15 | Duplicate when buffer is full | The fingerprint has no pending record, so mutated = classified. A first buffered write would have targeted the same campaign | D5 decides | No | Immediately (`realtime_socgraph:461`) | Its event time | `SOURCE`; not exercised. Later repeats differ from the buffered path, since each resolves afresh instead of merging. **Corrects the draft**, which said the attribution differs without this qualification |

---

## C. Duplicate-buffer semantics

| Question | Answer | Class |
|---|---|---|
| Fingerprint to record | One pending record per fingerprint (sha256 of attacker, victim, technique, rule, agent) | `SOURCE` `dedup_engine:42-51,164-180` |
| Is the first association retained? | **Yes.** `event_id` and `campaign_id` are set once and never updated while pending | `SOURCE` `:42-51`; `OBSERVED` |
| How are later duplicates represented? | Merged: `occurrences`, `total_tps`, `latest_payload`, `last_seen` overwritten with the latest arrival | `SOURCE` `:54-58` |
| Which time is kept? | **The latest arrival's, not the first's.** Fingerprint `9b729cf5…` first buffered at 03:27:29.702, flushed carrying 19:20:11.029. Assignment is not monotone inside the buffer, though Neo4j's write is | `OBSERVED`; `SOURCE` |
| Does a new event update the pending record? | **No.** `register_event` updates only the dedup cache (`:260-285`) | `SOURCE`; `OBSERVED` |
| Can a record outlive the campaign context? | **Yes.** All 18 flushed records had INACTIVE destinations, none in memory. Oldest waited 16.1 event-hours (harness) | `OBSERVED` |
| Does the buffer depend on memory? | It holds ids only and writes through `AttackEvent.campaign_id`. The write works with no context | `SOURCE` `neo4j_client:437`; `OBSERVED` |
| What does a flush write against? | The **retained** association. The record holds `event_id` and `campaign_id`. The Cypher finds the event by `r.event_id`, then the campaign by `e.campaign_id`, and doesn't read `r.campaign_id`. It doesn't write against the campaign the merged duplicate was classified to, when the two differ | `SOURCE` `neo4j_client:418,437`; `OBSERVED` 18 of 18 flushed records matched |
| What else follows the retained record? | On `Campaign`: `occurrences`, `total_tps`, `risk_score`. On `AttackEvent`: `last_seen`, `occurrences`, `tps`, `investigation_payload`. On `Technique`: `last_seen`, `occurrences`, `total_tps`. **So risk and evidence payload also follow it.** Field-by-field table in D12 | `SOURCE` `neo4j_client:421-443`. Size not measured |

**The four campaign terms in one line.** The alert has a classified campaign, the record has a retained campaign, the write has a mutated campaign, and the policy has an intended campaign. Today mutated = retained. Whether either should equal classified is D5.

### C.1 The observed mechanism

A pending record survives while its fingerprint gets re-registered to new events. A duplicate classified to the new event then merges into the old record, so its retained campaign is not its classified campaign.

**The 19:20:11 case** is two alerts, index 8942 and 8943, each with its own fingerprint. `OBSERVED`
- Classified campaign, for both: **C0106**, ACTIVE, in memory.
- Retained and mutated campaign, for both: **C0002**, closed 03:29:29, not in memory.
- Fingerprint `9b729cf5…` was registered to 24 events in total, the first at index 5.
- At flush, `C0002.last_seen` moved from 03:29:08.179 to 19:20:11.029.

**C0045** is the same case at 18:39:48: alerts 8766 and 8767 were classified to C0093 and C0092. `OBSERVED`

**Scale in this run:** 144 of 1,201 `BUFFER_ADD` events (12%) had retained ≠ classified, across 8 fingerprints. 17 campaigns received a duplicate credited elsewhere. `OBSERVED`, **under the harness flush pattern only.** These are cross-campaign retained-attribution cases. Whether they are defects depends on D5.

### C.2 The two flush patterns

| | Study harness | Live listener |
|---|---|---|
| Trigger | Restart and end of run | Every 7 s wall clock, plus stop |
| Location | `variants/CUR/replay/harness.py:382-386,393,411` | `dedup_engine.py:119-146`; `listener/wazuh_listener.py:320-322,337-338` |
| Time base | Alert order | Wall clock, while dedup uses event time |
| Batching | One batch | 200 per batch |
| On failure | Exception propagates | `restore()` re-buffers everything, including batches already written. Cases in F.2 |
| Thread | Alert thread | Separate thread |
| Evaluated? | Yes | **No** |

**The 12% is not a live rate and must not be cited as one.** Cross-campaign merging needs a fingerprint to lapse (over 90 s of event time) and re-register with no flush in between.
- `HYPOTHESIS`: at real-time pace, a 7 s flush would prevent most of this.
- `HYPOTHESIS`: it could still occur where event time runs faster than wall clock, such as startup catch-up or a backlog, or after flush failures.
- Neither has been measured.

### C.3 Flush timing, attribution and scores

| Effect | Class |
|---|---|
| Flush moved `last_seen` on 5 of 7 destination campaigns | `OBSERVED` |
| Splitting the buffer at index 4500 reproduces `last_seen` for all 116 campaigns in frozen `CUR_restart_r2` (C0002 → 06:10:57.532, C0034 → 19:20:11.029) | `PREDICTED`; restart path not traced |
| Reopen scoring reads Neo4j `last_seen`, so flush timing can change the temporal term | `SOURCE` `campaign_manager:360`; `campaign_feature_engine:92-118` |
| C0028 scores 0.8843, 0.8883, 0.8823 reproduce from the temporal term alone | `PREDICTED`; **restart score's input not directly read** |
| Reopen candidates are ordered by Neo4j `last_seen`, limited to 5. 36 of 39 candidate loads returned exactly 5, so a moved `last_seen` could change who is even considered | `SOURCE` `neo4j_client:612-613`; `OBSERVED` for the sizes; effect is a `HYPOTHESIS` |

---

## D. Lifecycle and timestamp semantics

### D.1 Intended relationships

**Correction from the draft.** The draft proposed "one defined `last_seen`" for expiry and reopen. That pre-selected D11b over D11a and D11c, so those two rows now say "Open".

| Relationship | Proposed | Today | Class |
|---|---|---|---|
| Time base | Event time for expiry, reopen, closure; processing time only as audit, or marked fallback | Holds in the alert path | `SOURCE` |
| Monotonicity | `Campaign.last_seen` non-decreasing | Holds | `SOURCE` |
| Expiry reference | Event-time watermark | Holds, but see D.4 | `SOURCE` |
| Expiry input | **Open, D11** | **Two:** memory if cached, Neo4j otherwise | `SOURCE`; `POLICY` |
| Reopen scoring input | **Open, D11** | **Neo4j**, while expiry of a cached campaign uses memory | `SOURCE`; `POLICY` |
| `closed_at` | Records closure in event time; relation to later activity per D3, D8 and D10 | See D.2, D.3 | |

### D.2 `closed_at` writers

| Writer | Value | Run count (`CUR_norestart`) |
|---|---|---|
| `close_expired_context:201-208` | Memory `last_seen` + 120 s | 96 |
| `resolve_campaign:601-604`, score below 0.35 | Triggering alert's event time | 21 |
| `expire_active_campaigns:499-505`, cache sweep | Memory `last_seen` + 120 s | 11 |
| `expire_stale_campaigns_db`, `neo4j_client:751` | Neo4j `last_seen` + 120 s | 0 in this run |

`OBSERVED` counts, `SOURCE` values. `close_campaign_db` overwrites any earlier value, and `reopen_campaign_db` does not clear it (`neo4j_client:624-640`; operations do clear theirs at `:835`). So only the latest closure survives: C0028's 06:19:05.224 was replaced by 06:26:53.353. `OBSERVED`

### D.3 When `closed_at < last_seen` can occur

| Case | Mechanism | Assessment | Class |
|---|---|---|---|
| 1 | Cross-campaign buffer credit: C0002, C0045, gaps of about 16 and 12 hours | **Suspicious.** The time belongs to another campaign's alert. D5a or d would accept it, and V9.2 would then list it | `OBSERVED` |
| 2 | Same, shifted by restart: C0034 in `CUR_restart_r2` | **Suspicious**, on the same terms as case 1 | Frozen value `OBSERVED`; mechanism `PREDICTED` |
| 3 | Duplicates arriving after their own campaign closed: A's C0023, C0082, C0083, C0089, gaps 4.4 to 39.8 s | **Unresolved**, depends on D3. Arguably true activity, or arguably noise | Frozen values `OBSERVED`; per-campaign mechanism `HYPOTHESIS`, A was never traced |
| 4 | Reopened, ACTIVE, with a stale `closed_at` | **Suspicious** while it lasts; D10 | `SOURCE`; transient, absent from end-state views (9 reopened campaigns all reclosed) |
| 5 | Score-based close triggered by a late alert: `closed_at` = that earlier event time | **Unresolved**, D8 | `SOURCE`; not exercised, 0 late alerts |
| 6 | Mixed time bases: fallback alert stamps processing time | **Unresolved**, D9 | `SOURCE`; not exercised |

A's restart case adds C0020, the same kind as case 2. **B's zero is not proof of correctness.** B doesn't record this activity on the campaign; the repeats still occur. `OBSERVED`

**A related condition without an inversion.** 30 of 101 expired campaigns closed before Neo4j `last_seen` + 120 s, though after `last_seen` in 28 of them, the other 2 being C0002 and C0045 above. 5 were moved by the flush; 25 have `last_seen` equal to their last new UNKNOWN alert. `OBSERVED`. The UNKNOWN cause is `SOURCE` (row 2).

**The 101 is a campaign count, not a write count.** Of 116 campaigns, the last close was an expiry for 101 (90 alert path, 11 cache sweep), a score for 14, and 1 ended ACTIVE. D.2's 96 and 11 count every close write, including re-closures. `OBSERVED`, this reconciliation.

**Scale of duplicates on closed campaigns.** `OBSERVED`
- 4,606 of 7,552 UNKNOWN duplicates (61%) arrived while their campaign was closed.
- 539 of 1,201 resolved duplicates did too: 478 same-campaign, 61 cross-campaign.
- This is possible because the dedup window slides independently of campaign state. `SOURCE`

### D.4 Other time hazards

| Hazard | Class |
|---|---|
| `EventClock.observe` advances the watermark for any alert (`event_time.py:162`). A **future-skewed** alert (more than five minutes ahead, `:65,140`) is only flagged, and a **fallback** alert uses wall clock. Maintenance expires against the watermark, so one such alert could expire every ACTIVE campaign older than it by more than 120 s, for example during a backlog | `SOURCE`; effect `HYPOTHESIS` |
| The `future_skew` flag reaches only the replay trace (`realtime_socgraph.py:325`). The listener passes no trace (`wazuh_listener.py:149`), so in live operation the flag is neither persisted nor logged | `SOURCE`; D9-S |
| `archive_old_campaigns` uses `is_expired` with no reference time, so wall clock. It has **no callers** in the backend, tests excluded | `SOURCE` |
| A late alert joins the currently active campaign, not the historical one | `SOURCE`; Step 2 §9.2 |
| Tied `last_seen` among candidates uses Neo4j's unspecified order | `SOURCE`; did not occur in these runs |

### D.5 Repairs and what they would cost

This contract prescribes none.

| Possible repair | Effect |
|---|---|
| Clamp `closed_at` up to `last_seen` | Hides the anomaly. A campaign may look alive until another campaign's alert. Destroys the signal in cases 1 to 2 |
| Clamp `last_seen` down to `closed_at` | Discards real sightings, and breaks I2 if applied later |
| Rewrite history in Neo4j | A data migration. Changes earlier reopen and candidate-order inputs after the fact; needs versioning, before/after counts, provenance |
| Change attribution only (D5b or c) | Removes cases 1 to 2 going forward; leaves history and cases 3 to 6 |

---

## E. Restart and recovery semantics

**Four modes, which must not be conflated.**

| Mode | What happens | Status |
|---|---|---|
| **H. Harness-simulated graceful restart** | Study harness flushes with its own call, then resets manager, dedup and clock. No listener, no worker, no `stop()` | **Run once**, before alert 4,500. `OBSERVED` |
| **L. Real listener shutdown** | Signal to the listener, its `finally` block, `DuplicateFlushWorker.stop()`, then a new process | **Never run.** `SOURCE` only |
| **C. Unclean process termination** | Process killed with records pending | **Never run.** `SOURCE` only |
| **D. Durable recovery after a crash** | Exists only if selected under D7. Doesn't exist today | **Doesn't exist** |

The B3 restart is mode H at one point in one archive. Crashes, multiple restarts, other restart points and the real shutdown path are **untested**. **The H results are not evidence of correct shutdown or crash recovery.**

| Guarantee wanted | Mode | Available with the existing design? | Needs a new mechanism? | Evidence |
|---|---|---|---|---|
| Graceful shutdown drains queued alerts before flushing | L | **By reading the code, in this order:** stop flag, queue join, worker join, flush (`wazuh_listener:333-338`). Offset isn't part of this order. See F.3 | Depends on D7-O | `SOURCE`; not exercised |
| Pending duplicates flushed at shutdown | L | **Intended**; undermined by the finding in F.1 | No, if F.1 is confirmed and resolved | `SOURCE`; runtime unverified |
| Grouping unchanged across graceful restart | H | Holds in the one case tested | Unknown beyond it | `OBSERVED`, all three policies |
| Persisted fields unchanged across restart | H | **No** for CUR, A. Yes for B, by not recording | Needs D3, D5, D11 | `OBSERVED` |
| Cache reload after restart | H, L, C | **Yes, lazily**: one active context per pair at next alert (`campaign_manager:400-423`). Absent campaigns are closed by the DB sweep using Neo4j `last_seen` | No, but D11 decides which value is right | `SOURCE`; `PREDICTED` for C0025 |
| Dedup continuity | H, L, C | **Partial:** rebuilt from `AttackEvent.last_seen` (`neo4j_client:1041-1066`). Pending duplicates aren't there, so after a crash a fingerprint may look lapsed | Durable buffer, or accept (D7) | `SOURCE`; paths identical across mode H restart `OBSERVED` |
| Watermark continuity | H, L, C | **No.** Empty until the first alert; lateness across a restart isn't detected | Persist or derive it | `SOURCE` |
| Campaign reconstruction | H, L, C | **Yes** from Neo4j, for resolved chains. `get_campaign_chain` omits UNKNOWN events (`:1023-1027`) | No | `SOURCE` |
| Multi-step consistency for one alert | C | **No.** Campaign writes, event creation and chain update are separate transactions | Transaction or recovery logic | `SOURCE`; crash not tested |
| Idempotent flush | L, C, D | **No.** Counters add. `restore()` after a partial failure would re-apply written batches (`dedup_engine:125-139`). See F.2 | Needed if retry or replay exists: record identity or a flush ledger | `SOURCE`; not exercised |
| No loss of pending duplicates on crash | C, D | **No.** Buffer is in memory only | Only if D7 doesn't accept the loss | `SOURCE` |
| No loss of alerts on crash | C, D | **No.** `offset.dat` advances when an alert is **queued**, not when processed (`wazuh_listener:272-288`). `seen_alerts` is memory only | Depends on D7-O | `SOURCE`; not exercised |

**The B3 restart didn't test the real path.** It calls its own flush (`harness.py:382-386`), not `DuplicateFlushWorker.stop()`. So its results say nothing about section F, V5.5 or V5.7.

---

## F. Shutdown and flush-failure findings

Everything in this section comes from reading the code. **None of it has been executed, and V6 and V8 have not passed.**

### F.1 Undefined `logger` in `stop()`

**Finding, `SOURCE`, runtime behaviour untested.** `DuplicateFlushWorker.stop()` references `logger` at `dedup_engine.py:110` and `:116`. The module defines no `logger` and doesn't import `logging`.

**Potential failure path, by reading the code, with pending records present:**
1. `stop()` takes the pending records (`:104`) and writes each batch (`:107-109`).
2. `logger.info` at `:110` would raise `NameError`, after the writes.
3. `except Exception` at `:114` would catch it and call `restore(pending)` at `:115`, re-buffering records already written.
4. `logger.exception` at `:116` would raise `NameError` out of `stop()`, into the listener's `finally` at `wazuh_listener:338`. `logger.info("Listener stopped.")` at `:339` would not run.

**If a batch write fails instead,** step 3 follows the failure and step 4 still raises. Pending records stay only in a process about to exit, so they would be lost. With no pending records, `logger` isn't reached.

**Why this remains unverified:**
- The listener wasn't run in any experiment.
- The harness doesn't call `stop()`.
- No test in `backend/tests/` references `DuplicateFlushWorker` or `duplicate_flush_worker`.
- A search of the backend found nothing that injects `logger` into this module, but static reading can't exclude every runtime path.

**Recommended separate task, not done here.** An isolated unit test of `stop()` with a fake batch writer and no database, covering pending records with a successful write, pending records with a failed write, and an empty buffer. It should run against a copy, and be approved on its own.

### F.2 Flush failure cases

Each batch is one Cypher statement in one auto-commit transaction (`neo4j_client.py:413-446`), so a batch should be all or nothing. That rests on Neo4j's transaction behaviour, and no test has exercised it. "Partial" therefore means across batches today, not within one.

| # | Case | Worker path, `run()` | Shutdown path, `stop()` |
|---|---|---|---|
| 1 | **Write fails before persistence** | `restore()`; retried next cycle. Correct, unless case 5 applies | `restore()`, then F.1's second `NameError`. Records stay only in a process about to exit, so they would be lost |
| 2 | **Batch partially persists.** Within a batch: not expected. Across batches: earlier batches commit, a later one fails | `restore()` re-buffers **all** records, so committed batches apply again. Counters double | Same re-buffering, but the process exits, so nothing is re-applied and uncommitted batches are lost |
| 3 | **Batch succeeds, then logging fails** | Only if `print` raises, which is unlikely. Same effect as case 2 | **F.1's case.** `NameError` after writes, `restore()`, second `NameError`. Neo4j holds one application. Re-buffered records die with the process |
| 4 | **Records restored, no newer data** | Record returns unchanged. No capacity check on restore | Not relevant at exit |
| 5 | **Restored records collide with newer data for the same fingerprint** | See below | Unlikely, since the queue has drained |
| 6 | **Commit outcome unknown:** client sees an error, server committed. `HYPOTHESIS`, general behaviour of a client and server over a network | Treated as a failure, so re-applied. Counters double | As case 3 |

**Case 5 in detail.** `restore()` (`dedup_engine.py:72-82`), given a newer pending record for the fingerprint:
- Adds the failed record's `occurrences` and `total_tps` to the newer record. Those counters now follow the newer record's `event_id` and campaign, whichever campaign they were first retained against.
- Overwrites the newer record's `last_seen` and `latest_payload` with the **older** values. The pending time and payload move backwards, and Neo4j's monotone write doesn't protect the payload.

The window exists because the worker snapshots and clears the buffer, then writes while alerts keep arriving.

**A record can also change less than it appears to.** If the event has no `MATCHES` edge, the statement stops after updating `AttackEvent`, and `Campaign` isn't reached. No error is raised. All 18 flushed records had the edge, so this didn't occur. `OBSERVED`, `SOURCE`.

### F.3 Shutdown sequence and input handling

**The sequence V8 requires isn't the sequence the listener has.** V8 sets out the required order and what the code does at each step. Findings:

| Finding | Class |
|---|---|
| The offset is written during polling, when an alert is queued, so it precedes processing and flush (`wazuh_listener.py:272-288`) | `SOURCE` |
| `alert_queue.join()` has no timeout, so a blocked `process_alert` would hold shutdown (`:334`) | `SOURCE` |
| An alert whose processing raises is logged and marked done, with no retry, and its offset has already advanced (`:153-156`) | `SOURCE` |
| `stop()` joins the worker for 2 s, while the worker sleeps 7 s (`dedup_engine.py:102,121`). After waking, the worker snapshots once more without checking `running`. It is a daemon thread, so an in-flight write can be cut off at exit | `SOURCE` |
| **Queue-full alerts can be lost silently.** `already_seen` records the id at `:264`, before `put_nowait` at `:272`. On a full queue the offset stays, and the next poll re-reads the line, finds it already seen, logs a raw duplicate and advances. Applies within the 600 s cache lifetime | `SOURCE`; not exercised |

---

## G. Policy decisions

Full alternatives, consequences and evidence are in `OPEN_POLICY_DECISIONS_FINAL.md`. No decision is taken here.

| # | Decision |
|---|---|
| D1 | Which event classes count as campaign activity |
| D2 | Do UNKNOWN new alerts extend activity |
| D3 | Do duplicates extend activity. **D3-C:** what a duplicate does when its campaign is closed |
| D4 | Which time a duplicate extends with |
| D5 | Is the first duplicate's association authoritative after the campaign changes |
| D6 | Flush policy. **D6-R:** is replay determinism required |
| D7 | Pending duplicates on unclean crash. **D7-O:** offset-commit boundary |
| D8 | Late events, `last_seen` and closure |
| D9 | Events with no trustworthy timestamp. **D9-S:** future-skewed events |
| D10 | `closed_at` across reopen (added in draft) |
| D11 | Which `last_seen` governs each operation (added in draft) |
| D12 | Which attribution duplicate writes follow, field by field, including `investigation_payload`. **D12-F:** field groups (added in draft) |

## H. Acceptance criteria

Testable criteria are in `VALIDATION_CRITERIA_FINAL.md`. They are written against decision outcomes, not against a chosen policy. **None has been run.**

---

## Findings preserved without policy change

| Finding | Class | Boundary |
|---|---|---|
| 30 early closures: 5 moved by flush, 25 with `last_seen` equal to their last new UNKNOWN alert | `OBSERVED` | Input to D2, D3 |
| C0028's winning reopen candidate led by only 0.0131; no threshold crossed | `OBSERVED` | One case. One other multi-candidate reopen in the run led by 0.0403 |
| Prediction and chain terms contributed 0.65 of C0028's score from one edge with count 1 | `OBSERVED` | Scoring work, not this contract |
| `chain_similarity` can exceed 1, reaching 30.0 (29 of 231 inputs above 1) | `OBSERVED` | **Separate scoring defect** |
| `duplicate_similarity` was 0 in all 231 scored decisions | `OBSERVED` | Scoring work; not a claim about live data |
| Policy B and CUR had identical assignments, operations and transition counts, but different persisted activity values | `OBSERVED` | This archive |
| B's zero `closed_at < last_seen` is not proof its policy is correct | `OBSERVED` counts; assessment by reasoning | |
| UNKNOWN-only campaigns in the replay received no operations (63/63) | `OBSERVED` | The UNKNOWN path skips correlation |
| UNKNOWN is not the only explanation: in the live graph, 31 of 102 campaigns without an operation have resolved events | `OBSERVED` | Second cause unexplained |

**Out of scope:** scoring changes, operation correlation, self-loop identity fallback, ATT&CK mapping.

# Step 2 — Event-time correctness and deterministic replay

**Date:** 2026-11-04 · **Base:** `3afe62c` + Step 1 (uncommitted working tree) · **Audit findings:** A8 (wall-clock time), A7 (timeout semantics, partially)
**Evidence:** `research/experiments/STEP2-event-time-replay-v1/` (6 full-archive runs; `six_run_comparison.csv`, `comparison.json`, `analysis.json`, per-run manifests/views/traces/decisions/graph snapshots, all scripts).

## 0. Three things this report keeps separate

| Property | Question | Step 2 result |
|---|---|---|
| **Determinism** | Same input + fresh database → same output? | **Yes.** 4 event-time runs at different speeds and maintenance timings are byte-identical (same SHA-256). The old code is not deterministic: 2 runs → 2 different results. |
| **Temporal correctness** | Are activity, inactivity, dedup and expiry decided on the alert's own time? | **Yes, for every campaign-lifecycle decision** (inventory §2). Stored lifecycle timestamps now lie inside the archive's event window (116/116); before, they were the processing date (0/15). |
| **Campaign quality** | Do campaigns match real attacks? | **Not measured, and not claimed.** The event-time code produces **more** campaigns (116 vs 15) on the same alerts. That is the consequence of applying the existing 120 s timeout to real 16-hour event time. It is not evidence of better or worse grouping. Quality needs ground truth and the Step 3/4 metrics. |

## 1. What changed

| File | Change |
|---|---|
| `backend/event_time.py` (new) | Timestamp parsing (`timestamp`, fallback `@timestamp`, converted to UTC); failure classes `MISSING`/`MALFORMED`/`TIMEZONE_MISSING`; policies `strict` (reject) and `processing_time_fallback` (use processing time, **marked**); `EventClock` watermark; `late` and `future_skew` flags. |
| `backend/config.py` | `EVENT_TIME_POLICY` (env; default `processing_time_fallback` for the live listener). The replay harness always uses `strict`. |
| `backend/realtime_socgraph.py` | `process_alert(alert, trace=None, event_time_policy=None)`: resolves event time once and passes it to dedup, campaign resolution/activation/creation, event writes, operation correlation. Optional `trace` dict for replay. |
| `backend/campaign_manager.py` | All lifecycle decisions take `event_time`. Monotone `last_seen`. **An alert finding its campaign expired closes it** (`close_expired_context`). `resolve_active_context_at` for the UNKNOWN path. Maintenance sweeps use a reference time; **cache sweep runs before the DB sweep** (§8). Optional `decision_log`. Legacy no-argument calls keep processing time. |
| `backend/campaign_feature_engine.py` | `temporal_similarity(context, reference_time)`. **Formula and thresholds unchanged**; only the reference moved to event time. |
| `backend/dedup_engine.py` | Window, cleanup and DB fallback in event time; monotone cache `last_seen`; cleanup clock seeded from the first event, not the wall clock; `reset()`. |
| `backend/neo4j_client.py` | Optional `event_time` on campaign/event/operation lifecycle writers; monotone `last_seen`; `AttackEvent.ingested_at` (processing time) and `event_time_source` added; expiry `closed_at = last_seen + timeout`; DB dedup fallback at event time; event lists keep same-timestamp events distinct (`event_id` in `collect(DISTINCT …)`) and order by `(first_seen, ingested_at, event_id)`. |
| `backend/campaign_correlation_engine.py`, `operation_manager.py` | Operation expiry at the alert's event time before choosing candidates; reopen at event time. **Operation scoring unchanged.** |
| `backend/listener/wazuh_listener.py` | Maintenance uses the event-time watermark; does nothing before the first alert. |
| `backend/replay/harness.py` (new) | Deterministic replay CLI; comparison mode. Refuses non-empty databases; MISP and outbound HTTP replaced by recorders. |
| Tests | New `tests/test_event_time.py` (40). `tests/test_chain_learning_regression.py`: fake graph extended with event-time semantics, fixture made reusable; Step 1 tests unchanged and passing. Fakes in `test_realtime_mitre_integration.py`, `test_campaign_manager_maintenance.py`, `test_failure_injection.py` and `benchmarks/run_benchmarks.py` accept the new keyword arguments. **No assertion was changed.** |

Not touched: transition learning rules, chain/prediction scoring, `CAMPAIGN_WEIGHTS`, `ACTIVE_CONTINUE_THRESHOLD`/`REOPEN_THRESHOLD`/`CAMPAIGN_TIMEOUT`/`DEDUP_WINDOW`, ATT&CK mappings, severity, XGBoost, GNN, attribution, MISP logic, production Neo4j.

## 2. Wall-clock inventory

| Use | Decides campaign assignment? | Now |
|---|---|---|
| `CampaignManager.is_expired`, `append_technique`, `create_campaign_context`, reopen paths, `activate_*` | yes | event time |
| `CampaignFeatureEngine.temporal_similarity` | yes (continuation/reopen score) | event time |
| `DeduplicationEngine` window/cleanup, `find_recent_duplicate` | yes (which alerts reach campaign logic) | event time |
| `expire_stale_campaigns_db`, `close_campaign_db`, `reopen_campaign_db`, `update_campaign_activity`, `create_campaign_db`, `create_(unattributed_)attack_event`, `update_duplicate_event`, `batch_update_duplicates` | yes (status / candidates) | event time, monotone |
| `expire_stale_operations_db`, `reopen_operation_db`, listener maintenance | yes (operation candidates) | event time / watermark |
| `prediction_generated_at`, Attacker/Host/Technique `first_seen`/`last_seen`, MISP event `date`, dashboard health timestamps, listener raw-alert-ID TTL, duplicate-buffer flush interval | no | **unchanged** (processing time) |
| Dashboard time-window filters (`datetime() - duration(...)` vs `AttackEvent.first_seen`) | no (display) | unchanged code. These now filter by **event time**: equivalent for live data; replayed historical data appears old. |

## 3. Policies for problematic timestamps

| Case | `strict` (replay) | `processing_time_fallback` (live default) |
|---|---|---|
| Missing / empty | rejected, recorded (`REJECTED_EVENT_TIME`, reason) | processing time used; `event_time_source=PROCESSING_TIME_FALLBACK`, reason kept, warning logged, stored on the AttackEvent |
| Malformed (incl. epoch numbers) | rejected | same as above |
| No timezone (naive) | rejected; **never assumed UTC or local** | same as above |
| Any valid offset (`+0000`, `Z`, `+0530`, `-0500`) | converted to UTC | converted to UTC |
| Future (> 5 min ahead of processing time) | accepted, flagged `future_skew` | same |
| Late / out-of-order (< watermark) | accepted, flagged `late`, processed in arrival order | same |

Late-alert semantics: a campaign's `last_seen` never moves backwards; a negative gap counts as "within activity". **Limitation (tested, documented):** a late alert is evaluated against the campaign state when it *arrives*; it is not retroactively placed into a campaign that has already closed. Replay sorts by event time, so this only arises in live operation.

## 4. Six full-archive runs

Input: `data/raw_logs/alerts.json`, 9,015 alerts, 2026-03-11 03:27:24 → 19:38:38 UTC (sha256 in each manifest). Each run used a **brand-new disposable `neo4j:5-community` container** (port 17690, no volume, removed afterwards). Runs E/F use the Step 1 accepted code rebuilt from `HEAD` + Step 1 edits; that rebuild passed all 13 Step 1 tests. All six runs fed alerts in the same order: (event time, file position).

| Run | Code | Pacing | Maintenance thread | Wall s | Campaigns | Operations | Transitions | Alerts reaching campaign logic | Duplicates | Output hash |
|---|---|---|---|---|---|---|---|---|---|---|
| A | event time | none | none (final sweep only) | 100 | 116 | 29 | 9 | 262 | 8,753 | `a0f93bb1…` |
| B | event time | none, fresh DB | none | 101 | 116 | 29 | 9 | 262 | 8,753 | `a0f93bb1…` |
| C | event time | 2000× | every 0.05 s | 128 | 116 | 29 | 9 | 262 | 8,753 | `a0f93bb1…` |
| D | event time | 200× | every 0.5 s | 393 | 116 | 29 | 9 | 262 | 8,753 | `a0f93bb1…` |
| E | Step 1 (wall clock) | none | none | 73 | 15 | 7 | 10 | 49 | 8,966 | `3ec5edb8…` |
| F | Step 1 (wall clock) | 200× | none | 372 | 20 | 8 | 10 | 71 | 8,944 | `3b196741…` |

Machine-readable: `six_run_comparison.csv`, `comparison.json` (pairwise section-level comparison), `analysis.json`.

### 4.1 Per-alert assignment consistency
The compared view holds, for every alert:
- its path (RESOLVED / UNATTRIBUTED / DUPLICATE / DUPLICATE_UNATTRIBUTED / rejected);
- its campaign, canonicalised by order of first appearance (random `CAMP_` ids differ per run);
- its event time and late flag.

It also holds every alert-driven lifecycle decision (with score) and the final campaign table: attacker, victim, status, first/last seen, `closed_at`, reopen count, ordered technique sequence. Finally it includes the `NEXT_TECHNIQUE` edges with counts and the operation membership, status and `closed_at`.
- **A = B = C = D in every section** (6/6 pairs identical; 9,015/9,015 alerts assigned identically).
- **E ≠ F** in assignments, campaigns, operations and transitions. Same code and same input; the only difference is replay speed. In F, 22 alerts had a different duplicate/non-duplicate status than in E.

### 4.2 Replay-speed independence
Unpaced vs 2000× vs 200× pacing, with the background maintenance thread firing 2–20 times per second (as in the live listener), give identical outputs. Old code: speed changed the campaign count (15 → 20), the number of alerts reaching campaign logic (49 → 71) and operations (7 → 8).

### 4.3 Event ordering, ties, late events
- All 9,015 alerts have a valid, timezone-aware timestamp: 0 rejected, 0 fallback.
- Processing order is non-decreasing in event time; 0 late alerts (expected, since replay sorts).
- 324 alerts share a timestamp with another alert (8,833 distinct timestamps, largest tie group 9). In every tie, processing order equals file order.
- Out-of-order, late-after-expiry, identical-timestamp and file-order-independence behaviour is covered by unit tests (§7) because the sorted archive cannot produce late events.

### 4.4 Lifecycle timestamps
(There is no `created_at` on Campaign; creation time is `first_seen`.)

| Check (run A) | Result |
|---|---|
| `first_seen ≤ last_seen` | 116/116 |
| first/last seen inside the archive's event window | **116/116** (run E: **0/15**, all stamped 2026-11-03, the processing date) |
| `closed_at < last_seen` | 0/115 closed |
| Expired campaigns with `closed_at = Neo4j last_seen + 120 s` | 75/101 |
| Expired campaigns with `closed_at` *earlier* than Neo4j `last_seen + 120 s` | 26/101, see §9.2 (pre-existing B3) |
| Final status | 115 INACTIVE, 1 ACTIVE (the last campaign at the end of the archive) |
| Lifecycle decisions | CREATE 116, CONTINUE 71, CLOSE_EXPIRED 96, MAINTENANCE_EXPIRE 11, CLOSE_BELOW_THRESHOLD 21, FAST_REOPEN 10, REOPEN 3 |

### 4.5 Operation expiry
- **A–D:** 29 operations (28 INACTIVE, 1 ACTIVE), identical membership, status and `closed_at` across all four. `closed_at` values are event times (e.g. 2026-03-11 03:29:33).
- **E/F (old code):** 7 and 8 operations, all ACTIVE. With wall-clock expiry, nothing went stale during a 1–6 minute replay of a 16-hour archive.
- 63 of 116 campaigns have no operation. That is UNKNOWN-only campaigns never reaching operation correlation (audit B3/C2, pre-existing), the same in every event-time run.

### 4.6 Deduplication
- **A–D:** 8,753 duplicates, 262 new events, identical per alert.
- **E/F:** 8,966 vs 8,944. The 90 s dedup window measured on the wall clock covered most of the 16-hour archive when replayed in minutes, so only 49/71 alerts ever reached campaign logic.
- With event time, 262 alerts reach campaign logic (5.3× E). The window now means "90 s of alert time".

## 5. Before vs after on the same alerts (expected changes, with an example)

Only 7 campaign partitions are identical between A and E; 1,250/9,015 alerts take the same path. Example, `192.168.56.106 → pes1ug23cs411-VirtualBox`:
- **E (old):** 3 campaigns. One holds the whole 06:18–06:24 session (13 events: UNKNOWN, UNKNOWN, T1595.002, T1068, T1068, T1055 ×3, …), plus the T1190 and T1210 splits already seen in Step 1. Everything is stamped 2026-11-03 18:56.
- **A (new):** 14 campaigns:
  - 9 UNKNOWN-only shells between 03:27 and 19:20, separated by hours. Previously those alerts were swallowed by the wall-clock dedup window.
  - The 06:18:46–06:24:53 session (14 events).
  - Score-based splits at 06:19:05 (T1190) and 06:19:05.795 (T1210), unchanged in kind from Step 1.
  - Two further fragments at 06:24:49 / 06:24:51 created after split/reopen interactions.

Classification:
- **Expected consequence of event-time semantics:** expiry over real gaps; separate shells for UNKNOWN bursts hours apart; more alerts passing dedup.
- **Pre-existing campaign-scoring behaviour, unchanged by Step 2** (Step 4 scope): the T1190/T1210 splits and the overlapping fragments.
- **Implementation defects of Step 2:** none found in these runs, after the §8 fix.

## 6. Tests

| Suite | Result |
|---|---|
| `tests/test_event_time.py` (new) | **40 passed**: parsing in 5 timezone forms; 7 invalid-timestamp classes; strict vs fallback; `@timestamp`; unknown policy; future skew; stored times are event times; within / exactly at / beyond threshold; expired campaign is a reopen candidate without maintenance; out-of-order; late-after-expiry; identical timestamps; timezone gap; missing/malformed/naive strict rejection without side effects; fallback marking; UNKNOWN-path expiry; watermark maintenance; dedup in event time and late duplicates; loader ordering and rejections; repeated fresh-DB replay; speed + maintenance-thread independence; file-order independence; maintenance race (×2) |
| `tests/test_event_time.py` + `tests/test_chain_learning_regression.py`, 5 consecutive runs | 51/51 passed each time (pre-race-test count) |
| Full backend suite | **916 passed, 29 skipped, 2 failed.** Baseline 863/29/2; Step 1 876/29/2. The same 2 Neo4j-dependent tests fail (`ServiceUnavailable`). |

## 7. Maintenance-race regression test (item 8)

`test_closed_at_does_not_depend_on_whether_maintenance_or_the_alert_path_closes[maintenance_first=True|False]`:

| Code | `maintenance_first=True` | `maintenance_first=False` |
|---|---|---|
| Sweep order before the fix (DB sweep, then cache), applied to a scratch copy | **FAILED**: `closed_at` 03:29:34 ≠ expected 03:29:24 | passed |
| Repository (cache sweep, then DB) | passed | passed |

## 8. Difference found during Step 2 and its root cause

The first version of the speed test failed. Unpaced and paced replays of the same scenario differed only in one campaign's `closed_at` (03:31:24 vs 03:31:34).

**Root cause (pre-existing, audit B3).** An UNKNOWN or duplicate alert refreshes the campaign's `last_seen` in Neo4j but not the in-memory `CampaignContext.last_seen`. The per-alert path and the maintenance cache sweep close with `memory last_seen + 120 s`; the maintenance DB sweep closes with `Neo4j last_seen + 120 s`. Whichever ran first set `closed_at`. The expiry *decision* was the same in both runs; only the recorded close time differed.

**Fix (no semantic change).** The maintenance cache sweep now runs **before** the DB sweep. A DB-expired cached campaign is always memory-expired, because the memory copy is never newer. So every cached campaign is closed by the memory rule, whoever runs first. The DB sweep then only closes campaigns absent from the cache (e.g. after a restart). After the fix: unit test passes, and 4 full-archive runs are identical.

## 9. Open issues (not fixed; outside Step 2 scope or needing a decision)

1. **B3: which alerts count as "activity"?** In memory, only resolved alerts extend a campaign; in Neo4j, UNKNOWN and duplicate alerts do too. Step 2 makes the outcome deterministic, but the two definitions still differ:
   - 26/101 expired campaigns closed earlier than their Neo4j `last_seen + 120 s`, though never before `last_seen`.
   - After a listener restart (context reloaded from Neo4j) the other definition would apply.

   Choosing one definition changes campaign semantics, so it needs your decision.
2. **Late alert after expiry** joins the currently active campaign, not the historical one (§3).
3. **UNKNOWN-only campaigns get no operation:** 63/116 (B3/C2).
4. **Self-loop identities** (`attacker := victim` fallback, C3): `Omkar→Omkar` 45 and `pes1ug23cs411-VirtualBox→itself` 36 of 116 campaigns in run A.
5. **Inactive reopen candidates with tied `last_seen`** are ordered by Neo4j's unspecified tie order. That didn't occur in these runs (identical outputs) and wasn't changed.
6. **Live-graph read-only validation is blocked**: `neo4j-soc` is still `Exited` (checked 2026-11-04).

## 10. Campaign quality: what can and cannot be said

- 116 campaigns over a 16-hour archive with 12 attacker/victim pairs is **not** a quality result. There is no ground truth for this archive (its attacks are not labelled), and the campaign scorer, thresholds and timeout are unchanged.
- The increase from 15 to 116 comes from applying the existing 120 s timeout and 90 s dedup window to real alert time. Before, they were applied to a compressed replay clock, which made the old count an artefact of replay speed (15 vs 20 for the same input).
- The step makes later measurements **reproducible and meaningful**. Inverse purity / campaigns-per-attack (Step 3) and the scoring corrections (Step 4) must be evaluated on labelled data with this harness before any statement about grouping quality.

## Addendum, 2026-11-04: corrections from the B3 v2 study

The text above is kept as first written. Where it conflicts with this addendum, the addendum applies.

| Location | As written | Corrected | Why |
|---|---|---|---|
| §4.4, §9.1 | 26/101 expired campaigns closed before Neo4j `last_seen + 120 s` | **30/101** (`CUR_norestart`) | The Step 2 harness never flushed the duplicate buffer, so buffered activity never reached Neo4j |
| §4.4, §9.1 | 0 campaigns with `closed_at < last_seen` ("never before `last_seen`") | **2** without restart, **3** with one restart | Same cause. Campaign grouping is unaffected; `view_hash` differs because stored timestamps differ |
| §8, §9.1 | UNKNOWN and duplicate alerts refresh Neo4j `last_seen` | UNKNOWN **new** alerts and **resolved** duplicates do. UNKNOWN duplicates do not | `update_duplicate_event` reaches the campaign through a `MATCHES` edge, which UNKNOWN events lack |
| §9.6 | Live-graph validation blocked | Done, read-only | `experiments/STEP2A-dedup-deadlock-v1/live_validation_20261104.json` |

The flushed figures also carry a harness artefact: the harness flushes only at a restart and at the end, while the live listener flushes every 7 s wall clock, so the size of the buffer-attribution effect is not representative of live operation.

Evidence: `experiments/b3_runs_v2/comparison/REPORT.md` §5, `experiments/b3_runs_v2/smoke/dup_update_CUR.json`. Decision: `B3_DECISION.md`.

# Evidence traceability: campaign activity contract (review copy)

Maps each major statement to where it can be checked. Tags are defined in `CAMPAIGN_ACTIVITY_CONTRACT_FINAL_DRAFT.md`. All statements carried over from the draft keep their ID and class. Statements added in reconciliation are in the section "Added in reconciliation" at the end, and corrected rows say so.

"Gap", "Assessment" and "Decision by owner" are not evidence classes. They mark absent evidence, reasoned judgement and recorded decisions.

## Paths

| Short name | Location |
|---|---|
| `backend/` | `isfcr/CYUKTI/backend/`, working tree. HEAD `3afe62c` plus uncommitted changes; backend diff sha256 `804bdeb06d2f949cc3340e3e116a17f222c19900ad97d52cc78c3ad485ddac6a`; 149 git status entries |
| `DIAG/` | `\\wsl.localhost\Ubuntu\home\omkar\cyukti_diag\b3_diag_v1\`, parity-validated; variant list hash `f6de9a4b…` |
| `TRACE` | `DIAG/diagnostic_traces/CUR_norestart.events.jsonl`, sha256 `4671339bece60610d3c36bca815d6dc305f02d089851e9f9de9664171a197470`, verified |
| `B3/` | `research/experiments/b3_runs_v2/`, 986 files verified against `FREEZE_20261104.sha256` (`0c838de7…`) |

**Sources inspected (10), sha256 prefix:** `campaign_manager.py` `ba5182db`, `dedup_engine.py` `af93dc3b`, `event_time.py` `5c186735`, `neo4j_client.py` `f2e9e786`, `realtime_socgraph.py` `7026cbd2`, `campaign_feature_engine.py` `712ef466`, `campaign_decision_engine.py` `4dc4ea73`, `config.py` `caeeb8b0`, `listener/wazuh_listener.py` `2f833491`, `replay/harness.py` `eb7a806e`.

**Version check:** `diff` of `backend/` and `DIAG/variants/CUR` differs only in `replay/harness.py` and the added `diag_trace.py`. Excluded from that comparison: tests, logs, `.env`, caches, datasets.

**Counts labelled "this inspection"** were computed read-only from `TRACE` and run outputs while drafting. They aren't in the parity report. The scripts are not saved in `DIAG/`, to leave that workspace unchanged; the method is given in each row.

## Terminology and values

| ID | Statement | Class | Evidence |
|---|---|---|---|
| T1 | Fingerprint is sha256 of attacker, victim, technique, rule, agent | `SOURCE` | `dedup_engine.py:164-180` |
| T2 | Dedup window slides, 90 s | `SOURCE` | `dedup_engine.py:241-243`; `config.py:24` |
| T3 | Dedup operates in event time | `SOURCE` | `dedup_engine.py:148-154` |
| T4 | Four `last_seen` values exist | `SOURCE` | `campaign_context.py`; `neo4j_client.py:137,312,381,421`; `dedup_engine.py:213,274` |
| T5 | B3 restart is a simulation and doesn't call `stop()` | `SOURCE` | `DIAG/variants/CUR/replay/harness.py:388-400` |
| T6 | `touch()` exists only in variant A | `SOURCE` | `B3/variants/A/campaign_manager.py:128`, `realtime_socgraph.py:212,278,486`; 0 matches in `backend/`, CUR, B |
| T7 | Repository harness never flushes | `SOURCE` | `backend/replay/harness.py:319`; no flush call in file |
| T8 | Live default is fallback; replay is strict | `SOURCE` | `config.py:22`; `event_time.py:24-33`; harness `POLICY_STRICT` |

## Activity matrix

| ID | Statement | Class | Evidence |
|---|---|---|---|
| M1 | New resolved alert updates memory and Neo4j | `OBSERVED` | `TRACE` `CAMPAIGN_DB_WRITE` `update_campaign_activity` = 110; `RESOLVED` = 110 (this inspection). `campaign_manager.py:774,280`; `neo4j_client.py:709,137` |
| M2 | New UNKNOWN updates Neo4j, not memory | `SOURCE` | `neo4j_client.py:312`; `realtime_socgraph.py:170-286` has no memory update; `campaign_manager.py:465` creation only |
| M3 | 152 new UNKNOWN alerts | `OBSERVED` | `DIAG/runs/CUR_norestart/manifest.json` |
| M4 | UNKNOWN duplicate updates `AttackEvent`, not `Campaign` | `OBSERVED` | `B3/smoke/dup_update_CUR.json`; `neo4j_client.py:388`; real Neo4j, synthetic alerts |
| M5 | 7,552 UNKNOWN duplicates; immediate update | `OBSERVED` | `TRACE` `IMMEDIATE_DUPLICATE_UPDATE` = 7,552, trigger `_process_unattributed_alert` |
| M6 | Resolved duplicates are buffered, written at flush | `OBSERVED` | `TRACE` `BUFFER_ADD` = 1,201; `realtime_socgraph.py:438` |
| M7 | 1,057 same-campaign adds: 579 open, 478 closed, all C0028 | `OBSERVED` | `TRACE`, this inspection: `BUFFER_ADD` joined in sequence to `close_campaign_db` and `reopen_campaign_db` writes |
| M8 | 144 cross-campaign adds: 83 open, 61 closed | `OBSERVED` | As M7; total also in `DIAG/analysis/followup_output.txt` |
| M9 | `touch()` in memory, C0025 06:21:25.702 | `PREDICTED` | `B3/runs/A_norestart/view.json`; `DIAG/analysis/followup_output.txt`. No `TOUCH` events exist |
| M10 | `touch()` absent, C0025 06:21:05.795 | `PREDICTED` | `B3/runs/A_restart/view.json`; duplicates at index 6890 to 6891, after restart boundary at 4500; `B3/variants/A/campaign_manager.py:128`. **No-op not observed** |
| M11 | Late alerts: monotone; 0 in archive | `SOURCE`, `OBSERVED` | `campaign_manager.py:89-94,181-190`; `TRACE` `late` count 0 (this inspection) |
| M12 | Bad timestamps: rejected or marked; 0 in archive | `SOURCE`, `OBSERVED` | `event_time.py:128-137`; manifest `rejected` 0; 9,015 `ALERT_TIMESTAMP` |
| M13 | Reopen monotone, 13 occurrences | `OBSERVED` | `TRACE` `reopen_campaign_db` = 13; `campaign_manager.py:643,708`; `neo4j_client.py:632` |
| M14 | Buffer-full path uses current event. **Corrected:** it runs only when the fingerprint has no pending record, so its first write targets the campaign a new record would retain. Later repeats differ | `SOURCE` | `realtime_socgraph.py:461-466`; `dedup_engine.py:36-40`; not exercised |

## Duplicate buffer

| ID | Statement | Class | Evidence |
|---|---|---|---|
| B1 | First association retained, later duplicates merged | `SOURCE` | `dedup_engine.py:42-58` |
| B2 | `register_event` doesn't touch the pending record | `SOURCE` | `dedup_engine.py:260-285` |
| B3 | 19:20:11 is two alerts, classified to C0106, credited to C0002 | `OBSERVED` | `TRACE` `BUFFER_ADD` index 8942, 8943; `DIAG/analysis/targets_output.txt` |
| B4 | `9b729cf5…` registered 24 times, first at index 5, buffered at index 18 | `OBSERVED` | `DIAG/analysis/targets_output.txt` |
| B5 | C0002 moved from 03:29:08.179 to 19:20:11.029 at flush | `OBSERVED` | `TRACE` `FLUSH_RECORD` |
| B6 | C0045 same, alerts 8766, 8767 | `OBSERVED` | `DIAG/analysis/followup_output.txt` |
| B7 | 144 of 1,201 (12%) retained ≠ classified, 8 fingerprints, 17 campaigns; harness pattern only. Whether these are defects depends on D5 | `OBSERVED` | `DIAG/analysis/followup_output.txt`, `targets.json`; `BUFFER_ADD` `retained_campaign`, `classified_campaign` |
| B8 | 18 records, 7 campaigns, 5 moved, all INACTIVE, none in memory; 16.1 event-hours | `OBSERVED` | `DIAG/analysis/followup_output.txt` |
| B9 | Counters, risk and payload follow retained record. Scope widened: see N2 | `SOURCE` | `neo4j_client.py:421-443` |
| B10 | Flush patterns differ | `SOURCE` | `DIAG/variants/CUR/replay/harness.py:382-386,393,411`; `dedup_engine.py:119-146`; `wazuh_listener.py:320-322,337-338`; `config.py:32-35` |
| B11 | Live 7 s flush not evaluated | Gap | No experiment ran it |
| B12 | Live flush would prevent most cross-campaign merging | `HYPOTHESIS` | Reasoning from T2 and B10 |
| B13 | All 116 restart `last_seen` values reproduced | `PREDICTED` | `DIAG/analysis/restart_shift_output.txt`; `B3/runs/CUR_restart_r2/view.json` |

## Lifecycle

| ID | Statement | Class | Evidence |
|---|---|---|---|
| L1 | Four `closed_at` writers with these rules | `SOURCE` | `campaign_manager.py:201-208,499-505,601-604`; `neo4j_client.py:751` |
| L2 | Close counts 96, 21, 11; DB sweep 0 | `OBSERVED` | `TRACE` callers (this inspection); `CAMPAIGN_DB_EXPIRE` empty |
| L3 | Reopen doesn't clear `closed_at`; operations do | `SOURCE` | `neo4j_client.py:624-640,835` |
| L4 | Only the latest closure survives: C0028 | `OBSERVED` | `TRACE` writes 06:19:05.224 then 06:26:53.353; `view.json` |
| L5 | Inversions: CUR 2 or 3, A 6 or 7, B 0 | `OBSERVED` | `B3/comparison/campaign_anomalies.csv`, `REPORT.md` §5; recomputed from views (this inspection) |
| L6 | C0002, C0045 inversions come from cross-campaign credit | `OBSERVED` | `TRACE` `FLUSH_RECORD`, `BUFFER_ADD` |
| L7 | C0034 inversion under restart | Frozen value `OBSERVED`; mechanism `PREDICTED` | `B3/runs/CUR_restart_r2/view.json`; B13 |
| L8 | A's short-gap inversions (C0023, C0082, C0083, C0089) are duplicates after own closure | `HYPOTHESIS` | Values in `B3/comparison/campaign_anomalies.csv`; A was never traced |
| L9 | 30 of 101 early: 5 flush, 25 equal last new UNKNOWN, 0 neither | `OBSERVED` | `DIAG/analysis/followup_output.txt` |
| L10 | Cause of the 25 is the memory/Neo4j split | `SOURCE` | M2; `campaign_manager.py:201` |
| L11 | 4,606 of 7,552 UNKNOWN duplicates, and 539 of 1,201 resolved, arrived with campaign closed | `OBSERVED` | `TRACE`, this inspection, method as M7 |
| L12 | Late alert can invert via score close | `SOURCE` | `campaign_manager.py:601-604`; not exercised |
| L13 | Watermark advances for any alert, including skewed or fallback | `SOURCE` | `event_time.py:158-171`; `wazuh_listener.py:170-173` |
| L14 | That could mis-expire campaigns in a backlog | `HYPOTHESIS` | L13 |
| L15 | `archive_old_campaigns` uses wall clock, and has no callers | `SOURCE` | `campaign_manager.py:852-873,884-885`; search of `backend/` excluding tests |

## Reopen scoring

| ID | Statement | Class | Evidence |
|---|---|---|---|
| R1 | Reopen uses Neo4j `last_seen` through `load_inactive_campaigns` | `SOURCE` | `campaign_manager.py:346-398`; `neo4j_client.py:589-622` |
| R2 | Cached expiry uses memory | `SOURCE` | `campaign_manager.py:181-190,400-416` |
| R3 | Candidates ordered by `last_seen`, limit 5 | `SOURCE` | `neo4j_client.py:612-613` |
| R4 | 36 of 39 loads returned 5 | `OBSERVED` | `TRACE` `CANDIDATES_LOADED`, this inspection |
| R5 | Moved `last_seen` could change candidates | `HYPOTHESIS` | R3, R4 |
| R6 | C0028 scores and inputs at alert 8117 | `OBSERVED` for 0.8843 | `TRACE` `score_id` 85; `DIAG/analysis/targets_output.txt` |
| R7 | 0.8883 and 0.8823 reproduced from temporal term | `PREDICTED` | `DIAG/analysis/followup_output.txt`; `B3/runs/*/decisions.jsonl`. Restart input not read |
| R8 | Winner's lead 0.0131; other multi-candidate reopen 0.0403 | `OBSERVED` | `TRACE` `SCORE_RESULT`, this inspection |
| R9 | 0.65 of C0028's score from one edge, count 1 | `OBSERVED` | `TRACE` `CHAIN_INPUT` and `PREDICTION_INPUT`, `score_id` 85; weights in `config.py:119` |
| R10 | `chain_similarity` over 1 in 29 of 231, max 30.0 | `OBSERVED` | `DIAG/analysis/targets.json`; `campaign_feature_engine.py:60,91` |
| R11 | `duplicate_similarity` 0 in all 231 | `OBSERVED` | `TRACE` `SCORE_RESULT`, this inspection |

## Restart and shutdown

| ID | Statement | Class | Evidence |
|---|---|---|---|
| E1 | Grouping unchanged across one harness-simulated restart (mode H), before alert 4,500, all policies. Not evidence for modes L, C or D | `OBSERVED` | `B3/comparison/REPORT.md` §2 |
| E2 | Persisted differences: CUR 2 `last_seen` and 1 score; A 2 `last_seen` and 1 `closed_at`; B none | `OBSERVED` | `B3/comparison/REPORT.md` §2 |
| E3 | Shutdown order drains queue, then flushes | `SOURCE` | `wazuh_listener.py:333-338`; not exercised |
| E4 | Reload is lazy and per pair | `SOURCE` | `campaign_manager.py:286-344,400-423` |
| E5 | Buffer is memory only; crash loses it | `SOURCE` | `dedup_engine.py:18-21`; not exercised |
| E6 | Offset advances at queue time | `SOURCE` | `wazuh_listener.py:272-288` |
| E7 | `seen_alerts` is memory only | `SOURCE` | `wazuh_listener.py:35,125-140` |
| E8 | Flush isn't idempotent; `restore` can re-apply | `SOURCE` | `neo4j_client.py:422-423,441-443`; `dedup_engine.py:72-82,125-139` |
| E9 | Dedup fallback reads `AttackEvent.last_seen` | `SOURCE` | `dedup_engine.py:245-256`; `neo4j_client.py:1041-1066` |
| E10 | No dedup DB fallback in `CUR_norestart` | `OBSERVED` | `TRACE` `DEDUP_CLASSIFY` source: 8,753 cache, 262 new |
| E11 | `stop()` references undefined `logger` | `SOURCE` | `dedup_engine.py:1-16,110,116`; **runtime untested** |
| E12 | No test references the flush worker | `SOURCE` | Search of `backend/tests/` |
| E13 | Nothing injects `logger` into the module | `SOURCE`, limited | Search of `backend/` for `dedup_engine.logger`, `setattr(dedup_engine`, star imports: none |
| E14 | No crash, multi-restart or real-shutdown test exists in this study | Gap | Run list, `B3/README.md` |

## Policy comparison

| ID | Statement | Class | Evidence |
|---|---|---|---|
| P1 | A: 97 campaigns, 17 of 19 fewer are self-loops; decision changes | `OBSERVED` | `B3/comparison/REPORT.md` §3, §4 |
| P2 | B matches CUR on assignment, operation, transition; differs in persisted values | `OBSERVED` | `B3/comparison/REPORT.md` §3 |
| P3 | B's zero isn't proof of correctness | Assessment | `research/B3_DECISION.md` §6; L5, M4 |
| P4 | Replay: 63 of 63 no-operation campaigns are UNKNOWN-only | `OBSERVED` | `B3/comparison/REPORT.md` §1; audit status log |
| P5 | Live: 102 without operation, 71 UNKNOWN-only, 31 with resolved events | `OBSERVED` | `research/BASELINE_AUDIT.md` status log; `research/experiments/STEP2A-dedup-deadlock-v1/live_validation_20261104.json` |
| P6 | Second cause of P5 unexplained | Gap | |
| P7 | Step 2's 26/101 and 0 inversions are superseded by flushed figures | `OBSERVED` | `research/STEP2_EVENT_TIME.md` addendum |
| P8 | CUR is retained as reference, not declared correct | Decision by owner | `research/B3_DECISION.md` §1, §10 |

## Known gaps

| Gap | Effect on the contract |
|---|---|
| Live flush cadence not measured | D5, D6 can't use a live rate |
| `touch()` never traced | Rows 8 and 9 stay `PREDICTED` |
| No instrumented restart run | C0028's restart input, restart flush: `PREDICTED` |
| No late or bad-timestamp alert in the archive | D8, D9 rest on code and earlier unit tests. Unit tests were not re-run for this document |
| Listener and shutdown never run | Section F, section E shutdown rows are `SOURCE` only |
| Crash never tested | D7 has no behavioural evidence |
| No ground truth | No alternative can be called more accurate |
| A's anomalies not traced | L8 stays `HYPOTHESIS` |
| B's C0017 score change not examined | Not used in the contract |
| Replay databases lack an ATT&CK corpus | ATT&CK-dependent values aren't representative |
| Flush failures never injected | F.2 cases, V6: `SOURCE` or `HYPOTHESIS` only. Per-batch atomicity is assumed, not tested |
| Size of counter, risk and payload misattribution not measured | D12 has no quantity behind it |
| Which `risk_score` each consumer receives not traced | D12 consequences are incomplete |
| Readers of `Campaign.closed_at` found by text search only | A generic read of node properties wouldn't appear. Confirm before acting on D10 |

## Added in reconciliation

Source statements were read while reconciling the documents, with the backend unchanged (diff sha256 as above). `OBSERVED` statements were computed read-only from `TRACE`. **No statement here was exercised by a new run.**

| ID | Statement | Class | Evidence |
|---|---|---|---|
| N1 | Flush reaches the campaign through retained `event_id`, then `AttackEvent.campaign_id`. The writer doesn't read `r.campaign_id` | `SOURCE` | `neo4j_client.py:418,437` |
| N2 | Flush changes `Campaign` `last_seen`, `occurrences`, `total_tps`, `risk_score`; `AttackEvent` `last_seen`, `occurrences`, `tps`, `investigation_payload`; `Technique` `last_seen`, `occurrences`, `total_tps` | `SOURCE` | `neo4j_client.py:421-443` |
| N3 | `investigation_payload` is an `AttackEvent` property, not a `Campaign` property, and is replaced rather than merged | `SOURCE` | `neo4j_client.py:186,330,384,424`; read at `dashboard_api.py:265,319` |
| N4 | Record's campaign equalled the destination in 18 of 18 flushed records, and all had a `MATCHES` edge | `OBSERVED` | `TRACE` `FLUSH_RECORD` `buffered_campaign`, `destination_campaign`, `has_matches_edge` |
| N5 | UNKNOWN duplicate replaces payload and adds to `AttackEvent.occurrences`, with `tps` 0 | `SOURCE` | `realtime_socgraph.py:206-211`; `neo4j_client.py:381-384` |
| N6 | `restore()` on collision adds counters to the newer record, keeps its ids, and overwrites its `last_seen` and payload with older values | `SOURCE` | `dedup_engine.py:72-82`; not exercised |
| N7 | Each batch is one auto-commit statement. Atomic per batch by Neo4j's transaction behaviour | `SOURCE` for the code; atomicity not tested | `neo4j_client.py:413-446` |
| N8 | Multiple batches fail independently, and `restore()` re-buffers all of them | `SOURCE` | `dedup_engine.py:107-115,127-139`; `config.py:34` |
| N9 | Client error after server commit would be re-applied | `HYPOTHESIS` | N7, N8; general client and server behaviour |
| N10 | Future-skew flag is five minutes, flag only, and doesn't stop the watermark advancing | `SOURCE` | `event_time.py:65,140,158-163` |
| N11 | The flag's only use outside the module is the replay trace, and the listener passes no trace | `SOURCE`, limited to a text search | `realtime_socgraph.py:319-326`; `wazuh_listener.py:149` |
| N12 | Signal handlers set `stop_event`; shutdown runs join, worker joins, flush, in `finally` | `SOURCE` | `wazuh_listener.py:111-116,323-339`; not exercised |
| N13 | `alert_queue.join()` has no timeout; a failed alert is marked done with no retry | `SOURCE` | `wazuh_listener.py:147-156,334` |
| N14 | `stop()` joins for 2 s, worker sleeps 7 s and snapshots after waking without rechecking; daemon thread | `SOURCE` | `dedup_engine.py:94-102,119-122` |
| N15 | Queue-full line is lost on re-read, within the cache lifetime | `SOURCE`; not exercised | `wazuh_listener.py:34,125-140,264-278`; `listener/utils.py:3-27` (id is deterministic) |
| N16 | Memory `risk_score` sums the chain and excludes duplicates; Neo4j `risk_score` is a running sum including them | `SOURCE` | `campaign_manager.py:285,315,370`; `neo4j_client.py:141,403,443` |
| N17 | `risk_recalculation.py` can set persisted risk from `AttackEvent.occurrences`. Its use wasn't examined | `SOURCE` | `risk_recalculation.py:7-13` |
| N18 | Risk consumers a search found | `SOURCE`, limited to a text search | `soar/generator.py:39`; `soar/response_plan.py:104`; `active_response/integration.py:65`; `dashboard_api.py:1371,1929` |
| N19 | Only reader of `Campaign.closed_at` found is the replay view. Dashboard and frontend `closed_at` belong to operations | `SOURCE`, limited to a text search | `replay/harness.py:256`; `dashboard_api.py:403`; `frontend/src/types/index.ts:74-81` |
| N20 | Continuation scoring reads memory. Reopen writes the Neo4j-derived value into memory | `SOURCE` | `campaign_manager.py:545-550,643,708` |
| N21 | Dedup boundaries: cache `<= 90`, fallback `>= reference - 90 s`, read from `AttackEvent.last_seen` | `SOURCE` | `dedup_engine.py:241`; `neo4j_client.py:1046` |
| N22 | Of 116 campaigns, the last close was expiry for 101 (90 alert path, 11 cache sweep), score for 14, and 1 ended ACTIVE. 128 close writes in all | `OBSERVED` | `TRACE` `CAMPAIGN_DB_WRITE`, walked in sequence |

**Scope of the check.** N4 and N22 were computed by `reconcile_stats.py`, a read-only script in the session's temporary folder and not in `DIAG/`. The method is given in each row, so they can be recomputed.

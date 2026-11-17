# Step 1 — Live ATT&CK transition learning fix

**Date:** 2026-11-03 · **Base:** `3afe62c` (uncommitted working-tree changes) · **Audit finding:** `BASELINE_AUDIT.md` A2b (and A2c)
**Feedback link:** (a) "only 3 attack patterns were ever learned"; (a)ii "it never records the transitions"; (b) "next-step prediction stays almost useless".

## 1. Defect: exact execution path (pre-fix, `HEAD`)

For each resolved alert, `realtime_socgraph.process_alert()` ran:

| # | Call | Effect on the "previous technique" |
|---|---|---|
| 1 | `campaign_manager.get_or_create_campaign(attacker, victim, T_cur)` → `resolve_campaign()` → `activate_campaign(context, T_cur)` | `append_technique()` sets in-memory `context.last_technique = T_cur`; **`neo4j_client.update_campaign_activity()` runs `SET c.last_technique = T_cur`** |
| 2 | `neo4j_client.create_attack_event(..., mitre_id=T_cur)` | **`SET c.last_technique = $mitre_id`** (second overwrite) |
| 3 | `chain_updater.update_attack_chain(campaign_id, T_cur)` | Reads `MATCH (c:Campaign) RETURN c.last_technique AS previous`, which is **T_cur**. `previous == current` → `"[CHAIN] Duplicate technique ignored"`. No `NEXT_TECHNIQUE` write. |

Captured trace (pre-fix, regression test): `Previous : T1110 / Current : T1110 / Skip Flag: False → [CHAIN] Duplicate technique ignored: T1110`.

**Root cause.** The previous technique was recovered by reading back a mutable property that has three writers, two of which run earlier in the same event. A second mutable channel, `CampaignContext.skip_next_chain_update`, was cleared by the chain updater on a shared cached object. The ordering is identical in every preserved version: all 42 commits, and the July/August copies in `_legacy_archive/soc_graph_backup_stripped/backend/`, incl. `.ipynb_checkpoints`.

## 2. Correction

No reordering and no new read-back. The campaign manager, which owns campaign state, captures an **immutable `TransitionContext`** at the single point where an event is attached to a campaign, before any state is mutated. That snapshot is passed explicitly to the chain updater.

| File | Change |
|---|---|
| `backend/campaign_context.py` | New `@dataclass(frozen=True) TransitionContext(campaign_id, current_technique, previous_technique, segment_start)` |
| `backend/campaign_manager.py` | New `activate_campaign_with_transition()`: snapshot first (`previous = context.last_technique` unless segment start), consume `skip_next_chain_update` once, then the original activation. New `get_or_create_campaign_with_transition()`. `activate_campaign()` and `get_or_create_campaign()` keep their contract (return `campaign_id`) as thin wrappers. |
| `backend/chain_updater.py` | `update_attack_chain(campaign_id, current_technique, transition)`: `transition` is required and validated against `campaign_id`/`current_technique`. **No longer reads or writes `Campaign.last_technique`** (still maintained by `update_campaign_activity`/`create_attack_event`, so its meaning for dashboard/API/reconstruction is unchanged). No longer imports `campaign_manager`. Edge-writing Cypher is unchanged. Returns the learned pair, or `None`, and warns if a Technique node is missing. |
| `backend/realtime_socgraph.py` | Uses `get_or_create_campaign_with_transition()` and passes `transition` to `update_attack_chain()`. |

**Deliberately unchanged** (later decisions):
- No transition for the first event of a new or reopened campaign (`segment_start`). Recording the split-causing transition is (a)ii, Step 4.
- Self-transitions A→A are still not recorded (Decision 2).
- UNKNOWN events still don't change the previous technique, so A, UNKNOWN, B learns A→B.
- No Neo4j schema change.

## 3. Tests

New: `backend/tests/test_chain_learning_regression.py`, 13 tests. They drive the **real** `process_alert → CampaignManager → chain_updater` path over a stateful fake graph. Its `last_technique` writes mirror the real Cypher in `update_campaign_activity`, `create_attack_event` and `create_unattributed_attack_event`. The `NEXT_TECHNIQUE` MERGE is chain_updater's own query, routed through a fake driver.

| Test | Pre-fix (`HEAD`) | Post-fix |
|---|---|---|
| consecutive A, B in one campaign learn A→B | **FAIL** (`{}`) | pass |
| three-step chain learns both transitions | **FAIL** | pass |
| A, UNKNOWN, B learns A→B | **FAIL** | pass |
| previous technique is per campaign (interleaved victims) | **FAIL** | pass |
| `Campaign.last_technique` still = latest technique | pass | pass |
| first event of a new campaign learns nothing | pass | pass |
| A, A records no self-transition | pass | pass |
| split → new segment records nothing (policy preserved) | — (added post-fix) | pass |
| `update_attack_chain` without a transition raises | — | pass |
| mismatched transition raises | — | pass |
| snapshot taken before mutation; immutable | — | pass |
| segment-start flag consumed exactly once | — | pass |
| legacy entry points still return `campaign_id` | — | pass |

Pre-fix run: **4 failed, 3 passed** (the 7 tests that existed then). Post-fix: **13 passed**.

Updated: `backend/tests/test_realtime_mitre_integration.py`. The stubs now patch `get_or_create_campaign_with_transition` and accept the `transition` argument; the stub also asserts the transition matches the event. All original assertions are unchanged.

Full suite: **876 passed, 29 skipped, 2 failed**, vs baseline 863/29/2. The same 2 tests fail before and after (`test_operation_attribution_integration.py::test_realistic_two_alert_scenario…`, `test_threat_attribution.py::test_attribution_evidence_derives…`), both `ServiceUnavailable: localhost:7687`. They need a live Neo4j and aren't marked to skip.

## 4. Live verification on a real Neo4j (disposable instance)

The user's `neo4j-soc` database was **not** used (read-only restriction). A throwaway `neo4j:5-community` container (same image, port 17687, no volume, removed afterwards) was started per run. The real `process_alert()` ran end to end with MISP pointed at a closed port (no publication) and GNN disabled. Pre-fix code came from `git archive HEAD`.

Expected transitions are derived independently from each campaign's own stored `AttackEvent` sequence: consecutive resolved events with different techniques, excluding the segment-start event.

| Input | Code | Campaigns | `NEXT_TECHNIQUE` edges recorded | Expected from event sequences | Match |
|---|---|---|---|---|---|
| 3 synthetic alerts, one pair (T1595, T1110, T1078) | pre-fix | 1 | **none** | T1595→T1110, T1110→T1078 | ✗ |
| same | fixed | 1 | T1595→T1110, T1110→T1078 | same | ✓ |
| 1,159 real native-mapped alerts from `data/raw_logs/alerts.json` (192.168.56.106 → pes1ug23cs411-VirtualBox), file order | pre-fix | 1 | **none** | 7 transitions | ✗ |
| same | fixed | **3** | 6 transitions (T1595.002→T1068, T1068→T1055, T1055→T1190, T1190→T1059.007, T1059.007→T1059, T1059→T1190) | identical 6 | ✓ |

The fixed-code archive replay was run twice on fresh instances, with the same campaigns and edges both times. Outputs: `research/experiments/STEP1-chain-learning-live-replay-v1/` (script, 4 JSON results, gzipped logs).

### 4.1 Important consequence (expected; not a regression of this fix)
With learning working, the **feedback (a) fragmentation mechanism is now active on live traffic.** The same real alerts produced 3 campaigns instead of 1. Both splits were scored at 0.32–0.33 < 0.35 with `prediction = chain = 0.00`:
- **Split 1:** `T1190` → `T1190` (a repeat; self-transitions score 0 once T1190 has any learned successor).
- **Split 2:** `T1190` → `T1210` (an unseen successor). Under the unchanged policy, this split-causing transition is not recorded.

The pre-fix single campaign wasn't correct grouping either. With no learned edges, every step took the "no information → continue" branch (score ≈ 0.51). **Do not run the live listener to produce experimental numbers until Steps 2–4 are done**; campaign counts from the Step 1 code alone would overstate fragmentation relative to the final design.

Also visible in the replay and addressed by Step 2: dedup and expiry use wall-clock time. Replaying 1,159 alerts in seconds collapsed them to 12 events, a different result from the original 16-hour timeline.

## 5. Origin of the three historical patterns (A2c): UNRESOLVED, narrowed

Recorded: `T1110.001→T1110 (6)`, `T1110→T1078 (5)`, `T1078→T1110.001 (1)`, each with `confidence 1.0` (`evaluation/results/prediction_metrics.json`).

| Hypothesis | Evidence | Status |
|---|---|---|
| Hard-coded seed `attack_chain_builder.py` | Its pairs (T1595→T1110, T1110→T1110.001, T1110.001→T1078, T1078→T1053.003) don't match. It never sets `r.confidence`, but all 3 edges carry it. | **Ruled out** for these edges |
| Live path of any preserved version, single process | The overwrite ordering is present in every preserved version; the listener has one queue worker (`listener/wazuh_listener.py:141-158`), so `process_alert` is serialized. Only `chain_updater`'s learning branch sets `r.confidence`, so the edges *were* written by that branch at some point. | Not reachable in preserved code |
| Two listener processes overlapping (cross-process race on `c.last_technique`) | 11 listener start-ups in `backend/logs/prerana_listener.log` (2026-09-15 → 09-26); overlap can't be determined from the log | Open |
| Code version older than the July legacy copy, or a manual/notebook write | No artefact found | Open |

`backend/logs/prerana_listener.log` can't settle this: `chain_updater` reports via `print()`, which that log does not capture.
**Resolution needs the live graph:** `research/queries/readonly_live_graph_validation.py` returns each edge's `created_at`/`last_seen` and the campaigns and event timestamps where src was immediately followed by dst. That lets the edges be dated against the listener start-ups and the code history. The script uses `execute_read` only, contains no write clause (asserted), and was self-tested on the disposable instance.

## 6. Outstanding issues
1. Live-graph validation (A2c and the other audit items) is **blocked until `neo4j-soc` is started**; the script is ready.
2. Splitting behaviour is now live, as above (Steps 2–4).
3. `CampaignManager.resolve_campaign` + `activate` aren't one critical section. That's safe under the single-worker listener, but not if `process_alert` is ever parallelised or two listeners run concurrently (the second open hypothesis above).
4. The 2 Neo4j-dependent tests should be marked `requires_live_neo4j` (not changed here; out of Step 1 scope).

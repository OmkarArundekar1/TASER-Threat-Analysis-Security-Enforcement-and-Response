# Phase 21 Real Investigation Validation

**Correction notice:** an earlier version of this document, produced minutes before this one, reported `BLOCKED — Neo4j unavailable` after `docker ps -a` showed `neo4j-soc` as `Exited (255)`. On a follow-up check, Neo4j was found to be genuinely reachable and serving real data. Root cause: the whole Docker daemon had restarted (all containers showed "Up 48 minutes" measured from the same instant, and `dockerd`'s own pid started at the same timestamp), and the Neo4j container process came back up but is no longer tracked as a container object by that daemon (`docker inspect <cgroup-id>` returns "no such object" even though its process is live and serving on 7687 — an orphaned/untracked container, not one I started, recreated, or restarted). A direct read-only Cypher check confirmed real, non-decreased data (2405 nodes; 858 Technique, up from a prior 65/124/858 baseline to 71 campaigns / 130 attack events / 858 techniques — growth only, no loss). Given this, the real experiment was then actually run. This document supersedes the earlier BLOCKED version.

## 1. Objective

Determine — by attempting to falsify it, not assuming it — whether CYUKTI's evidence-aware investigation engine performs state-dependent, adaptive evidence selection on **real** CYUKTI campaigns, with explicit uncertainty and no promotion of unsupported hypotheses into attack-chain ground truth.

## 2. Environment

| Component | Status |
|---|---|
| Neo4j (`neo4j-soc` container process) | **Up**, serving on `bolt://localhost:7687` (orphaned from `docker ps -a`'s registry, but live and queryable) |
| Neo4j data | 2405 total nodes, 71 `Campaign`, 858 `Technique`, 130 `AttackEvent` — consistent growth from the last recorded baseline, no data loss |
| Docker daemon | Restarted at 15:18 today; not touched by this session — `docker start`/`stop`/`rm` were never invoked |
| Python environment | `soc_env`, activated from `/home/omkar/capstone/soc_env` |
| Trained XGBoost model | Present, loaded, real inference confirmed (`ml/models/xgb_severity.json`) |
| `scripts/run_real_investigations.py` | Run **unmodified**, exactly as it exists in the repository |

I did not start, stop, restart, or recreate any container. I only ran a read-only Cypher query to confirm the database was genuinely live before running the real experiment.

## 3. Real campaigns investigated

All 3 script-defined targets, executed against live Neo4j:

| Campaign | Attacker | Victim | Techniques observed | Last technique | Risk score |
|---|---|---|---|---|---|
| `CAMP_427A075C` | 192.168.56.106 | 192.168.56.105 | `T1595` | T1595 | 8330 |
| `CAMP_1429ADB4` | 192.168.56.105 | pes1ug23cs411-VirtualBox | `T1055, T1059, T1059.007, T1190, T1210, T1595.002` | T1210 | 34820 |
| `CAMP_D8605E81` | 192.168.56.106 | pes1ug23cs411-VirtualBox | `T1110, T1110.001` | T1110 | 630 |

None of the 3 target campaigns' current technique is UNKNOWN — see Section 10.

## 4. Methodology

Unmodified `scripts/run_real_investigations.py`: real Cypher-loaded `CampaignContext`, real `default_action_executor` (7 live evidence collectors), real `default_model_predictor` (real XGBoost inference), real `run_investigation(confidence_threshold=0.75, max_steps=8)`. No evidence was fabricated, no collector was mocked, no probability was hand-inserted.

**Known, disclosed gap (unchanged from before the run, not fixed):** the script's print statements surface only the pre-existing `ActionValue` fields (`reliability, novelty, uncertainty_reduction, cost, latency`, and the aggregate `selection_value`) — not the newly-added `why_selected`, `redundancy_penalty`, full `action_scores`, or `candidate_hypotheses`. The `redundancy_penalty` value is present in memory (confirmed by reading `next_best_evidence.py`) but not printed, so it is inferred analytically below (Section 6), not read directly off stdout.

## 5. Investigation traces

Full raw stdout for all 3 campaigns (all 8 steps each) captured and preserved; per-campaign summaries below. Total real evidence items collected: 76, 73, 76 respectively across the 8 actions available.

## 6. Adaptive evidence-selection results — the falsification test

**Concrete before/after numeric evidence, as required:**

| Step | Action | `selection_value` — CAMP_427A075C | CAMP_1429ADB4 | CAMP_D8605E81 |
|---|---|---|---|---|
| 1 | mitre_knowledge | 1.74 | 1.74 | 1.74 |
| 2 | xgboost_prediction | 1.73 | 1.73 | 1.73 |
| 3 | graph_structure | 1.62 | 1.62 | 1.62 |
| 4 | campaign_history | 1.49 | 1.49 | 1.49 |
| 5 | detection_check | 1.485 | 1.485 | 1.485 |
| 6 | cti_lookup | 1.135 | 1.135 | 1.135 |
| 7 | attribution_match | 0.875 | 0.875 | 0.875 |
| 8 | mitre_semantic_search | 0.3 | 0.3 | 0.3 |

**Finding: the action order and every `selection_value` were identical, to 3 decimal places, across all 3 real campaigns**, despite the campaigns having different techniques, different attacker/victim identities, different risk scores (630–34820), and materially different XGBoost model outputs.

**Root cause (verified by reading `investigation/next_best_evidence.py`, not guessed):** `score_action()`'s `uncertainty_reduction` term is nonzero *only* for `XGBOOST_PREDICTION` — every other action always scores `uncertainty_reduction=0.0`, independent of the current investigation's actual uncertainty. `novelty` and `redundancy_penalty` depend only on *which action types have already been taken*, not on what evidence content those actions returned. `reliability`, `cost`, and `latency` are static `ActionMeta` constants. The consequence: for this fixed 8-action universe — which every one of the 3 real investigations exhausted completely within `max_steps=8` — the ranking formula is a deterministic function of the *sequence of action types taken so far*, not of the *evidence values* those actions returned. Since all 3 campaigns exhaust the same 8 actions, they necessarily produce the same order.

**This is a genuine, disclosed, negative-leaning result, not a bug.** Nothing crashed, no invariant was violated, and the formula behaves exactly as its own docstring describes (a transparent heuristic, not a learned/optimal policy). But it directly falsifies the strong form of the adaptivity claim — "the investigator selects evidence differently depending on the campaign's observed state" — for this specific real-data condition (small, always-fully-exhausted action universe). It does **not** falsify the weaker, still-real claim that the underlying *confidence/uncertainty computation* is state- and evidence-dependent — see Section 7, which shows real, materially different numbers per campaign.

**What this experiment cannot rule out:** adaptivity could still manifest (a) with a larger action universe than 8, (b) if an investigation stopped early via the confidence threshold before exhausting all actions (never happened here — see Section 9), or (c) if `uncertainty_reduction` were computed per-action rather than being reserved for the single model action — none of which this experiment tests.

## 7. Confidence/uncertainty trajectories — real, per-campaign numbers

| Metric | CAMP_427A075C | CAMP_1429ADB4 | CAMP_D8605E81 |
|---|---|---|---|
| Final `investigation_confidence` | 0.1835 | 0.0687 | 0.0963 |
| Final `uncertainty` | 0.8165 | 0.9313 | 0.9037 |
| Final `model_probabilities` | Critical 0.5002 / Low 0.0003 / Medium 0.4995 | Critical 0.0997 / Low 0.4907 / Medium 0.4095 | Critical 0.0983 / Low 0.3301 / Medium 0.5716 |
| Final `model_uncertainty` | 0.6332 | 0.8600 | 0.8316 |
| Final `evidence_reliability` | 1.0 | 0.9287 | 0.9814 |
| Final `evidence_coverage` | 0.5279 | 0.4884 | 0.4528 |

These are all genuinely campaign-dependent, real numbers — this part of the architecture is confirmed adaptive on live data. `investigation_confidence` never approached the 0.75 stopping threshold in any of the 3 real campaigns.

## 8. Evidence dependency/redundancy behavior

The one declared dependency (`ATTRIBUTION_MATCH → CAMPAIGN_HISTORY`) was exercised in all 3 runs — `campaign_history` (step 4) always preceded `attribution_match` (step 7), so the redundancy discount was live-active in every run. **However, because the action order itself was identical across all 3 campaigns (Section 6), this specific experiment cannot demonstrate that the discount *differs* by campaign** — it can only confirm the discount fires when its precondition is met, which it consistently did. A genuinely differentiating test would require an investigation where the two actions' relative order changes, which did not happen here.
*Planned:* re-run this test once the NBE formula (or the action set) is changed enough that the two actions' relative order can vary between campaigns.

## 9. Stopping behavior

All 3 real investigations stopped for the identical reason: `"reached maximum investigation depth (8 steps)"`. The confidence-threshold stopping branch (`investigation_confidence >= 0.75`) was **never exercised** in this run — all 3 final confidences were far below threshold (0.07–0.18). This is a real, measured limitation of this specific experiment: max-depth stopping is confirmed live; confidence-triggered early stopping is not (NOT MEASURED on real data this session).
*Planned:* re-run against a real campaign whose confidence trajectory is expected to cross 0.75, once one is identified, to measure the confidence-threshold stopping branch live.

## 10. UNKNOWN safety validation

**Not exercised live** — none of the 3 script-defined target campaigns' current technique is UNKNOWN (`T1595`, `T1210`, `T1110` are all resolved techniques). The structural (source-inspection) guarantee established in the prior implementation session — that the investigation module never references a Neo4j write function capable of setting `attack_id`/creating `Technique`/`MATCHES`/`NEXT_TECHNIQUE` — is unaffected and still holds, but a live UNKNOWN-campaign case was not part of this specific run.
*Planned:* select a real UNKNOWN-technique campaign as a target in a future verification session and re-run this validation against it.

## 11. Hypothesis/ground-truth separation

No violation observed: `model_probabilities` (candidate hypotheses) remained distinct from any Neo4j write throughout all 3 runs — the script performs no writes back to Neo4j based on investigation output. Consistent with the structural guarantee; not independently re-verified against a live write attempt this session (none was made, by design).
*Planned:* add an integration test that deliberately attempts a disallowed Neo4j write from the investigation module and asserts it is structurally impossible, complementing the existing source-inspection proof.

## 12. Bugs discovered

**None.** The non-adaptive ranking-order finding (Section 6) is classified as a disclosed design limitation of the current heuristic formula under a specific condition (small, fully-exhausted action set), not a defect — no exception, incorrect state, or safety violation occurred. No stop-fix-regression cycle was triggered.

## 13. Quantitative results

| Metric | CAMP_427A075C | CAMP_1429ADB4 | CAMP_D8605E81 |
|---|---|---|---|
| Investigation steps | 8/8 (max reached) | 8/8 | 8/8 |
| Total evidence gathered | 76 | 73 | 76 |
| Distinct action-order sequences observed (n=3) | 1 (identical across all 3) | | |
| Confidence-threshold stops observed | 0/3 | | |
| Max-depth stops observed | 3/3 | | |
| UNKNOWN campaigns tested | 0/3 | | |

Full backend test suite: 164/164 passing (unchanged; no code was modified this phase).

## 14. What the experiment proves

1. The full real pipeline (Neo4j → 7 live evidence collectors → real XGBoost model → confidence/uncertainty computation → stopping logic) runs end-to-end against genuine production campaign data without error, across 3 campaigns of very different risk profiles.
2. Confidence and uncertainty outputs are genuinely campaign-dependent on real data (Section 7) — this part of the claim holds.
3. The dependency/redundancy discount mechanism fires correctly when its precondition is met on real data (Section 8).
4. Max-depth stopping behaves correctly on real data (Section 9).

## 15. What it does NOT prove

1. That evidence-selection **order** adapts to campaign-specific evidence content — this experiment's concrete numeric evidence (Section 6) shows the opposite for this fixed 8-action universe: order was identical across all 3 real campaigns, for an identifiable, code-verified structural reason.
2. That confidence-threshold-triggered stopping works on real data (never exercised — all confidences stayed low).
3. UNKNOWN safety or hypothesis/ground-truth separation under a live UNKNOWN case (no UNKNOWN campaign was in the target set).
4. Anything about generalization, attribution accuracy, or ML performance — none of those were in scope and none are claimed here.

## 16. Limitations

- Only 3 campaigns, all script-selected in an earlier phase, none currently UNKNOWN.
  *Planned:* the Phase 19 dataset-expansion spec is expected to provide enough real campaigns, including UNKNOWN ones, to broaden this validation.
- The action universe (8 actions) is small enough that every real investigation exhausted it completely within `max_steps=8`, which is precisely the condition under which the current formula cannot show order-adaptivity (Section 6) — a structural property of this experiment's scale, not of a larger/held-out one.
  *Planned:* re-test with a larger action universe or a higher `max_steps` once one is available, per Phase 22's recommended next step.
- `scripts/run_real_investigations.py` does not print `redundancy_penalty`, `why_selected`, `action_scores`, or `candidate_hypotheses` — those fields exist in memory but were not captured this run; the redundancy finding in Section 8 is a code-level inference, not a direct printout.
  *Planned:* extend the script's logging to print these fields directly so future runs don't rely on code-level inference.
- No timing/latency measurement was made.
  *Planned:* add timing instrumentation to `scripts/run_real_investigations.py` alongside the existing benchmark harness.

## 17. Research conclusion

The real-campaign experiment ran successfully end-to-end and is genuinely informative, but it **falsifies the strong claim** that CYUKTI's evidence-selection order is campaign-adaptive under real data, for the specific (and currently only) condition tested: a small action universe that gets fully exhausted every time. It **confirms** that confidence/uncertainty computation is real-data-adaptive, and that the redundancy-discount mechanism fires correctly, though not differentially, in this run. This is reported as a mixed, honest result — not a full success and not a full failure — consistent with the task's own standard: a negative result on one sub-claim is acceptable to report; it was not hidden or downplayed.

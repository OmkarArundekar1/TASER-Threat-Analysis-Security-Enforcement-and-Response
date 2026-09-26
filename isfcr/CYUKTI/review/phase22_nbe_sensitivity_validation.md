# Phase 22 — NBE Sensitivity & Ablation Study

## 1. Objective

Determine *why* the current Next-Best-Evidence (NBE) policy produced identical evidence-selection ordering across all 3 real Phase 21 campaigns, and determine whether the policy's nominally adaptive components (uncertainty reduction, dependency/redundancy discounting, novelty) are actually capable of influencing ranking at all — or whether the invariance is structural.

## 2. Hypothesis

Two competing explanations for Phase 21's finding were possible going in:

- **H1 — magnitude dominance:** adaptive terms (`uncertainty_reduction`, `redundancy_penalty`, `novelty`) do vary with real evidence, but static terms (`reliability`, `cost`, `latency`) are large enough to dominate the sum, so variation exists but never flips the ranking.
- **H2 — no content channel:** the adaptive terms don't actually encode evidence *content* at all — they encode only coarse, sequence-level state (which action *types* have run), which is itself deterministic given the formula, so there is no variation to dominate in the first place.

This phase's job was to distinguish H1 from H2 with numbers, not assume either.

## 3. Experimental setup

**Method:** run the real, unmodified `investigation.loop.run_investigation()` against the same 3 real Phase 21 campaigns via the real `default_action_executor`/`default_model_predictor`, capture the full `InvestigationRecord.to_dict()` trace (which already includes every `ActionValue` component — `expected_gain`, `reliability`, `novelty`, `uncertainty_reduction`, `redundancy_penalty`, `cost`, `latency`, `value` — for every candidate action at every step, not just the one chosen), then re-weight those *already-real* component values through 8 alternate linear combinations ("ablations") to see which terms can move the ranking. No ablation re-executes `score_action()` with modified code, and the production module was not touched.

**Script:** `scripts/phase22_nbe_sensitivity.py` (new, diagnostic only).

## 4. Environment

Identical to the corrected Phase 21 environment: `neo4j-soc`'s process live on `bolt://localhost:7687` (still the same orphaned-from-`docker ps -a` process established in Phase 21 — not restarted or touched this phase), `soc_env` Python 3.12.3, `scipy` 1.11.4 (used for Kendall's tau / Spearman's rho only — not a production dependency). Repository base commit: `888c2e0` (working tree has the Phase 20+/21 uncommitted modifications already on disk, none touched this phase; only `scripts/phase22_nbe_sensitivity.py` and `backend/tests/test_phase22_nbe_sensitivity.py` are new).

## 5. Campaigns tested

Same 3 as Phase 21, same infrastructure fact preserved: `CAMP_427A075C`, `CAMP_1429ADB4`, `CAMP_D8605E81`. Re-running produced the identical final confidences Phase 21 reported (0.1835 / 0.0687 / 0.0963) — an exact reproduction, not a new/different run.

## 6. Exact NBE formulation (from code, not assumed)

From `investigation/next_best_evidence.py::score_action()`:

```
value = W_GAIN * expected_gain
      + W_RELIABILITY * reliability
      + W_NOVELTY * novelty
      + W_UNCERTAINTY_REDUCTION * uncertainty_reduction
      - W_COST * cost
      - W_LATENCY * latency
      - W_REDUNDANCY * redundancy_penalty

where:
  expected_gain       = meta.reliability * novelty
  novelty             = 1.0 if this action's EvidenceSource has never been
                         consulted yet, else 0.15 (REPEAT_QUERY_NOVELTY) —
                         a boolean-derived constant, not a function of what
                         evidence was actually found
  uncertainty_reduction = current_uncertainty  IF action == XGBOOST_PREDICTION
                           ELSE 0.0             (unconditionally, for all 7 other actions)
  redundancy_penalty  = (# of this action's declared depends_on actions
                         already taken) * 0.5   — again a function of which
                         action TYPES have run, not evidence content
  reliability, cost, latency = static ActionMeta constants (investigation/actions.py)

Weights (verified against the live module, not retyped from memory):
  W_GAIN=1.0  W_RELIABILITY=0.3  W_NOVELTY=0.5
  W_UNCERTAINTY_REDUCTION=0.6  W_COST=0.4  W_LATENCY=0.2  W_REDUNDANCY=0.5
```

**Key structural fact, confirmed by reading the code (not inferred from behavior):** of the 8 candidate actions, only `XGBOOST_PREDICTION` has a formula term (`uncertainty_reduction`) that can take a value other than a static constant or a step-sequence-derived one. Every other action's score is, by construction, a pure function of (a) fixed `ActionMeta` numbers and (b) which action *types* have already run — never of the *values* any evidence collector actually returned.

## 7. Score-component analysis — step 1, all 8 candidates (representative; identical across all 3 campaigns)

| Action | expected_gain | reliability | novelty | uncertainty_reduction | redundancy_penalty | cost | latency | **value** |
|---|---|---|---|---|---|---|---|---|
| mitre_knowledge | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.1 | 0.1 | **1.74** |
| xgboost_prediction | 0.6 | 0.6 | 1.0 | 1.0 | 0.0 | 0.25 | 0.25 | **1.73** |
| graph_structure | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.3 | 0.3 | **1.62** |
| campaign_history | 0.9 | 0.9 | 1.0 | 0.0 | 0.0 | 0.3 | 0.3 | **1.49** |
| detection_check | 0.85 | 0.85 | 1.0 | 0.0 | 0.0 | 0.2 | 0.2 | **1.485** |
| mitre_semantic_search | 0.7 | 0.7 | 1.0 | 0.0 | 0.0 | 0.15 | 0.15 | **1.32** |
| cti_lookup | 0.75 | 0.75 | 1.0 | 0.0 | 0.0 | 0.5 | 0.7 | **1.135** |
| attribution_match | 0.65 | 0.65 | 1.0 | 0.0 | 0.0 | 0.4 | 0.3 | **1.125** |

(Phase 21's printed trace showed `attribution_match=0.875` and `mitre_semantic_search=0.3` as their *step-7/step-8 taken* values, after novelty/redundancy discounts from later state — the table above is each action's **step-1, pre-discount** candidate value, which differs; both are real, correctly computed numbers from different points in the same trace, not a contradiction.)

## 8. Baseline reproduction

`Ablation-A reproduction max abs error vs. real recorded value: 0.0` across all 3 campaigns × 8 steps × (up to 8) candidates. Ablation A's re-weighting formula is bit-for-bit identical to `score_action()`'s real output — this is the sanity check that everything downstream is analyzing genuine, correctly-reproduced numbers, not a divergent re-implementation.

## 9. Ablation methodology

| Ablation | Definition |
|---|---|
| A — full current policy | All 7 terms, real weights (reproduces the actual system) |
| B — no uncertainty-reduction | Zero the `W_UNCERTAINTY_REDUCTION * uncertainty_reduction` term only |
| C — no model-probability contribution | Same operation as B — see note below |
| D — no dependency | Zero the `W_REDUNDANCY * redundancy_penalty` term only |
| E — static/base-only | `W_RELIABILITY*reliability - W_COST*cost - W_LATENCY*latency` only (pure `ActionMeta` constants; `expected_gain` excluded because it includes `novelty`, which is state-, not action-, dependent) |
| F1 — base + novelty/gain only | E + `W_GAIN*expected_gain` |
| F2 — base + uncertainty-reduction only | E + `W_UNCERTAINTY_REDUCTION*uncertainty_reduction` |
| F3 — base + redundancy only | E − `W_REDUNDANCY*redundancy_penalty` |

**Finding, not an artifact of test design:** ablations B and C are numerically identical under the current architecture. This was not contrived — it reflects a real structural property: `model_probabilities` has exactly one channel into `score_action()` (`current_uncertainty`, feeding `uncertainty_reduction`, which is gated to a single action). There is no second, independent term through which model-probability information reaches the ranking formula. Reported directly rather than inventing an artificial distinction to satisfy the "6 ablations" framing.

## 10. Ranking metrics — full results

192 total pairwise comparisons were computed (8 steps × 8 ablations × 3 campaign-pairs per step, since `C(3,2)=3`):

| Metric | Result |
|---|---|
| Identical rankings | **192 / 192** |
| Identical raw scores (not just order) | **192 / 192** |
| Total pairwise rank swaps (summed across all 192 comparisons) | **0** |
| Kendall's tau (non-degenerate comparisons, n=168; 24 undefined at step 8, where only 1 candidate remains and tau is mathematically undefined for n=1) | **min = max = 1.0** |
| Top-1 agreement | 192 / 192 |
| Top-3 overlap fraction | 1.0 in every case with ≥3 candidates |
| Score variance across campaigns, per action, per step, per ablation | **0.0 in every single case** (checked at all 8 steps under the full policy specifically, and confirmed for every ablation) |

This is a stronger result than "the resulting order never changed" — the underlying **numeric scores themselves never differed by even one part in 10⁴** across any of the 3 real campaigns, under any of the 8 tested scoring configurations, at any point in the 8-step investigation.

## 11. The XGBoost-specific test (Section 6 of the task)

1. **Which actions receive nonzero `uncertainty_reduction`?** Only `xgboost_prediction` — confirmed both by reading `CONCLUSION_ACTIONS = frozenset({InvestigationAction.XGBOOST_PREDICTION})` in the code and by scanning all 24 real candidate-score records (3 campaigns × 8 steps) for nonzero `uncertainty_reduction`: exactly one action ever shows it.
2. **Does XGBoost's score change between campaigns?** No. `xgboost_prediction`'s `value` was exactly `1.73` in all 3 campaigns, at both points it was scored (step 1, as a losing candidate; step 2, as the chosen action).
3. **Is that score change sufficient to alter its rank?** N/A — there was no change to evaluate.
4. **Root cause, traced through the confidence pipeline (not just the NBE formula):** `current_uncertainty` at step 2 (the only moment `XGBOOST_PREDICTION`'s score matters) comes from `estimate_confidence(store, model_probabilities=None)` computed after step 1. Step 1 is always `mitre_knowledge` (highest static score, 1.74, in every campaign). `evidence/collectors/mitre_collector.py` never sets `Evidence.relevance` — it stays at its dataclass default, `0.0` (confirmed by reading `evidence/schema.py`'s own docstring: relevance "starts at 0.0 and is filled in by a retriever, never by the collector that produced the evidence"). `EvidenceStore.weighted_confidence()` and `evidence_coverage()` both explicitly give **zero weight** to relevance-0.0 evidence (a deliberate anti-saturation fix from an earlier phase, documented in `store.py`'s own docstring). Consequence: after step 1, `evidence_reliability=0.0` and `evidence_coverage=0.0` in **every** campaign, unconditionally — so `investigation_confidence=0.0`, `uncertainty=1.0`, and `uncertainty_reduction` for XGBoost is pinned to exactly `1.0` regardless of which real campaign is under investigation. This is not a coincidence observed in 3 campaigns; it is a structural consequence of `mitre_knowledge` always being scored first (an invariant property of the formula on an empty store) combined with its collector never producing relevance-bearing evidence.
5. **Do other adaptive components affect competing actions?** Within a single investigation, yes — `novelty` and `redundancy_penalty` genuinely change step-to-step as actions are taken (e.g. `mitre_semantic_search`'s value drops from 1.32 at step 1 to 0.3 once `mitre_knowledge`'s same-source repeat discount applies). But this evolution is a function of the **action-type sequence only**, and that sequence is itself always identical across campaigns (Section 12) — so the evolution, real as it is, never diverges between campaigns either.
6. **Is invariant ordering caused by magnitude dominance from static terms?** **No — this hypothesis (H1) is falsified.** Ablations F1/F2/F3, which isolate `novelty`, `uncertainty_reduction`, and `redundancy_penalty` respectively *without* any static-term competition, each independently show **zero cross-campaign score variance** on their own. If H1 were correct, at least one of F1/F2/F3 — freed from being outweighed by static terms — should have shown campaigns diverging. None did. **H2 is confirmed instead:** the adaptive terms don't encode evidence content at all; they encode only coarse action-sequence state, which is deterministic given the formula, so there was never any content-derived variation for static terms to dominate in the first place.

## 12. Why the invariance is self-reinforcing

`novelty` depends on "has this `EvidenceSource` been touched" (a boolean), and `redundancy_penalty` depends on "has a declared dependency already been taken" (a count over the taken-action set) — both are functions of the **sequence of action types executed**, not of the **values** any collector returned. Because the ranking formula is itself deterministic given that same sequence, and the sequence is produced by that same formula, the process is a closed loop: identical starting conditions (empty store, `max_steps=8` always sufficient to exhaust the small 8-action menu) produce an identical first choice, which produces identical downstream state, which produces an identical second choice, and so on for all 8 steps — for any campaign, regardless of its actual evidence content.

## 13. Interpretation

**H2 confirmed, H1 falsified.** The current NBE policy does not fail to be adaptive because content-sensitive signals are outweighed by static ones — it never has a content-sensitive signal to weigh in the first place, for 7 of its 8 candidate actions, and the 8th (`XGBOOST_PREDICTION`) is structurally evaluated only in a state (`current_uncertainty=1.0`, pre-model) that itself never varies by campaign, given the always-identical `mitre_knowledge`-first sequence.

## 14. Limitations

- Only 3 real campaigns, and — as in Phase 21 — all 3 fully exhaust the same fixed 8-action menu within `max_steps=8`; the study cannot speak to a larger action universe or an investigation that stops early.
  *Planned:* the Phase 19 dataset-expansion spec is expected to provide enough real campaigns to test a larger action universe and early-stopping cases.
- The ablations are re-weightings of real, already-collected component values, not independent re-executions with alternate code — appropriate for isolating *which terms* can move ranking, but cannot test whether a differently-*structured* formula (e.g., one that reads `Evidence.confidence`/`Evidence.relevance` values directly into scoring, which none of the current terms do) would behave differently. That question is out of scope for an ablation of the *existing* formula.
  *Planned:* prototype and re-evaluate a redesigned NBE formula that reads `Evidence.confidence`/`Evidence.relevance` directly, once that redesign is scoped.
- `scipy.stats.kendalltau`/`spearmanr` return `nan` for single-element vectors (step 8, 1 remaining candidate); those 24/192 comparisons are correctly excluded from the tau statistics rather than treated as agreement, and are called out explicitly rather than silently dropped.
- No claim is made here about whether this is a *problem* worth fixing immediately — that judgment belongs to Section "Recommended next step," not this results section.
  *Planned:* revisit the NBE scoring formula (Section 16's proposed content-sensitive redesign) so adaptive terms can encode evidence content rather than only action-type sequence.

## 15. Conclusion

**Effectively static ranking.** Not "adaptive score variation but ranking invariance" (that would require the scores themselves to vary while still producing the same order) — the scores show literally zero cross-campaign variance, in every one of 192 pairwise comparisons, under 8 different scoring configurations including 3 that isolate individual adaptive terms. The current NBE formulation does not convert investigation-specific uncertainty or model state into a campaign-differentiated evidence-selection policy, for this real 3-campaign, 8-action, 8-step dataset. This is reported as the actual, unforced result of the ablation study.

## 16. Implications for CYUKTI's research claims

- The Phase 21 finding ("evidence-selection order was identical across 3 real campaigns") is **not weakened or contradicted** by this study — it is explained down to the specific code lines responsible, and shown to be the expected consequence of the formula's structure rather than a fluke of these 3 particular campaigns.
- The claim "the investigator adapts what it looks at based on the campaign" remains unsupported by real data and should not be made.
- The claim "the confidence/uncertainty computation is genuinely data-dependent on real campaigns" (Phase 21, Section 7) is **unaffected** — this study did not re-test confidence computation itself, only the NBE ranking formula that consumes one output of it (`current_uncertainty`).
- A new, more precise claim is now defensible: **"the current evidence-dependency/redundancy and novelty mechanisms are structurally activated by which action types run, not by the substance of what those actions found — this has been verified both by code inspection and by an 8-way ablation across 192 real-data comparisons showing zero exceptions."**

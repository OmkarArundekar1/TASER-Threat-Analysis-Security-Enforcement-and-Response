# CYUKTI — Evidence-Aware Adaptive Investigation

Implemented and tested 2026-09-14. This document describes a real, coded extension to CYUKTI's existing investigation architecture — not a proposal.

## 1. Problem statement

Given incomplete and uncertain cybersecurity telemetry, what evidence should an autonomous investigator acquire next, how should that evidence update the investigation's confidence, and when should it stop? A naive implementation collapses "this fact is true" (source reliability) with "this fact resolves the question" (investigative confidence) — and stops the moment it finds one reliable fact, regardless of relevance or breadth.

## 2. Existing CYUKTI architecture (found on inspection, largely already correct)

Full inspection of `investigation/` and `evidence/` found the core separation this task asked for **already implemented** from an earlier project phase: `investigation/confidence.py` already distinguishes `evidence_reliability`, `evidence_coverage`, `model_confidence`, `model_uncertainty`, and `investigation_confidence` as five distinct values; `investigation/next_best_evidence.py` already scores candidate actions by an explicit, interpretable value formula; `investigation/stopping.py` already checks confidence, uncertainty, and conflicts independently; `investigation/loop.py` already wires a real trained XGBoost model into the confidence calculation via an injected `model_predictor` callback. This is documented precisely so the review doesn't imply these were newly invented here.

## 3. Previous investigation-loop failure (historical, already fixed prior to this task)

Two real bugs, found by running the loop against real campaigns in an earlier phase:
1. `EvidenceStore.weighted_confidence()` gave relevance=0 evidence full weight instead of zero — fixed by excluding it from the weighted mean entirely.
2. A single reliable fact from one source (e.g. graph_structure, confidence=1.0) was read as investigation-resolved — fixed by introducing `evidence_coverage` (fraction of the 6 evidence-source categories actually consulted) as an independent gating factor, so reliability alone can no longer saturate confidence.

## 4. Root cause

Source reliability answers "is this observation true"; investigation confidence must answer "does the available evidence, taken as a whole, resolve the current hypothesis." Conflating them lets one narrow fact terminate an investigation regardless of how much of the evidence space remains unexplored.

## 5. New architecture (implemented this task)

Three genuine gaps were found and closed, on top of the already-correct foundation:

### 5a. Evidence dependency / redundancy (new — `investigation/actions.py`, `investigation/next_best_evidence.py`, `investigation/loop.py`)
`ActionMeta.depends_on` declares actions whose underlying data is **code-verified** (not assumed) to overlap another action's — confirmed by reading `threat_attribution_engine.attribute()`, which calls `context.load_historical_campaigns()` internally, the same data `CAMPAIGN_HISTORY`'s own collector queries directly. Once a depended-on action has been taken, the dependent action's expected value is discounted by `DEPENDENCY_REDUNDANCY_PENALTY` per satisfied dependency — bounded by the number of *declared* dependencies (currently one), not by how many unrelated actions have been taken, so it cannot grow unboundedly. `Evidence.derived_from` (new field) is populated per-instance at collection time, recording which specific evidence_ids a new item's information overlaps.

### 5b. Multi-hypothesis state (new — `investigation/loop.py`)
`InvestigationState` gained `candidate_hypotheses: list[tuple[str, float]]` — every model hypothesis with non-zero probability, ranked — alongside the existing `conclusion` (the single argmax label). This preserves the full competing-hypothesis picture (e.g. Critical 0.40 / Medium 0.35 / Low 0.25 is a materially different state from Critical 0.90 / Medium 0.06 / Low 0.04 despite sharing a top label) as an inspectable, per-hypothesis structure in the trace, not only as the aggregate entropy number that already existed.

### 5c. Conflict-severity scaling (fixed — `investigation/confidence.py`)
The conflict penalty was a flat 0.85 multiplier regardless of how many unresolved conflicts existed. It now compounds per conflict (`CONFLICT_PENALTY ** len(conflicts)`), so one disagreement and five are no longer treated identically. Behavior for 0 or 1 conflicts is byte-identical to before (verified: `0.85**1 == 0.85`); only 2+ conflicts are now handled differently, and only in the more conservative direction.

## 6. Evidence confidence vs. investigative confidence

Unchanged from the existing (already-correct) design — see Section 2. This task did not alter the core formula; it closed adjacent gaps (redundancy, multi-hypothesis, conflict scaling) around it.

## 7. Next-best-evidence formulation

```
value = W_GAIN·expected_gain + W_RELIABILITY·reliability + W_NOVELTY·novelty
        + W_UNCERTAINTY_REDUCTION·uncertainty_reduction
        - W_COST·cost - W_LATENCY·latency
        - W_REDUNDANCY·redundancy_penalty   <- new this task
```
All weights are documented, simple, non-fitted constants (no labeled outcome data exists yet to fit them against) — stated as a heuristic policy, not an optimal one, exactly as the existing code already documented for the pre-existing terms.

## 8. Uncertainty update

Unchanged mechanism (predictive entropy of `model_probabilities`, normalized Shannon entropy in [0,1]); newly exposed per-hypothesis via `candidate_hypotheses` (Section 5b) rather than only as the aggregate scalar.

## 9. Evidence dependency model

See Section 5a. This is a declared, static, code-verified dependency graph (one edge: ATTRIBUTION_MATCH → CAMPAIGN_HISTORY), not a general probabilistic dependency model — the task's own instructions permit this ("if full probabilistic dependency modeling would require excessive architectural changes, implement a clear dependency/redundancy model that can later be upgraded"). It is upgradable: `Evidence.derived_from` already records per-instance dependencies generically; extending `depends_on` to additional real, verified pairs requires no further architectural change.

## 10. Stopping criterion

Unchanged (already correct): stop when (confidence ≥ threshold AND model_uncertainty ≤ max AND no unresolved conflicts), OR max depth reached, OR no remaining action has positive expected value. Each `StoppingDecision` carries a human-readable reason, already distinguishing all four cases the task asked for.

## 11. UNKNOWN safety invariant

**Verified, not merely asserted.** Three new structural tests (`test_investigation_evidence_aware.py`) prove, by inspecting the actual source of `default_action_executor` and the whole `investigation/loop.py` module, that none of the Neo4j write functions capable of creating an `AttackEvent.attack_id`, a `Technique`/`MATCHES` relationship, or a `NEXT_TECHNIQUE` edge are ever referenced. The investigation module is read-only with respect to attribution ground truth by construction — it forms hypotheses (`InvestigationState.candidate_hypotheses`) and never has the capability to promote one into a graph fact. This is the same class of proof already used to certify the UNKNOWN ingestion path in Phase 20, applied here to a second, independent module.

## 12. Investigation trace design

`InvestigationRecord.to_dict()` now includes, per step: `why_selected` (a human-readable justification naming the runner-up action and value margin), `candidate_actions` and full `action_scores` for every action considered (not only the one chosen), `previous_confidence`/`previous_uncertainty` alongside the existing after-values, and `candidate_hypotheses`. A researcher can now answer both "why did CYUKTI choose this evidence next" and "why did CYUKTI stop" directly from the JSON trace, per-step.

## 13. Tests performed

164/164 backend tests passing (150 pre-existing + 14 new, zero regressions). New tests specifically cover: redundancy penalty triggers only for declared, satisfied dependencies and is bounded (not unlimited); ranking is state-dependent (the core adaptivity claim); `derived_from` is populated correctly from real dependency data; conflict penalty compounds with conflict count; `candidate_hypotheses` correctly ranks/filters; the trace exposes full candidate scoring; and three structural tests proving the UNKNOWN/hypothesis-vs-fact separation holds by source-code inspection.

## 14. Real integration results

**BLOCKED — Neo4j (`neo4j-soc`) unavailable at the time of this task** (`nc -z localhost 7687` → connection refused, confirmed both before and after implementation). Per this task's explicit instruction, infrastructure was not restarted to force a result. `scripts/run_real_investigations.py` already exists, unmodified, targeting three real campaigns (`CAMP_427A075C`, `CAMP_1429ADB4`, `CAMP_D8605E81`) with the real `default_action_executor`/`default_model_predictor` — it is ready to run the moment Neo4j is available, and doing so is the recommended immediate next step (see `review_story.md`'s future work). No fake executor was used to manufacture a substitute "integration" result — the test suite (Section 13), which exercises the identical real scoring/confidence/stopping code paths with controlled evidence content, is offered as the available evidence instead, correctly labeled as test-verified, not live-validated.
*Planned:* re-run the live-Neo4j integration validation via `scripts/run_real_investigations.py` once the infrastructure is available in a future session.

## 15. Current limitations

- Live, real-Neo4j integration validation is currently blocked by infrastructure availability, not attempted with substitutes.
  *Planned:* re-run via `scripts/run_real_investigations.py` once the infrastructure is available in a future session.
- The evidence-dependency model covers exactly one verified pair (ATTRIBUTION_MATCH/CAMPAIGN_HISTORY); other plausible overlaps (e.g. MITRE_SEMANTIC_SEARCH's partial overlap with MITRE_KNOWLEDGE) were deliberately left to the existing same-source discount rather than double-declared, to avoid an unverified/redundant dependency claim.
  *Planned:* verify and declare additional dependency pairs (e.g. MITRE_SEMANTIC_SEARCH/MITRE_KNOWLEDGE) once their overlap is confirmed by code inspection, following the same evidence standard used for the existing pair.
- Next-best-evidence weights remain hand-set constants, not fitted to any outcome data — stated as such in code, unchanged from the pre-existing design.
  *Planned:* fit these weights to real outcome data once a sufficient log of investigation-outcome pairs has been collected.
- The previously-reported real XGBoost probabilities for 3 named campaigns (from earlier project history) were not re-generated this task, since that requires the same unavailable Neo4j infrastructure.
  *Planned:* regenerate them in the same session as the live-Neo4j integration re-run above.

## 16. What remains NOT MEASURED

- Live-system investigation traces (blocked, Section 14).
  *Planned:* schedule a fresh live end-to-end investigation trace against Neo4j in the next verification session.
- Whether the redundancy-penalty/multi-hypothesis additions measurably change real investigation outcomes (step count, final confidence) versus the pre-existing behavior on real campaigns — untestable without Neo4j this session.
  *Planned:* compare before/after investigation traces on the same real campaigns once Neo4j is available.
- Any comparison of the heuristic action-value weights against a learned or outcome-calibrated policy — no such calibration data exists.
  *Planned:* collect a log of investigation outcomes and use it to fit or validate the action-value weights in a future phase.

## 17. Research novelty statement

This work demonstrates **evidence-aware autonomous investigation under incomplete and uncertain telemetry**: an investigator that explicitly separates source reliability from investigative conclusion confidence, selects its next evidence-gathering action by an interpretable, state-dependent expected-value calculation (not a fixed priority list — proven by test to re-rank as state changes), recognizes and discounts evidence whose information content is verified to overlap already-collected evidence, and is structurally incapable of promoting an unresolved hypothesis into ground-truth attribution. This is not a claim that CYUKTI surpasses or is better than any named commercial platform (Chronicle or otherwise) — no such comparison exists in this repository, and none is asserted here.

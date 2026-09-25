# CYUKTI — Research Claims Matrix

> Last verified: 2026-09-25. Counts reflect live Neo4j query and pytest run from this date. Only claims 1, 4, and 10's supporting "current state" numbers were refreshed; the Phase 20/21/22 historical claims (5-6, 9, 11-14) retain their original numbers as a record of what was true when each experiment ran — see `review/phase21_real_investigation_validation.md`/`review/phase22_nbe_sensitivity_validation.md` for those frozen results.

Each candidate claim classified using only evidence already in `review/quantitative_results.md` / `results_audit.md`.

| # | Claim | Classification | Why |
|---|---|---|---|
| 1 | CYUKTI integrates heterogeneous cybersecurity telemetry into a campaign-centric graph | **STRONGLY DEMONSTRATED** | Live graph (2026-09-25): 111 Campaigns, 212 AttackEvents, 858 Technique nodes, 9 Attacker, 12 Host, 50 Operation nodes — 2,539 total nodes, 20,804 total relationships (was 65/124/858/5/4/43, 119 MATCHES, 0 orphans on 2026-09-12; graph has grown via continued live ingestion, orphan/MATCHES counts not re-verified at the new scale). This is real accumulated Wazuh-sourced telemetry (not synthetic), structured around Campaign as the central object, with 689/689 backend tests (65 files) + 109/109 frontend tests (17 files) = 798 passing on the full system. |
| 2 | CYUKTI preserves provenance when MITRE attribution is unavailable | **STRONGLY DEMONSTRATED** | Every resolved `AttackEvent` carries `mitre_status`/`mitre_provenance`/`mitre_confidence`/`mitre_reason`/`mitre_resolver_version`; live-verified on 5 real UNKNOWN events across two sessions (2026-08-31 and 2026-09-12, surviving an infrastructure outage in between with all fields intact). |
| 3 | CYUKTI prevents fabricated ATT&CK attribution through an explicit UNKNOWN state | **STRONGLY DEMONSTRATED** | 0/5 live UNKNOWN events have a fabricated `attack_id` (property absent entirely, not a placeholder string); structurally proven (control-flow analysis) that the UNKNOWN branch can never reach `Technique` creation, `chain_updater`, `predict_next()`, or MISP publish; live-confirmed that `NEXT_TECHNIQUE` edge count stayed at 3 across UNKNOWN ingestion. This is the best-evidenced claim in the project — both structural (code-level) and empirical (live-data) support exist together. |
| 4 | Campaign reconstruction repairs real graph-integrity failures | **STRONGLY DEMONSTRATED** | A real, found-in-production defect (44 AttackEvents with no parent Campaign) was root-caused, repaired via evidence-only reconstruction (never guessing), proven idempotent by 10 passing tests, and the repair held as of the last orphan check (0/124 orphans, 2026-09-12; not re-run at the current 212-event scale). This is a genuine before/after real-data result, not a synthetic demonstration. |
| 5 | Evidence-aware investigation provides a foundation for uncertainty-aware SOC reasoning | **PARTIALLY DEMONSTRATED** | The architecture is real and implemented (`investigation/confidence.py` separates evidence_reliability, evidence_coverage, model_confidence, model_uncertainty, investigation_confidence as distinct signals) and extensively unit-tested (25 tests, including tests that specifically verify the system does *not* manufacture false confidence from absent evidence). What's missing: no fresh live end-to-end investigation trace was generated in the two most recent sessions, and the previously-reported real XGBoost probabilities for 3 named campaigns were not independently re-verified this session — so the claim rests on test-level evidence plus an unverified prior report, not a current live demonstration. |
| 6 | ML prediction is currently validated quantitatively | **PARTIALLY DEMONSTRATED** | XGBoost severity classification has real, measured in-sample metrics (accuracy 0.917, macro F1 0.676, per-class breakdown, confusion matrix) — this is genuine quantitative evidence. But "validated" in the normal ML sense (generalization) is not established: no held-out split, no cross-validation, no ROC-AUC/PR-AUC exist anywhere in the repository. NEXT_TECHNIQUE prediction has a real measured accuracy (33.3%) but the project's own audit classifies that as `INSUFFICIENT_FOR_SUPERVISED_ML`. Quantitative measurement exists; validation in the sense a reviewer would mean it does not. |
| 7 | Threat attribution is currently validated quantitatively | **NOT DEMONSTRATED** | No ground-truth attacker-identity dataset exists in the repository. The attribution engine is implemented and its evidence-collector is unit-tested (2 tests), but zero accuracy/precision/recall/F1/AUC figures exist. This is an honest, disclosed gap, not a hidden one. |
| 8 | RAG retrieval quality is currently validated quantitatively | **NOT DEMONSTRATED** | Only scale is measured (>500 real ATT&CK documents indexed, confirmed by test) — no retrieval precision/recall benchmark, no labeled query set exists. Index scale is real evidence of corpus grounding, but it is not a retrieval-quality validation. |

## Interpretation for the review

Claims 1-4 are the project's genuine, defensible strengths — each has both a structural/code-level argument and independent live-data confirmation. Claim 3 in particular (the explicit UNKNOWN state preventing fabricated attribution) is the strongest single research claim in the project because it is proven two ways at once (control-flow proof + live evidence) rather than resting on either alone. Claims 5-6 have real implementation and real numbers, but those numbers don't yet support the stronger claim a reviewer might assume from the words "validated" or "foundation." Claims 7-8 should be presented candidly as future work, not as demonstrated capabilities — the repository evidence simply doesn't exist yet, and inventing it would be indefensible.

---

## NEW PHASE RESULTS (2026-09-14): Evidence-Aware Adaptive Investigation

The following is new evidence, added after (and clearly separate from) everything above. It does not alter any previous claim's classification.

| # | Claim | Classification | Why |
|---|---|---|---|
| 9 | The investigation loop's next-best-evidence selection is genuinely state-dependent, not a fixed priority list | **STRONGLY DEMONSTRATED** | New test `test_ranking_changes_as_investigation_state_changes` proves the same action's score changes once a code-verified dependency (ATTRIBUTION_MATCH ↔ CAMPAIGN_HISTORY, confirmed by reading `threat_attribution_engine.attribute()`) has been satisfied. 14/14 new tests pass; 164/164 full suite, zero regressions. |
| 10 | Hypotheses formed during investigation can never be promoted into ground-truth ATT&CK attribution | **STRONGLY DEMONSTRATED** | Three new structural tests inspect the actual source of `default_action_executor` and `investigation/loop.py` and prove no Neo4j write function (`create_attack_event`, `append_technique`, `update_attack_chain`, etc.) is ever referenced — the same class of proof already used to certify UNKNOWN-path safety, applied to a second, independent module. |
| 11 | This capability has been validated against live production data | **NOT DEMONSTRATED** | Neo4j was unavailable throughout this implementation session; `scripts/run_real_investigations.py` exists and is ready, but was not run. This is disclosed as a real, current limitation, not hidden. |

See `review/evidence_aware_investigation.md` for the full research writeup.

---

## PHASE 21 REAL-CAMPAIGN RESULTS (2026-09-14): supersedes claim 11, refines claim 9

Distinct from the "NEW PHASE RESULTS" section above, which was produced with Neo4j unavailable. Neo4j later came back reachable this same day (see `review/phase21_real_investigation_validation.md` for the full root-cause note); the real experiment was then run against 3 live campaigns.

| # | Claim | Classification | Why |
|---|---|---|---|
| 11 (updated) | This capability has been validated against live production data | **PARTIALLY DEMONSTRATED** | `scripts/run_real_investigations.py` was run unmodified against 3 real Neo4j campaigns. The pipeline ran end-to-end without error and produced real, campaign-dependent confidence/uncertainty numbers (Section 7 of the validation doc). This upgrades claim 11 from NOT DEMONSTRATED, but with the caveat below. |
| 12 | Evidence-selection order adapts to real campaign-specific content | **FALSIFIED FOR THE TESTED CONDITION** | Concrete numeric evidence: the action order and every per-step `selection_value` were identical (to 3 decimals) across all 3 real campaigns, despite different techniques/risk scores/model outputs. Root cause verified by reading `next_best_evidence.py`: the ranking formula's only content-sensitive term (`uncertainty_reduction`) is nonzero solely for the XGBoost action; every other term depends on *which action types* were taken, not on the evidence *values* they returned. Since all 3 real investigations exhausted the same fixed 8-action universe, order was necessarily identical. This is a disclosed negative result, not a bug — see claim 9 above for the test-verified mechanism this does not contradict (dependency discounting still fires correctly, just not differentially in this run). |

**Net effect on claim 9 (state-dependent ranking):** claim 9's unit-test evidence (ranking changes once a specific dependency is satisfied, within one investigation) still stands and is unaffected. What real data adds is a boundary condition: across *different* campaigns that each exhaust the full action set, the resulting order does not differ. Both statements are true and are not in tension — they describe different things (within-investigation state changes vs. across-campaign comparison).

---

## PHASE 22 REAL-CAMPAIGN RESULTS (2026-09-14): root-causes claim 12, does not reopen it

An 8-way ablation study (`review/phase22_nbe_sensitivity_validation.md`) re-weighted the real, already-collected `score_action()` component values from the same 3 Phase-21 campaigns through 8 alternate linear combinations, isolating each nominally-adaptive term individually.

| # | Claim | Classification | Why |
|---|---|---|---|
| 13 | The current NBE formula's adaptive terms encode evidence *content* | **FALSIFIED** | 192/192 pairwise campaign comparisons (8 steps × 8 ablations × 3 pairs) showed zero score variance and zero ranking divergence — including 3 ablations (F1/F2/F3) that isolate `novelty`, `uncertainty_reduction`, and `redundancy_penalty` individually, with no static-term competition to hide behind. This directly falsifies "static terms dominate/mask real adaptive variation" (H1) and confirms instead that the adaptive terms are functions of the *action-type sequence*, not of the *evidence values* any collector returned (H2). |
| 14 | The XGBoost action's `uncertainty_reduction` term is campaign-sensitive | **FALSIFIED, root cause identified** | Traced through the confidence pipeline: `mitre_knowledge` is always taken first (highest static score on an empty store, in every campaign); its collector (`mitre_collector.py`) never sets `Evidence.relevance`, which stays at its 0.0 default; `EvidenceStore` gives zero weight to relevance-0.0 evidence; so `evidence_reliability`/`evidence_coverage` are exactly `0.0` after step 1 in every campaign, unconditionally — pinning `current_uncertainty=1.0` at the one moment XGBoost's score matters, regardless of real campaign content. |

This does not reopen or weaken claim 12 (Phase 21's finding that ranking was invariant) — it explains the exact code-level mechanism producing it, with a decisive (not merely suggestive) ablation result.

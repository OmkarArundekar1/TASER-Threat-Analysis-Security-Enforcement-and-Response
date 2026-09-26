# CYUKTI — Novelty Argument

## 1. One-sentence novelty claim

CYUKTI structurally separates telemetry ingestion from ATT&CK attribution, enforcing — both by code-level design and by live-verified evidence — that an explicit, provenance-tracked UNKNOWN state can never contaminate downstream attack-chain learning, risk scoring, or prediction, rather than forcing a choice between discarding unattributed evidence and fabricating attribution for it.

## 2. Three supporting mechanisms

1. **Provenance-tracked, precedence-ordered MITRE resolution** (`mitre_resolver.py`): four tiers — native Wazuh mapping, reviewed rule registry, deterministic structural inference, UNKNOWN — where a lower-confidence tier can never override a higher one, and every result carries an ordinal confidence (never a fabricated probability) plus a human-readable reason.
2. **Structural non-contamination guarantee**: the UNKNOWN code path is architecturally incapable of reaching `Technique` node creation, `append_technique()`, `chain_updater.update_attack_chain()`, `predict_next()`, or MISP publication — proven by direct control-flow analysis (the branch returns before those call sites are reachable), not merely by convention or documentation.
3. **Evidence-aware confidence separation**: `investigation/confidence.py` treats evidence reliability, evidence coverage, model confidence, and model uncertainty as four distinct signals rather than one conflated scalar, specifically to prevent the system from manufacturing high confidence from sparse evidence — a real bug (confidence=1.0 after one evidence item) that was found and fixed.

## 3. Quantitative evidence supporting each

1. Live measurement, 391 real alerts: 39 (10.0%) NATIVE_WAZUH, 0 reviewed, 0 inferred, 352 (90.0%) UNKNOWN — a real, measured precedence outcome, not a synthetic demonstration.
2. Live Neo4j inspection of 5 real UNKNOWN events across two sessions separated by a real infrastructure outage: 0 fabricated `attack_id` (property absent, not placeholder), 0 `Technique`/`MATCHES` relationships, `NEXT_TECHNIQUE` edge count unchanged (3→3), `tps=0` on all 5.
3. 25 passing tests in `test_investigation.py`, including tests specifically constructed to verify the system does *not* falsely report near-certainty from uncertain model probabilities.

## 4. What cannot yet be claimed

- That this design measurably improves any downstream task's accuracy relative to a naive discard-or-fabricate baseline — no comparative experiment exists.
  *Planned:* run a controlled comparative experiment against a naive discard-or-fabricate baseline once the Phase 19 dataset expansion provides enough scale to make the comparison meaningful.
- That the mechanism generalizes beyond this specific Wazuh/Neo4j pipeline — it has been validated in exactly one deployment.
  *Planned:* validate the same design on a second, independent Wazuh/Neo4j-style deployment to test generalization.
- Any comparison to existing commercial or open-source SOC platforms — no such comparison exists in the repository, and none should be asserted without one.
  *Planned:* benchmark against an open-source SIEM/SOAR baseline once a comparable test environment can be set up.
- Novelty of the underlying components themselves (graph databases, TF-IDF, gradient boosting, GraphSAGE) — all are standard techniques; the contribution is their application and the specific safety guarantee around attribution, not the techniques themselves.

## 5. A 30-second oral explanation

"Most real security telemetry doesn't come with a machine-readable attack-technique label — in our own measurement, 90% of real alerts didn't. The standard failure mode is either discarding that evidence or guessing a label for it, and guessing is worse, because a fabricated label silently poisons every model trained on it downstream. CYUKTI instead resolves attribution as a separate, provenance-tracked step with an explicit UNKNOWN outcome, and we can prove — not just claim — that UNKNOWN evidence never leaks into the learned attack-chain graph, even after a real infrastructure outage and recovery. That's the contribution: not a new algorithm, but a verifiably safe way to handle the attribution gap that real security telemetry actually has."

---

## NEW PHASE RESULTS (2026-09-14): Evidence-Aware Adaptive Investigation

Extends, does not replace, the novelty claim above. The same principle — never let source reliability alone stand in for the answer to the actual question being asked — now also governs *evidence selection* during investigation, not only MITRE attribution: the investigator recognizes when a candidate evidence-gathering action's information is code-verified to overlap already-collected evidence (e.g. ATTRIBUTION_MATCH vs. CAMPAIGN_HISTORY) and discounts it accordingly, rather than treating every additional fact as independent corroboration. This is a real, tested (164/164 passing) extension of the same underlying design discipline to a second investigative concern (evidence redundancy) beyond the first (attribution provenance). Live-data validation of this specific extension is `NOT MEASURED` this session — Neo4j was unavailable — and is disclosed as such rather than assumed. See `review/evidence_aware_investigation.md`.
*Planned:* re-run the live-Neo4j integration validation once the infrastructure is available in a future session.

---

## PHASE 21 REAL-CAMPAIGN RESULTS (2026-09-14): honest update, not a stronger novelty claim

Neo4j became reachable later the same day and the real experiment was run (`review/phase21_real_investigation_validation.md`). The honest result **does not strengthen** the novelty argument above — if anything it sharpens its boundary. Across 3 real campaigns, evidence-selection *order* was identical every time, for a code-verified, mechanistic reason (the ranking formula's only content-sensitive term applies to a single action type; the rest depends on which action *types*, not which evidence *values*, have already been seen). The redundancy-discount mechanism itself did fire correctly on real data (it just wasn't put in a position to differ across these 3 campaigns). The defensible novelty claim remains what it was before this run — a structurally-verified, provenance-aware separation of detection from attribution, now also extended (with test-level, not yet differentiated live-data, evidence) to evidence redundancy. "The investigator adapts what it looks at next, campaign to campaign" is **not** a claim this data supports; it should not be made in review.

---

## PHASE 22 REAL-CAMPAIGN RESULTS (2026-09-14): the negative result is now mechanistically explained, not just observed

An ablation study (`review/phase22_nbe_sensitivity_validation.md`) isolated each of the NBE formula's nominally-adaptive terms (novelty, uncertainty-reduction, dependency/redundancy) individually, across all 3 real campaigns and all 8 investigation steps — 192 pairwise comparisons total, zero divergence in any of them, including the 3 isolated-term configurations. This rules out "static terms are simply outweighing real adaptive signal" as the explanation and confirms instead that the adaptive terms currently only read *which action types have run*, never *what evidence content those actions returned*. **This does not change the novelty argument's boundary from what Phase 21 already established** — it replaces "we observed invariant ranking" with "we know precisely why, at the level of specific code paths, and have ruled out the alternative explanation." No stronger claim is licensed by this; if anything, it sharpens the honest boundary further: the current evidence-redundancy mechanism (the extension claimed above) is real and tested, but is provably insensitive to evidence content — it is a same-action-type-repeat guard, not a content-aware redundancy detector, and should be described that way if asked directly.
*Planned:* give `mitre_collector.py` (and other static-scoring collectors) a real, non-zero relevance signal so novelty/redundancy discounting becomes content-aware rather than type-only — identified but out of scope for this project's remaining time.

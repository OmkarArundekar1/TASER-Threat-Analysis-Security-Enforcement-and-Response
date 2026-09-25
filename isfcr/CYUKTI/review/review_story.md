# CYUKTI Review Story

> Last verified: 2026-09-25. Counts reflect live Neo4j query and pytest run from this date. Sections 1-7 (current-state narrative) have been refreshed; the Phase-dated sections below the divider are historical records of specific experiments and retain their original numbers.

## 1. Problem

Real-world SOC telemetry (Wazuh, in this project's case) mostly arrives without a native MITRE ATT&CK tag — verified empirically: of the current 120-alert live snapshot (2026-09-25, post rule-fix/reboot), only 26 (21.7%) carried a native technique mapping (an earlier, larger 391-alert snapshot from 2026-08-31 measured 10.0%). A pipeline that requires attribution before ingestion discards most of its own evidence either way. A pipeline that fabricates attribution to avoid that corrupts every downstream consumer that trusts the technique label as ground truth — attack-chain learning, risk scoring, and any ML trained on it. This is the "detection paradox": detection (a sensor firing) and attribution (defensibly naming the technique) are different questions, and conflating them forces a false choice between losing evidence and fabricating conclusions.

## 2. Design response

CYUKTI separates ingestion from attribution with an explicit, provenance-tracked, four-tier MITRE resolver (`mitre_resolver.py`): native Wazuh mapping → reviewed rule registry → deterministic structural inference → an explicit `UNKNOWN` state. Every result carries `provenance`, an ordinal `confidence` (never a fabricated probability), and a `reason`. Events that cannot be defensibly attributed are still ingested as evidence — but structurally barred (not just by convention) from participating in attack-chain learning, prediction, risk scoring, or MISP publication.

## 3. Evidence

- Live measurement (120 alerts, 2026-09-25, post rule-fix/reboot): 26 native (21.7%), 94 UNKNOWN (78.3%), 0 reviewed/inferred/ambiguous. (Earlier snapshot, 391 alerts, 2026-08-31: 39 native (10.0%), 352 UNKNOWN (90.0%) — preserved as historical, superseded traffic composition.)
- Live Neo4j inspection of 5 real UNKNOWN events, across two sessions separated by an infrastructure outage: 0 fabricated `attack_id`, 0 `Technique` relationships, `NEXT_TECHNIQUE` edge count unchanged (3→3), `tps=0` on all 5, raw payload preserved on all 5.
- Structural (control-flow) proof that the UNKNOWN code branch can never reach `append_technique()`, `chain_updater.update_attack_chain()`, `predict_next()`, or MISP publish.
- 798/798 automated tests passing (689 backend across 65 files + 109 frontend across 17 files, as of 2026-09-25), including 15 resolver-specific and 8 integration tests built specifically to catch regressions in this mechanism.

## 4. Key result

**The explicit UNKNOWN state is both structurally guaranteed and empirically confirmed to never contaminate attack-chain learning.** This is the single strongest result because it's supported two independent ways at once — a code-level proof that the corrupting call sites are unreachable from the UNKNOWN branch, *and* live graph evidence (identical `NEXT_TECHNIQUE` count before and after real UNKNOWN ingestion, across a real infrastructure outage and recovery) that it actually holds in practice, not just in theory.

## 5. Important negative result

The ML components do not currently generalize, and the project says so explicitly rather than hiding it:
- XGBoost severity classifier: 91.7% accuracy, but **in-sample only** (n=60, the same rows it was trained on) — no held-out split exists anywhere in the repository.
- NEXT_TECHNIQUE prediction: 33.3% accuracy (4/12 evaluable predictions) — the project's own dataset-validity audit (Phase 17) and NEXT_TECHNIQUE audit (Phase 18) independently concluded, respectively, `NOT_READY_FOR_CALIBRATION` and `INSUFFICIENT_FOR_SUPERVISED_ML`, **before** these numbers were even computed.
- Threat attribution: zero accuracy metric exists, because no ground-truth attacker-identity dataset exists.

## 6. Why the negative results matter (and strengthen, not weaken, academic credibility)

A system that reports 91.7% accuracy without disclosing it's in-sample, or 33.3% next-technique accuracy without disclosing the sample size is 12, would be making an indefensible claim the moment a reviewer asks "what was your test set?" Here, that question has a real, pre-existing answer that was reached *before* the numbers were computed: the project ran a dedicated dataset-validity audit (Phase 17) that concluded the accumulated real data — only 3 attacker identities, 2 victims, 23/57 zero-variance features, 11 exact duplicate rows — was not statistically sufficient for calibration, and only then computed the in-sample metrics, labeled accordingly. Reporting 25% Critical-class recall (3 of 4 Critical campaigns missed, even in-sample) rather than only the flattering 91.7% aggregate accuracy demonstrates the same discipline. A reviewer who probes any of these numbers finds a project that already asked the harder question first — that is what distinguishes a defensible research result from an inflated one.

## 7. Current maturity: RESEARCH PROTOTYPE

- Real, live, multi-service integration (Wazuh + Neo4j + a trained ML stack) with tested, working recovery from a real infrastructure outage — beyond a bare prototype.
- A genuine production defect (44 orphaned AttackEvents) was found, root-caused, and durably repaired with proven idempotency — evidence of engineering rigor, not just design intent.
- No held-out evaluation methodology exists for any ML component, and the project's own internal audits explicitly rule several core modules not-yet-viable for the claims a mature system would make — this rules out "Production-like Research System" or higher.
- At least one disclosed, unfixed defect remains live (a maintenance-thread exception triggered by a specific UNKNOWN-first-event condition) — non-blocking, but unresolved by deliberate choice pending broader scope authorization.
- Several major modules (attribution, GNN on real data, RAG retrieval quality, MISP live publication) have zero quantitative benchmark beyond unit-test-level verification.

## 8. Future experimental path (smallest set to move toward EMPIRICALLY VALIDATED RESEARCH SYSTEM)

Only the minimum needed — not a general engineering wishlist:

1. **Deliberate, diversity-targeted data collection** (not just more volume): the Phase 19 spec already exists — 6-8 attacker identities, 5+ victims, at least one real High-severity example, ≥20 technique-set compositions. This single step unblocks both severity calibration and a meaningful held-out ML split, since the current blocker for both is diversity, not row count.
2. **One held-out ML evaluation**, once (1) makes it statistically meaningful — a single train/test split on the expanded dataset, reporting accuracy, macro F1, and per-class recall the same way the in-sample numbers are reported now.
3. **A minimal labeled attribution benchmark**: even a small set of campaigns with known, manually-confirmed attacker identity would let the existing attribution engine be scored for the first time, rather than remaining permanently `NOT MEASURED`.
4. **Fix the disclosed maintenance-worker defect** — small, scoped, already root-caused; the remaining work is authorization, not investigation.

No new subsystem needs to be built to take this step — every item above evaluates or extends something that already exists.

---

## NEW PHASE RESULTS (2026-09-14): Evidence-Aware Adaptive Investigation

Everything above this line reflects the project state through the previous session. This section documents new, real, coded work — not a revision of the prior sections.

**What was found**: the investigation architecture's core confidence separation (evidence reliability vs. coverage vs. model confidence vs. investigation confidence) was already correctly implemented from an earlier phase. Three genuine gaps were identified and closed: (1) no mechanism existed to discount an evidence-gathering action once a code-verified overlapping dependency had already been satisfied (redundancy could accumulate unbounded credit); (2) the investigation state tracked only a single top hypothesis, not the full competing set; (3) the conflict penalty was a flat multiplier regardless of how many unresolved conflicts existed.

**What was implemented**: `ActionMeta.depends_on` (one real, verified dependency: ATTRIBUTION_MATCH → CAMPAIGN_HISTORY, confirmed by reading `threat_attribution_engine.attribute()`'s source), a bounded `redundancy_penalty` term in the next-best-evidence value function, per-instance `Evidence.derived_from` tracking, `InvestigationState.candidate_hypotheses`, and a compounding (rather than flat) conflict penalty. 14 new tests, 164/164 total passing, zero regressions.

**What was not established**: live-Neo4j validation. Infrastructure was unavailable throughout this session and was not restarted to force a result — this is disclosed plainly, consistent with the rest of this project's evidence discipline, rather than substituted with a fake demonstration.

**Full detail**: `review/evidence_aware_investigation.md`.

---

## PHASE 21 REAL-CAMPAIGN RESULTS (2026-09-14)

Everything above reflects the state through the implementation session, when Neo4j was unavailable. Later the same day, Neo4j was found genuinely reachable on a follow-up check (the container had come back after an independent Docker daemon restart — not something this session did) and the real experiment was finally run.

**What was found, running unmodified `scripts/run_real_investigations.py` against 3 real campaigns:** confidence and uncertainty outputs are genuinely campaign-dependent on real data — final `investigation_confidence` ranged 0.069-0.184, `model_probabilities` differed substantially, all consistent with real evidence, not fixed numbers. But evidence-selection *order* was identical across all 3 real campaigns, to 3 decimal places on every per-step score. Reading `next_best_evidence.py` explains why: the ranking formula's only evidence-content-sensitive term applies to exactly one action (the model prediction); everything else depends on which action *types* have run, not what they returned — and all 3 real investigations exhausted the same fixed 8-action universe, so order was necessarily identical every time.

**How this is being told in review**: as a real, mixed result, not a clean win. The strong form of "the investigator adapts what it looks at based on the campaign" is not supported by this data — say so directly if asked. The confidence/uncertainty architecture being genuinely data-dependent on real campaigns, and the dependency-discount mechanism firing correctly when its precondition is met, are supported and can be stated as such.

**Full detail**: `review/phase21_real_investigation_validation.md`, `review/phase21_real_investigation_results.json`.

---

## PHASE 22 RESULTS (2026-09-14): why the order never changed

This is an evaluation/ablation study, not new features — nothing in `investigation/next_best_evidence.py` or any other production module was touched.

**What was done:** took the real component values `score_action()` already computes and records for every candidate action, at every step, of the same 3 real Phase 21 campaigns, and re-weighted them through 8 alternate configurations — the full current formula, three "turn one term off" ablations, a pure-static baseline, and three "turn on exactly one adaptive term, nothing else" configurations.

**What was found:** all 8 configurations agree, perfectly, across all 3 campaigns, at all 8 steps — 192 pairwise comparisons, 0 ranking differences, 0 score differences, not even a rounding-level discrepancy. Critically, isolating each adaptive term individually (novelty alone, uncertainty-reduction alone, redundancy alone) still produced zero cross-campaign variance — ruling out "the real signal is there but static terms are drowning it out" as the explanation. The actual explanation, traced to specific lines of code: the formula's "adaptive" terms respond only to *which action types have already run* (a coarse, sequence-level fact), never to *what those actions actually found*. The one term that could in principle carry real content (XGBoost's model-state term) is evaluated only at a moment that's structurally pinned to the same value every time, because the action that always runs immediately before it (`mitre_knowledge`) produces evidence with zero `relevance` by construction of its collector — so it can never move the coverage/reliability numbers that term depends on.

**How this is being told in review:** as a root-cause finding, not a new negative result layered on the old one. Phase 21 said "the order didn't change"; Phase 22 says, with code-level precision and an exhaustive ablation, exactly why — and rules out the more forgiving explanation (signal present, just outweighed) in favor of the less forgiving one (no content-sensitive signal exists in the formula as written, for 7 of 8 actions).

**Full detail**: `review/phase22_nbe_sensitivity_validation.md`, `review/phase22_nbe_sensitivity_results.json`.

# CYUKTI — One-Page Review Cheat Sheet

## CYUKTI in one sentence
CYUKTI ingests real Wazuh security telemetry into a Neo4j campaign graph and resolves MITRE ATT&CK attribution through a provenance-tracked, four-tier resolver that preserves unattributable evidence as an explicit, structurally-isolated UNKNOWN state instead of discarding or fabricating it.

## 10 numbers to memorize
1. **391** — real alerts in the representative MITRE-coverage snapshot (2026-08-31)
2. **39/391 (10.0%)** — native ATT&CK-mapped alerts
3. **352/391 (90.0%)** — preserved as UNKNOWN (by design, not a deficiency)
4. **0/5** — fabricated `attack_id` values across all live-inspected UNKNOWN events
5. **3 → 3** — NEXT_TECHNIQUE edge count, unchanged across UNKNOWN ingestion
6. **44 → 0** — orphaned AttackEvents repaired
7. **150/150** — automated tests passing
8. **858** — real ATT&CK Technique nodes imported (STIX v19.1)
9. **91.7%** — XGBoost accuracy, **in-sample only**, n=60
10. **4/12 (33.3%)** — NEXT_TECHNIQUE accuracy, labeled `INSUFFICIENT_FOR_SUPERVISED_ML`

## 5 strongest contributions
1. Provenance-aware, precedence-ordered MITRE resolution with mandatory confidence/reason on every result.
2. Structurally-proven + live-verified UNKNOWN state that cannot contaminate attack-chain learning.
3. Real production data-integrity defect (44 orphaned events) found, root-caused, and durably repaired.
4. Evidence-aware confidence architecture that separates reliability/coverage/model-confidence/uncertainty to prevent manufactured confidence.
5. Demonstrated operational resilience — recovered from a real Neo4j infrastructure outage with zero data loss.

## 5 biggest limitations
1. No held-out ML evaluation exists anywhere in the repository.
2. No ground-truth attribution dataset — attribution accuracy is `NOT MEASURED`.
3. Real dataset diversity is limited: 3 attackers, 2 victims, 0 High-severity examples.
4. One disclosed, unfixed live defect (maintenance-thread `last_seen=None` exception) — non-blocking.
5. No latency, throughput, or RAG-retrieval-quality benchmark exists.

## 5 difficult questions + answers
1. **"Isn't 90% UNKNOWN bad?"** → It's the correct outcome of refusing to fabricate attribution — the real question is contamination, and we can show 0%.
2. **"Is 91.7% a test accuracy?"** → No — in-sample, n=60, no held-out split exists; we say so explicitly.
3. **"How do you validate attribution?"** → We don't yet — no ground-truth dataset exists, and we report that honestly rather than inventing a number.
4. **"What's actually novel here?"** → Not the algorithms — the structurally-enforced, live-proven separation of detection from attribution.
5. **"Does 150/150 tests mean it's bug-free?"** → No — we found a real production bug live that no test anticipated, and disclose it.

## One sentence on ML
The ML pipeline (XGBoost severity, SSL autoencoder, GNN) is real, trained, and tested end-to-end, but every reported metric is in-sample or synthetic-only — no held-out generalization claim exists yet, by deliberate, evidence-based choice.

## One sentence on UNKNOWN handling
UNKNOWN is a first-class, provenance-tracked outcome — not a failure or a discard — verified live to preserve full raw evidence while being structurally incapable of touching technique attribution, attack-chain learning, prediction, or risk scoring.

## One sentence on attribution
Threat attribution is implemented and unit-tested at the evidence-collector level, but has zero quantitative accuracy measurement because no ground-truth attacker-identity dataset currently exists.

## One sentence on future work
The single highest-leverage next step is deliberate, diversity-targeted real-data expansion (more attacker/victim identities, at least one real High-severity example) — not more data volume — because it simultaneously unblocks a meaningful ML held-out evaluation and a first attribution benchmark.

## Final maturity:
**RESEARCH PROTOTYPE**

---

## NEW PHASE RESULTS (2026-09-14): Evidence-Aware Adaptive Investigation
Closed three real gaps in the existing (already-largely-correct) investigation architecture: bounded evidence-redundancy discounting (one verified dependency: attribution ↔ campaign history), multi-hypothesis state tracking, and conflict-severity-scaled confidence. 164/164 tests passing (14 new, zero regressions). Structurally proven the investigation module can never promote a hypothesis into ground-truth ATT&CK attribution. **Live-Neo4j validation was not performed — infrastructure was unavailable and was not restarted to force a result.** Full detail: `review/evidence_aware_investigation.md`. Maturity classification is unchanged: **RESEARCH PROTOTYPE**.

## PHASE 21 REAL-CAMPAIGN RESULTS (2026-09-14, same day, Neo4j came back)
Ran the real, unmodified `scripts/run_real_investigations.py` against 3 live campaigns once Neo4j was confirmed reachable. **Honest mixed result — say this if asked:** confidence/uncertainty numbers ARE genuinely campaign-dependent on real data (0.069-0.184 final confidence, all different); evidence-SELECTION-ORDER was NOT — identical across all 3 real campaigns to 3 decimal places, because the ranking formula's only content-sensitive term applies to one action type only, and all 3 investigations exhausted the same fixed 8-action set. This is a disclosed negative result on the strong adaptivity claim, not a bug. Do not say "the investigator adapts evidence choice per campaign" — the real data available so far says otherwise. Full detail: `review/phase21_real_investigation_validation.md`.

## PHASE 22 RESULTS (2026-09-14): root cause of the invariant ranking, nailed down
Ablation study (8 configurations × 3 real campaigns × 8 steps = 192 pairwise comparisons): zero score/ranking differences in every single one, including 3 configurations that isolate each adaptive term (novelty, model-uncertainty, dependency-redundancy) alone with no static-term competition. **Root cause, not just "it doesn't adapt":** the formula's adaptive terms respond only to which action TYPES have already run, never to what those actions actually found; the one term that could carry real model state (XGBoost's) is always evaluated at a moment pinned to the same value, because the action always taken right before it produces evidence with `relevance=0.0` by construction of its collector, which structurally zeroes the coverage signal that term depends on. Classification: **effectively static ranking** (not "adaptive scores that don't change order" — the raw scores themselves never differ). Say this if asked "is it actually adaptive": no, and we know exactly which code path explains it. Full detail: `review/phase22_nbe_sensitivity_validation.md`.

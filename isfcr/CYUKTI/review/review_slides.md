# CYUKTI — 8-Slide Technical Review Structure

> Last verified: 2026-09-25. Counts reflect live Neo4j query and pytest run from this date.

## Slide 1 — CYUKTI: Research Objective
**Objective**: frame the detection-paradox problem and CYUKTI's response before any architecture detail.
- Real SOC telemetry mostly arrives without native ATT&CK attribution (currently measured: 21.7% of 120 real alerts, post rule-fix/reboot, 2026-09-25; a larger historical 391-alert snapshot from 2026-08-31 measured 10.0%).
- Requiring attribution before ingestion discards most real evidence; fabricating attribution corrupts everything downstream that trusts it.
- CYUKTI's response: separate ingestion from attribution, with an explicit, provenance-tracked UNKNOWN state.
- Numbers to display: **26/120 (21.7%) native**, **94/120 (78.3%) UNKNOWN** (2026-09-25).
- Figure/table: none — text framing slide.
- **Say**: "Most real security telemetry doesn't come pre-labeled with an attack technique, and what you do about the other 90% defines the system."
- **Likely question**: "Isn't this just a data-cleaning problem?" 
- **Defensible answer**: "It's a ground-truth-integrity problem — mislabeling here doesn't just look messy, it silently corrupts every downstream model trained on it."

## Slide 2 — System Architecture
**Objective**: show the actual pipeline, not a generic SOC diagram.
- Wazuh → Listener → MITRE Resolver → {Resolved, Unknown} → Neo4j → chain/prediction/correlation/attribution/ML/RAG → Dashboard.
- Central graph: Campaign, AttackEvent, Technique, Attacker, Host, Operation.
- Real, vendored ATT&CK STIX corpus (Enterprise v19.1) — no runtime internet dependency.
- Numbers to display: **858 Technique nodes**.
- Figure/table: `cyukti_detailed_architecture.mmd` (from `review_pack/`).
- **Say**: "Everything downstream reads from one graph that both resolved and unresolved evidence feed into."
- **Likely question**: "Why Neo4j specifically?"
- **Defensible answer**: "Campaigns, events, and techniques are naturally graph-structured — attack-chain and correlation queries map directly onto Cypher pattern matching rather than repeated joins."

## Slide 3 — Campaign-Centric Security Graph
**Objective**: demonstrate real scale and integrity, not synthetic data.
- 111 Campaigns, 212 AttackEvents, 50 Operations (2,539 total nodes, 20,804 total relationships), as of 2026-09-25 (was 65/124/43 on 2026-09-12).
- 44 real orphaned events (a genuine production defect) were found, root-caused, and repaired — idempotently, evidence-only, never guessing.
- Repair held as of the last orphan check (0/124, 2026-09-12); not re-run at the current 212-event scale.
- Numbers to display: **111 Campaigns / 212 AttackEvents / 2,539 nodes / 20,804 relationships / 44→0 repaired**.
- Figure/table: simple before/after bar (orphans: 44 → 0).
- **Say**: "We found and fixed a real data-integrity defect in our own production graph, and proved the fix is stable and idempotent."
- **Likely question**: "How do you know the 27 reconstructed campaigns are correct, not guessed?"
- **Defensible answer**: "Reconstruction only uses data already present on the orphaned events themselves — attacker, victim, technique, timestamps — it never infers or guesses a value that isn't already recorded on a child node."

## Slide 4 — MITRE Resolution + UNKNOWN Safety
**Objective**: this is the core contribution slide.
- Four-tier resolver: NATIVE_WAZUH → REVIEWED_RULE_MAPPING → DETERMINISTIC_INFERENCE → UNKNOWN, each with provenance/confidence/reason.
- 78.3% UNKNOWN (current, post rule-fix/reboot snapshot) is the *correct*, intended outcome — not a coverage failure.
- 5 live UNKNOWN events verified twice (across an infrastructure outage): 0 fabricated `attack_id`, 0 Technique links, `NEXT_TECHNIQUE` unchanged.
- Numbers to display: **26/120 (21.7%) native, 94/120 (78.3%) UNKNOWN (2026-09-25), 0/5 fabricated attack_id, 3→3 NEXT_TECHNIQUE edges**.
- Figure/table: stacked bar of provenance distribution (native/reviewed/inferred/unknown).
- **Say**: "The large majority unresolved is what correct behavior looks like when you refuse to fabricate an answer — and we can prove, not just claim, that it never contaminates the learned attack chain."
- **Likely question**: "Isn't a ~78% unresolved rate a weak result?"
- **Defensible answer**: "Only if the goal were maximizing attribution rate. Ours is never attributing something we can't defend — the correct comparison is 0% fabricated, not the unresolved percentage."

## Slide 5 — Quantitative System Validation
**Objective**: show test rigor and reproducibility.
- 689/689 backend tests (65 files) + 109/109 frontend tests (17 files) = 798/798 passing (unit + integration + regression), as of 2026-09-25 — was 150/150 backend-only on 2026-09-12.
- Coverage spans every major module, including a new SOAR/playbook layer (83 tests, 9 files) added since the earlier snapshot.
- Clean compile check across the full backend.
- Numbers to display: **798/798**.
- Figure/table: simple table of test counts per module (from `quantitative_results.md`).
- **Say**: "Every regression we've found in this project has become a permanent test, including the ones we found live in production."
- **Likely question**: "Does 798/798 mean the system is bug-free?"
- **Defensible answer**: "No — we found a real production bug weeks ago that no existing test caught, specifically because it depended on an interaction the test suite hadn't anticipated. We disclose it rather than claim otherwise."

## Slide 6 — ML / Prediction Results
**Objective**: present real numbers with correct, unambiguous caveats — this is the slide most likely to be challenged.
- XGBoost severity classifier: 91.7% accuracy — **in-sample, n=60, no held-out split**.
- Critical-class recall only 25% (3 of 4 missed, even in-sample) — disclosed, not hidden.
- NEXT_TECHNIQUE: 4/12 correct (33.3%) — labeled `INSUFFICIENT_FOR_SUPERVISED_ML` by the project's own prior audit.
- Numbers to display: **91.7% (in-sample) / macro F1 0.676 / Critical recall 25% / 33.3% (4/12)**.
- Figure/table: confusion matrix `[[1,2,1],[0,51,2],[0,0,3]]`; do not chart NEXT_TECHNIQUE (n=12 too small for a meaningful chart — present as a plain fraction).
- **Say**: "We're showing you the in-sample numbers and the class-level breakdown, including the one that looks worst, because that's the honest picture at 60 campaigns."
- **Likely question**: "Why report ML numbers at all if they're not generalizable?"
- **Defensible answer**: "Because the pipeline mechanics — feature extraction, leakage prevention, label correctness — are real and tested, and the numbers demonstrate the pipeline runs correctly end-to-end; we're explicit that the accuracy figures aren't a generalization claim."

## Slide 7 — Limitations + What the Results Actually Mean
**Objective**: pre-empt the hardest questions by answering them yourself.
- No held-out ML evaluation exists anywhere in the repository.
- No ground-truth attribution dataset — attribution accuracy is `NOT MEASURED`, not poor.
- Dataset diversity is limited: 3 attackers, 2 victims, 4 pairs, 0 High-severity examples.
- One disclosed, unfixed live defect (maintenance-thread exception, non-blocking).
- Numbers to display: **3 attackers / 2 victims / 0 High-severity / NOT MEASURED (attribution)**.
- Figure/table: the "cannot claim" list from `quantitative_results.md`.
- **Say**: "These aren't gaps we're hiding — they're gaps our own internal audits found and formally documented before we were asked about them."
- **Likely question**: "Given all these limitations, what can you actually claim?"
- **Defensible answer**: "That the ingestion, resolution, and graph-integrity mechanisms are real, tested, and live-verified — and that we know exactly what would be needed to extend that same rigor to the ML and attribution components."

## Slide 8 — Contributions + Next Experimental Step
**Objective**: close on the defensible novelty claim and a concrete, minimal roadmap.
- Core contribution: provenance-aware MITRE resolution with a structurally-proven, live-verified explicit UNKNOWN state.
- Not claiming novelty for individual algorithms (graph DB, TF-IDF, XGBoost, GraphSAGE are all standard).
- Next step: deliberate diversity-targeted data expansion (not more volume) → enables a real held-out ML split and a first attribution benchmark.
- Numbers to display: none new — reference Slide 4's numbers as the closing evidence.
- Figure/table: none — closing slide.
- **Say**: "The contribution isn't a new algorithm — it's a correctly-implemented, verifiably-safe design decision applied to a real problem in our own pipeline."
- **Likely question**: "What's the single most important next experiment?"
- **Defensible answer**: "Expanding attacker/victim diversity in the real dataset, because that's the one step that unblocks both a real ML held-out evaluation and a first attribution benchmark at the same time."

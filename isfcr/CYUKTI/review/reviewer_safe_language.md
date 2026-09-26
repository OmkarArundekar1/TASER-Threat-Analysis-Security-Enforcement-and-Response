# CYUKTI — "Do Not Say This" Sheet

> Last verified: 2026-09-25. Counts reflect live Neo4j query and pytest run from this date.

Bad claim → Safe claim, for every major result in the review package. Each safe claim is followed by the planned next step that closes the gap (see `10_limitations_and_future_work.md` for the full roadmap).

**"CYUKTI achieves 91.7% accuracy"**
→ "XGBoost achieved 91.7% in-sample accuracy on 60 campaigns; held-out generalization has not been established."
*Planned:* the Phase 19 dataset expansion (6-8 attackers, 5+ victims, ≥1 real High example) is designed to reach the scale needed to support a proper held-out train/test evaluation.

**"CYUKTI has 10% MITRE coverage"**
→ "In the current 120-alert snapshot (2026-09-25, post rule-fix/reboot), 26 alerts carried native Wazuh ATT&CK mappings (21.7%); the remaining 94 (78.3%) were explicitly preserved as UNKNOWN, not discarded. An earlier 391-alert snapshot from 2026-08-31 measured 10.0% — both are traffic-composition-dependent, not fixed."
*Planned:* continue extending `MITRE_TO_STAGE` coverage (e.g. the disclosed `T1548.003` gap) as new native mappings are confirmed; the UNKNOWN rate itself is intended behavior, not a defect to close.

**"CYUKTI predicts the next attack technique with 33% accuracy"**
→ "An offline evaluation on 12 evaluable real transitions produced 4 correct predictions; the project formally classifies this as `INSUFFICIENT_FOR_SUPERVISED_ML`."
*Planned:* the same Phase 19 dataset expansion is intended to accumulate enough real technique transitions to move this out of `INSUFFICIENT_FOR_SUPERVISED_ML`.

**"Threat attribution works"**
→ "Threat attribution is implemented and exercised at the evidence-collector level, but quantitative attribution accuracy is not measured because no ground-truth attacker-identity dataset exists."
*Planned:* build a ground-truth attacker-identity benchmark dataset so a real attribution-accuracy metric can be computed.

**"RAG has high retrieval accuracy"**
→ "More than 500 real ATT&CK documents are indexed and the RAG pipeline is test-verified, but retrieval precision/recall has not been measured."
*Planned:* construct a labeled query set to measure retrieval precision/recall directly.

**"798/798 tests means CYUKTI is reliable"**
→ "689 backend (65 files) + 109 frontend (17 files) = 798 automated tests passed, as of 2026-09-25 (up from 150 backend-only on 2026-09-12); this demonstrates tested behavior for known cases, not the absence of undiscovered production failures — we found one such failure live, outside test coverage, during this project."
*Planned:* the maintenance-worker failure that test coverage missed is root-caused and tracked as Current Issue 1 in `10_limitations_and_future_work.md`, with a regression test to be added alongside the fix.

**"0 orphaned events means the graph can never become inconsistent"**
→ "The current graph contains 0 orphaned AttackEvents as of the last integrity check, and the reconstruction mechanism is proven idempotent by test — this is a current-state measurement, not a standing guarantee."
*Planned:* run the integrity check on a recurring schedule (rather than ad hoc) so any regression is caught promptly instead of relying on point-in-time snapshots.

**"CYUKTI has 858 ATT&CK techniques"** (implying all are observed/relevant)
→ "858 real Technique nodes are imported from the vendored MITRE ATT&CK STIX corpus (v19.1); only 15 distinct techniques have actually been observed in real campaign data."
*Planned:* the Phase 19 dataset expansion is expected to broaden the slice of the imported corpus actually observed in real campaigns.

**"The system detects 78% unknown threats"**
→ "78.3% of alerts in the current snapshot (2026-09-25) could not be defensibly attributed to a specific ATT&CK technique and were preserved as UNKNOWN evidence — this is an attribution-coverage measurement, not a detection or threat-relevance measurement."
*Not a gap to close*: preserving UNKNOWN rather than fabricating an attribution is the intended, correct behavior, per `10_limitations_and_future_work.md`.

**"The ML model correctly classifies severity"**
→ "The XGBoost model reaches 91.7% in-sample accuracy overall, but only 25% recall on the Critical class (3 of 4 missed) — reported alongside the aggregate number, not instead of it."
*Planned:* the Phase 19 dataset expansion explicitly targets adding real High/Critical-severity examples, which the current dataset lacks entirely.

**"CYUKTI's investigation engine reasons under uncertainty"**
→ "The investigation architecture implements distinct evidence-reliability, coverage, model-confidence, and model-uncertainty signals, and 25 tests verify it does not manufacture false confidence from absent evidence — no fresh live end-to-end investigation trace was generated in the most recent sessions."
*Planned:* schedule a fresh live end-to-end investigation trace against Neo4j in the next verification session.

**"MISP integration is working"**
→ "The MISP client initializes correctly and is confirmed non-blocking for ingestion; live publication success is not measured because the MISP service is not currently running in this environment."
*Planned:* reconfirm live publication once `MISP_API_KEY` is configured and the MISP service is running for a verification session.

**"The GNN model works on real attack graphs"**
→ "The GNN implementation (encoder, SAGEConv layer, training loop) is unit-tested and correct on synthetic graphs; no benchmark exists yet on real campaign graphs."
*Planned:* build a real-campaign GNN benchmark once the Phase 19 dataset expansion provides enough real campaign graphs to make one meaningful.

**"CYUKTI is production-ready"**
→ "CYUKTI is a research prototype: core ingestion, MITRE resolution, and graph-integrity mechanisms are live-validated, but no held-out ML evaluation, attribution benchmark, or performance (latency/throughput) measurement exists yet."
*Planned:* per-stage latency is now measured (see `BENCHMARKS.md`); the remaining gaps — held-out ML evaluation and an attribution benchmark — are tracked under the same Phase 19 dataset-expansion and benchmark work referenced above.

**"We fixed all known issues"**
→ "One known issue remains disclosed and unfixed: a maintenance-thread exception triggered by a specific UNKNOWN-first-event condition. It is non-blocking — ingestion and safety invariants are unaffected — but it has not been resolved."
*Planned:* the root cause is already identified (`create_campaign_context` bypasses `activate_campaign()` on the UNKNOWN-first-event path, leaving `last_seen=None`); the fix is to initialize `last_seen` there or guard `expire_active_campaigns()` against `None`, tracked as Current Issue 1 in `10_limitations_and_future_work.md`.

**"CYUKTI's architecture is novel"** (unqualified)
→ "The individual components (graph database, TF-IDF retrieval, gradient boosting) are standard; the defensible novelty claim is the system-level design — provenance-aware, structurally-enforced separation of detection from attribution, with a live-verified explicit UNKNOWN state."

---

## NEW PHASE RESULTS (2026-09-14): Evidence-Aware Adaptive Investigation

**"CYUKTI's investigator learned to avoid redundant evidence"**
→ "One real, code-verified evidence dependency is declared and discounted once satisfied — not a learned redundancy model, and not yet validated against live investigation outcomes."
*Planned:* validate the dependency-discounting behavior against live investigation outcomes once enough live investigation runs have been collected.

**"The adaptive investigation was tested end-to-end"**
→ "164/164 tests pass, including new tests proving state-dependent ranking and the hypothesis/fact separation by source inspection — live-Neo4j integration validation is `NOT MEASURED` this session because infrastructure was unavailable, and was not forced."
*Planned:* re-run the live-Neo4j integration validation once the infrastructure is available in a future session.

**"CYUKTI's investigator can never fabricate an attack technique"**
→ "Structurally proven for the investigation module specifically (it never references any Neo4j write function for `attack_id`/`Technique`/`NEXT_TECHNIQUE`), the same class of proof already established for the separate UNKNOWN-ingestion path — this is a code-level guarantee, not a runtime-tested one, for this specific module."
*Planned:* add a runtime/integration test against a live Neo4j instance to complement the existing structural proof for this module.

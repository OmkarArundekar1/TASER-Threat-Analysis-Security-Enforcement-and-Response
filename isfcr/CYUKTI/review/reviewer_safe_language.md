# CYUKTI — "Do Not Say This" Sheet

Bad claim → Safe claim, for every major result in the review package.

**"CYUKTI achieves 91.7% accuracy"**
→ "XGBoost achieved 91.7% in-sample accuracy on 60 campaigns; held-out generalization has not been established."

**"CYUKTI has 10% MITRE coverage"**
→ "In the representative 391-alert snapshot (2026-08-31), 39 alerts carried native Wazuh ATT&CK mappings; the remaining 352 were explicitly preserved as UNKNOWN, not discarded."

**"CYUKTI predicts the next attack technique with 33% accuracy"**
→ "An offline evaluation on 12 evaluable real transitions produced 4 correct predictions; the project formally classifies this as `INSUFFICIENT_FOR_SUPERVISED_ML`."

**"Threat attribution works"**
→ "Threat attribution is implemented and exercised at the evidence-collector level, but quantitative attribution accuracy is not measured because no ground-truth attacker-identity dataset exists."

**"RAG has high retrieval accuracy"**
→ "More than 500 real ATT&CK documents are indexed and the RAG pipeline is test-verified, but retrieval precision/recall has not been measured."

**"150/150 tests means CYUKTI is reliable"**
→ "150/150 automated tests passed; this demonstrates tested behavior for known cases, not the absence of undiscovered production failures — we found one such failure live, outside test coverage, during this project."

**"0 orphaned events means the graph can never become inconsistent"**
→ "The current graph contains 0 orphaned AttackEvents as of the last integrity check, and the reconstruction mechanism is proven idempotent by test — this is a current-state measurement, not a standing guarantee."

**"CYUKTI has 858 ATT&CK techniques"** (implying all are observed/relevant)
→ "858 real Technique nodes are imported from the vendored MITRE ATT&CK STIX corpus (v19.1); only 15 distinct techniques have actually been observed in real campaign data."

**"The system detects 90% unknown threats"**
→ "90% of alerts in the representative snapshot could not be defensibly attributed to a specific ATT&CK technique and were preserved as UNKNOWN evidence — this is an attribution-coverage measurement, not a detection or threat-relevance measurement."

**"The ML model correctly classifies severity"**
→ "The XGBoost model reaches 91.7% in-sample accuracy overall, but only 25% recall on the Critical class (3 of 4 missed) — reported alongside the aggregate number, not instead of it."

**"CYUKTI's investigation engine reasons under uncertainty"**
→ "The investigation architecture implements distinct evidence-reliability, coverage, model-confidence, and model-uncertainty signals, and 25 tests verify it does not manufacture false confidence from absent evidence — no fresh live end-to-end investigation trace was generated in the most recent sessions."

**"MISP integration is working"**
→ "The MISP client initializes correctly and is confirmed non-blocking for ingestion; live publication success is not measured because the MISP service is not currently running in this environment."

**"The GNN model works on real attack graphs"**
→ "The GNN implementation (encoder, SAGEConv layer, training loop) is unit-tested and correct on synthetic graphs; no benchmark exists yet on real campaign graphs."

**"CYUKTI is production-ready"**
→ "CYUKTI is a research prototype: core ingestion, MITRE resolution, and graph-integrity mechanisms are live-validated, but no held-out ML evaluation, attribution benchmark, or performance (latency/throughput) measurement exists yet."

**"We fixed all known issues"**
→ "One known issue remains disclosed and unfixed: a maintenance-thread exception triggered by a specific UNKNOWN-first-event condition. It is non-blocking — ingestion and safety invariants are unaffected — but it has not been resolved."

**"CYUKTI's architecture is novel"** (unqualified)
→ "The individual components (graph database, TF-IDF retrieval, gradient boosting) are standard; the defensible novelty claim is the system-level design — provenance-aware, structurally-enforced separation of detection from attribution, with a live-verified explicit UNKNOWN state."

---

## NEW PHASE RESULTS (2026-09-14): Evidence-Aware Adaptive Investigation

**"CYUKTI's investigator learned to avoid redundant evidence"**
→ "One real, code-verified evidence dependency is declared and discounted once satisfied — not a learned redundancy model, and not yet validated against live investigation outcomes."

**"The adaptive investigation was tested end-to-end"**
→ "164/164 tests pass, including new tests proving state-dependent ranking and the hypothesis/fact separation by source inspection — live-Neo4j integration validation is `NOT MEASURED` this session because infrastructure was unavailable, and was not forced."

**"CYUKTI's investigator can never fabricate an attack technique"**
→ "Structurally proven for the investigation module specifically (it never references any Neo4j write function for `attack_id`/`Technique`/`NEXT_TECHNIQUE`), the same class of proof already established for the separate UNKNOWN-ingestion path — this is a code-level guarantee, not a runtime-tested one, for this specific module."

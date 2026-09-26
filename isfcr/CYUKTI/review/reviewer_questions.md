# CYUKTI — Reviewer Cross-Examination (25 questions)

> Last verified: 2026-09-25. Counts reflect live Neo4j query and pytest run from this date.

## 1. Why is ~78% UNKNOWN acceptable?
**20-second answer**: Because it measures Wazuh's own native ATT&CK coverage gap, not CYUKTI's capability — and the alternative (fabricating attribution to lower that number) would corrupt every downstream learning component.
**If they push further**: "The right question isn't 'how do we raise this number' — it's 'does anything downstream ever get contaminated by it,' and we can show live evidence that it doesn't."
**Evidence you can show**: current 120-alert coverage table (26 native/21.7%, 94 UNKNOWN/78.3%, 2026-09-25, post rule-fix/reboot); an earlier 391-alert snapshot (10.0% native, 2026-08-31); 5 live UNKNOWN events with 0 fabricated `attack_id`.
**What NOT to claim**: That either number is a target or a good score — both are measured, expected consequences of the design choice, and vary with whatever traffic actually arrives.

## 2. Does UNKNOWN mean CYUKTI failed?
**20-second answer**: No — UNKNOWN means Wazuh's rule engine detected something, but CYUKTI could not defensibly name the ATT&CK technique. Detection and attribution are separate questions.
**If they push further**: Walk through the six-concept table (Detection/Attribution/Investigation/Confidence/Prediction/Response) from the research contribution notes.
**Evidence you can show**: the resolver's 4-tier precedence and the structural proof UNKNOWN never contaminates learning.
**What NOT to claim**: That UNKNOWN events are irrelevant — they're preserved specifically because they aren't.

## 3. Why only ~10-22% MITRE coverage?
**20-second answer**: That's the real, measured native-mapping rate of Wazuh's own out-of-the-box ruleset against the telemetry we observed (10.0% in an August snapshot, 21.7% in the current September one — it moves with whatever traffic mix actually arrives) — most of that telemetry (systemd failures, package-manager activity, disk warnings) isn't attacker behavior at all.
**If they push further**: Point out that Wazuh's own rule authors didn't tag rules like 40704/52002/2904 with ATT&CK either — this isn't a CYUKTI gap, it's inherited.
**Evidence you can show**: the rule-level breakdown table (which specific rules are native vs. unknown).
**What NOT to claim**: That this will necessarily improve — it depends on what telemetry arrives, not on CYUKTI's resolver logic.
*Planned:* continue extending `MITRE_TO_STAGE` coverage (e.g. the disclosed `T1548.003` gap) as new native mappings are confirmed; the coverage rate itself is expected to keep varying with traffic, not to hit a fixed target.

## 4. How was 91.7% XGBoost accuracy obtained?
**20-second answer**: By running `evaluate_model.py` against the trained model on the same 60-row dataset it was trained on — it's an in-sample metric.
**If they push further**: "We don't have enough real campaign diversity yet for a held-out split — our own Phase 17 audit concluded that before we ever computed this number."
**Evidence you can show**: confusion matrix, per-class precision/recall/F1.
**What NOT to claim**: Test accuracy, validation accuracy, or generalization performance — say "in-sample" every time.
*Planned:* the Phase 19 dataset expansion (6-8 attackers, 5+ victims, ≥1 real High example, ≥20 technique compositions) is designed to reach the scale needed to support a proper held-out train/test evaluation.

## 5. Why no train/test split?
**20-second answer**: Because 60 campaigns from only 3 attacker identities isn't enough to make a held-out split statistically meaningful — a random split would likely separate near-duplicate rows (11 exact duplicate feature vectors already exist) rather than test true generalization.
**If they push further**: cite the Phase 17 validity gate's specific findings (23/57 zero-variance features, attacker/severity confounding).
**Evidence you can show**: Phase 17 dataset statistics.
**What NOT to claim**: That a split wasn't done due to time constraints — it was a deliberate methodological decision based on measured data insufficiency.
*Planned:* introduce a held-out split once the Phase 19 dataset expansion provides enough campaigns to make one meaningful.

## 6. Why is Critical recall only 25%?
**20-second answer**: Only 4 Critical-severity campaigns exist in the entire dataset, and the model missed 3 of them — even in-sample. It's a direct symptom of severe class imbalance (4 Critical vs. 53 Low).
**If they push further**: "We're showing this number specifically because it's the least flattering one — hiding it would undermine the accuracy figure's credibility more than reporting it does."
**Evidence you can show**: full per-class table, confusion matrix.
**What NOT to claim**: That this reflects model quality in general — with 4 examples, no classifier could be expected to generalize on this class.
*Planned:* the Phase 19 dataset expansion explicitly targets adding real High/Critical-severity examples, which the current dataset lacks entirely.

## 7. Why is NEXT_TECHNIQUE only 33.3%?
**20-second answer**: Because only 12 real transitions were evaluable across the entire dataset, and the project's own audit (Phase 18) had already concluded this is `INSUFFICIENT_FOR_SUPERVISED_ML` before computing accuracy.
**If they push further**: explain that 7 of the 8 incorrect predictions were self-repeat transitions (attacker repeats the same technique), which the current model has no representation for by design.
**Evidence you can show**: the 4/12 breakdown and the self-repeat pattern.
**What NOT to claim**: That next-technique prediction works — it's presented as a diagnosed limitation, not a capability.
*Planned:* the same Phase 19 dataset expansion is intended to accumulate enough real technique transitions to revisit this verdict.

## 8. Why use ML with only 60 campaigns?
**20-second answer**: To validate that the full pipeline — feature extraction, leakage prevention, label generation — runs correctly end-to-end on real data, not to claim a production-ready classifier.
**If they push further**: point to the leakage fix (risk_score excluded from features after discovering 0.31 feature importance leakage) as evidence the pipeline itself is being taken seriously even at small scale.
**Evidence you can show**: `test_dataset_leakage.py` passing; the leakage discovery narrative.
**What NOT to claim**: That 60 is an adequate sample size for the claims being made about the model.
*Planned:* the Phase 19 dataset-expansion spec directly targets this gap with more attackers, victims, and campaigns.

## 9. Why only 3 attackers?
**20-second answer**: Because that's the real, accumulated lab data to date — CYUKTI doesn't synthesize attacker identities, and the project has a dedicated, not-yet-executed data-expansion plan (Phase 19) specifically to address this.
**If they push further**: describe the Phase 19 target (6-8 attacker identities, 5+ victims, ≥1 real High-severity example).
**Evidence you can show**: Phase 17's diversity analysis.
**What NOT to claim**: That 3 attackers is sufficient for any calibration claim — the project's own verdict says the opposite.
*Planned:* the Phase 19 dataset-expansion spec is the concrete plan to reach 6-8 attacker identities and 5+ victims.

## 10. How do you know the 44 repaired events were correctly reconstructed?
**20-second answer**: Reconstruction only uses data already present on the orphaned events themselves (attacker, victim, technique, timestamps) — it never guesses a value that isn't already recorded, and the process is proven idempotent by test.
**If they push further**: explain the refusal behavior — if orphaned events for one campaign_id have inconsistent attacker/victim pairs, the code raises rather than guessing which is correct.
**Evidence you can show**: `test_campaign_reconstruction.py` (10 tests), including the "refuses inconsistent attacker/victim" test.
**What NOT to claim**: That every possible edge case is covered — only that the implemented cases never guess.
*Planned:* add additional edge-case tests for campaign reconstruction as new inconsistent-data patterns are discovered in live traffic.

## 11. How do you know UNKNOWN does not contaminate the graph?
**20-second answer**: Two independent ways: a code-level control-flow proof that the UNKNOWN branch can never reach the contaminating function calls, and live Neo4j evidence that the `NEXT_TECHNIQUE` edge count stayed at exactly 3 across real UNKNOWN ingestion.
**If they push further**: walk through the exact line numbers where the UNKNOWN branch returns before reaching `chain_updater`/`predict_next`/MISP calls.
**Evidence you can show**: the control-flow proof plus the live before/after count.
**What NOT to claim**: Nothing to soften here — this is one of the best-evidenced claims in the project.

## 12. Why 858 ATT&CK techniques?
**20-second answer**: That's the real count of techniques in the vendored MITRE ATT&CK Enterprise STIX release (v19.1) we imported — not a number we chose.
**If they push further**: note that most of the 858 are reference metadata, not observed in real campaign data (only 15 distinct techniques have actually been observed).
**Evidence you can show**: live Technique node count; the STIX file itself.
**What NOT to claim**: That all 858 techniques are relevant to observed attacker behavior — they're the full reference corpus.
*Planned:* the Phase 19 dataset expansion is expected to broaden the slice of the imported corpus actually observed in real campaigns.

## 13. Did you create the ATT&CK knowledge base yourself?
**20-second answer**: No — we vendor the official MITRE STIX release as-is and import it directly into Neo4j; every technique ID CYUKTI ever proposes is validated against this real, unmodified corpus, including revoked/deprecated status.
**If they push further**: explain why (no runtime internet dependency, reproducibility, avoiding a second, drifting "shadow" database).
**Evidence you can show**: the STIX file path/version; the validation logic in `mitre_resolver.py`.
**What NOT to claim**: Any originality in the ATT&CK data itself — the contribution is in how it's used (validation gate), not in creating it.

## 14. How is RAG evaluated?
**20-second answer**: Only at the scale level — we've confirmed the retriever indexes over 500 real ATT&CK documents via a passing test. Retrieval precision/recall has not been measured; there's no labeled query set to measure it against.
**If they push further**: "That would require constructing a labeled query benchmark, which is a real future-work item, not something we're claiming today."
**Evidence you can show**: `test_rag.py`'s index-size assertion.
**What NOT to claim**: Retrieval quality, relevance ranking accuracy, or any comparison to a baseline retriever.
*Planned:* construct a labeled query set to measure retrieval precision/recall directly.

## 15. How is attribution evaluated?
**20-second answer**: It isn't, quantitatively — no ground-truth attacker-identity dataset exists, so accuracy is honestly reported as `NOT MEASURED`.
**If they push further**: describe what the attribution engine does (historical campaign similarity by attacker_ip) and that it's exercised at the evidence-collector unit-test level.
**Evidence you can show**: the 2 evidence-collector tests for attribution.
**What NOT to claim**: Any accuracy, precision, recall, or "works well" — there's no data to support such a claim.
*Planned:* build a ground-truth attacker-identity benchmark dataset so a real attribution-accuracy metric can be computed.

## 16. Why is attribution accuracy unavailable?
**20-second answer**: Because scoring attribution requires knowing the *true* attacker identity independent of what the system infers, and the current lab environment doesn't have that independent ground truth recorded.
**If they push further**: propose the minimal fix — a small set of campaigns with manually-confirmed attacker identity, which would let the existing engine be scored for the first time.
**Evidence you can show**: nothing to show here except the honest gap — that's the point.
**What NOT to claim**: That this is a fundamental architectural limitation — it's a data-collection gap, addressable without redesigning anything.
*Planned:* build the same ground-truth attacker-identity benchmark dataset proposed above so the existing attribution engine can be scored for the first time.

## 17. What does evidence-aware investigation actually contribute?
**20-second answer**: It separates evidence reliability, evidence coverage, model confidence, and model uncertainty as distinct signals, specifically so the system can't manufacture high confidence just because little evidence has been gathered — this was a real bug we found and fixed (investigations previously reached confidence=1.0 after a single evidence item).
**If they push further**: cite the specific test verifying uncertain model probabilities don't falsely report near-certainty.
**Evidence you can show**: 25 passing tests in `test_investigation.py`.
**What NOT to claim**: A live, fresh end-to-end investigation trace this session — the evidence here is test-level, not a fresh live demonstration.
*Planned:* schedule a fresh live end-to-end investigation trace against Neo4j in the next verification session.

## 18. Why isn't MISP quantitatively evaluated?
**20-second answer**: Because the MISP containers are currently absent from the environment — there's nothing to publish to right now, so live publication success can't be measured.
**If they push further**: clarify that MISP is confirmed non-blocking for both listener startup and UNKNOWN ingestion (verified by reading the client's constructor and the code's control flow).
**Evidence you can show**: the listener's own log line confirming client initialization with no network call.
**What NOT to claim**: That MISP publication works — only that it's correctly gated and doesn't block anything else.
*Planned:* reconfirm once `MISP_API_KEY` is configured and the MISP service is running for a verification session.

## 19. Why are GNN results only synthetic?
**20-second answer**: Because a real-campaign GNN benchmark hasn't been built yet — the 11 passing tests verify the encoder, layer, and training mechanics are implemented correctly, on synthetic graphs.
**If they push further**: "The mechanism is real and tested; applying it to real campaign graphs at meaningful scale is future work, gated on the same data-diversity expansion as the other ML components."
**Evidence you can show**: `test_gnn.py`'s 11 tests.
**What NOT to claim**: Any real-campaign GNN performance number — none exists.
*Planned:* build a real-campaign benchmark once the Phase 19 dataset expansion provides enough real campaign graphs to make one meaningful.

## 20. What does 798/798 tests actually prove?
**20-second answer**: That every currently-known behavior the project has tests for behaves as expected — it does not prove the absence of undiscovered bugs, including production issues found live that no prior test anticipated.
**If they push further**: cite the maintenance-worker exception found live, which had zero test coverage before it was discovered in production, as a concrete counter-example to "100% pass rate means bug-free."
**Evidence you can show**: the disclosed, unfixed maintenance-worker issue.
**What NOT to claim**: System reliability or bug-freedom from the pass rate alone.
*Planned:* the maintenance-worker failure is already root-caused (Current Issue 1 in `10_limitations_and_future_work.md`); the fix plus a regression test for the UNKNOWN-first-event path is queued once `campaign_manager.py` is back in scope.

## 21. Can the system operate in real time?
**20-second answer**: The listener does process live Wazuh telemetry as it arrives (byte-offset polling, ~0.5s interval) rather than in batch, and we've verified it recovers correctly from both restarts and infrastructure outages.
**If they push further**: be precise that "real-time" here means near-real-time polling, not sub-second guarantees, and that no latency has been formally measured.
**Evidence you can show**: the live offset-recovery log line; the outage-recovery evidence.
**What NOT to claim**: Any specific latency figure or real-time performance guarantee.
*Planned:* extend the benchmark harness (see `BENCHMARKS.md`) further back into the alert-processing path so per-alert latency, not just the API round trip, is measured.

## 22. What is the latency?
**20-second answer**: Not measured — no timing instrumentation exists in the alert-processing path.
**If they push further**: state this plainly; do not estimate a number.
**Evidence you can show**: nothing — this is a genuine gap.
**What NOT to claim**: Any latency figure, even qualitatively ("fast," "low-latency") without data.
*Planned:* the same benchmark-harness extension referenced above is intended to close this gap for the alert-processing path specifically.

## 23. What is the throughput?
**20-second answer**: Not measured — no throughput instrumentation exists.
**If they push further**: same as latency — state the gap plainly.
**Evidence you can show**: nothing.
**What NOT to claim**: Any throughput figure.
*Planned:* extend the benchmark harness to a sustained-load ingestion test once a suitable traffic generator is in place.

## 24. What is novel compared with existing SOC platforms?
**20-second answer**: Not the individual components (graph databases, TF-IDF retrieval, gradient boosting are all standard) — the defensible claim is the system-level design: separating detection from attribution with mandatory provenance and a structurally-enforced, live-verified UNKNOWN state that can't contaminate learning.
**If they push further**: be explicit that no external product comparison exists in the repository, so any comparative novelty claim beyond "this design decision, correctly implemented" is not something we can currently defend with evidence.
**Evidence you can show**: the dual (structural + live) proof of UNKNOWN safety.
**What NOT to claim**: Novelty relative to any named commercial or open-source SOC platform — no such comparison has been done.

## 25. What experiment would most strongly validate CYUKTI next?
**20-second answer**: Deliberate, diversity-targeted real-data expansion (more attacker identities, more victims, at least one real High-severity example) — it's the single step that unblocks both a meaningful held-out ML evaluation and a first attribution benchmark.
**If they push further**: explain why volume alone wouldn't help (the current bottleneck is diversity/confounding, not row count, per Phase 17).
**Evidence you can show**: the Phase 19 expansion specification.
**What NOT to claim**: That this experiment is already underway or scheduled — it's the recommended next step, not a completed or in-progress one.

---

## NEW PHASE RESULTS (2026-09-14): Evidence-Aware Adaptive Investigation

### 26. How does the investigator decide what evidence to gather next?
**20-second answer**: Each candidate action gets an explicit, interpretable score (expected gain, reliability, novelty, uncertainty reduction, cost, latency, and now a redundancy penalty) and the highest-scoring one is chosen — proven by test to actually re-rank as the investigation state changes, not a fixed sequence.
**If they push further**: walk through `test_ranking_changes_as_investigation_state_changes`.
**Evidence you can show**: the full per-candidate `action_scores` now recorded in every trace step.
**What NOT to claim**: That the weights are learned or optimal — they're documented, hand-set constants, stated as such in code.
*Planned:* revisit the hand-set weights once enough live investigation outcomes are collected to tune or validate them empirically.

### 27. How do you prevent double-counting redundant evidence?
**20-second answer**: One real, code-verified dependency is modeled explicitly (attribution's verdict is computed from the same historical-campaign data a separate campaign-history lookup would return) and discounted once satisfied — bounded by the number of declared dependencies, not by how many actions have been taken overall.
**If they push further**: explain why only one dependency is declared — it's the only one independently verified by reading the engine source; others weren't asserted without that verification.
**Evidence you can show**: `test_redundant_evidence_does_not_receive_unlimited_weight`.
**What NOT to claim**: A general probabilistic dependency model — this is a declared, upgradable graph, not full Bayesian evidence fusion.
*Planned:* declare additional dependencies as they are independently verified in the engine source, and validate the discounting behavior against live investigation outcomes once enough runs have been collected.

### 28. Can a hypothesis the investigator forms ever become a "fact" in the graph?
**20-second answer**: No — structurally proven by inspecting the actual source of the investigation module: it never references any Neo4j write function capable of setting `attack_id`, creating a `Technique`/`MATCHES` relationship, or a `NEXT_TECHNIQUE` edge.
**If they push further**: note this is the same class of proof already used to certify the separate UNKNOWN-ingestion safety guarantee, applied to this second module.
**Evidence you can show**: the three structural tests in `test_investigation_evidence_aware.py`.
**What NOT to claim**: That this was tested at runtime against a live write attempt — it's a static source-inspection proof, labeled as such.
*Planned:* add a runtime/integration test against a live Neo4j instance to complement the existing structural proof for this module.

### 29. Was this new capability validated against live data?
**20-second answer**: No — Neo4j was unavailable for the entire implementation session, and it was not restarted to force a result.
**If they push further**: point to `scripts/run_real_investigations.py`, unmodified and ready to run the moment infrastructure is available.
**Evidence you can show**: nothing live — say so directly.
**What NOT to claim**: Any live investigation trace from this work — only test-verified evidence exists for it currently.
*Planned:* re-run the live-Neo4j integration validation once the infrastructure is available in a future session.

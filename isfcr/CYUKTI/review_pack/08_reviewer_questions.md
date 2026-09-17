# CYUKTI — 20 Likely Reviewer Questions and Defensible Answers

**1. Why Neo4j?**
Campaigns, attack events, techniques, attackers, and hosts are naturally graph-structured (an attacker launches a campaign that targets a host and has events matching techniques that transition to other techniques). Cypher's pattern matching maps directly onto attack-chain and campaign-correlation queries that would require repeated joins in a relational model. We use it as the single source of truth for 65 Campaign nodes, 124 AttackEvents, and 858 real ATT&CK Technique nodes.

**2. Why Wazuh?**
It's a real, open-source, widely-deployed HIDS/SIEM with native MITRE ATT&CK tagging on a subset of its ruleset, giving us genuine (not synthetic) native attribution to validate against — and a real gap (most rules untagged) that motivated Phase 20's design.

**3. Why MITRE ATT&CK?**
It's the de facto standard adversary-behavior taxonomy, with a maintained, versioned, machine-readable STIX corpus (we vendor v19.1, 25,843 objects) — giving us a real knowledge base to validate proposed technique IDs against (existence, revoked/deprecated status) rather than trusting free-text labels.

**4. How is ground truth generated?**
For severity: a deterministic function of the normalized cumulative TPS score (`risk_scoring.py`), not a label anyone assigns by hand. For next-technique: the true chronological order of real `AttackEvent.first_seen` timestamps (Phase 18 fix) — not the model's own prediction, which was the bug we found and corrected. We do not hand-label; we derive from real Neo4j event data.

**5. How do you prevent data leakage?**
`risk_score` is explicitly excluded from `FEATURE_COLUMNS` (`LEAKAGE_COLUMNS`) because severity is a deterministic function of it — we found this leak (0.31 feature importance) via real-data inspection and fixed it. `test_dataset_leakage.py` regression-tests this.

**6. Why XGBoost / why this ML model?**
Gradient-boosted trees handle the tabular, mixed-scale, partially-sparse 57-feature vector well without extensive tuning, and give interpretable feature importances — useful for catching leakage (see Q5). We are explicit that its current 0.917 in-sample accuracy is not a generalization claim; no held-out split exists yet.

**7. How is attribution validated?**
Honestly: it isn't, quantitatively. `threat_attribution_engine.py` is implemented and exercised at the evidence-collector unit-test level, but no ground-truth attacker-identity dataset exists to compute an accuracy metric against. We report this as `NOT MEASURED`, not as a hidden gap.

**8. How do you handle missing MITRE mappings?**
That's Phase 20's entire subject: a 4-tier resolver (native Wazuh mapping, reviewed rule registry, deterministic inference, UNKNOWN), each carrying explicit provenance/confidence/reason. Missing mapping is not an ingestion blocker — it results in a preserved `UNKNOWN` AttackEvent instead.

**9. Why not simply discard UNKNOWN alerts?**
That was the pre-Phase-20 behavior, and it discarded ~90-99% of real telemetry — including legitimate operational context that later investigation might need. Discarding is a decision that permanently destroys evidence; preserving it as UNKNOWN costs nothing (tps=0, no graph contamination) and can always be filtered out later if truly irrelevant.

**10. How is UNKNOWN different from a false positive?**
A false positive is a detection judgment ("this rule fired but the behavior wasn't malicious"). UNKNOWN is an attribution judgment ("this rule fired, and we cannot defensibly say which ATT&CK technique it represents"). CYUKTI doesn't currently make false-positive judgments at all — Wazuh's own rule engine decides what fires; CYUKTI decides only whether it can attribute what fired.

**11. How do you prevent UNKNOWN from corrupting the attack chain?**
Structurally: `create_unattributed_attack_event()` never creates a `Technique` node or `MATCHES` relationship, and the UNKNOWN code branch in `realtime_socgraph.py` never calls `append_technique()`, `chain_updater.update_attack_chain()`, or `prediction_engine.predict_next()` — verified both by a line-number control-flow proof (the UNKNOWN branch returns before those call sites) and live Neo4j evidence (NEXT_TECHNIQUE edge count unchanged at 3 across 5 new UNKNOWN events).

**12. How is multi-technique Wazuh mapping handled?**
All native technique IDs Wazuh supplies are preserved in `AttackEvent.mitre_technique_ids` (a list); the first is used as the primary `attack_id` to preserve the existing one-event-per-alert schema used throughout the rest of the codebase, rather than fabricating multiple AttackEvents from one alert. Tested (`test_multiple_native_mitre_ids_are_preserved`).

**13. How is confidence represented?**
As ordinal categories (`CONFIRMED`, `REVIEWED`, `CANDIDATE`, `NONE`) for MITRE resolution — deliberately not a fabricated probability, since nothing in the deterministic resolver produces a real probability distribution. Separately, the investigation architecture computes `investigation_confidence` from real evidence_reliability/coverage or model probability/uncertainty, gated so it can't be manufactured from absent evidence.

**14. What makes your evidence reliable?**
Every `Evidence` object carries source, confidence, relevance, and provenance explicitly (`evidence/schema.py`), and the store detects conflicting evidence about the same entity rather than silently averaging it away. Reliability isn't asserted — it's a first-class, inspectable field.

**15. How does RAG differ from ordinary retrieval?**
It's not fundamentally different — it's TF-IDF over the real, vendored ATT&CK STIX corpus, used to surface relevant technique documents as an evidence source. We don't claim a novel retrieval algorithm; the contribution is that it's grounded in real ATT&CK data (858 techniques) rather than a synthetic or toy corpus.

**16. How is MISP used?**
As a CTI-sharing sink for resolved (attributed) incidents only — `cti_publisher.py`/`misp_sync.py` publish after the Neo4j write, with failures caught and logged non-fatally. UNKNOWN events are never published to MISP (nothing to attribute). Live publish success has not been reconfirmed this session (`MISP_API_KEY` was empty as of a prior audit).

**17. What happens when Neo4j is unavailable?**
Not formally tested this session. The driver is constructed eagerly at import time; a connection failure would surface as an exception on first query, not gracefully degraded. This is a real, disclosed gap — we have not built or tested a Neo4j-unavailable fallback path.

**18. What are the current system limitations?**
See `10_limitations_and_future_work.md` in full; top items: dataset not ready for calibration (Phase 17), NEXT_TECHNIQUE insufficient for supervised ML (Phase 18), a known non-blocking maintenance-thread exception, and the unresolved `100500`/`100501` duplicate rule question.

**19. What is actually novel?**
Not the individual techniques (graph DBs, TF-IDF, XGBoost, GraphSAGE are all standard). The defensible claim is the *system-level design decision*, correctly and verifiably implemented: separating telemetry ingestion from ATT&CK attribution with mandatory provenance, an explicit UNKNOWN outcome, and structurally-proven non-contamination of downstream learning — applied specifically to the real problem of Wazuh's incomplete native MITRE coverage.

**20. What would you improve next?**
Deliberate, diversity-focused dataset expansion (not just more data) per the Phase 19 spec; a held-out ML evaluation split; fixing the two disclosed known issues; and only then reconsidering severity calibration and next-technique supervised learning, which the system itself has determined are not yet supportable.

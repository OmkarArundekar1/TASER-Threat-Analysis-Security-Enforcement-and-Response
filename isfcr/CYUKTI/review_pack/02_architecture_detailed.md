# CYUKTI — Detailed Investigation / Data-Flow Architecture

Companion to `cyukti_detailed_architecture.mmd`. Plain-text version below for when Mermaid isn't available.

## Full pipeline, annotated

```
1. Raw Wazuh Alert (JSON: rule, agent, data, timestamp)
        |
2. extract_iocs(alert)  [realtime_socgraph.py]
   attacker_ip/victim_ip from data.src_ip|srcip / data.dest_ip|dstip,
   falls back to agent.name/agent.id for host-based (non-network) alerts
   e.g. systemd/Apparmor/PAM events -- this is why self-directed
   "attacker == victim" campaigns exist in the real dataset.
        |
3. dedup_engine.is_duplicate()  [dedup_engine.py]
   fingerprint = SHA256(attacker|victim|technique_id|rule_id|agent_id)
   checked against in-memory cache + Neo4j find_recent_duplicate()
        |
4. resolve_mitre(alert)  [mitre_resolver.py] -- PHASE 20
   Tier 1 NATIVE_WAZUH: rule.mitre.id, ALL ids preserved (not just [0])
   Tier 2 REVIEWED_RULE_MAPPING: mitre_rule_registry.py, keyed by rule ID
          (currently EMPTY -- no entries added)
   Tier 3 DETERMINISTIC_INFERENCE: explicit structural rules only,
          no fuzzy/semantic matching (currently ZERO live rules)
   Tier 4 UNKNOWN / AMBIGUOUS: technique_ids=(), confidence=NONE
   Tiers 2/3 results are validated against existing Neo4j Technique
   nodes (MATCH (t:Technique {attack_id}) -> reject if missing,
   revoked, or deprecated). Tier 1 (native) is NEVER vetoed.
        |
   +----+----------------------------------+
   |  RESOLVED                             |  UNKNOWN/AMBIGUOUS
   v                                       v
5a. create_attack_event()               5b. create_unattributed_attack_event()
    [neo4j_client.py]                       [neo4j_client.py]
    - AttackEvent.attack_id = primary       - NO attack_id property
      native/reviewed/inferred technique    - NO Technique node/MATCHES rel
    - mitre_technique_ids = ALL native ids  - tps = 0 (hardcoded, not
    - MERGE Technique, MATCHES rel            parameterized -- cannot
    - tps computed via MITRE_TO_STAGE ->      be overridden)
      TPS_MAP                               - mitre_status = "UNKNOWN"
    - mitre_status = "RESOLVED"
        |                                       |
6a. campaign_manager.get_or_create_campaign()  6b. campaign_manager.
    -> resolve_campaign() -> activate_campaign()   resolve_campaign_context()
    -> append_technique()                          (include_inactive=False
    -> Campaign.last_technique UPDATED              -- Phase 20E fix)
                                                     OR create_campaign_context
                                                     (current_technique=None)
                                                     if no active campaign
                                                     -> Campaign.last_technique
                                                        NOT touched
        |                                       |
        +-------------------+-------------------+
                             v
7. Neo4j: Campaign -[:HAS_EVENT]-> AttackEvent
   Attacker -[:LAUNCHED]-> Campaign -[:TARGETS]-> Host
        |
   (RESOLVED path only, from here on)
        |
8. chain_updater.update_attack_chain()
   Technique -[:NEXT_TECHNIQUE {count, confidence}]-> Technique
   self-transitions (technique repeats) deliberately excluded
        |
9. prediction_engine.predict_next()
   top-1 frequency lookup from NEXT_TECHNIQUE, gated by
   MIN_TRANSITION_OBSERVATIONS (3) and MIN_PREDICTION_CONFIDENCE (30%)
        |
10. feature_orchestrator.extract_features()
    graph + mitre + runtime + threat-intel + detection sub-features
    -> CampaignDatasetRecord (57 declared ML features)
        |
11. campaign_correlation_engine.correlate() -> Operation match/create
        |
12. threat_attribution_engine.attribute()
    historical campaign similarity, ranked by attacker_ip match
        |
13. evidence/orchestrator.py
    collectors: mitre_collector, graph_collector, detection_collector,
    cti_collector, campaign_history_collector, attribution_collector
    each returns Evidence(source, confidence, relevance, provenance)
        |
14. rag/mitre_retriever.py
    TF-IDF index over the real vendored ATT&CK STIX corpus
    (25,843 objects, Enterprise v19.1) -- used as an evidence source
        |
15. investigation/confidence.py
    evidence_reliability   = weighted avg of collected evidence confidence
    evidence_coverage      = fraction of 6 evidence-source categories used
    model_confidence       = XGBoost top-class probability (when model used)
    model_uncertainty      = predictive entropy of model_probabilities
    investigation_confidence =
        with model:    model_confidence * (1 - model_uncertainty), gated
                        by coverage >= MIN_COVERAGE_FOR_MODEL_TRUST (0.15)
        without model: evidence_reliability * evidence_coverage
        |
16. investigation/next_best_evidence.py + stopping.py
    selects the unexplored action with highest expected value
    (uncertainty-reduction term only for the XGBOOST_PREDICTION action)
    stops when investigation_confidence >= threshold AND
    model_uncertainty <= max_uncertainty AND conflict_count == 0,
    or at max depth
        |
17a. cti_publisher.py / misp_sync.py .publish_campaign()
     RESOLVED path only; runs AFTER Neo4j writes; failure is caught
     and logged, never fatal to ingestion; SKIPPED entirely for UNKNOWN
        |
18. dashboard_api.py (Flask) serves the resulting state to
    frontend/ (React/TypeScript/Vite dashboard) for the analyst.
```

## Notes on what is exercised in production vs. tested-only

- Steps 1-9 (ingestion through prediction) are confirmed exercised against **live** Wazuh/Neo4j traffic this session (see `04_results_and_metrics.md` and `05_phase20_results.md`).
- Steps 10-16 (features -> investigation loop) are confirmed implemented and unit/integration tested (25 tests in `test_investigation.py`, 13 in `test_evidence.py`) but were **not** re-exercised against a fresh live campaign during this specific audit session — evidence for their correctness is test-based, not a fresh live trace.
- Step 17 (MISP) has a real client (`cti_publisher.CTIPublisher`) that initializes successfully against the local MISP instance (confirmed via listener startup logs: `Initialized CTIPublisher -> https://localhost:8443`), but actual event publication was last confirmed configured with an **empty `MISP_API_KEY`** as of the Phase 19 infrastructure audit — not re-verified this session. Treat MISP publication as code-path-present, not confirmed-functional, unless re-checked.

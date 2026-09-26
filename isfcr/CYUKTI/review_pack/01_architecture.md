# CYUKTI — System Architecture

All statements below are grounded in direct repository inspection and live system checks performed 2026-08-31. Where a claim could not be independently verified this session, it is marked as such.

## 1. High-level architecture (plain text)

```
ATTACK / LAB ACTIVITY (Kali attacker VM, Ubuntu victim VM, VirtualBox host-only net)
        |
        v
Wazuh Agent (Ubuntu, agent ID 001) --alerts--> Wazuh Manager (WSL2, native service)
        |
        v
alerts.json  (Wazuh rule engine output; byte-offset polled, not tailed)
        |
        v
CYUKTI Listener  [backend/listener/wazuh_listener.py]
   - dedup (SHA-256 fingerprint + Neo4j lookup)
   - queue worker thread, maintenance thread, duplicate-flush thread
        |
        v
realtime_socgraph.process_alert()
        |
        v
mitre_resolver.resolve_mitre()  -- Phase 20
   NATIVE_WAZUH -> REVIEWED_RULE_MAPPING -> DETERMINISTIC_INFERENCE -> UNKNOWN/AMBIGUOUS
        |
   +----+----------------------------+
   |                                 |
   v                                 v
RESOLVED                          UNKNOWN
create_attack_event()             create_unattributed_attack_event()
  - Technique node/MATCHES rel      - no attack_id, no Technique rel
  - TPS contribution                - tps = 0
        |                                 |
   get_or_create_campaign()      resolve_campaign_context() (active only)
   append_technique()            or create_campaign_context(technique=None)
        |                                 |
        +----------------+----------------+
                          v
                  Neo4j Graph
     Campaign -[:HAS_EVENT]-> AttackEvent -[:MATCHES]-> Technique
     Attacker -[:LAUNCHED]-> Campaign -[:TARGETS]-> Host
        |
   +----+------+------------+-------------+
   |           |            |             |
   v           v            v             v
chain_updater  prediction  correlation  ML feature
(NEXT_TECHNIQUE) engine    (Operation)   extraction
   |           |            |             |
   +-----------+------------+-------------+
                     |
                     v
        evidence/orchestrator.py collectors
   (mitre, graph, detection, cti, campaign_history, attribution)
                     |
                     v
        investigation/ loop + confidence.py
   (evidence_reliability, evidence_coverage, model_confidence,
    model_uncertainty, investigation_confidence)
                     |
              +------+------+
              |             |
              v             v
     misp_sync.py       dashboard_api.py (Flask)
   (RESOLVED path only)         |
                                 v
                        React dashboard (frontend/)
                                 |
                                 v
                          Analyst / Reviewer
```

## 2. Component table

| Component | Input | Output | Storage | Algorithm/Tech | Purpose |
|---|---|---|---|---|---|
| Wazuh Listener | `alerts.json` lines | queued alert dicts | `listener/offset.dat` | byte-offset polling, threaded queue | Ingest raw Wazuh telemetry without loss |
| MITRE Resolver | raw alert dict | `MitreResolution(technique_ids, provenance, confidence, reason, resolver_version)` | none (pure function; validates against Neo4j `Technique` nodes) | rule-based precedence, no ML/LLM | Separate telemetry ingestion from ATT&CK attribution |
| Neo4j Graph | AttackEvents, Campaigns | graph nodes/relationships | Neo4j (bolt://localhost:7687) | property graph | Central knowledge store |
| Chain Updater | consecutive real techniques per campaign | `NEXT_TECHNIQUE` edges (count, confidence) | Neo4j | frequency counting, self-transitions excluded | Learn technique transitions |
| Prediction Engine | current technique | predicted next technique + confidence | Neo4j `LIKELY_NEXT` rel | top-1 frequency lookup, thresholded | Live "what's likely next" |
| Campaign Correlation | campaign features | matched/created Operation | Neo4j `Operation` nodes | feature-similarity scoring | Group related campaigns |
| Threat Attribution | campaign context | ranked actor candidates | Neo4j | historical similarity by attacker_ip | Attribute campaign to a known actor identity |
| ML models | 57-feature vector | severity class / autoencoder score | `ml/models/*.pth,*.json,*.pkl` | XGBoost, autoencoder (SSL), hand-built GraphSAGE (GNN) | Severity classification, anomaly scoring |
| RAG retriever | text query | ranked ATT&CK technique documents | in-memory TF-IDF index over vendored STIX | TF-IDF | Real-time technique lookup for evidence |
| Investigation loop | Campaign context + model | `ConfidenceEstimate`, action trace | in-memory (per request) | evidence-value scoring, entropy-based uncertainty | Adaptive evidence gathering with honest stopping |
| MISP publisher | resolved incident | MISP event (if API key configured) | external MISP instance | REST API | CTI sharing |
| Dashboard API | HTTP requests | JSON | Flask, reads Neo4j | REST | Serve UI |
| React dashboard | API responses | rendered UI | browser | React/TypeScript/Vite | Analyst interface |

## 3. What is NOT in this architecture (verified absent or out of scope)

- No LLM/semantic-similarity component in the MITRE resolver (explicitly excluded by design).
- No live internet dependency for ATT&CK metadata (vendored STIX snapshot, `backend/mitredata/attack-stix-data/enterprise-attack/enterprise-attack.json`, ATT&CK Enterprise v19.1, 25,843 objects).
- `DETERMINISTIC_INFERENCE` tier exists as infrastructure with **zero live rules** — not currently contributing to resolution.
  *Planned:* populate this tier with rules as new, confirmed deterministic-mapping gaps (e.g. the disclosed `T1548.003` gap) are batched into a taxonomy update.
- `REVIEWED_RULE_MAPPING` registry (`mitre_rule_registry.py`) is **empty** by design.
  *Planned:* add entries as human-reviewed rule-to-technique mappings are confirmed, rather than pre-populating unverified ones.

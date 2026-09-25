# CYUKTI Reproducibility

Real, introspected values from this environment (2026-09-25), not assumed/typical versions. No credentials included — see `CONFIGURATION_SNAPSHOT.md`.

## Software versions

| Component | Version |
|---|---|
| Python | 3.12.3 |
| Node.js | v18.20.8 |
| Neo4j | 5-community (Docker image `neo4j:5-community`) |
| Wazuh manager / indexer / dashboard | 4.7.5-1 (`dpkg -l \| grep wazuh`) |
| Flask | 3.1.3 |
| flask-cors | 5.0.1 |
| Flask-SocketIO | 5.5.1 |
| neo4j (Python driver) | 6.2.0 |
| scikit-learn | 1.3.2 |
| torch | 2.10.0 |
| torchaudio | 2.10.0 |
| torchvision | 0.25.0 |
| xgboost | 3.4.1 |
| jsonschema | 4.26.0 |

Live, current values are also served at runtime via `GET /api/system/config-snapshot` (`config_snapshot.get_software_versions()`), so this table can be cross-checked against a running instance rather than trusted as a point-in-time snapshot alone.

## Model versions

- GNN autoencoder artifact: `gnn_autoencoder.pt@1790069556-13923b` (embedding_dim=8, 2-layer SAGEConv) — see `GNN_PRODUCTION_INTEGRATION.md` for training provenance.
- XGBoost severity model: `ml/models/xgb_severity.json` (trained on real resolved-campaign records; see `ml/train_xgboost.py`).

## ATT&CK version

Vendored `backend/mitredata/attack-stix-data/enterprise-attack/enterprise-attack.json`: `x_mitre_version: "19.1"`, `x_mitre_attack_spec_version: "3.3.0"`. **Known caveat**: this phase's engineering audit found the vendored file's own `kill_chain_phases` data contains non-standard tactic labels for ~31% of techniques (see `MITRE_MAPPING.md`) — a data-quality issue in the vendored bundle, not in the import code. Re-fetching from `mitre-attack/attack-stix-data` and diffing is the recommended next step before treating tactic labels from `enrich_technique_metadata()` as fully authoritative.

## Configuration relevant to reproducing an experiment

See `CONFIGURATION_SNAPSHOT.md` / `GET /api/system/config-snapshot` — reports `GNN_ENABLED`, `CAMPAIGN_TIMEOUT`, `DEDUP_WINDOW`, thresholds, and real software versions, with credentials excluded by an explicit allowlist.

## Random seeds

- GNN training used a fixed seed (`training_seed` field in `GNNModelMetadata`, exposed via `/api/gnn/status`) — see `GNN_PRODUCTION_INTEGRATION.md` for the exact value and training-run details.
- Nothing in the live inference/decision path (MITRE resolution, threat qualification, campaign selection, ResponsePlan generation) uses randomness at all — see `DETERMINISM.md`; there is no seed to record for those because there is no stochastic behavior to seed.

## Dataset versions

No fixed "dataset" in the traditional ML sense drives live decisions — CYUKTI's live behavior is a function of whatever is actually in the running Neo4j instance at query time (100+ real campaigns as of this phase, growing). For reproducing a *specific* experiment, the relevant "dataset version" is the Neo4j database's own state at that timestamp — not separately versioned today. A `neo4j-admin database dump` snapshot would be the correct mechanism if point-in-time reproducibility of the full graph becomes a requirement; not implemented this phase (no clear immediate need beyond re-running the same queries against the live, growing instance).

## Repeating the procedure

1. Start Neo4j: `docker start neo4j-soc` (or provision a fresh `neo4j:5-community` container and re-run the MITRE import — see `mitre_import/`).
2. Start the backend: `python dashboard_api.py` (from `backend/`).
3. Optionally start the listener (`python listener/wazuh_listener.py`) if a real Wazuh manager is available (see `MITRE_MAPPING.md` for this environment's specific Wazuh setup).
4. Start the frontend: `npm run dev` (from `frontend/`).
5. Run the benchmark harness (`BENCHMARKS.md`) and/or record a new experiment (`experiments/README.md`) against the now-live system.

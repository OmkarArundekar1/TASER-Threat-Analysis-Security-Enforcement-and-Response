# GNN Production Integration

Companion to `GNN_FEASIBILITY.md`, `GNN_OBJECTIVE_DECISION.md`,
`GNN_REPRESENTATION_DESIGN.md`, `GNN_REPRESENTATION_IMPLEMENTATION.md`,
`GNN_XGBOOST_ABLATION.md` (null result, preserved, not reopened), and
`GNN_RETRIEVAL_EVALUATION.md` (promising-but-limited retrieval
evidence). This phase takes the research-stage GNN from "evaluated in
isolation" to "additively wired into the live system, off by default,
fails safe everywhere" — real code, real tests, real live-Neo4j
verification, not a redesign of anything existing.

All numbers/behavior described below were verified this phase against
this environment's real, live Neo4j instance and the real trained
artifact (`ml/models/gnn_autoencoder.pt`) — not assumed.

---

## 1. Current production architecture

Unchanged (this phase is additive only):

```
Wazuh -> Raw Alert Stream -> MITRE Resolution -> IOC Extraction ->
Enterprise Deduplication -> Campaign Manager -> Operation Correlation ->
Neo4j Attack Graph -> Graph/MITRE Features -> Severity Engine ->
XGBoost Prediction -> Threat Attribution -> Evidence-Aware Investigation ->
Next-Best-Evidence -> Multi-RAG -> MISP/CTI -> Dashboard API -> React Frontend
```

## 2. GNN architecture (unchanged from the research phases)

2-layer `SAGEConvLayer` encoder (`hidden_dim=16`), mean-pool +
`Linear(16,8)` + `tanh` bottleneck (`embedding_dim=8`). Trained via
graph autoencoding (edge existence + edge type + standardized node
feature reconstruction) — see `GNN_REPRESENTATION_IMPLEMENTATION.md`
for the full design.

## 3. GNN model artifact

**Verified this phase, not assumed**: `ml/models/gnn_autoencoder.pt`
(gitignored, present on disk in this environment, ~13.9 KB). Loaded
via the existing, unmodified `train_autoencoder.load_autoencoder`.
Metadata now explicitly exposed via `GNNModelMetadata`
(`ml/gnn/inference.py`) — constructed at load time from the checkpoint
dict, not stored redundantly in the artifact file itself:

```json
{
  "model_type": "graph_autoencoder",
  "architecture": "2-layer SAGEConv encoder, mean-pool + Linear+tanh bottleneck",
  "hidden_dim": 16, "embedding_dim": 8, "num_layers": 2,
  "node_feature_schema": ["Attacker","Campaign","AttackEvent","Technique","Host","Unknown","occurrences","total_tps","tps","rule_level","vt_reputation","threat_actor_reputation","malware_confidence","tool_confidence","misp_confidence","ioc_confidence"],
  "edge_feature_schema": ["LAUNCHED","HAS_EVENT","MATCHES","TARGETS","Unknown"],
  "normalization": "z-score on the 10 numeric node properties, fit on the training split only",
  "training_seed": 42,
  "model_version": "gnn_autoencoder.pt@<mtime>-<size>b",
  "artifact_scope": "single all-population split (not cross-validated)"
}
```

**Critical distinction, explicitly documented and enforced in the
metadata text itself** (`training_dataset_description`): this artifact
is the **single all-71-campaign-population split** (59 train / 12
val) from `GNN_REPRESENTATION_IMPLEMENTATION.md` — **NOT** the
fold-safe Leave-One-Attacker-Group-Out cross-validation embeddings
reported in `GNN_XGBOOST_ABLATION.md`/`GNN_RETRIEVAL_EVALUATION.md`,
which came from 7 (or 3) separately-trained fold-specific models never
exposed to production. **Production inference always uses this one
frozen artifact for every campaign** — the strong retrieval numbers in
`GNN_RETRIEVAL_EVALUATION.md` (temporal MRR 0.377, attacker/host MRR
~0.97) were measured under stricter, fold-safe conditions this
artifact does not reproduce exactly; production topology-similarity
values should be read as coming from the same *kind* of model, not a
byte-identical reproduction of those cross-validated numbers.

**Not retrained this phase, per explicit instruction** — frozen as-is.

## 4. GNN inference path

New: `ml/gnn/inference.py` — `GNNInferenceService` (module-level
singleton `gnn_inference_service`). Verified this phase there was
**no pre-existing production inference layer** (GNN had zero runtime
consumers before this phase, confirmed repeatedly across
`GNN_FEASIBILITY.md`/`GNN_OBJECTIVE_DECISION.md`) — this is new, not a
duplicate.

Guarantees, all live-verified this phase:
- Never trains; loads a frozen artifact once, lazily, on first use.
- `model.eval()` + `torch.no_grad()` (the latter already inside
  `GraphAutoencoder.embed_graph`, called again here for defense in
  depth).
- **Deterministic**: verified live — two calls to `embed_campaign` on
  the same real campaign (`CAMP_7331E223`), with caching bypassed,
  produce bit-identical tensors (`torch.equal`).
- **Fails safe on every tested failure mode** (all with real,
  passing tests in `tests/test_gnn_inference.py`): `GNN_ENABLED=false`,
  missing artifact file, corrupt artifact file, artifact missing
  required metadata keys, campaign not found, empty (0-node) graph,
  non-finite embedding output, embedding-dimension mismatch, and any
  unexpected exception during graph construction or model inference —
  every one of these returns `None` and logs, never raises.

### A real bug found and fixed this phase (not a synthetic test)

Live end-to-end investigation testing (Section 8) surfaced a genuine
**Windows DLL load-order conflict**: importing `torch`
(`ml/gnn/inference.py`) *after* XGBoost has already loaded its own
bundled native runtime intermittently — in this environment,
*reliably* — fails with `WinError 1114` ("DLL initialization routine
failed" loading torch's `c10.dll`), a known class of conflict between
two ML libraries' bundled native binaries on Windows. Reproduced 3/3
times via the full investigation path (`XGBOOST_PREDICTION` action
runs before `GNN_TOPOLOGY_RETRIEVAL` in NBE's selected order), and
confirmed **not** to occur when torch is imported standalone/first.

**Fixed, not just caught**: `dashboard_api.py` and `realtime_socgraph.py`
now eagerly "pre-warm" `gnn_inference_service.available` at process
startup, *only when `GNN_ENABLED=true`* (zero cost otherwise) — this
forces torch's native runtime to initialize before XGBoost gets a
chance to load a conflicting one. Verified fully fixed: 3/3 repeated
live investigation runs after the fix returned real GNN topology
evidence (5 items each), where all 3 pre-fix runs returned 0. The
fail-safe design's correctness was also validated by this real
failure: before the fix, the DLL error was caught, logged, and the
investigation completed successfully anyway (70 real evidence items,
normal stopping reason) — GNN was never a single point of failure,
demonstrated under a genuine, naturally-occurring failure, not only a
synthetic one.

## 5. Topology similarity

New: `ml/gnn/topology_similarity.py` — `gnn_graph_similarity(a, b)`
(pure cosine-similarity function) and
`gnn_topology_similarity_between_campaigns(id_a, id_b)` (fail-safe
end-to-end version via the inference service). Named explicitly so its
source is never ambiguous next to CYUKTI's other similarity signals.
**Does not replace** technique/attacker/victim/temporal/chain/prediction
similarity, or either pre-existing `graph_similarity` slot (Section 6).

## 6. The two pre-existing `graph_similarity` paths — re-verified, not merged

Traced independently again this phase (unchanged from prior findings):

1. **`OperationFeatures.graph_similarity`** (`operation_feature_engine.py`):
   a literal `return 0.0`, weight `0.00` in
   `OperationDecisionEngine.weights`. **Untouched.** Not merged with
   GNN topology similarity, not redefined.
2. **`CampaignDecisionEngine`'s graph_similarity**
   (`campaign_feature_engine.CampaignFeatureEngine.graph_similarity`):
   real (not a stub), weight `0.01` in `config.CAMPAIGN_WEIGHTS`,
   feeding `campaign_manager.py`'s active-campaign continue/close
   decision. **Untouched** — `campaign_decision_engine.py`,
   `campaign_feature_engine.py`, and `CAMPAIGN_WEIGHTS` are unmodified.

Neither engine's scoring, weights, or thresholds were changed. GNN
topology similarity is exposed **alongside** both, never substituted.

## 7. Correlation/attribution/retrieval integration points

All additive, all fail-safe to `None`/`[]` when GNN is
disabled/unavailable, all with tests proving the pre-existing decision
is byte-identical regardless of GNN state:

| Integration point | What was added | What was NOT touched |
|---|---|---|
| `campaign_correlation_engine.py` | `CorrelationResult.gnn_topology_similarity: float \| None` (new field, default `None`) — max GNN cosine similarity between the current campaign and the matched operation's existing campaigns | `OperationDecisionEngine.evaluate()`, its `weights` dict, its `0.70` threshold — zero lines changed |
| `campaign_manager.py` | `_log_gnn_nearest_topology_neighbor()` — logs the nearest historical campaign by GNN topology, diagnostic only | `CampaignDecisionEngine.evaluate()`, `CAMPAIGN_WEIGHTS`, `0.35` continue/close threshold — zero lines changed |
| `threat_attribution_engine.py` / `threat_actor_context.py` | `ThreatActorContext.topology_similarity: float \| None` (new field) computed per candidate, attached after `similarity`/`total_score` are already decided | `coverage`/`precision`/`chain_similarity` computation and the `total_score`-based sort — verified by test that ranking is byte-identical GNN-on vs GNN-off |
| `rag/gnn_topology_retriever.py` (new file) | `GNNTopologyRetriever` — ranks the same historical-campaign population TF-IDF/CAMPAIGN_HISTORY already draw from, by GNN embedding similarity instead of text/technique overlap; every result's `provenance` says `ml.gnn.inference` explicitly | `rag/campaign_retriever.py` (TF-IDF) and `evidence/collectors/campaign_history_collector.py` (exhaustive technique-overlap) — both unmodified, both still run |
| `investigation/actions.py` / `investigation/loop.py` | `InvestigationAction.GNN_TOPOLOGY_RETRIEVAL`, `EvidenceSource.GNN_TOPOLOGY` (`evidence/schema.py`), wired into `default_action_executor`, `reliability=0.5` (deliberately not high — see the code comment citing `GNN_RETRIEVAL_EVALUATION.md`'s own "promising but limited" verdict), `depends_on={CAMPAIGN_HISTORY}` (a real, code-verified overlap: both query `attribution_context.context.load_historical_campaigns()`) | The NBE scoring algorithm itself, every other action's metadata |
| `dashboard_api.py` | `GET /api/gnn/status`, `GET /api/gnn/topology/<campaign_id>` (both additive, return `gnn_available: false` — 200, not an error — when disabled) | All 24 pre-existing routes — regression-tested, unchanged |

## 8. Failure fallback — live-verified, not just unit-tested

See Section 4's "real bug found and fixed" — the fail-safe design was
exercised by a genuine environment failure (not a mock), and correctly
degraded: investigation completed with 70 real evidence items from 6
other sources while GNN evidence was silently unavailable, before the
DLL fix. After the fix, the same real campaign's investigation
produces 75 evidence items (70 + 5 real GNN topology matches).

## 9. XGBoost non-integration (preserved null result)

**Unchanged, not reopened.** `GNN_XGBOOST_ABLATION.md`'s finding
stands: baseline and GNN-augmented XGBoost produced bit-identical
metrics across all folds; all 8 `z_G` feature columns had exactly
`0.0` importance. `z_G` is **not** added to `FEATURE_COLUMNS`, not
added to any XGBoost training/inference path, and `ml/train_xgboost.py`/
`ml/runtime_predictor.py`/`ml/dataset_utils.py` are byte-for-byte
unmodified this phase (verified via `git status`).

## 10. SSFT limitation (unchanged)

The SSL/SSFT pipeline expects 312-dimensional CICIDS2017 network-flow
windows; the live campaign pipeline does not produce that
representation. **SSFT → live campaign integration remains
infrastructure-blocked** — not attempted, not claimed complete.
*Planned:* revisit once a live network-flow/packet-capture pipeline
producing a real per-campaign window exists — an infrastructure
question, not an engineering one.

## 11. MISP authentication limitation (re-verified this phase)

`config.MISP_API_KEY` is empty in this environment
(`os.environ.get("MISP_API_KEY", "")`, verified live this phase:
`len(config.MISP_API_KEY) == 0`). MISP event generation/validation
code exists and is unchanged; **authenticated publication was not
attempted and is not claimed to have succeeded** — genuinely blocked
on a missing credential, not a code limitation.

## 12. WebSocket limitation (unchanged)

The dashboard's real-time behavior is polling-based. No
Flask-SocketIO or equivalent backend exists. **Not implemented, not
claimed.** *Planned:* implement it once the listener/dashboard-API
process-boundary question (see `ARCHITECTURE_AUDIT.md` Section D) is
settled.

## 13. Neo4j — no new persistence (as instructed)

Per this phase's explicit instruction, **no new relationship type
(e.g. a `GNN_SIMILAR_TO`) was created in Neo4j.** GNN topology
similarity is entirely runtime-derived (computed on demand from the
frozen artifact) — the model-versioning/staleness/recomputation/
deletion-semantics questions that would need answering before
persisting GNN-derived edges were not resolved this phase, so nothing
is written. `SIMILAR_TO`/`RESEMBLES` remain exactly as they were,
untouched, unread by any new GNN code (verified: zero occurrences of
either string in `ml/gnn/inference.py`, `ml/gnn/topology_similarity.py`,
or `rag/gnn_topology_retriever.py`).

## 14. Feature flag

`config.GNN_ENABLED` (default `False`) + `config.GNN_MODEL_PATH`
(defaults to `ml/models/gnn_autoencoder.pt`). **Verified live this
phase**: with `GNN_ENABLED=false` (the shipped default), every
integration point above returns `None`/`[]`/`gnn_available: false`
without importing torch or touching the filesystem — confirmed via a
dedicated test
(`test_feature_flag_disabled_never_touches_filesystem_or_torch`) and
live smoke-testing of `/api/gnn/status` and `/api/gnn/topology/<id>`
both disabled and enabled.

---

## Frontend (Phase 12)

**Not modified**, per this phase's own permitted alternative
("otherwise, keep the backend/API integration complete and document
the frontend presentation point"). Frontend regression-verified
unchanged: 61/61 tests pass, TypeScript build (`tsc -b && vite build`)
succeeds, production bundle builds cleanly (496.89 kB, 147.87 kB
gzipped) — all identical to pre-phase state (`git status` confirms
zero frontend files touched).

**Documented presentation point for a future phase**: `/api/gnn/status`
and `/api/gnn/topology/<campaign_id>` are ready to consume. The most
natural integration points, based on existing component boundaries
(not designed or built this phase): `ThreatCorrelation.tsx` (operation
correlation already renders `breakdown` fields — `gnn_topology_similarity`
could sit alongside them, clearly labeled as a separate signal) and
`ThreatActorAttribution.tsx` (attribution results already render
per-candidate similarity breakdowns — `topology_similarity` fits the
same card layout). A dedicated small panel is also plausible but was
judged unnecessary scope for this phase given the "do not create
unnecessary UI complexity" instruction.

---

## Security check (Phase 19)

- **No secrets in GNN code**: grep-verified zero occurrences of
  `api_key`/`password`/`MISP_API_KEY`/`NEO4J_PASSWORD` anywhere in
  `ml/gnn/inference.py`, `ml/gnn/topology_similarity.py`,
  `rag/gnn_topology_retriever.py`.
- **Model path never returned by the API**: `/api/gnn/status`'s
  `model_version` field is `os.path.basename(model_path)@mtime-size`
  (`_model_version_from_path`) — the filename only, never the full
  filesystem path.
- **No debug endpoint added.** `app.run(host='0.0.0.0', port=5002)`
  (pre-existing, unmodified) has no `debug=True`.
- **GNN failures cannot kill alert processing or investigation**:
  demonstrated live (Section 4/8) under a real failure, not only
  asserted.
- **Neo4j failures**: unchanged, pre-existing `_try_load_campaign_context`
  handling (`ServiceUnavailable`/`Neo4jError` → clean 503) still
  applies to the new `/api/gnn/topology/<id>` route (it calls the same
  helper).
- **Malformed input**: `/api/gnn/topology/<id>` for a nonexistent
  campaign returns a controlled 404, not a crash (tested).

---

## Testing (Phase 14)

```bash
cd backend && python -m pytest tests/ -q
```

| | Backend | Frontend | Total |
|---|---|---|---|
| Before this phase | 430 | 61 | 491 |
| After this phase | 470 | 61 | 531 |

(The task brief's stated baseline of "401/61/462" was one phase stale
— the XGBoost ablation and retrieval evaluation phases had already
brought backend to 430 before this phase began; noted as a factual
correction, not a discrepancy introduced here.)

40 new backend tests (`tests/test_gnn_inference.py`,
`tests/test_gnn_production_integration.py`) covering every item Phase
14 named: model loading, deterministic inference, embedding
dimensionality, finite embeddings, topology similarity, the feature
flag, missing model, corrupt model, malformed/empty graphs, failure
fallback, operation/campaign/attribution/retrieval/investigation/API
integration, and regression proof that XGBoost, campaign scoring,
attribution ranking, TF-IDF retrieval, and Neo4j behavior are
unchanged. Zero regressions; zero existing tests weakened.

---

## Live verification (Phases 15-17)

**Phase 15 (live Neo4j)** — real campaigns, GNN enabled: deterministic
(`torch.equal` across two calls), finite, correct 8-dim shape,
self-similarity exactly `1.0`, real cross-campaign topology retrieval
(e.g. `CAMP_7331E223`'s nearest neighbor: `CAMP_947A7084`, similarity
`0.9994`) — no `SIMILAR_TO`/`RESEMBLES` involved anywhere (grep-verified).

**Phase 16 (live investigation)** — real campaign `CAMP_7331E223`
through the full `/api/investigate/<id>` path (MITRE → XGBoost severity
→ graph structure → campaign history → detection → CTI → attribution →
GNN topology retrieval): 75 total evidence items, 5 from GNN, normal
completion (`stopping_reason: "reached maximum investigation depth (8
steps)"`), coexists cleanly with every existing evidence source.
Phase 21/22 frozen findings were not read or modified.

**Phase 17 (live Wazuh)** — **infrastructure unavailable in this
environment, confirmed, not assumed**: this session runs on a local
Windows development machine (MSYS2/Git Bash, `MINGW64_NT`), not the
Kali/Ubuntu/Wazuh lab the architecture assumes. `/var/ossec` does not
exist, no `wazuh-agent`/`wazuh-manager`/`ossec-control` binaries are on
`PATH`, no Wazuh alert file exists at the configured
`WAZUH_ALERT_FILE` path. **No live Wazuh test was performed, and none
is fabricated.** This is the one Phase 17 item genuinely blocked on
external infrastructure this session cannot provide.

---

## Component status table (Phase 21)

| Component | Status |
|---|---|
| Wazuh listener | **NOT INTEGRATED THIS SESSION** (code exists, unmodified; no live Wazuh reachable to verify against, Phase 17) |
| MITRE resolver | IMPLEMENTED |
| IOC extraction | IMPLEMENTED |
| Enterprise deduplication | IMPLEMENTED |
| Campaign manager | IMPLEMENTED (+ additive GNN diagnostic log) |
| Operation correlation | IMPLEMENTED (+ additive `gnn_topology_similarity` field, decision unchanged) |
| Neo4j | IMPLEMENTED (no new GNN-derived persistence, by design — Section 13) |
| Graph feature engine | IMPLEMENTED (untouched) |
| GNN | IMPLEMENTED — extraction, encoding, training, production inference, topology similarity, fail-safe integration into correlation/attribution/retrieval/investigation/API, all live-verified; **disabled by default** |
| XGBoost | IMPLEMENTED (untouched; GNN explicitly NOT integrated — preserved null result) |
| Threat attribution | IMPLEMENTED (+ additive `topology_similarity` field, ranking unchanged) |
| Evidence engine | IMPLEMENTED (+ new `EvidenceSource.GNN_TOPOLOGY`) |
| Next-best-evidence | IMPLEMENTED (+ `GNN_TOPOLOGY_RETRIEVAL` in the action menu) |
| MITRE RAG | IMPLEMENTED (untouched) |
| Historical campaign RAG | IMPLEMENTED (TF-IDF, untouched) + new GNN topology retriever alongside it |
| MISP | PARTIALLY IMPLEMENTED — event generation/validation real; authenticated publication INFRASTRUCTURE-BLOCKED (no credential) |
| Dashboard API | IMPLEMENTED (+ 2 additive `/api/gnn/*` routes, all 24 pre-existing routes regression-verified) |
| React frontend | IMPLEMENTED (untouched; GNN presentation point documented, not built) |
| SSFT | INFRASTRUCTURE-BLOCKED (no 312-dim live network-flow pipeline) |
| WebSocket | NOT INTEGRATED (polling only, as designed; not attempted) |

---

## Limitations

- Live Wazuh end-to-end verification (Phase 17) could not be performed
  — no such infrastructure exists in this session's environment.
  *Planned:* run this verification once a real Wazuh lab environment
  (Kali → Ubuntu → Wazuh Manager) is reachable from a future session.
- The production GNN artifact is a single all-population split, not
  the fold-safe cross-validated model the strongest retrieval evidence
  (`GNN_RETRIEVAL_EVALUATION.md`) was measured against — documented
  explicitly in the artifact's own metadata (Section 3) so this is
  never silently conflated. *Planned:* retrain/ship a fold-safe
  artifact once the project decides production should reflect the
  cross-validated numbers rather than the single-split one.
- GNN is disabled by default; every "live-verified" result above was
  obtained with `GNN_ENABLED=true` set explicitly for verification —
  the shipped default behavior is byte-identical to pre-GNN-integration
  CYUKTI.
- No frontend UI was built for GNN data — the API is ready, the
  presentation point is documented, not implemented. *Planned:* build
  the documented presentation points (`ThreatCorrelation.tsx`,
  `ThreatActorAttribution.tsx`) in a future frontend-scoped phase.
- MISP authenticated publication remains blocked on a missing
  credential; not attempted, not claimed. *Planned:* reconfirm once
  `MISP_API_KEY` is configured and the MISP service is running for a
  verification session.

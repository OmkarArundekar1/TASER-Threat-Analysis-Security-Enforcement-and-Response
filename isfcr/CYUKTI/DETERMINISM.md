# CYUKTI Determinism Verification

`backend/tests/test_determinism.py` (7 tests) calls each component twice with byte-identical input and asserts byte-identical output.

| Component | Deterministic? | Verified how |
|---|---|---|
| MITRE mapping (`resolve_mitre`) | **Yes** | Same alert dict twice → identical `MitreResolution` (dataclass equality), for both a native-mapped and an unmapped alert. |
| Campaign selection composite scoring | **Yes** | `build_candidate()` with fixed signal values twice → identical `composite_score` (pure weighted-average arithmetic, no randomness). |
| Campaign selection ranking + explanation | **Yes** | `select()` over the same two candidates twice → identical `selected`, `composite_score`, `explanation` string, and `confidence` tier. |
| Threat qualification | **Yes** | Same `_Incident` stub twice → identical `classification`, `may_publish_to_misp`, per-check `passed` list, and `reason` string. |
| ResponsePlan generation | **Yes** | Same `CampaignContext` + `Playbook` twice → identical `threat_summary`, `severity`, `misp_status`. |
| GNN embedding | **Yes — live-verified, not mocked.** | `gnn_inference_service.embed_campaign(real_campaign_id, use_cache=False)` called twice against the real trained model and a real campaign's real graph → `torch.allclose()` on the two embeddings. Both calls bypass the cache, so this proves the actual forward pass is deterministic (the model is in eval mode with no dropout/batchnorm training-mode randomness), not just that a cache returns the same object twice. |

## Intentionally NOT tested here (documented, not silently skipped)

- **GNN topology retrieval / campaign selection's *candidate discovery*** — the ranked list of historical campaigns can change between two calls if new campaigns are created in Neo4j between them (a live, growing dataset). The *scoring function* given a fixed candidate set is deterministic (tested above); the *candidate set itself* is not fixed over time, by design — this is expected, not a defect.
- **Campaign/operation correlation's live decision** — `CampaignDecisionEngine`/`OperationDecisionEngine` are pure functions of their feature inputs (and would be deterministic given a frozen snapshot of those inputs), but the *real* inputs include live Neo4j state that changes as new events are ingested — testing "the same alert processed twice produces the same campaign" isn't meaningful once campaign state itself has moved on. The underlying weighted-sum math is exercised by `campaign_decision_engine`'s own existing test suite.
- **Anything depending on wall-clock time** (e.g. `_now_iso()` timestamps stamped onto new records) — deterministic in the sense that the same code always produces "the current time," but the literal output value differs between calls by design; this is not a determinism defect, it's a timestamp.

## Regression baseline

`test_determinism.py` is part of the standard backend test suite (`pytest`, 651+ tests total as of this phase) — it runs on every full-suite invocation, not as a separate opt-in benchmark.

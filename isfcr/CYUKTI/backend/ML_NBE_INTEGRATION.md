# ML → NBE Integration: what exists and what this session added

This documents the closed-loop adaptive-investigation pipeline —
`IncidentContext → features → ML prediction → NBE scoring → next
action → evidence → context update → next decision` — as it actually
exists in the repository, and specifically what this BUILD/INTEGRATE
session changed versus what was already there.

## Architecture (already implemented, verified this session)

```
CampaignContext (campaign_context.py)          <- the IncidentContext
        |
feature_extractors.extractor.extract(...)      <- SAME function used by
        |                                          both training (dataset_builder.py)
        v                                          and runtime (runtime_predictor.py) —
CampaignDatasetRecord (feature_schema.py)          train/runtime feature parity by
        |                                          construction, not by convention
        v
ml.runtime_predictor.RuntimeCampaignPredictor
        |  (FEATURE_COLUMNS order, dataset_utils.py)
        v
ml.train_xgboost.XGBoostCampaignClassifier.predict()
        |  {label, confidence, probabilities, top_k, model_metadata}
        v
investigation.loop.default_model_predictor()   <- returns probabilities only;
        |                                          model verdict != investigation confidence
        v
investigation.loop.run_investigation()
        |  InvestigationAction.XGBOOST_PREDICTION is ONE candidate action
        |  among the evidence-gathering menu (investigation/actions.py)
        v
investigation.next_best_evidence.score_action()
        |  uncertainty_reduction_term: XGBOOST_PREDICTION's value scales
        |  with current investigation uncertainty (confidence.py) — once
        |  it has run and uncertainty is low, running it again earns no
        |  further credit. This is how the ML signal becomes one input to
        |  NBE's existing evidence-value/cost/latency/redundancy formula,
        |  without NBE losing interpretability or the model being treated
        |  as ground truth.
        v
best_action_value -> action executed -> Evidence -> EvidenceStore
        |
investigation.confidence.estimate_confidence(store, model_probabilities)
        |  evidence_reliability/coverage (deterministic, evidence-only)
        |  is kept separate from model_confidence/model_uncertainty
        |  (entropy of model_probabilities) — combined, not conflated,
        |  into investigation_confidence. See confidence.py docstring.
        v
InvestigationState (loop.py) — conclusion, candidate_hypotheses (full
ranked distribution, not just argmax), model_probabilities,
uncertainty, evidence_coverage, investigation_cost -> next decision
```

This loop already ran for real, against live Neo4j, for real campaigns
(`scripts/run_real_investigations.py`, `review/phase21_real_investigation_*`).
**Phase 21's findings are frozen and were not touched this session.**

## What this session added

The integration above existed but had two real gaps:

1. **`default_action_executor` and `default_model_predictor` — CYUKTI's
   real-engine wiring layer — had zero behavioral test coverage.** Both
   were imported by `tests/test_investigation_evidence_aware.py` but
   only ever inspected structurally (the no-Neo4j-write-function proof),
   never actually invoked. `tests/test_default_wiring.py` (new) exercises
   every `InvestigationAction` branch against the real engine singletons
   (`mitre_feature_engine.engine`, `threat_intelligence_engine.engine`,
   `detection_confidence_engine.engine`, `threat_attribution_engine.engine`,
   `attribution_context.context`, `graph_feature_engine.graph_analytics`,
   `rag.mitre_retriever.mitre_retriever`) and the real evidence collectors,
   with only the engine/DB call itself stubbed — and includes a full
   closed-loop sanity test (`test_full_closed_loop_incident_context_to_next_decision`)
   that runs real `run_investigation` + real `default_action_executor` +
   real `default_model_predictor` + a real trained XGBoost model end to
   end, asserting the model action is genuinely selected by NBE (not
   hardcoded first/last) and that its verdict actually propagates into
   every later step's confidence estimate.

2. **The prediction API exposed no model/version metadata**, despite
   `XGBoostCampaignClassifier` already tracking `target_column` and
   `classes` internally. `predict()` now also returns:
   - `top_k`: the full class list sorted by probability (convenience —
     `probabilities` already had the same information unsorted)
   - `model_metadata`: `target_column`, `classes`, `feature_schema_version`
     (new: `ml.dataset_utils.FEATURE_SCHEMA_VERSION`, bumped only when
     `CampaignDatasetRecord`'s feature columns change shape/meaning),
     `n_features`, `trained_at`
   - `RuntimeCampaignPredictor.predict_for_campaign` additionally attaches
     `prediction_context` (`campaign_id`, `attack_id`, `event_id`) so a
     prediction can be traced back to the investigation state that
     produced it.

   Models saved before this change (including the currently committed
   `ml/models/xgb_severity.meta.joblib`) still load correctly —
   `load()` falls back to `"unknown"` for the two new metadata fields
   rather than raising `KeyError`.

3. **`default_model_predictor` gained an optional `model_path` override**
   (default `None` = the existing hardcoded production path). Production
   callers (`dashboard_api.py`) are unaffected; it exists purely so tests
   can point at a small model trained on the synthetic fixture generator
   instead of requiring the committed production artifact.

No existing public interface's default behavior changed. All 185
backend tests pass (`python -m pytest tests/ -q`), including the 172
that existed before this session.

## Production model artifact — regeneration (RESOLVED, follow-up session)

A follow-up session found the committed production artifact
(`ml/models/xgb_severity.json` + `.meta.joblib`) failed to load under
this environment's xgboost 3.4.1 (`XGBoostError: input stream
corrupted` unpickling the calibrated model inside `.meta.joblib`).

**Root cause**: `ml/models/` is gitignored (`isfcr/CYUKTI/.gitignore`)
— the artifact was never a committed file, only a local build product
left over from whatever xgboost version last trained it (dated Aug 28,
per file mtime, predating this repository's `requirements.txt` pin of
`xgboost==3.4.1`). Loading `xgb_severity.json` directly via
`xgb.XGBClassifier().load_model(...)` (the self-describing JSON format)
succeeded — only the joblib-pickled `CalibratedClassifierCV` in
`.meta.joblib` (which pickles its wrapped booster in xgboost's raw
binary format, a different and less version-stable path than
`save_model()`'s JSON) was actually corrupted-relative-to-this-xgboost.

**Fix**: regenerated the artifact with the existing, unmodified
pipeline — `python -m ml.train_xgboost` run from `backend/` — against
the real, frozen 60-campaign dataset (`ml/datasets/campaign_dataset.csv`,
untouched). This reproduced `val_accuracy = 0.9167` (rounds to the
`0.917` already documented in `review_pack/03_module_status.md`),
confirming the regeneration is a faithful, deterministic reproduction
of the original result (same data, same `random_state=42` split, same
hyperparameters) and not a new/different model. Same feature schema
(57 columns, unchanged), same target (`severity`), same model interface
(`predict()` contract, including the `top_k`/`model_metadata` fields
added earlier this integration). No training code was changed to
produce this fix — only the on-disk artifact was regenerated.

`tests/test_production_model_artifact.py` (new) now exercises this
exact committed artifact directly (not a synthetic fixture model):
load, `RuntimeCampaignPredictor` wiring, feature-count-matches-schema,
prediction on a real historical feature row, full prediction-contract
retention, `prediction_context` tracing, `default_model_predictor`'s
production (`model_path=None`) code path, and a full closed-loop
sanity run — `CampaignContext → feature extraction → **this** artifact
→ prediction → NBE → investigation action/evidence → context update`
— with only the Neo4j/MISP engine boundary mocked. It is skipped (not
failed) if the artifact is absent, since it's a local, gitignored file.

To regenerate the artifact yourself: `cd backend && python -m ml.train_xgboost`.

## How to run

```bash
cd backend
python -m pytest tests/ -q                                # full suite (193 tests)
python -m pytest tests/test_default_wiring.py -q           # real-engine wiring coverage
python -m pytest tests/test_production_model_artifact.py -q  # committed artifact regression
python -m pytest tests/test_ml_pipeline.py tests/test_investigation.py \
    tests/test_investigation_evidence_aware.py -q   # ML + NBE + investigation loop
```

Mocked end-to-end sanity check (no live infra required, fresh fixture model):
`tests/test_default_wiring.py::test_full_closed_loop_incident_context_to_next_decision`.

Mocked end-to-end sanity check using the real committed model:
`tests/test_production_model_artifact.py::test_full_closed_loop_with_production_model_artifact`.

Real end-to-end run against live Neo4j (requires the infrastructure
described in `review_pack/03_module_status.md`):
`python scripts/run_real_investigations.py`.

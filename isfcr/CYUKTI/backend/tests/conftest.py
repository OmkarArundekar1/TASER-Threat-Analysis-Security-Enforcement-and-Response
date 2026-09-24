import os
import sys

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


@pytest.fixture(autouse=True)
def _reset_gnn_inference_singleton():
    """ml.gnn.inference.gnn_inference_service is a process-wide singleton
    that, by design, loads the model artifact once and caches that
    decision for the process's lifetime (real production behavior: a
    feature flag read once at startup, not re-checked per request).

    That design makes tests order-dependent: whichever test runs FIRST
    with real config.GNN_ENABLED (now True by default in this
    environment's .env, per the user's request) permanently marks the
    shared singleton "loaded", so a LATER test's
    `monkeypatch.setattr(config, "GNN_ENABLED", False)` has no effect on
    it -- the singleton never re-reads the flag once `_loaded=True`.
    Resetting its private state before every test restores isolation
    without changing the singleton's real (correct) production
    behavior."""
    from ml.gnn.inference import gnn_inference_service

    gnn_inference_service._loaded = False
    gnn_inference_service._model = None
    gnn_inference_service._metadata = None
    gnn_inference_service._embedding_cache.clear()
    yield
    gnn_inference_service._loaded = False
    gnn_inference_service._model = None
    gnn_inference_service._metadata = None
    gnn_inference_service._embedding_cache.clear()

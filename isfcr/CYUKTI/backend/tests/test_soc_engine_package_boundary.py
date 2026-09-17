"""
Regression test for the Generation-1/Generation-2 architectural boundary
fix made to soc_engine/__init__.py during the Generation-1 disposition
audit (see ../../GENERATION1_DISPOSITION.md).

Before this fix, soc_engine/__init__.py eagerly re-exported
FeatureExplainer (soc_engine/explainer.py) and TemporalSmoother
(soc_engine/temporal_engine.py) at package level. Because Python
package imports are eager, this meant Generation-2's own real,
production import path -- `from soc_engine.model import Autoencoder`,
used by ml/ssl_pipeline.py and ml/ssft.py -- always executed
soc_engine/__init__.py first, which transitively imported explainer.py
(a Generation-1-only module with zero consumers anywhere in this
repository) and temporal_engine.py (used only by the demo notebooks,
not by any Generation-2 code) as an unrequested side effect.

These tests prove the decoupling without deleting either file: the
package-level re-export is gone, but both modules remain fully
importable via their own direct submodule path -- exactly how
agents/soc_agent.py and the demo notebooks already consume them, so
nothing that genuinely needs FeatureExplainer/TemporalSmoother breaks.
"""

import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def test_soc_engine_package_does_not_reexport_generation1_only_helpers():
    import soc_engine

    assert not hasattr(soc_engine, "FeatureExplainer")
    assert not hasattr(soc_engine, "TemporalSmoother")
    assert "FeatureExplainer" not in soc_engine.__all__
    assert "TemporalSmoother" not in soc_engine.__all__


def test_soc_engine_reusable_generation2_primitives_still_reexported():
    """model/scorer/threshold/trainer are real, currently-used Generation-2
    dependencies (ml/ssl_pipeline.py, ml/ssft.py) -- the fix must not
    touch these."""
    import soc_engine

    assert soc_engine.Autoencoder is not None
    assert soc_engine.load_autoencoder is not None
    assert soc_engine.SeverityScorer is not None
    assert soc_engine.compute_scores is not None
    assert soc_engine.AdaptiveThreshold is not None
    assert soc_engine.AutoencoderTrainer is not None


def test_explainer_and_temporal_engine_remain_directly_importable():
    """Neither file was deleted -- only the accidental package-level
    re-export was removed. Direct submodule imports (what soc_agent.py
    and the demo notebooks actually use) must still work."""
    from soc_engine.explainer import FeatureExplainer
    from soc_engine.temporal_engine import TemporalSmoother, apply_temporal_smoothing

    assert FeatureExplainer is not None
    assert TemporalSmoother is not None
    assert apply_temporal_smoothing is not None


def test_generation2_import_path_no_longer_pulls_in_generation1_only_modules():
    """The actual proof of decoupling: importing soc_engine.model (the
    real statement ml/ssl_pipeline.py and ml/ssft.py execute) must not,
    as a side effect, load explainer.py or temporal_engine.py into
    sys.modules. Run in a fresh subprocess so no other test's imports
    can mask the result."""
    code = (
        "import sys\n"
        "from soc_engine.model import Autoencoder\n"
        "assert 'soc_engine.explainer' not in sys.modules, "
        "'explainer.py must not be eagerly imported via soc_engine.model'\n"
        "assert 'soc_engine.temporal_engine' not in sys.modules, "
        "'temporal_engine.py must not be eagerly imported via soc_engine.model'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(BACKEND_DIR),
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout

"""
Regression tests for ml.label_generator.

Covers two distinct bugs found during real-data validation:

1. Severity thresholds were applied to the raw, unnormalized cumulative
   risk_score instead of the normalized 0-100 scale.

2. (Phase 18) next_technique/prediction_correct were defined as
   "does the campaign's own live predicted_next equal its final
   technique" -- structurally near-impossible to satisfy, since the live
   loop (chain_updater.py/realtime_socgraph.py) always overwrites the
   stored prediction with one made FROM the campaign's own
   most-recently-observed technique, leaving no further same-campaign
   event to validate it against once the campaign stops. The corrected
   definition evaluates the one real within-campaign transition that can
   ever be checked: technique_sequence[-2] -> technique_sequence[-1],
   using a fresh, side-effect-free read of the same learned-transition
   model (prediction_engine.predict_next_readonly), and returns
   NOT_APPLICABLE (None) rather than 0 when there is no transition to
   evaluate or no learned prediction exists.
"""

from dataclasses import dataclass, field


@dataclass
class _FakeCampaignContext:
    risk_score: float
    attack_chain: list = field(default_factory=list)


def test_label_generator_normalizes_before_thresholding():
    from ml.label_generator import generator

    # Real campaign values (CAMP_A93237FD=90, CAMP_D8605E81=630, CAMP_427A075C=8330)
    low = generator.generate(_FakeCampaignContext(risk_score=90), ["T1110"], "T1110", "unknown")
    medium = generator.generate(_FakeCampaignContext(risk_score=630), ["T1110"], "T1110", "unknown")
    critical = generator.generate(_FakeCampaignContext(risk_score=8330), ["T1110"], "T1110", "unknown")

    assert low.severity == "Low"
    assert medium.severity == "Medium"
    assert critical.severity == "Critical"


def test_single_event_campaign_yields_not_applicable(monkeypatch):
    """A campaign with only one observed technique has no transition to
    evaluate at all -- must be NA, never silently 0."""
    from ml import label_generator

    def _should_not_be_called(_current):
        raise AssertionError("predict_next_readonly must not be called with < 2 techniques")

    monkeypatch.setattr(label_generator, "predict_next_readonly", _should_not_be_called)

    ctx = _FakeCampaignContext(risk_score=0)
    labels = label_generator.generator.generate(ctx, ["T1595"], "T1595", "unknown")

    assert labels.next_technique == ""
    assert labels.prediction_correct is None


def test_no_learned_transition_yields_not_applicable(monkeypatch):
    """Two real events exist, but the learned NEXT_TECHNIQUE graph has no
    (or insufficient/low-confidence) transition from the earlier one --
    still NA, not 0: there was nothing to predict against."""
    from ml import label_generator

    monkeypatch.setattr(label_generator, "predict_next_readonly", lambda current: None)

    ctx = _FakeCampaignContext(risk_score=0)
    labels = label_generator.generator.generate(ctx, ["T1595", "T1595"], "T1595", "unknown")

    assert labels.next_technique == "T1595"
    assert labels.prediction_correct is None


def test_correct_prediction_recomputed_from_second_to_last_technique(monkeypatch):
    from ml import label_generator

    calls = []

    def _fake_predict(current):
        calls.append(current)
        return {"predicted": "T1110", "confidence": 100.0}

    monkeypatch.setattr(label_generator, "predict_next_readonly", _fake_predict)

    ctx = _FakeCampaignContext(risk_score=0)
    labels = label_generator.generator.generate(ctx, ["T1110.001", "T1110"], "T1110", "unknown")

    assert calls == ["T1110.001"]  # queried from the PREVIOUS technique, not the final one
    assert labels.next_technique == "T1110"
    assert labels.prediction_correct == 1


def test_incorrect_prediction_recomputed_from_second_to_last_technique(monkeypatch):
    from ml import label_generator

    monkeypatch.setattr(
        label_generator, "predict_next_readonly",
        lambda current: {"predicted": "T1078", "confidence": 100.0},
    )

    ctx = _FakeCampaignContext(risk_score=0)
    labels = label_generator.generator.generate(ctx, ["T1110.001", "T1110"], "T1110", "unknown")

    assert labels.next_technique == "T1110"
    assert labels.prediction_correct == 0


def test_longer_sequence_only_evaluates_the_final_transition(monkeypatch):
    """Only the last transition in the sequence is evaluable (it's the
    only one with a real 'what happened next' to check) -- earlier
    technique churn must not affect the result."""
    from ml import label_generator

    calls = []

    def _fake_predict(current):
        calls.append(current)
        return {"predicted": "T1078", "confidence": 100.0}

    monkeypatch.setattr(label_generator, "predict_next_readonly", _fake_predict)

    ctx = _FakeCampaignContext(risk_score=0)
    labels = label_generator.generator.generate(
        ctx, ["T1110.001", "T1110", "T1078"], "T1078", "unknown"
    )

    assert calls == ["T1110"]
    assert labels.prediction_correct == 1

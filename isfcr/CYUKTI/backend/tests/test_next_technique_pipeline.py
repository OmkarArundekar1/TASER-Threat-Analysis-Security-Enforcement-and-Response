"""
Phase 18 regression tests: next_technique / prediction_correct pipeline.

Covers the two concrete bugs found while tracing next_technique/
predicted_next/prediction_correct from the persisted dataset column back
to their production sources:

Bug A (ml/feature_extractors.py): the persisted `next_technique` column
was wired to campaign_context.predicted_next (the live model's own,
frequently-stale or absent prediction) instead of the ground-truth value
computed by ml/label_generator.py -- which was computed correctly but
silently discarded, never passed to DatasetFeatureExtractor.extract().

Bug B (ml/dataset_validator.py): the blanket integer-field coercion
`max(0, int(value))` would either crash on prediction_correct=None (the
NOT_APPLICABLE sentinel) or, if guarded naively, could turn NA into 0 --
exactly the anti-pattern the audit was commissioned to catch.

prediction_engine.predict_next_readonly is exercised directly against a
fake Neo4j session (same pattern as test_campaign_reconstruction.py) to
confirm the threshold logic behaves identically to the live predict_next()
path it was extracted from, with zero write side effects.
"""

from dataclasses import fields


# ---------------------------------------------------------------- Bug A: feature_extractors wiring

class _FakeCampaignContext:
    campaign_id = "CAMP_TEST"
    attacker_ip = "1.2.3.4"
    victim_ip = "5.6.7.8"
    risk_score = 90.0
    last_technique = "T1110"
    predicted_next = "T1999"  # deliberately different from the real next_technique
    last_seen = None


def _install_fake_feature_engines(monkeypatch):
    """Bypasses every engine that would otherwise need a live Neo4j
    connection (graph_similarity, chain_similarity, mitre/runtime/threat/
    detection extractors) so extractor.extract() is a pure unit under
    test for the next_technique wiring specifically."""
    from ml import feature_extractors as fe_module

    class _FakeGraph:
        campaign_size = 1
        unique_techniques = 1
        campaign_duration = 0.0
        node_count = 0
        edge_count = 0
        graph_density = 0.0
        graph_connectivity = 0.0
        average_degree = 0.0
        attacker_degree = 0
        victim_degree = 0
        technique_degree = 0
        attack_chain_depth = 0
        average_path_length = 0.0
        graph_diameter = 0
        branching_factor = 0.0
        average_clustering = 0.0
        average_betweenness = 0.0
        average_closeness = 0.0
        community_count = 0
        largest_community = 0
        campaign_complexity = 0.0
        structural_risk = 0.0
        evolution_rate = 0.0

    class _FakeRuntime:
        campaign_count = 0
        attack_event_count = 0
        graph_degree = 0
        incoming_chain_count = 0
        outgoing_chain_count = 0
        prediction_frequency = 0
        duplicate_frequency = 0

    class _FakeMitre:
        platform_count = 0
        domain_count = 0
        kill_chain_count = 0
        threat_actor_count = 0
        malware_count = 0
        tool_count = 0
        mitigation_count = 0
        subtechnique_count = 0

    class _FakeThreat:
        ip_reputation = 0.0
        threat_actor_reputation = 0.0
        malware_confidence = 0.0
        tool_confidence = 0.0
        misp_confidence = 0.0
        ioc_confidence = 0.0

    class _FakeDetection:
        wazuh_level = 0
        suricata_score = 0.0
        zeek_score = 0.0
        sigma_score = 0.0
        yara_score = 0.0
        detection_confidence = 0.0

    class _FakeFeatures:
        graph = _FakeGraph()
        runtime = _FakeRuntime()
        mitre = _FakeMitre()
        threat_intel = _FakeThreat()
        detection = _FakeDetection()

    class _FakeCampaignFeatures:
        prediction_similarity = 0.0
        chain_similarity = 0.0
        temporal_similarity = 0.0
        attacker_similarity = 0.0
        duplicate_similarity = 0.0
        graph_similarity = 0.0
        runtime_similarity = 0.0

    monkeypatch.setattr(fe_module.orchestrator, "extract_features", lambda **_: _FakeFeatures())
    monkeypatch.setattr(fe_module.campaign_engine, "extract", lambda *a, **k: _FakeCampaignFeatures())


def test_persisted_next_technique_comes_from_label_not_from_predicted_next(monkeypatch):
    """Regression for Bug A: next_technique must reflect the value passed
    explicitly (the label_generator-computed ground truth), never
    campaign_context.predicted_next."""
    _install_fake_feature_engines(monkeypatch)
    from ml.feature_extractors import extractor

    record = extractor.extract(
        campaign_context=_FakeCampaignContext(),
        current_attack_id="T1110",
        attacker_ip="1.2.3.4",
        event_id="evt-1",
        severity="Low",
        attributed_actor="unknown",
        prediction_correct=1,
        attribution_correct=1,
        next_technique="T1110.001",  # the real ground truth
    )

    assert record.next_technique == "T1110.001"
    assert record.next_technique != _FakeCampaignContext.predicted_next


def test_extract_accepts_none_prediction_correct(monkeypatch):
    _install_fake_feature_engines(monkeypatch)
    from ml.feature_extractors import extractor

    record = extractor.extract(
        campaign_context=_FakeCampaignContext(),
        current_attack_id="T1110",
        attacker_ip="1.2.3.4",
        event_id="evt-1",
        severity="Low",
        attributed_actor="unknown",
        prediction_correct=None,
        attribution_correct=1,
        next_technique="",
    )
    assert record.prediction_correct is None


# ---------------------------------------------------------------- Bug B: dataset_validator NA handling

def _make_record(**overrides):
    from ml.feature_schema import CampaignDatasetRecord

    values = {f.name: 0 for f in fields(CampaignDatasetRecord)}
    values.update({
        "campaign_id": "CAMP_TEST", "attacker_ip": "1.2.3.4", "victim_ip": "5.6.7.8",
        "severity": "Low", "attributed_actor": "unknown", "next_technique": "",
        "attribution_correct": 1, "prediction_correct": None,
    })
    values.update(overrides)
    return CampaignDatasetRecord(**values)


def test_validator_preserves_none_prediction_correct():
    from ml.dataset_validator import validator

    record = validator.validate(_make_record(prediction_correct=None))
    assert record.prediction_correct is None


def test_validator_still_coerces_real_prediction_correct_values():
    from ml.dataset_validator import validator

    record = validator.validate(_make_record(prediction_correct=1))
    assert record.prediction_correct == 1

    record = validator.validate(_make_record(prediction_correct=-5))
    assert record.prediction_correct == 0  # still clamped to >= 0, unchanged behavior for real values


# ---------------------------------------------------------------- predict_next_readonly (pure lookup)

class _FakeTransitionResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeTransitionSession:
    def __init__(self, rows):
        self._rows = rows

    def run(self, query, **kwargs):
        return _FakeTransitionResult(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeTransitionDriver:
    def __init__(self, rows):
        self._rows = rows

    def session(self):
        return _FakeTransitionSession(self._rows)


def test_predict_next_readonly_returns_none_when_no_edges(monkeypatch):
    import prediction_engine

    monkeypatch.setattr(prediction_engine, "driver", _FakeTransitionDriver([]))
    assert prediction_engine.predict_next_readonly("T1595") is None


def test_predict_next_readonly_respects_min_observations(monkeypatch):
    import prediction_engine

    monkeypatch.setattr(prediction_engine, "MIN_TRANSITION_OBSERVATIONS", 10)
    monkeypatch.setattr(
        prediction_engine, "driver",
        _FakeTransitionDriver([{"next": "T1110", "count": 3}]),
    )
    assert prediction_engine.predict_next_readonly("T1110.001") is None


def test_predict_next_readonly_respects_min_confidence(monkeypatch):
    import prediction_engine

    monkeypatch.setattr(prediction_engine, "MIN_TRANSITION_OBSERVATIONS", 1)
    monkeypatch.setattr(prediction_engine, "MIN_PREDICTION_CONFIDENCE", 90)
    monkeypatch.setattr(
        prediction_engine, "driver",
        _FakeTransitionDriver([{"next": "T1110", "count": 5}, {"next": "T1078", "count": 5}]),
    )
    assert prediction_engine.predict_next_readonly("T1110.001") is None  # 50% confidence < 90%


def test_predict_next_readonly_returns_best_candidate_with_no_side_effects(monkeypatch):
    import prediction_engine

    monkeypatch.setattr(prediction_engine, "MIN_TRANSITION_OBSERVATIONS", 1)
    monkeypatch.setattr(prediction_engine, "MIN_PREDICTION_CONFIDENCE", 50)
    monkeypatch.setattr(
        prediction_engine, "driver",
        _FakeTransitionDriver([{"next": "T1110", "count": 6}]),
    )

    def _fail_if_called(*a, **k):
        raise AssertionError("predict_next_readonly must not write to campaign_manager")

    monkeypatch.setattr(prediction_engine.campaign_manager, "update_prediction", _fail_if_called)

    result = prediction_engine.predict_next_readonly("T1110.001")
    assert result["predicted"] == "T1110"
    assert result["confidence"] == 100.0


# ---------------------------------------------------------------- dataset_builder technique_sequence threading

def test_dataset_builder_defaults_technique_sequence_to_single_element(monkeypatch):
    """When no technique_sequence is supplied, build() must default to a
    single-element sequence (yielding NA labels) rather than fabricating
    a transition it never observed."""
    import ml.dataset_builder as db_module
    from ml.label_generator import CampaignLabels

    captured = {}

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        return CampaignLabels(
            risk_score=0, severity="Low", attributed_actor="unknown",
            next_technique="", prediction_correct=None, attribution_correct=1,
        )

    monkeypatch.setattr(db_module.generator, "generate", _fake_generate)
    monkeypatch.setattr(db_module.extractor, "extract", lambda **kwargs: _make_record())
    monkeypatch.setattr(db_module.validator, "validate", lambda record: record)
    monkeypatch.setattr(db_module.writer, "append", lambda record: None)

    db_module.builder.build(
        campaign_context=_FakeCampaignContext(),
        final_attack_id="T1110",
        attacker_ip="1.2.3.4",
        event_id="evt-1",
        attributed_actor="unknown",
    )

    assert captured["technique_sequence"] == ["T1110"]

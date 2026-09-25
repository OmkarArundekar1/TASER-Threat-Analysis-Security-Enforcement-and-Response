from threat_qualification import (
    ThreatQualificationEngine,
    NOT_THREAT,
    SUSPICIOUS,
    QUALIFIED_THREAT,
)


class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Incident:
    def __init__(self, **kw):
        defaults = dict(
            campaign_id="CAMP_1", attacker_ip="1.2.3.4", technique="T1110",
            timestamp="2026-01-01T00:00:00Z", cti=None,
        )
        defaults.update(kw)
        self.__dict__.update(defaults)


engine = ThreatQualificationEngine()


def test_qualified_threat_with_all_fields_present_may_publish():
    incident = _Incident(cti=_Stub(threat_classification=QUALIFIED_THREAT, score=90.0))
    result = engine.qualify(incident)
    assert result.classification == QUALIFIED_THREAT
    assert result.may_publish_to_misp is True
    assert all(c.passed for c in result.checks)


def test_not_threat_never_may_publish_even_with_all_other_fields_present():
    incident = _Incident(cti=_Stub(threat_classification=NOT_THREAT, score=5.0))
    result = engine.qualify(incident)
    assert result.may_publish_to_misp is False
    failed = [c.name for c in result.checks if not c.passed]
    assert "threat_classification_qualified" in failed


def test_suspicious_never_automatically_may_publish():
    incident = _Incident(cti=_Stub(threat_classification=SUSPICIOUS, score=30.0))
    result = engine.qualify(incident)
    assert result.classification == SUSPICIOUS
    assert result.may_publish_to_misp is False


def test_missing_cti_confidence_defaults_to_not_threat_not_fabricated():
    incident = _Incident(cti=None)
    result = engine.qualify(incident)
    assert result.classification == NOT_THREAT
    assert result.cti_score is None
    assert result.may_publish_to_misp is False


def test_qualified_threat_blocked_by_missing_ioc():
    incident = _Incident(cti=_Stub(threat_classification=QUALIFIED_THREAT, score=90.0), attacker_ip=None)
    result = engine.qualify(incident)
    assert result.may_publish_to_misp is False
    assert any(c.name == "has_ioc" and not c.passed for c in result.checks)


def test_qualified_threat_blocked_by_unknown_mitre_technique():
    incident = _Incident(cti=_Stub(threat_classification=QUALIFIED_THREAT, score=90.0), technique="UNKNOWN")
    result = engine.qualify(incident)
    assert result.may_publish_to_misp is False
    assert any(c.name == "valid_mitre_provenance" and not c.passed for c in result.checks)


def test_qualified_threat_blocked_if_already_published():
    incident = _Incident(cti=_Stub(threat_classification=QUALIFIED_THREAT, score=90.0))
    result = engine.qualify(incident, already_published=True)
    assert result.may_publish_to_misp is False
    assert any(c.name == "not_already_published" and not c.passed for c in result.checks)


def test_reason_string_names_the_failed_checks():
    incident = _Incident(cti=_Stub(threat_classification=NOT_THREAT, score=1.0), attacker_ip=None)
    result = engine.qualify(incident)
    assert "threat_classification_qualified" in result.reason
    assert "has_ioc" in result.reason


def test_to_dict_is_json_serializable_shape():
    incident = _Incident(cti=_Stub(threat_classification=QUALIFIED_THREAT, score=90.0))
    result = engine.qualify(incident).to_dict()
    assert result["classification"] == QUALIFIED_THREAT
    assert isinstance(result["checks"], list)
    assert result["checks"][0]["name"] == "threat_classification_qualified"


def test_missing_incident_fields_do_not_crash_the_engine():
    class _Bare:
        pass
    result = engine.qualify(_Bare())
    assert result.classification == NOT_THREAT
    assert result.may_publish_to_misp is False

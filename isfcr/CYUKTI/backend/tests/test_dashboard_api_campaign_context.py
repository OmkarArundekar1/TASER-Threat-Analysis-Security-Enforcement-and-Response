"""
Regression test for a real bug found running the investigation/severity
endpoints against real campaigns: _load_campaign_context never populated
CampaignContext.first_seen/last_seen, so campaign_feature_engine's
temporal_similarity() crashed with
'unsupported operand type(s) for -: datetime and NoneType' on every
real campaign — a live 500 on both /api/investigate and
/api/ml/predict/severity.
"""

from datetime import datetime, timezone

import dashboard_api


class _FakeNeo4jDateTime:
    def __init__(self, dt):
        self._dt = dt

    def to_native(self):
        return self._dt


class _FakeRecord(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row


class _FakeSession:
    def __init__(self, campaign_props, attacker_ip, victim_ip, techniques,
                 predicted_next=None, prediction_confidence=None):
        self._row = {
            "c": _FakeRecord(campaign_props),
            "attacker_ip": attacker_ip,
            "victim_ip": victim_ip,
            "techniques": techniques,
            "predicted_next": predicted_next,
            "prediction_confidence": prediction_confidence,
        }

    def run(self, *args, **kwargs):
        return _FakeResult(self._row)


def test_load_campaign_context_populates_first_and_last_seen():
    first = _FakeNeo4jDateTime(datetime(2026, 8, 1, tzinfo=timezone.utc))
    last = _FakeNeo4jDateTime(datetime(2026, 8, 2, tzinfo=timezone.utc))
    session = _FakeSession(
        campaign_props={"risk_score": 90.0, "last_technique": "T1110.001",
                         "first_seen": first, "last_seen": last},
        attacker_ip="1.2.3.4", victim_ip="5.6.7.8", techniques=["T1110.001"],
    )

    context = dashboard_api._load_campaign_context(session, "camp-1")

    assert context.first_seen == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert context.last_seen == datetime(2026, 8, 2, tzinfo=timezone.utc)


def test_load_campaign_context_populates_predicted_next():
    session = _FakeSession(
        campaign_props={"risk_score": 90.0, "last_technique": "T1110.001"},
        attacker_ip="1.2.3.4", victim_ip="5.6.7.8", techniques=["T1110.001"],
        predicted_next="T1110", prediction_confidence=100.0,
    )

    context = dashboard_api._load_campaign_context(session, "camp-1")

    assert context.predicted_next == "T1110"
    assert context.prediction_confidence == 100.0


def test_load_campaign_context_missing_campaign_returns_none():
    class _EmptySession:
        def run(self, *a, **k):
            return _FakeResult(None)

    assert dashboard_api._load_campaign_context(_EmptySession(), "nope") is None

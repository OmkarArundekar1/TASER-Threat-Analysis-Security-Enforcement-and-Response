"""
Tests for campaign_reconstruction.py, built after discovering 44
AttackEvent nodes orphaned by a historical gap in Campaign persistence.
Uses a fake Neo4j session (no live database needed) to verify the pure
planning logic: refuses to guess when attacker/victim are inconsistent,
is a no-op when a Campaign already exists, and computes risk_score via
the same corrected mapping as risk_recalculation.py.
"""

import campaign_reconstruction


class _FakeSingleResult:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row


class _FakeDataResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, existing_campaign, event_rows):
        self._existing_campaign = existing_campaign
        self._event_rows = event_rows
        self.merge_calls = []

    def run(self, query, **kwargs):
        if "RETURN e.event_id" in query:
            return _FakeDataResult(self._event_rows)
        if "RETURN c" in query:
            return _FakeSingleResult(self._existing_campaign)
        # any MERGE/write call — record it
        self.merge_calls.append((query, kwargs))
        return _FakeDataResult([])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    def __init__(self, existing_campaign=None, event_rows=None):
        self._existing_campaign = existing_campaign
        self._event_rows = event_rows or []
        self.last_session = None

    def session(self):
        self.last_session = _FakeSession(self._existing_campaign, self._event_rows)
        return self.last_session


def test_plan_reconstruction_is_none_when_campaign_already_exists(monkeypatch):
    driver = _FakeDriver(existing_campaign={"c": {"campaign_id": "CAMP_X"}})
    monkeypatch.setattr(campaign_reconstruction, "driver", driver)

    plan = campaign_reconstruction.plan_reconstruction("CAMP_X")
    assert plan is None


def test_plan_reconstruction_refuses_inconsistent_attacker_victim(monkeypatch):
    rows = [
        {"event_id": "e1", "attacker_ip": "1.1.1.1", "victim_ip": "v1",
         "attack_id": "T1595", "occurrences": 1, "first_seen": "t1", "last_seen": "t1"},
        {"event_id": "e2", "attacker_ip": "2.2.2.2", "victim_ip": "v1",  # different attacker!
         "attack_id": "T1595", "occurrences": 1, "first_seen": "t2", "last_seen": "t2"},
    ]
    driver = _FakeDriver(existing_campaign=None, event_rows=rows)
    monkeypatch.setattr(campaign_reconstruction, "driver", driver)

    import pytest
    with pytest.raises(ValueError, match="don't share a single attacker/victim pair"):
        campaign_reconstruction.plan_reconstruction("CAMP_MIXED")


def test_plan_reconstruction_computes_risk_score_via_corrected_mapping(monkeypatch):
    rows = [
        {"event_id": "e1", "attacker_ip": "1.1.1.1", "victim_ip": "v1",
         "attack_id": "T1595", "occurrences": 25, "first_seen": "t1", "last_seen": "t1"},
    ]
    driver = _FakeDriver(existing_campaign=None, event_rows=rows)
    monkeypatch.setattr(campaign_reconstruction, "driver", driver)

    plan = campaign_reconstruction.plan_reconstruction("CAMP_SOLO")

    assert plan is not None
    assert plan.attacker_ip == "1.1.1.1"
    assert plan.victim_ip == "v1"
    assert plan.event_ids == ["e1"]
    assert plan.risk_score == 10 * 25  # Reconnaissance tps * occurrences
    assert plan.last_technique == "T1595"


def test_plan_reconstruction_empty_events_returns_none(monkeypatch):
    driver = _FakeDriver(existing_campaign=None, event_rows=[])
    monkeypatch.setattr(campaign_reconstruction, "driver", driver)

    plan = campaign_reconstruction.plan_reconstruction("CAMP_EMPTY")
    assert plan is None


class _FakeIntegritySession:
    def __init__(self, total, linked, orphan_ids):
        self._total, self._linked, self._orphan_ids = total, linked, orphan_ids

    def run(self, query, **kwargs):
        if "count(e) AS n" in query and "HAS_EVENT" not in query:
            return _FakeSingleResult({"n": self._total})
        if "count(DISTINCT e) AS n" in query:
            return _FakeSingleResult({"n": self._linked})
        return _FakeDataResult([{"campaign_id": cid} for cid in self._orphan_ids])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeIntegrityDriver:
    def __init__(self, total, linked, orphan_ids):
        self._session = _FakeIntegritySession(total, linked, orphan_ids)

    def session(self):
        return self._session


def test_check_integrity_reports_clean_when_no_orphans(monkeypatch):
    monkeypatch.setattr(campaign_reconstruction, "driver", _FakeIntegrityDriver(117, 117, []))
    report = campaign_reconstruction.check_integrity()
    assert report.is_clean
    assert report.orphaned_events == 0


def test_check_integrity_reports_dirty_when_orphans_exist(monkeypatch):
    monkeypatch.setattr(campaign_reconstruction, "driver", _FakeIntegrityDriver(117, 73, ["CAMP_X", "CAMP_Y"]))
    report = campaign_reconstruction.check_integrity()
    assert not report.is_clean
    assert report.orphaned_events == 44
    assert report.orphan_campaign_ids == ["CAMP_X", "CAMP_Y"]

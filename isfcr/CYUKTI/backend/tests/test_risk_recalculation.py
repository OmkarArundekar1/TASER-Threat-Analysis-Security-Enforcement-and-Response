"""
Tests for risk_recalculation.py's compute_deltas() — verifies the
recomputation formula is correct and, critically, idempotent (running
it against its own output changes nothing further), which is what
makes it safe to run against real production data without risk of
double-counting.
"""

import risk_recalculation


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _FakeSession:
    def __init__(self, event_rows):
        self._event_rows = event_rows

    def run(self, query, **kwargs):
        return _FakeResult(self._event_rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    def __init__(self, event_rows):
        self._event_rows = event_rows

    def session(self):
        return _FakeSession(self._event_rows)


def test_compute_deltas_recomputes_tps_from_occurrences(monkeypatch):
    rows = [
        {"campaign_id": "C1", "old_campaign_risk": 8330.0, "event_id": "e1",
         "attack_id": "T1595", "occurrences": 833, "old_stage": "Reconnaissance", "old_tps": 8330},
    ]
    monkeypatch.setattr(risk_recalculation, "driver", _FakeDriver(rows))

    deltas = risk_recalculation.compute_deltas()

    assert len(deltas) == 1
    assert deltas[0].campaign_id == "C1"
    # T1595 -> Reconnaissance -> tps=10 per occurrence, 833 occurrences
    assert deltas[0].new_risk_score == 8330  # unchanged: T1595 was already correctly mapped
    assert deltas[0].events[0].new_stage == "Reconnaissance"


def test_compute_deltas_fixes_previously_invalid_stage(monkeypatch):
    rows = [
        {"campaign_id": "C2", "old_campaign_risk": 0.0, "event_id": "e2",
         "attack_id": "T1053.003", "occurrences": 5, "old_stage": "Cron abuse", "old_tps": 0},
    ]
    monkeypatch.setattr(risk_recalculation, "driver", _FakeDriver(rows))

    deltas = risk_recalculation.compute_deltas()

    event = deltas[0].events[0]
    assert event.old_stage == "Cron abuse"
    assert event.new_stage == "Privilege Escalation"
    assert event.old_tps == 0
    assert event.new_tps == 90 * 5  # Privilege Escalation weight * occurrences
    assert deltas[0].new_risk_score == 450


def test_compute_deltas_is_idempotent(monkeypatch):
    # simulate: apply the recalculation once, then compute deltas again
    # against the ALREADY-CORRECTED values — nothing should change further
    rows = [
        {"campaign_id": "C3", "old_campaign_risk": 450.0, "event_id": "e3",
         "attack_id": "T1053.003", "occurrences": 5, "old_stage": "Privilege Escalation", "old_tps": 450},
    ]
    monkeypatch.setattr(risk_recalculation, "driver", _FakeDriver(rows))

    deltas = risk_recalculation.compute_deltas()

    event = deltas[0].events[0]
    assert event.old_stage == event.new_stage
    assert event.old_tps == event.new_tps
    assert deltas[0].old_risk_score == deltas[0].new_risk_score


def test_compute_deltas_unmapped_technique_yields_zero_tps(monkeypatch):
    rows = [
        {"campaign_id": "C4", "old_campaign_risk": 100.0, "event_id": "e4",
         "attack_id": "T1059", "occurrences": 10, "old_stage": "Unknown", "old_tps": 0},
    ]
    monkeypatch.setattr(risk_recalculation, "driver", _FakeDriver(rows))

    deltas = risk_recalculation.compute_deltas()

    assert deltas[0].events[0].new_stage == "Unknown"
    assert deltas[0].events[0].new_tps == 0

"""
Step 2 regression tests: event-time correctness and deterministic replay
(research/STEP2_EVENT_TIME.md).

All pipeline tests drive the real process_alert() -> CampaignManager path
over tests.test_chain_learning_regression's stateful fake graph. Alerts are
processed back-to-back in milliseconds of wall-clock time while their event
times span minutes/hours: any decision still made on wall-clock time would
make these tests fail.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

import realtime_socgraph as rsg
from config import CAMPAIGN_TIMEOUT, DEDUP_WINDOW
from event_time import (
    POLICY_PROCESSING_TIME_FALLBACK, POLICY_STRICT, REASON_MALFORMED, REASON_MISSING,
    REASON_TIMEZONE_MISSING, SOURCE_ALERT, SOURCE_PROCESSING_TIME_FALLBACK,
    EventClock, EventTimeError, event_clock, parse_timestamp, resolve_event_time,
)
from replay.harness import comparable_view, load_alerts, replay, view_hash
from tests.test_chain_learning_regression import install_fake_pipeline

T0 = datetime(2026, 3, 11, 3, 27, 24, tzinfo=timezone.utc)


def _ts(dt: datetime) -> str:
    """Wazuh's own format: 2026-03-11T03:27:24.000+0000"""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}" + dt.strftime("%z")


def _alert(technique, when, src="10.0.0.5", dst="10.0.0.9", rule_id=None, raw_timestamp=None):
    rule = {"id": rule_id or f"9{abs(hash((technique, when))) % 10**5:05d}", "level": 10,
            "description": "test", "groups": []}
    if technique is not None:
        rule["mitre"] = {"id": [technique], "technique": [technique], "tactic": []}
    alert = {"rule": rule, "agent": {"id": "001", "name": "victim-host"},
             "data": {"src_ip": src, "dest_ip": dst}}
    if raw_timestamp is not None:
        alert["timestamp"] = raw_timestamp
    elif when is not None:
        alert["timestamp"] = _ts(when)
    return alert


@pytest.fixture(autouse=True)
def _fresh_clock():
    event_clock.reset()
    yield
    event_clock.reset()


@pytest.fixture
def pipe(monkeypatch):
    graph = install_fake_pipeline(monkeypatch)
    graph.manager.decision_log = []
    return graph


def _decisions(graph):
    return [d["decision"] for d in graph.manager.decision_log]


# ==========================================================================
# 1. Timestamp parsing, timezone handling, missing / malformed
# ==========================================================================

@pytest.mark.parametrize("raw", [
    "2026-03-11T03:27:24.000+0000",     # Wazuh alerts.json
    "2026-03-11T03:27:24.000Z",
    "2026-03-11T03:27:24+00:00",
    "2026-03-11T08:57:24.000+0530",     # same instant, IST
    "2026-03-10T22:27:24.000-0500",     # same instant, EST
])
def test_equivalent_timestamps_in_any_timezone_resolve_to_the_same_utc_instant(raw):
    assert parse_timestamp(raw) == T0
    assert parse_timestamp(raw).tzinfo == timezone.utc


@pytest.mark.parametrize("raw,reason", [
    (None, REASON_MISSING),
    ("", REASON_MISSING),
    ("   ", REASON_MISSING),
    ("not-a-time", REASON_MALFORMED),
    ("2026-13-45T99:99:99+0000", REASON_MALFORMED),
    (1710127644, REASON_MALFORMED),                     # epoch number: not silently guessed
    ("2026-03-11T03:27:24.000", REASON_TIMEZONE_MISSING),  # naive: never assumed UTC or local
])
def test_invalid_timestamps_are_classified(raw, reason):
    with pytest.raises(EventTimeError) as err:
        parse_timestamp(raw)
    assert err.value.reason == reason


def test_strict_policy_rejects_and_fallback_policy_marks_processing_time():
    now = datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc)
    alert = _alert("T1595", None)  # no timestamp at all

    with pytest.raises(EventTimeError):
        resolve_event_time(alert, POLICY_STRICT, now=now)

    fallback = resolve_event_time(alert, POLICY_PROCESSING_TIME_FALLBACK, now=now)
    assert fallback.value == now
    assert fallback.source == SOURCE_PROCESSING_TIME_FALLBACK
    assert fallback.fallback_reason == REASON_MISSING


def test_valid_timestamp_is_never_replaced_by_processing_time():
    now = datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc)
    resolved = resolve_event_time(_alert("T1595", T0), POLICY_PROCESSING_TIME_FALLBACK, now=now)
    assert resolved.value == T0 and resolved.source == SOURCE_ALERT


def test_at_timestamp_field_is_accepted_when_timestamp_is_absent():
    alert = _alert("T1595", None)
    alert["@timestamp"] = "2026-03-11T03:27:24.000Z"
    assert resolve_event_time(alert, POLICY_STRICT).value == T0


def test_unknown_policy_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError):
        resolve_event_time(_alert("T1595", T0), "whatever")


def test_future_timestamp_is_flagged_not_altered():
    now = T0
    resolved = resolve_event_time(_alert("T1595", T0 + timedelta(hours=1)), POLICY_STRICT, now=now)
    assert resolved.future_skew is True
    assert resolved.value == T0 + timedelta(hours=1)


# ==========================================================================
# 2. Pipeline: event time drives every lifecycle decision
# ==========================================================================

def test_pipeline_stores_alert_event_time_not_processing_time(pipe):
    rsg.process_alert(_alert("T1595", T0))
    (event,) = pipe.events
    assert event["event_time"] == T0
    assert event["event_time_source"] == SOURCE_ALERT
    (campaign,) = pipe.campaigns.values()
    assert campaign["first_seen"] == T0 and campaign["last_seen"] == T0


def test_events_within_the_inactivity_threshold_stay_in_one_campaign(pipe):
    rsg.process_alert(_alert("T1595", T0))
    rsg.process_alert(_alert("T1110", T0 + timedelta(seconds=CAMPAIGN_TIMEOUT - 1)))

    assert len(pipe.campaigns) == 1
    assert _decisions(pipe) == ["CREATE", "CONTINUE"]


def test_gap_exactly_equal_to_the_threshold_is_not_expired(pipe):
    rsg.process_alert(_alert("T1595", T0))
    rsg.process_alert(_alert("T1110", T0 + timedelta(seconds=CAMPAIGN_TIMEOUT)))
    assert len(pipe.campaigns) == 1


def test_events_beyond_the_threshold_expire_the_campaign_in_event_time(pipe):
    """Processed milliseconds apart in wall-clock time; 1 s past the
    threshold in event time -> the campaign expires."""
    pipe.score = 0.0  # no reopen: isolate expiry
    rsg.process_alert(_alert("T1595", T0))
    rsg.process_alert(_alert("T1110", T0 + timedelta(seconds=CAMPAIGN_TIMEOUT + 1)))

    first, second = pipe.campaigns.values()
    assert _decisions(pipe) == ["CREATE", "CLOSE_EXPIRED", "CREATE"]
    assert first["status"] == "INACTIVE"
    # closed_at = the event time at which it became inactive, not "now"
    assert first["closed_at"] == T0 + timedelta(seconds=CAMPAIGN_TIMEOUT)
    assert second["first_seen"] == T0 + timedelta(seconds=CAMPAIGN_TIMEOUT + 1)


def test_expired_campaign_becomes_a_reopen_candidate_without_background_maintenance(pipe):
    """Determinism: the alert path closes the expired campaign itself, so it is
    a reopen candidate exactly as if the maintenance thread had closed it
    first. (Before Step 2 it stayed ACTIVE and was skipped as a candidate.)"""
    pipe.score = 1.0  # reopen score >= REOPEN_THRESHOLD
    rsg.process_alert(_alert("T1595", T0))
    rsg.process_alert(_alert("T1110", T0 + timedelta(seconds=CAMPAIGN_TIMEOUT + 60)))

    assert len(pipe.campaigns) == 1
    assert _decisions(pipe) == ["CREATE", "CLOSE_EXPIRED", "REOPEN"]


def test_out_of_order_alert_joins_the_campaign_and_never_moves_last_seen_backwards(pipe):
    rsg.process_alert(_alert("T1595", T0 + timedelta(seconds=100)))
    trace = {}
    rsg.process_alert(_alert("T1110", T0 + timedelta(seconds=10)), trace=trace)

    assert trace["late"] is True
    (campaign,) = pipe.campaigns.values()
    assert campaign["last_seen"] == T0 + timedelta(seconds=100)
    context = next(iter(pipe.manager.cache.values()))
    assert context.last_seen == T0 + timedelta(seconds=100)


def test_late_alert_after_its_campaign_expired_attaches_to_the_current_campaign(pipe):
    """Documented limitation: a late alert is evaluated against the campaign
    state at the time it ARRIVES; it is not retroactively placed into an
    already-closed historical campaign."""
    pipe.score = 0.0
    rsg.process_alert(_alert("T1595", T0))
    rsg.process_alert(_alert("T1190", T0 + timedelta(seconds=500)))      # expires #1, creates #2
    pipe.score = 1.0
    trace = {}
    rsg.process_alert(_alert("T1110", T0 + timedelta(seconds=50)), trace=trace)  # late

    first, second = pipe.campaigns.values()
    assert trace["late"] is True
    assert trace["campaign_id"] == second["campaign_id"]


def test_identical_timestamps_keep_arrival_order(pipe):
    rsg.process_alert(_alert("T1595", T0))
    rsg.process_alert(_alert("T1110", T0))
    rsg.process_alert(_alert("T1078", T0))

    assert len(pipe.campaigns) == 1
    assert [e["mitre_id"] for e in pipe.events] == ["T1595", "T1110", "T1078"]
    assert pipe.edges == {("T1595", "T1110"): 1, ("T1110", "T1078"): 1}


def test_timezone_offsets_do_not_distort_the_inactivity_gap(pipe):
    """+05:30 and Z timestamps 60 s apart in real time: one campaign.
    (Read naively, the wall-clock digits differ by 5.5 h.)"""
    rsg.process_alert(_alert("T1595", None, raw_timestamp="2026-03-11T08:57:24.000+0530"))
    rsg.process_alert(_alert("T1110", None, raw_timestamp="2026-03-11T03:28:24.000Z"))
    assert len(pipe.campaigns) == 1


def test_missing_timestamp_strict_rejects_without_side_effects(pipe):
    trace = {}
    ok = rsg.process_alert(_alert("T1595", None), trace=trace, event_time_policy=POLICY_STRICT)
    assert ok is False
    assert trace == {"path": "REJECTED_EVENT_TIME", "event_time_error": REASON_MISSING}
    assert pipe.campaigns == {} and pipe.events == []


@pytest.mark.parametrize("raw", ["garbage", "2026-03-11T03:27:24"])
def test_malformed_or_naive_timestamp_strict_rejects(pipe, raw):
    trace = {}
    assert rsg.process_alert(_alert("T1595", None, raw_timestamp=raw), trace=trace,
                             event_time_policy=POLICY_STRICT) is False
    assert trace["path"] == "REJECTED_EVENT_TIME"
    assert pipe.events == []


def test_missing_timestamp_fallback_is_explicitly_marked_on_the_stored_event(pipe):
    trace = {}
    ok = rsg.process_alert(_alert("T1595", None), trace=trace,
                           event_time_policy=POLICY_PROCESSING_TIME_FALLBACK)
    assert ok is True
    assert trace["event_time_source"] == SOURCE_PROCESSING_TIME_FALLBACK
    assert trace["event_time_fallback_reason"] == REASON_MISSING
    assert pipe.events[0]["event_time_source"] == SOURCE_PROCESSING_TIME_FALLBACK


def test_unknown_path_uses_event_time_for_expiry(pipe):
    """An UNKNOWN alert after the campaign expired (event time) gets a new
    campaign shell instead of being attached to the stale one."""
    rsg.process_alert(_alert("T1595", T0))
    trace = {}
    rsg.process_alert(_alert(None, T0 + timedelta(seconds=CAMPAIGN_TIMEOUT + 5), rule_id="31101"),
                      trace=trace)
    first, second = pipe.campaigns.values()
    assert first["status"] == "INACTIVE"
    assert trace["path"] == "UNATTRIBUTED" and trace["campaign_id"] == second["campaign_id"]


def test_maintenance_uses_the_watermark_and_writes_a_deterministic_closed_at(pipe):
    rsg.process_alert(_alert("T1595", T0))
    rsg.process_alert(_alert("T1190", T0 + timedelta(seconds=1000), dst="10.0.0.10"))

    pipe.manager.expire_active_campaigns(reference_time=event_clock.watermark)

    first = next(c for c in pipe.campaigns.values() if c["victim_ip"] == "10.0.0.9")
    assert first["status"] == "INACTIVE"
    assert first["closed_at"] == T0 + timedelta(seconds=CAMPAIGN_TIMEOUT)


@pytest.mark.parametrize("maintenance_first", [True, False])
def test_closed_at_does_not_depend_on_whether_maintenance_or_the_alert_path_closes(monkeypatch, maintenance_first):
    """Found by the replay speed test: an UNKNOWN alert refreshes the campaign's
    Neo4j last_seen but not the in-memory one (pre-existing, audit B3). Before
    the sweep reorder, the DB sweep closed it at db_last_seen + timeout and the
    alert path at memory_last_seen + timeout -- whichever ran first won."""
    event_clock.reset()
    graph = install_fake_pipeline(monkeypatch)
    graph.score = 0.0
    rsg.process_alert(_alert("T1595", T0))
    rsg.process_alert(_alert(None, T0 + timedelta(seconds=10), rule_id="31101"))   # DB-only refresh
    later = T0 + timedelta(seconds=CAMPAIGN_TIMEOUT + 30)
    if maintenance_first:
        graph.manager.expire_active_campaigns(reference_time=later)
    rsg.process_alert(_alert("T1110", later))

    first = next(iter(graph.campaigns.values()))
    assert first["status"] == "INACTIVE"
    assert first["closed_at"] == T0 + timedelta(seconds=CAMPAIGN_TIMEOUT)


# ==========================================================================
# 3. Dedup window in event time
# ==========================================================================

@pytest.fixture
def dedup(monkeypatch):
    import dedup_engine as dedup_module
    monkeypatch.setattr(dedup_module, "find_recent_duplicate", lambda fingerprint, reference_time=None: None)
    return dedup_module.DeduplicationEngine()


def test_dedup_window_is_measured_in_event_time(dedup):
    args = ("1.2.3.4", "10.0.0.5", "T1110", "5716", "001")
    dup, fp = dedup.is_duplicate(*args, event_time=T0)
    dedup.register_event(fp, campaign_id="C", event_id="E", event_time=T0)

    assert dedup.is_duplicate(*args, event_time=T0 + timedelta(seconds=DEDUP_WINDOW - 1))[0] is True
    # sliding: last_seen moved to T0+89 s; 89+DEDUP_WINDOW+1 is outside it
    assert dedup.is_duplicate(*args, event_time=T0 + timedelta(seconds=2 * DEDUP_WINDOW + 1))[0] is False


def test_dedup_late_alert_is_a_duplicate_and_does_not_rewind_the_window(dedup):
    args = ("1.2.3.4", "10.0.0.5", "T1110", "5716", "001")
    _, fp = dedup.is_duplicate(*args, event_time=T0 + timedelta(seconds=100))
    dedup.register_event(fp, event_time=T0 + timedelta(seconds=100))
    assert dedup.is_duplicate(*args, event_time=T0 + timedelta(seconds=10))[0] is True
    assert dedup.get_metadata(fp)["last_seen"] == (T0 + timedelta(seconds=100)).timestamp()


# ==========================================================================
# 4. Deterministic replay: fresh databases, identical timestamps, speed
# ==========================================================================

def _scenario():
    """Two attacker->victim pairs, gaps on both sides of the threshold,
    identical timestamps, an UNKNOWN alert, a timestamp-less alert."""
    rows = []
    for i, t in enumerate(["T1595", "T1595.002", "T1110", "T1110.001", "T1078"]):
        rows.append(_alert(t, T0 + timedelta(seconds=30 * i), rule_id=f"8{i:04d}"))
    rows.append(_alert("T1059", T0 + timedelta(seconds=120), rule_id="81000"))          # same ts as row 4
    rows.append(_alert(None, T0 + timedelta(seconds=130), rule_id="31101"))
    rows.append(_alert("T1068", T0 + timedelta(seconds=600), rule_id="81001"))          # beyond timeout
    rows.append(_alert("T1190", T0 + timedelta(seconds=15), dst="10.0.0.10", rule_id="81002"))
    rows.append(_alert("T1210", T0 + timedelta(seconds=400), dst="10.0.0.10", rule_id="81003"))
    rows.append(_alert("T1083", None, rule_id="81004"))                                 # no timestamp
    return rows


def _write_jsonl(tmp_path, rows, name="alerts.json"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def _snapshot(graph):
    return {
        "campaigns": [
            {"campaign_id": c["campaign_id"], "attacker_ip": c["attacker_ip"], "victim_ip": c["victim_ip"],
             "status": c["status"], "first_seen": c["first_seen"].isoformat(),
             "last_seen": c["last_seen"].isoformat(),
             "closed_at": c.get("closed_at").isoformat() if c.get("closed_at") else None,
             "reopened_count": c["reopened_count"],
             "sequence": [e.get("mitre_id") or "UNKNOWN" for e in graph.events
                          if e["campaign_id"] == c["campaign_id"]]}
            for c in graph.campaigns.values()
        ],
        "transitions": [{"src": a, "dst": b, "count": n} for (a, b), n in sorted(graph.edges.items())],
        "operations": [],
    }


def _run(monkeypatch, path, **kw):
    """One replay against a brand-new fake database and fresh in-memory state."""
    event_clock.reset()
    graph = install_fake_pipeline(monkeypatch)
    graph.score = 0.0   # realistic: below both thresholds unless continuing
    loaded = load_alerts(path)
    manager = graph.manager

    def maintenance(reference_time):
        manager.expire_active_campaigns(reference_time=reference_time)

    result = replay(loaded.ordered, rsg.process_alert, manager, maintenance=maintenance, **kw)
    return loaded, comparable_view(result.traces, result.decisions, _snapshot(graph))


def test_loader_orders_by_event_time_then_file_position_and_records_rejections(tmp_path):
    loaded = load_alerts(_write_jsonl(tmp_path, _scenario()))
    times = [t for _, t, _ in loaded.ordered]
    assert times == sorted(times)
    tied = [pos for pos, t, _ in loaded.ordered if t == T0 + timedelta(seconds=120)]
    assert tied == sorted(tied) == [4, 5]   # identical timestamps keep file order
    assert loaded.rejected == [{"position": 10, "reason": REASON_MISSING, "raw": None}]


def test_repeated_replay_against_fresh_databases_is_identical(monkeypatch, tmp_path):
    path = _write_jsonl(tmp_path, _scenario())
    _, first = _run(monkeypatch, path)
    _, second = _run(monkeypatch, path)

    assert first == second
    assert view_hash(first) == view_hash(second)
    assert len(first["campaigns"]) >= 2   # the scenario really exercises expiry


def test_replay_speed_and_maintenance_timing_do_not_change_outcomes(monkeypatch, tmp_path):
    """Unpaced vs paced (real sleeps) with a background maintenance thread
    firing every millisecond against the event-time watermark."""
    path = _write_jsonl(tmp_path, _scenario())
    _, unpaced = _run(monkeypatch, path)
    _, paced = _run(monkeypatch, path, speed=20000.0, maintenance_interval=0.001)
    _, paced_slower = _run(monkeypatch, path, speed=5000.0, maintenance_interval=0.0005)

    assert unpaced == paced == paced_slower


def test_input_order_in_the_file_does_not_matter_for_distinct_timestamps(monkeypatch, tmp_path):
    rows = [r for r in _scenario() if "timestamp" in r]
    distinct = {r["timestamp"] for r in rows}
    rows = [r for r in rows if sum(x["timestamp"] == r["timestamp"] for x in rows) == 1]
    assert len(rows) == len(distinct) - 1   # dropped the one tied pair
    _, forward = _run(monkeypatch, _write_jsonl(tmp_path, rows, "fwd.json"))
    _, backward = _run(monkeypatch, _write_jsonl(tmp_path, list(reversed(rows)), "bwd.json"))
    # positions differ by construction; everything decided must not
    strip = lambda v: {**v, "assignments": [{k: x for k, x in a.items() if k != "position"}
                                             for a in v["assignments"]]}
    assert strip(forward) == strip(backward)

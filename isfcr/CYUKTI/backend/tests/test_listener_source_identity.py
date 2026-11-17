"""Source identity and duplicate suppression in the listener: Phase 2.6C.

The defect: generate_alert_id hashed a few fields and missed Suricata's src_ip and dest_ip, so accept_alert refused distinct
alerts, and the saved offset then moved past them. On the 9,015-alert sample that lost 70.

The real listener source runs here, with everything that could reach Neo4j, MISP or campaign logic stubbed, and ALERT_FILE,
offset.dat and logs in a temporary directory. start_listener() is never called. Real process death uses the isolated
subprocess. Each test states what happened to each alert: accepted, a legitimate duplicate, retained for retry, or failed.
Tests named known_gap pin behaviour that's still a real limitation, so that a change to it gets noticed.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import importlib.util
import json
import logging
import os
import signal
import sys
import threading
import time
import types
from pathlib import Path
from queue import Empty, Full, Queue

import pytest

import neo4j_idempotency_ledger as ledger_module
import realtime_socgraph as rsg
from tests._isolated_listener import run_isolated_listener
from tests.test_realtime_mitre_integration import _native_mapped_alert, _no_mitre_alert, stub_everything  # noqa: F401

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LISTENER_DIR = os.path.join(BACKEND_DIR, "listener")
SAMPLE = Path(BACKEND_DIR).parent / "data" / "raw_logs" / "alerts.json"
MANAGER = "manager-01"
needs_sample = pytest.mark.skipif(not SAMPLE.exists(), reason="the archived alert sample isn't on this machine")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


identity = _load("listener_utils_for_tests", os.path.join(LISTENER_DIR, "utils.py"))


@pytest.fixture
def listener(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("listener test reached a stubbed dependency")

    stubs = {
        "realtime_socgraph": {"process_alert": forbidden},
        "dedup_engine": {"duplicate_flush_worker": object(), "duplicate_buffer": object()},
        "neo4j_client": {"batch_update_duplicates": forbidden},
        "campaign_manager": {"campaign_manager": object()},
        "operation_manager": {"operation_manager": object()},
        "event_time": {"event_clock": object()},
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "utils", _load("listener_utils_under_test", os.path.join(LISTENER_DIR, "utils.py")))
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    module = _load("wazuh_listener_under_test", os.path.join(LISTENER_DIR, "wazuh_listener.py"))
    module.__file__ = str(tmp_path / "wazuh_listener.py")
    module.ALERT_FILE = str(tmp_path / "alerts.json")
    module.alert_queue = Queue(maxsize=100)
    (tmp_path / "alerts.json").write_text("")
    yield module
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


def _alert(number=1, *, manager=MANAGER, **fields):
    alert = {"timestamp": "2026-03-11T00:00:00.000+0000", "rule": {"id": "86601", "firedtimes": 1}, "agent": {"id": "001"},
             "location": "/var/log/suricata/eve.json", "id": f"1773187200.{number}", "data": {}}
    if manager is not None:
        alert["manager"] = {"name": manager}
    alert.update(fields)
    return alert


def _write(listener, *alerts):
    with open(listener.ALERT_FILE, "a") as handle:
        for alert in alerts:
            handle.write((alert if isinstance(alert, str) else json.dumps(alert) + "\n"))


def _handler(listener):
    handler = listener.AlertFileHandler()
    handler.offset = 0
    return handler


def _read(listener, handler=None):
    handler = handler or _handler(listener)
    handler.read_new_lines()
    return handler


def _drain(listener):
    alerts = []
    while True:
        try:
            alerts.append(listener.alert_queue.get_nowait())
        except Empty:
            return alerts


def _delivered(listener, *alerts):
    _write(listener, *alerts)
    _read(listener)
    return _drain(listener)


def _at_end(listener, handler):
    return handler.offset == os.path.getsize(listener.ALERT_FILE)


def _saved(tmp_path):
    return int((tmp_path / "offset.dat").read_text())


# ================================================================ D1: extraction

def test_a_valid_alert_gives_the_documented_identity_exactly():
    found = identity.extract_source_identity(_alert(number=1234, manager="manager-01"))
    assert found.valid and found.status == identity.VALID
    assert (found.manager, found.source_id) == ("manager-01", "1773187200.1234")
    assert found.source_key == "wazuh:manager-01/1773187200.1234"
    assert identity.request_key(found.source_key, "duplicate_event.update") == "cyukti.v1|wazuh:manager-01/1773187200.1234|duplicate_event.update"


@pytest.mark.parametrize("raw", ["1773187200.10", "1773187200.1", "0001.10", "1.0", "01.00"])
def test_the_id_is_kept_as_the_exact_string_and_never_parsed(raw):
    found = identity.extract_source_identity(_alert(id=raw))
    assert found.source_id == raw and found.source_key.endswith("/" + raw)


def test_ids_that_are_numerically_equal_stay_different():
    keys = {identity.extract_source_identity(_alert(id=raw)).source_key for raw in ("1.1", "1.10", "01.1")}
    assert len(keys) == 3


@pytest.mark.parametrize("change, status", [
    ({"id": None}, identity.MISSING_ID),
    ({"id": ""}, identity.EMPTY_ID),
    ({"id": 1773187200.1234}, identity.MALFORMED_ID),
    ({"id": 1773187200}, identity.MALFORMED_ID),
    ({"id": True}, identity.MALFORMED_ID),
    ({"id": ["1.2"]}, identity.MALFORMED_ID),
    ({"id": {"v": "1.2"}}, identity.MALFORMED_ID),
    ({"id": " 1.2"}, identity.MALFORMED_ID),
    ({"id": "1.2 "}, identity.MALFORMED_ID),
    ({"id": "1.2\n"}, identity.MALFORMED_ID),
    ({"id": "1"}, identity.MALFORMED_ID),
    ({"id": "1."}, identity.MALFORMED_ID),
    ({"id": ".2"}, identity.MALFORMED_ID),
    ({"id": "1.2.3"}, identity.MALFORMED_ID),
    ({"id": "a.b"}, identity.MALFORMED_ID),
    ({"id": "-1.2"}, identity.MALFORMED_ID),
    ({"id": "١.٢"}, identity.MALFORMED_ID),
    ({"id": "1." + "2" * 64}, identity.MALFORMED_ID),
    ({"manager": None}, identity.MISSING_MANAGER),
    ({"manager": {}}, identity.MISSING_MANAGER),
    ({"manager": {"name": None}}, identity.MISSING_MANAGER),
    ({"manager": {"name": ""}}, identity.MISSING_MANAGER),
    ({"manager": "manager-01"}, identity.MALFORMED_MANAGER),
    ({"manager": ["manager-01"]}, identity.MALFORMED_MANAGER),
    ({"manager": {"name": 5}}, identity.MALFORMED_MANAGER),
    ({"manager": {"name": "\ud800"}}, identity.MALFORMED_MANAGER),
    ({"manager": {"name": "m" * 201}}, identity.MALFORMED_MANAGER),
], ids=lambda value: repr(value)[:36] if isinstance(value, dict) else value)
def test_unusable_identity_is_named_and_gives_no_key(change, status):
    alert = _alert()
    alert.update(change)
    found = identity.extract_source_identity(alert)
    assert found.status == status and not found.valid
    assert found.source_key is None and (found.manager, found.source_id) == (None, None)


def test_a_missing_id_field_is_missing_rather_than_malformed():
    alert = _alert()
    del alert["id"]
    assert identity.extract_source_identity(alert).status == identity.MISSING_ID


def test_a_missing_manager_field_is_never_filled_from_anything_else():
    """An agent name or a hostname doesn't identify the manager, even where they happen to match it."""
    alert = _alert(manager=None, agent={"id": "000", "name": "manager-01"}, hostname="manager-01", location="manager-01", cluster={"name": "wazuh", "node": "manager-01"})
    found = identity.extract_source_identity(alert)
    assert found.status == identity.MISSING_MANAGER and found.source_key is None


@pytest.mark.parametrize("block", [{}, {"name": None}, {"name": ""}], ids=["empty block", "null name", "empty name"])
def test_a_manager_block_without_a_name_is_never_filled_from_the_agent(block):
    alert = _alert(agent={"id": "000", "name": "manager-01"}, hostname="manager-01")
    alert["manager"] = block
    found = identity.extract_source_identity(alert)
    assert found.status == identity.MISSING_MANAGER and found.source_key is None


@pytest.mark.parametrize("manager, encoded", [("a/b", "a%2Fb"), ("a|b", "a%7Cb"), ("a:b", "a%3Ab"), ("a b", "a%20b"), ("é", "%C3%A9"), ("A-._~z", "A-._~z"), ("%2F", "%252F")])
def test_a_manager_name_cannot_forge_the_key_separators(manager, encoded):
    found = identity.extract_source_identity(_alert(number=1, manager=manager))
    assert found.manager == manager, "the original is kept as it was"
    assert found.source_key == f"wazuh:{encoded}/1773187200.1"


def test_names_that_differ_only_by_encoding_stay_different():
    assert (identity.extract_source_identity(_alert(manager="a/b")).source_key != identity.extract_source_identity(_alert(manager="a%2Fb")).source_key)


def test_extraction_is_deterministic_and_reads_only_its_two_fields():
    base = _alert(number=7)
    other = {**base, "timestamp": "other", "rule": {"id": "x"}, "agent": {"id": "y"}, "location": "z", "data": {"src_ip": "10.0.0.1"}, "full_log": "changed"}
    assert identity.extract_source_identity(base) == identity.extract_source_identity(other) == identity.extract_source_identity(json.loads(json.dumps(base)))


def test_keys_are_acceptable_to_the_ledger_as_they_are():
    for manager in ("manager-01", "a b", "é☃", "m" * 66):
        key = identity.request_key(identity.extract_source_identity(_alert(manager=manager)).source_key, "duplicate_event.update")
        assert ledger_module._checked_key(key) == key


@pytest.mark.parametrize("source_key, operation", [("wazuh:m/1.2", ""), ("wazuh:m/1.2", "a|b"), ("wazuh:m/1.2", "a b"), ("wazuh:m/1.2", None), (None, "op"), ("sha:abc", "op")])
def test_a_request_key_refuses_a_bad_source_or_operation(source_key, operation):
    with pytest.raises(ValueError):
        identity.request_key(source_key, operation)


def test_one_alert_has_one_source_key_and_a_request_key_per_operation():
    key = identity.extract_source_identity(_alert()).source_key
    assert identity.request_key(key, "a") != identity.request_key(key, "b")
    assert identity.request_key(key, "a").split("|")[1] == key


def test_the_identity_on_the_alert_can_be_serialized_and_leaves_the_rest_as_it_was():
    alert = _alert(data={"src_ip": "10.0.0.1"})
    original = json.loads(json.dumps(alert))
    identity.assign_source_identity(alert)
    assert json.loads(json.dumps(alert["source_identity"])) == {"status": "VALID", "manager": MANAGER, "id": "1773187200.1", "source_key": f"wazuh:{MANAGER}/1773187200.1"}
    assert alert["alert_id"] == alert["source_identity"]["source_key"]
    assert {k: v for k, v in alert.items() if k not in ("alert_id", "source_identity")} == original


# ================================================================ D2: field names

def _lifted_extract_iocs():
    tree = ast.parse(Path(BACKEND_DIR, "realtime_socgraph.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "extract_iocs")
    namespace = {}
    exec(compile(ast.Module([function], []), "realtime_socgraph.py", "exec"), namespace)
    return namespace["extract_iocs"]


@pytest.mark.parametrize("data, top, expected, conflict", [
    ({"srcip": "1.1.1.1", "dstip": "2.2.2.2"}, {}, ("1.1.1.1", "2.2.2.2"), False),
    ({"src_ip": "1.1.1.1", "dest_ip": "2.2.2.2"}, {}, ("1.1.1.1", "2.2.2.2"), False),
    ({"src_ip": "1.1.1.1", "dstip": "2.2.2.2"}, {}, ("1.1.1.1", "2.2.2.2"), False),
    ({"srcip": "1.1.1.1", "dest_ip": "2.2.2.2"}, {}, ("1.1.1.1", "2.2.2.2"), False),
    ({"src_ip": "1.1.1.1", "srcip": "1.1.1.1", "dest_ip": "2.2.2.2", "dstip": "2.2.2.2"}, {}, ("1.1.1.1", "2.2.2.2"), False),
    ({"src_ip": "1.1.1.1", "srcip": "9.9.9.9", "dest_ip": "2.2.2.2", "dstip": "8.8.8.8"}, {}, ("1.1.1.1", "2.2.2.2"), True),
    ({"src_ip": "1.1.1.1", "srcip": "9.9.9.9"}, {}, ("1.1.1.1", None), True),
    ({"dest_ip": "2.2.2.2", "dstip": "8.8.8.8"}, {}, (None, "2.2.2.2"), True),
    ({"src_ip": "", "srcip": "9.9.9.9"}, {}, ("9.9.9.9", None), False),
    ({}, {"srcip": "1.1.1.1", "dstip": "2.2.2.2"}, ("1.1.1.1", "2.2.2.2"), False),
    ({}, {"src_ip": "1.1.1.1", "srcip": "9.9.9.9"}, ("1.1.1.1", None), True),
    ({"srcip": "1.1.1.1"}, {"srcip": "9.9.9.9"}, ("1.1.1.1", None), True),
    ({}, {}, (None, None), False),
], ids=["srcip", "src_ip", "mixed one", "mixed two", "both agreeing", "both conflicting", "conflict source only", "conflict destination only",
        "empty first form", "top level", "top level conflict", "data beats top level", "none"])
def test_network_fields_follow_one_order_and_values_are_unchanged(data, top, expected, conflict):
    alert = {"data": data, **top}
    assert identity.network_endpoints(alert) == (*expected, conflict)


@pytest.mark.parametrize("data", [{"src_ip": "1.1.1.1", "srcip": "9.9.9.9", "dest_ip": "2.2.2.2", "dstip": "8.8.8.8"}, {"srcip": "1.1.1.1", "dstip": "2.2.2.2"},
                                  {"src_ip": "1.1.1.1", "dest_ip": "2.2.2.2"}, {"src_ip": "", "srcip": "1.1.1.1", "dest_ip": "", "dstip": "2.2.2.2"}])
def test_the_order_is_the_one_that_decides_the_attacker_and_victim(data):
    source, destination, _ = identity.network_endpoints({"data": data})
    assert (source, destination) == _lifted_extract_iocs()({"data": data, "agent": {}})


@pytest.mark.parametrize("bad", [None, "text", 5, ["x"]])
def test_odd_data_blocks_do_not_stop_the_helpers(bad):
    alert = {"data": bad, "rule": bad, "agent": bad, "timestamp": 5, "location": None}
    assert identity.network_endpoints(alert) == (None, None, False)
    assert len(identity.content_fingerprint(alert)) == 64


def test_the_old_collision_no_longer_decides_anything(listener):
    """The heart of it. Same timestamp, rule, agent and location, and Suricata's spelling that the old key never read."""
    a = _alert(1, data={"src_ip": "10.0.0.1", "dest_ip": "10.0.0.2"})
    b = _alert(2, data={"src_ip": "10.0.0.3", "dest_ip": "10.0.0.4"})
    assert identity.content_fingerprint(a) != identity.content_fingerprint(b), "the corrected fingerprint separates these"
    c = _alert(3, data={"src_ip": "10.0.0.1", "dest_ip": "10.0.0.2"})
    assert identity.content_fingerprint(a) == identity.content_fingerprint(c), "and these share one"
    assert [x["id"] for x in _delivered(listener, a, b, c)] == [a["id"], b["id"], c["id"]], "yet all three are delivered, by source identity"


# ================================================================ D2: duplicate behaviour through the real reader

@pytest.mark.parametrize("first, second, delivered", [
    (_alert(1), _alert(1), 1),
    (_alert(1, manager="manager-01"), _alert(1, manager="manager-02"), 2),
    (_alert(1), _alert(2), 2),
    (_alert(1, data={"srcip": "1.1.1.1", "dstip": "2.2.2.2"}), _alert(2, data={"srcip": "1.1.1.1", "dstip": "2.2.2.2"}), 2),
    (_alert(1, data={"src_ip": "1.1.1.1", "dest_ip": "2.2.2.2"}), _alert(2, data={"src_ip": "1.1.1.1", "dest_ip": "2.2.2.2"}), 2),
    (_alert(1, data={"src_ip": "1.1.1.1", "dest_ip": "2.2.2.2"}), _alert(2, data={"srcip": "1.1.1.1", "dstip": "2.2.2.2"}), 2),
    (_alert(1, data={"src_ip": "1.1.1.1", "srcip": "9.9.9.9"}), _alert(2, data={"src_ip": "1.1.1.1", "srcip": "8.8.8.8"}), 2),
    (_alert(1, data={"src_ip": "1.1.1.1"}), _alert(2, data={"src_ip": "3.3.3.3"}), 2),
    (_alert(1, rule={"id": "86601", "firedtimes": 1}), _alert(2, rule={"id": "86601", "firedtimes": 2}), 2),
    (_alert(1, rule={"id": "86601", "firedtimes": 1}), _alert(1, rule={"id": "86601", "firedtimes": 2}), 1),
    (_alert(1, data={"src_ip": "1.1.1.1"}), _alert(1, data={"src_ip": "3.3.3.3"}), 1),
    (_alert(1, timestamp="2026-03-11T00:00:00.000+0000"), _alert(1, timestamp="2026-03-11T00:00:09.000+0000"), 1),
], ids=["same manager and id, replay", "same id, different managers", "different ids, identical content", "srcip form", "src_ip form", "mixed forms",
        "both forms present", "different attacker", "different firedtimes", "same identity, firedtimes differ", "same identity, attacker differs", "same identity, timestamp differs"])
def test_source_identity_alone_decides_what_is_a_repeat(listener, first, second, delivered):
    _write(listener, first, second)
    handler = _read(listener)
    assert len(_drain(listener)) == delivered
    assert _at_end(listener, handler)


@pytest.mark.parametrize("change", [{"id": None}, {"id": ""}, {"id": "not-an-id"}, {"id": 5}, {"manager": None}, {"manager": {"name": ""}}],
                         ids=["missing", "empty", "malformed", "not a string", "no manager", "empty manager"])
def test_without_identity_identical_alerts_are_all_delivered_never_collapsed(listener, change):
    alert = _alert()
    alert.update(change)
    delivered = _delivered(listener, alert, alert, alert)
    assert len(delivered) == 3
    assert all(a["alert_id"] is None and not a["source_identity"]["source_key"] for a in delivered)
    assert listener.seen_alerts == {}, "nothing is recorded, so nothing can be refused later"
    assert sum(listener.identity_unavailable.values()) == 3


def test_an_alert_with_no_id_field_at_all_is_delivered(listener):
    alert = _alert()
    del alert["id"]
    assert len(_delivered(listener, alert, alert)) == 2
    assert listener.identity_unavailable == {"MISSING_ID": 2}


def test_alerts_with_and_without_identity_do_not_interfere(listener):
    no_id = _alert(id=None)
    delivered = _delivered(listener, _alert(1), no_id, _alert(1), no_id, _alert(2))
    assert [a["id"] for a in delivered] == ["1773187200.1", None, None, "1773187200.2"]


def test_a_refused_alert_was_already_accepted_under_that_identity(listener, tmp_path):
    _write(listener, _alert(1), _alert(1))
    refused = []
    real = listener.accept_alert

    def watched(alert):
        was_there = alert["alert_id"] in listener.seen_alerts
        answer = real(alert)
        if not answer:
            refused.append(alert["alert_id"])
            assert was_there
        return answer
    listener.accept_alert = watched
    handler = _read(listener)
    assert refused == [f"wazuh:{MANAGER}/1773187200.1"]
    assert len(_drain(listener)) == 1, "the one accepted reached the queue"
    assert _saved(tmp_path) == handler.offset == os.path.getsize(listener.ALERT_FILE)


def test_identity_travels_with_the_alert_through_the_queue_to_process_alert(listener):
    real = _alert(4)
    _write(listener, real)
    _read(listener)
    seen = []
    listener.process_alert = lambda alert: seen.append(alert) or True

    def stop_after_one():
        while not seen:
            time.sleep(0.005)
        listener.stop_event.set()
    threading.Thread(target=stop_after_one, daemon=True).start()
    listener.queue_worker()
    assert seen[0]["source_identity"] == {"status": "VALID", "manager": MANAGER, "id": real["id"], "source_key": f"wazuh:{MANAGER}/{real['id']}"}
    assert seen[0]["id"] == real["id"] and seen[0]["manager"] == real["manager"]


def test_unavailable_identity_is_reported_counted_and_quotes_nothing(listener, caplog):
    caplog.set_level(logging.INFO)
    secret_manager = "secret-manager-name"
    _delivered(listener, _alert(id="oops", manager=secret_manager, data={"src_ip": "10.9.9.9"}))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "MALFORMED_ID" in warnings[0] and "occurrence 1" in warnings[0]
    assert secret_manager not in caplog.text and "10.9.9.9" not in caplog.text and "oops" not in caplog.text


def test_a_flood_of_unavailable_identity_is_logged_sparingly_and_counted_fully(listener, caplog):
    caplog.set_level(logging.WARNING)
    listener.alert_queue = Queue(maxsize=3000)
    _write(listener, *[_alert(id=None)] * 2500)
    _read(listener)
    assert listener.identity_unavailable["MISSING_ID"] == 2500
    assert len(listener.alert_queue.queue) == 2500
    assert sum("no usable source identity" in r.getMessage() for r in caplog.records) == 3, "the 1st, 1000th and 2000th"


def test_valid_alerts_never_log_their_key(listener, caplog):
    caplog.set_level(logging.INFO)
    _delivered(listener, _alert(1), _alert(1))
    key = f"wazuh:{MANAGER}/1773187200.1"
    assert MANAGER not in caplog.text and "1773187200.1" not in caplog.text
    assert hashlib.sha256(key.encode()).hexdigest()[:12] in caplog.text


def test_conflicting_spellings_are_reported_without_their_values(listener, caplog):
    caplog.set_level(logging.WARNING)
    delivered = _delivered(listener, _alert(1, data={"src_ip": "10.1.1.1", "srcip": "10.2.2.2"}))
    assert len(delivered) == 1 and delivered[0]["data"] == {"src_ip": "10.1.1.1", "srcip": "10.2.2.2"}, "both originals stay on the alert"
    assert "disagreeing" in caplog.text and "10.1.1.1" not in caplog.text


# ================================================================ attributed and unattributed paths

@pytest.mark.parametrize("make, expected, other", [
    (_native_mapped_alert, "create_attack_event", "create_unattributed_attack_event"),
    (_no_mitre_alert, "create_unattributed_attack_event", "create_attack_event"),
], ids=["attributed", "unattributed"])
def test_the_identity_fields_do_not_change_routing(stub_everything, make, expected, other):  # noqa: F811
    plain_trace, tagged_trace = {}, {}
    assert rsg.process_alert(make(), trace=plain_trace) is True
    plain = dict(stub_everything[expected])
    assert stub_everything[other] is None
    stub_everything[expected] = None
    tagged = make()
    tagged.update(id="1773187200.1", manager={"name": MANAGER})
    identity.assign_source_identity(tagged)
    assert rsg.process_alert(tagged, trace=tagged_trace) is True
    assert stub_everything[other] is None
    assert plain_trace["path"] == tagged_trace["path"]
    variable = {"event_id"}
    assert {k: v for k, v in stub_everything[expected].items() if k not in variable} == {k: v for k, v in plain.items() if k not in variable}
    assert "source_identity" not in json.dumps(stub_everything[expected], default=str)


def test_a_duplicate_on_the_unattributed_path_still_updates_its_event(stub_everything, monkeypatch):  # noqa: F811
    updates = []
    monkeypatch.setattr(rsg.dedup_engine, "is_duplicate", lambda **kw: (True, "fp-1"))
    monkeypatch.setattr(rsg.dedup_engine, "get_metadata", lambda fingerprint: {"event_id": "E1", "campaign_id": "C1"})
    monkeypatch.setattr(rsg, "update_duplicate_event", lambda **kw: updates.append(kw))
    alert = _no_mitre_alert()
    alert.update(id="1773187200.1", manager={"name": MANAGER})
    identity.assign_source_identity(alert)
    trace = {}
    assert rsg.process_alert(alert, trace=trace) is True
    assert trace["path"] == "DUPLICATE_UNATTRIBUTED"
    assert [u["event_id"] for u in updates] == ["E1"] and updates[0]["tps"] == 0
    assert "source_identity" not in updates[0]["investigation_payload"]


def test_a_duplicate_on_the_attributed_path_still_goes_to_the_buffer(stub_everything, monkeypatch):  # noqa: F811
    added = []
    monkeypatch.setattr(rsg.dedup_engine, "is_duplicate", lambda **kw: (True, "fp-1"))
    monkeypatch.setattr(rsg.dedup_engine, "get_metadata", lambda fingerprint: {"event_id": "E1", "campaign_id": "C1"})
    monkeypatch.setattr(rsg.duplicate_buffer, "add_duplicate", lambda **kw: added.append(kw) or True)
    alert = _native_mapped_alert()
    alert.update(id="1773187200.1", manager={"name": MANAGER})
    identity.assign_source_identity(alert)
    trace = {}
    assert rsg.process_alert(alert, trace=trace) is True
    assert trace["path"] == "DUPLICATE" and [a["event_id"] for a in added] == ["E1"]


# ================================================================ reading: what each line becomes

def test_a_normal_complete_read_accepts_all_and_saves_the_end(listener, tmp_path):
    _write(listener, _alert(1), _alert(2), _alert(3))
    handler = _read(listener)
    assert [a["id"] for a in _drain(listener)] == ["1773187200.1", "1773187200.2", "1773187200.3"]
    assert _at_end(listener, handler) and _saved(tmp_path) == handler.offset
    assert (tmp_path / "offset.dat").read_text().strip().isdigit(), "the same plain format"
    assert not (tmp_path / "offset.dat.tmp").exists()


def test_a_partial_final_line_is_retained_and_delivered_once_complete(listener, tmp_path):
    whole = json.dumps(_alert(2))
    _write(listener, _alert(1), whole[:20])
    handler = _read(listener)
    assert [a["id"] for a in _drain(listener)] == ["1773187200.1"], "the partial line isn't accepted or dropped"
    position = handler.offset
    assert position == len(json.dumps(_alert(1))) + 1 and _saved(tmp_path) == position
    _write(listener, whole[20:] + "\n")
    handler.read_new_lines()
    assert [a["id"] for a in _drain(listener)] == ["1773187200.2"]
    assert _at_end(listener, handler)


def test_a_partial_line_ending_in_a_brace_is_also_retained(listener):
    _write(listener, '{"a": {"b": 1}')
    handler = _read(listener)
    assert _drain(listener) == [] and handler.offset == 0


@pytest.mark.parametrize("line, kind", [("{not json}\n", "malformed"), ('{"id": "1.2", "trunc\n', "malformed"), ('{"id": "1.2",\n', "malformed"), ("[1, 2]\n", "not_an_object"),
                                        ("5\n", "not_an_object"), ('"text"\n', "not_an_object"), ("null\n", "malformed")],
                         ids=["ends in a brace", "truncated string", "truncated object", "array", "number", "string", "null"])
def test_a_complete_line_that_can_never_be_an_alert_is_failed_explicitly_and_does_not_stop_the_reader(listener, tmp_path, line, kind):
    """A truncated line ending in a newline used to stop the reader for good, and a JSON non-object raised on every poll."""
    _write(listener, _alert(1), line, _alert(2))
    handler = _read(listener)
    assert [a["id"] for a in _drain(listener)] == ["1773187200.1", "1773187200.2"], "the alerts on both sides are accepted"
    assert listener.unreadable_lines == {kind: 1}
    assert _at_end(listener, handler) and _saved(tmp_path) == handler.offset


def test_a_blank_line_is_passed_over(listener):
    assert [a["id"] for a in _delivered(listener, "\n", _alert(1), "   \n", _alert(2))] == ["1773187200.1", "1773187200.2"]


def test_known_gap_a_line_that_is_not_utf8_stops_the_reader_at_that_point_and_loses_nothing_read_so_far(listener, tmp_path, caplog):
    """Failed explicitly, by a logged exception on each poll, and retained: nothing behind it is delivered until someone acts.
    The sample has no such line. Not fixed here: replacing bytes would alter alert content."""
    _write(listener, _alert(1))
    with open(listener.ALERT_FILE, "ab") as handle:
        handle.write(b'{"id": "\xff\xfe"}\n')
    _write(listener, _alert(2))
    handler = _handler(listener)
    for _ in range(2):
        handler.read_new_lines()
    delivered = _drain(listener)
    assert delivered == [] or [a["id"] for a in delivered] == ["1773187200.1"]
    assert not _at_end(listener, handler)
    assert "Failure" in caplog.text


# ================================================================ queue failures

class _Refusing:
    def __init__(self, error):
        self.error = error

    def put_nowait(self, alert):
        raise self.error


@pytest.mark.parametrize("error", [Full(), RuntimeError("boom")], ids=["queue full", "other failure"])
@pytest.mark.parametrize("alert", [_alert(1), _alert(id=None)], ids=["with identity", "without identity"])
def test_a_queue_failure_retains_the_alert_and_a_retry_accepts_it_once(listener, tmp_path, error, alert):
    _write(listener, alert)
    listener.alert_queue = _Refusing(error)
    handler = _read(listener)
    assert handler.offset == 0 and _saved(tmp_path) == 0, "the offset doesn't pass an alert the queue refused"
    assert listener.seen_alerts == {}
    listener.alert_queue = Queue(maxsize=10)
    handler.read_new_lines()
    assert len(_drain(listener)) == 1
    assert _at_end(listener, handler)


def test_a_full_queue_stops_at_the_first_refused_alert_with_no_alert_passed_over(listener, tmp_path):
    listener.alert_queue = Queue(maxsize=2)
    _write(listener, _alert(1), _alert(id=None), _alert(3), _alert(4))
    handler = _handler(listener)
    received = []
    for _ in range(4):
        handler.read_new_lines()
        received += _drain(listener)
    assert [a["id"] for a in received] == ["1773187200.1", None, "1773187200.3", "1773187200.4"], "each once, in order"
    assert _at_end(listener, handler)


# ================================================================ offset persistence

def test_the_offset_file_is_replaced_in_one_step_and_a_failure_leaves_the_old_value(listener, tmp_path, monkeypatch):
    _write(listener, _alert(1))
    handler = _read(listener)
    first = handler.offset
    assert _saved(tmp_path) == first
    _drain(listener)
    _write(listener, _alert(2))
    real = os.replace

    def refuses(source, target):
        raise OSError("injected")
    monkeypatch.setattr(os, "replace", refuses)
    handler.read_new_lines()
    assert _saved(tmp_path) == first, "still a whole, valid, older value. Older means re-read, never skipped"
    assert handler.offset > first
    monkeypatch.setattr(os, "replace", real)
    handler.read_new_lines()
    assert _saved(tmp_path) == handler.offset, "and the next poll saves it, though no new line arrived"


def test_the_offset_is_not_rewritten_when_it_has_not_moved(listener, tmp_path, monkeypatch):
    _write(listener, _alert(1))
    handler = _read(listener)
    calls = []
    monkeypatch.setattr(listener, "_save_offset", lambda path, offset: calls.append(offset))
    handler.read_new_lines()
    handler.read_new_lines()
    assert calls == []


def test_a_first_start_saves_its_starting_point_at_once(listener, tmp_path):
    _write(listener, _alert(1))
    handler = listener.AlertFileHandler()
    assert _saved(tmp_path) == handler.offset == os.path.getsize(listener.ALERT_FILE), "the existing rule: start at the end"


def test_a_first_start_that_cannot_save_still_starts(listener, tmp_path, monkeypatch):
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("injected")))
    assert listener.AlertFileHandler().offset == 0


@pytest.mark.parametrize("content, expected", [("7", 7), ("0", 0)])
def test_the_existing_offset_format_is_still_read(listener, tmp_path, content, expected):
    _write(listener, _alert(1))
    (tmp_path / "offset.dat").write_text(content)
    assert listener.AlertFileHandler().offset == expected


@pytest.mark.parametrize("content", ["", "garbage", "-5"], ids=["empty", "garbage", "negative"])
def test_known_gap_an_unusable_offset_file_still_means_the_end_of_the_log(listener, tmp_path, content):
    """Existing rule, kept. It skips whatever was unread. The atomic save means this code can't now produce such a file."""
    _write(listener, _alert(1))
    (tmp_path / "offset.dat").write_text(content)
    assert listener.AlertFileHandler().offset == os.path.getsize(listener.ALERT_FILE)


def test_an_offset_past_the_end_is_brought_back_to_the_end(listener, tmp_path):
    _write(listener, _alert(1))
    (tmp_path / "offset.dat").write_text("999999")
    assert listener.AlertFileHandler().offset == os.path.getsize(listener.ALERT_FILE)


# ================================================================ replay and restart in one process

def test_replaying_a_file_in_one_process_recognizes_every_identity(listener):
    _write(listener, _alert(1), _alert(2))
    handler = _read(listener)
    assert len(_drain(listener)) == 2
    handler.offset = 0
    handler.read_new_lines()
    assert _drain(listener) == [], "both were legitimate duplicates"
    assert _at_end(listener, handler)


def test_repeated_timestamps_are_not_repeats(listener):
    same = "2026-03-11T00:00:00.000+0000"
    assert len(_delivered(listener, *[_alert(n, timestamp=same) for n in range(1, 21)])) == 20


def test_suppression_ends_when_the_cache_forgets(listener):
    """It's bounded. After ALERT_CACHE_TTL or MAX_ALERT_CACHE, a replayed alert is delivered again."""
    _write(listener, _alert(1))
    handler = _read(listener)
    _drain(listener)
    key = f"wazuh:{MANAGER}/1773187200.1"
    listener.seen_alerts[key] = time.time() - listener.ALERT_CACHE_TTL - 1
    handler.offset = 0
    handler.read_new_lines()
    assert len(_drain(listener)) == 1
    listener.MAX_ALERT_CACHE = 1
    _write(listener, _alert(2))
    handler.read_new_lines()
    _drain(listener)
    assert list(listener.seen_alerts) == [f"wazuh:{MANAGER}/1773187200.2"], "the oldest was evicted"


# ================================================================ rotation, with a controlled fixture

def _rotate(listener, *new_alerts):
    os.replace(listener.ALERT_FILE, listener.ALERT_FILE + ".1")
    open(listener.ALERT_FILE, "w").close()
    _write(listener, *new_alerts)


def test_rotation_to_a_smaller_file_starts_it_from_the_beginning(listener):
    _write(listener, *[_alert(n) for n in range(1, 6)])
    handler = _read(listener)
    _drain(listener)
    _rotate(listener, _alert(6))
    handler.read_new_lines()
    assert [a["id"] for a in _drain(listener)] == ["1773187200.6"]
    assert _at_end(listener, handler)


def test_known_gap_a_new_file_already_larger_than_the_old_offset_is_read_from_the_wrong_place(listener):
    """The offset can't tell one file from another, so alerts before it are skipped. Fixing it means recording the file's
    identity, which changes the offset format, so it's reported rather than changed."""
    _write(listener, _alert(1))
    handler = _read(listener)
    _drain(listener)
    position = handler.offset
    _rotate(listener, *[_alert(n) for n in range(2, 7)])
    handler.read_new_lines()
    delivered = {a["id"] for a in _drain(listener)}
    assert handler.offset > position
    assert "1773187200.2" not in delivered, "the first alert of the new file was skipped"
    assert _at_end(listener, handler)


def test_known_gap_alerts_appended_just_before_rotation_are_not_read(listener):
    """Written after the last poll, then rotated away. The reader never looks at the old file again."""
    _write(listener, *[_alert(n) for n in range(1, 4)])
    handler = _read(listener)
    _drain(listener)
    _write(listener, _alert(4))
    _rotate(listener, _alert(5))
    handler.read_new_lines()
    assert "1773187200.4" not in {a["id"] for a in _drain(listener)}


# ================================================================ real process death

_RESTART = """
import json
handler = wl.AlertFileHandler()
handler.read_new_lines()
delivered = []
while not wl.alert_queue.empty():
    delivered.append(wl.alert_queue.get_nowait()["id"])
print(json.dumps({"delivered": delivered, "offset": handler.offset, "end": os.path.getsize(wl.ALERT_FILE)}))
"""


def _alerts_file(tmp_path, numbers, name="alerts.json"):
    with open(tmp_path / name, "a") as handle:
        for number in numbers:
            handle.write(json.dumps(_alert(number)) + "\n")


def _died(result):
    assert result.returncode == -signal.SIGKILL, result.stderr[-800:]


def _restart(tmp_path):
    result = run_isolated_listener(_RESTART, cwd=str(tmp_path), timeout=60)
    assert result.returncode == 0, result.stderr[-800:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _ids(numbers):
    return [f"1773187200.{n}" for n in numbers]


def test_killed_while_reading_nothing_unread_is_skipped_and_what_was_queued_is_delivered_again(tmp_path):
    _alerts_file(tmp_path, [1, 2, 3])
    (tmp_path / "offset.dat").write_text("0")
    code = """
    import signal
    real, count = wl.accept_alert, [0]
    def dying(alert):
        real(alert)
        count[0] += 1
        if count[0] == 2:
            os.kill(os.getpid(), signal.SIGKILL)
        return True
    wl.accept_alert = dying
    wl.AlertFileHandler().read_new_lines()
    """
    _died(run_isolated_listener(code, cwd=str(tmp_path), timeout=60))
    assert (tmp_path / "offset.dat").read_text() == "0"
    seen = _restart(tmp_path)
    assert seen["delivered"] == _ids([1, 2, 3]), "alert 3 was never read, and it isn't skipped. Alerts 1 and 2 are redelivered"
    assert seen["offset"] == seen["end"]


def test_killed_after_queue_submission_before_the_offset_is_saved_alerts_are_redelivered_not_lost(tmp_path):
    _alerts_file(tmp_path, [1, 2])
    (tmp_path / "offset.dat").write_text("0")
    code = """
    import signal
    wl._save_offset = lambda path, offset: os.kill(os.getpid(), signal.SIGKILL)
    wl.AlertFileHandler().read_new_lines()
    """
    _died(run_isolated_listener(code, cwd=str(tmp_path), timeout=60))
    assert (tmp_path / "offset.dat").read_text() == "0"
    assert _restart(tmp_path)["delivered"] == _ids([1, 2])


def test_killed_in_the_middle_of_saving_the_offset_leaves_a_valid_older_value(tmp_path):
    _alerts_file(tmp_path, [1, 2])
    (tmp_path / "offset.dat").write_text("0")
    code = """
    import signal
    os.replace = lambda source, target: os.kill(os.getpid(), signal.SIGKILL)
    wl.AlertFileHandler().read_new_lines()
    """
    _died(run_isolated_listener(code, cwd=str(tmp_path), timeout=60))
    assert (tmp_path / "offset.dat").read_text() == "0", "not truncated, which would have meant skipping to the end"
    assert (tmp_path / "offset.dat.tmp").exists()
    assert _restart(tmp_path)["delivered"] == _ids([1, 2]), "and the leftover temporary file does no harm"


def test_known_gap_killed_after_the_offset_is_saved_alerts_still_in_the_queue_are_lost(tmp_path):
    """The queue is memory. This is D10, and fixing it needs the durable-queue design that this phase was told not to build."""
    _alerts_file(tmp_path, [1, 2])
    (tmp_path / "offset.dat").write_text("0")
    code = """
    import signal
    handler = wl.AlertFileHandler()
    handler.read_new_lines()
    assert wl.alert_queue.qsize() == 2
    os.kill(os.getpid(), signal.SIGKILL)
    """
    _died(run_isolated_listener(code, cwd=str(tmp_path), timeout=60))
    assert _restart(tmp_path)["delivered"] == []


def test_killed_after_a_first_start_what_arrives_later_is_not_skipped(tmp_path):
    _alerts_file(tmp_path, [1])
    code = """
    import signal
    wl.AlertFileHandler()
    os.kill(os.getpid(), signal.SIGKILL)
    """
    _died(run_isolated_listener(code, cwd=str(tmp_path), timeout=60))
    _alerts_file(tmp_path, [2])
    assert _restart(tmp_path)["delivered"] == _ids([2])


def test_a_clean_restart_at_the_saved_offset_delivers_only_what_is_new(tmp_path):
    _alerts_file(tmp_path, [1, 2])
    (tmp_path / "offset.dat").write_text("0")
    assert _restart(tmp_path)["delivered"] == _ids([1, 2])
    _alerts_file(tmp_path, [3])
    assert _restart(tmp_path)["delivered"] == _ids([3])
    assert _restart(tmp_path)["delivered"] == []


def test_restart_replay_of_the_whole_file_delivers_each_alert_again_because_the_cache_is_memory(tmp_path):
    _alerts_file(tmp_path, [1, 2])
    (tmp_path / "offset.dat").write_text("0")
    _restart(tmp_path)
    (tmp_path / "offset.dat").write_text("0")
    assert _restart(tmp_path)["delivered"] == _ids([1, 2]), "no loss, and no suppression across a restart: at-least-once in this window"


# ================================================================ the actual concurrency: one reader, one worker

def test_the_real_reader_and_worker_with_a_small_queue_deliver_every_alert_once_in_order(listener):
    listener.alert_queue = Queue(maxsize=5)
    numbers = list(range(1, 1001))
    _write(listener, *[_alert(n) if n % 50 else _alert(id=None) for n in numbers])
    seen = []
    listener.process_alert = lambda alert: seen.append(alert["id"]) or True
    worker = threading.Thread(target=listener.queue_worker, daemon=True)
    worker.start()
    handler = _handler(listener)
    deadline = time.monotonic() + 60
    while not _at_end(listener, handler) and time.monotonic() < deadline:
        handler.read_new_lines()
        time.sleep(0.001)
    listener.alert_queue.join()
    listener.stop_event.set()
    worker.join(5)
    assert seen == [f"1773187200.{n}" if n % 50 else None for n in numbers]


def test_a_writer_appending_in_pieces_while_the_reader_polls_loses_and_repeats_nothing(listener):
    listener.alert_queue = Queue(maxsize=5000)
    lines = [json.dumps(_alert(n)) + "\n" for n in range(1, 301)]
    done = threading.Event()

    def writer():
        with open(listener.ALERT_FILE, "a") as handle:
            for line in lines:
                handle.write(line[:17])
                handle.flush()
                time.sleep(0.0005)
                handle.write(line[17:])
                handle.flush()
        done.set()

    thread = threading.Thread(target=writer)
    handler = _handler(listener)
    thread.start()
    while not done.is_set():
        handler.read_new_lines()
        time.sleep(0.002)
    thread.join()
    handler.read_new_lines()
    assert [a["id"] for a in _drain(listener)] == _ids(range(1, 301))
    assert listener.unreadable_lines == {}, "a partial line must be waited for, and never treated as malformed"
    assert _at_end(listener, handler)


def test_threads_presenting_one_identity_at_once_get_one_acceptance(listener):
    """The listener has one reader, so this is beyond its model. It checks the lock anyway, because the claim costs little."""
    answers = []
    barrier = threading.Barrier(16)

    def present():
        barrier.wait()
        alert = _alert(1)
        identity.assign_source_identity(alert)
        answers.append(listener.accept_alert(alert))

    threads = [threading.Thread(target=present) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert sorted(answers) == [False] * 15 + [True]
    assert listener.alert_queue.qsize() == 1


def test_there_is_one_reader_and_one_queue_worker():
    """If this changes, the concurrency tests above no longer describe the listener."""
    source = Path(LISTENER_DIR, "wazuh_listener.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    start = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "start_listener")
    targets = [k.value.id for n in ast.walk(start) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "Thread" for k in n.keywords if k.arg == "target"]
    assert targets == ["queue_worker", "maintenance_worker"]
    assert source.count("accept_alert(") == 2, "its definition and the reader's one call"


# ================================================================ the original defect, on the original sample

def _legacy_key(alert):
    """generate_alert_id before Phase 2.6C, kept here only to name the alerts it collapsed."""
    data = alert.get("data", {})
    raw = (f"{alert.get('timestamp', '')}|{alert.get('rule', {}).get('id', '')}|{alert.get('agent', {}).get('id', '')}|"
           f"{data.get('srcip') or alert.get('srcip', '')}|{data.get('dstip') or alert.get('dstip', '')}|{alert.get('location', '')}")
    return hashlib.sha256(raw.encode()).hexdigest()


@pytest.fixture(scope="module")
def sample():
    return [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _collapsed(alerts):
    groups = collections.defaultdict(list)
    for alert in alerts:
        groups[_legacy_key(alert)].append(alert)
    return {members[0]["id"]: members[1:] for members in groups.values() if len(members) > 1}


def _read_through(listener, tmp_path, handler=None):
    listener.alert_queue = Queue(maxsize=20000)
    handler = handler or _handler(listener)
    handler.read_new_lines()
    return handler, _drain(listener)


@needs_sample
def test_the_old_key_really_did_collapse_seventy_of_them(sample):
    collapsed = _collapsed(sample)
    refused = [a for members in collapsed.values() for a in members]
    assert (len(sample), len(collapsed), len(refused)) == (9015, 39, 70)
    assert len({a["id"] for a in refused}) == 70


@needs_sample
def test_the_sample_through_the_real_reader_all_accepted_then_all_recognized(listener, tmp_path, sample):
    Path(listener.ALERT_FILE).write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    handler, first = _read_through(listener, tmp_path)
    assert len(first) == 9015 and _at_end(listener, handler)
    assert len({a["id"] for a in first}) == 9015
    assert [a["id"] for a in first] == [a["id"] for a in sample], "in file order, each once"
    assert all(a["source_identity"]["status"] == "VALID" for a in first)
    assert listener.identity_unavailable == {} and listener.unreadable_lines == {}
    delivered = {a["id"]: a for a in first}
    refused_before = [a for members in _collapsed(sample).values() for a in members]
    assert all(a["id"] in delivered for a in refused_before), "all 70 are preserved"
    assert all(delivered[a["id"]]["source_identity"]["id"] == a["id"] for a in refused_before), "under their own ids"
    handler.offset = 0
    handler, second = _read_through(listener, tmp_path, handler)
    assert second == [] and _at_end(listener, handler), "the second pass recognizes all 9,015"


@needs_sample
def test_the_seven_with_a_different_attacker_or_victim_reach_the_pipeline_with_their_own_values(listener, tmp_path, sample):
    extract = _lifted_extract_iocs()
    groups = _collapsed(sample)
    by_id = {a["id"]: a for a in sample}
    different = [a for first, members in groups.items() for a in members if extract(a) != extract(by_id[first])]
    assert len(different) == 7
    Path(listener.ALERT_FILE).write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    _, first = _read_through(listener, tmp_path)
    delivered = {a["id"]: a for a in first}
    for alert in different:
        assert extract(delivered[alert["id"]]) == extract(alert)


@needs_sample
def test_sample_managers_and_ids_have_the_form_the_contract_requires(sample):
    found = [identity.extract_source_identity(a) for a in sample]
    assert all(f.valid and f.manager == a["manager"]["name"] and f.source_id == a["id"] for f, a in zip(found, sample))
    assert len({f.source_key for f in found}) == 9015

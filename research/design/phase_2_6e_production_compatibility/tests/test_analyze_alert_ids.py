"""Focused tests for the offline analyzer, on small synthetic files in a temporary directory. They touch no real alert file and no service."""
import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_alert_ids as a  # noqa: E402

STAMP = "2026-03-11T00:00:00.000+0000"
SECONDS = 1773187200


def alert(offset=10, manager="m1", seconds=SECONDS, **extra):
    row = {"timestamp": STAMP, "rule": {"id": "1"}, "agent": {"id": "001"}, "location": "l", "decoder": {"name": "d"}, "id": f"{seconds}.{offset}", "manager": {"name": manager}}
    row.update(extra)
    return row


def write(path, rows):
    with open(path, "wb") as handle:
        for row in rows:
            handle.write(row if isinstance(row, bytes) else (json.dumps(row) + "\n").encode())
    return path


def scan(tmp_path, *files):
    s = a.Scan()
    for index, rows in enumerate(files):
        s.current = f"f{index}"
        s.add_file(write(tmp_path / f"f{index}.json", rows), f"f{index}")
    return s, s.report()


@pytest.mark.parametrize("value, kind", [(None, "null_or_missing"), ("", "empty_string"), (5, "number"), (1.5, "number"), (True, "boolean"), (["1.2"], "list"), ("1.2", "string_seconds_dot_offset"), ("a.b", "string_other_form"), ("1", "string_other_form"), ("1.2 ", "string_other_form")])
def test_id_kinds(value, kind):
    assert a.id_kind(value) == kind


@pytest.mark.parametrize("value, accepted", [("1.2", True), ("abc-def", True), ("a/b", True), ("x" * 64, True), ("", False), ("x" * 65, False), ("a b", False), ("a|b", False), ("a\n", False), ("a\x00", False), (None, False), (5, False), (True, False)])
def test_the_tolerant_policy_refuses_what_could_forge_a_key(value, accepted):
    assert a.tolerant_policy_accepts(value) is accepted


def test_counts_and_manager_hashing_hide_the_name(tmp_path):
    _, r = scan(tmp_path, [alert(10), alert(20)])
    assert r["totals"]["alerts"] == 2 and r["id_kinds"] == {"string_seconds_dot_offset": 2}
    assert "m1" not in json.dumps(r["managers_by_short_hash"]) and a.short_hash("m1") in r["managers_by_short_hash"]


def test_same_id_in_another_manager_is_not_a_repeat_but_is_noted(tmp_path):
    _, r = scan(tmp_path, [alert(10, "m1"), alert(10, "m2")])
    assert r["unique_manager_and_id"] == 2 and r["same_id_under_different_manager"] == 1 and r["repeats_identical_line"] == 0


def test_an_identical_line_in_two_files_is_a_copy(tmp_path):
    _, r = scan(tmp_path, [alert(10)], [alert(10)])
    assert r["repeats_identical_line"] == 1 and r["totals"].get("duplicates_different_content", 0) == 0
    assert r["repeats_identical_line_where"] == {"f0 and f1": 1}


def test_the_same_id_with_different_content_is_a_real_collision(tmp_path):
    _, r = scan(tmp_path, [alert(10, location="a"), alert(10, location="b")])
    assert r["totals"]["duplicates_different_content"] == 1 and r["repeats_identical_line"] == 0
    assert r["repeats_different_content_examples"][0]["id"] == f"{SECONDS}.10"


def test_missing_and_malformed_are_counted_not_dropped(tmp_path):
    no_id = alert()
    del no_id["id"]
    no_manager = alert(11)
    del no_manager["manager"]
    _, r = scan(tmp_path, [no_id, alert(id=""), alert(id=7), no_manager, alert(12, manager=""), alert(13, manager=5)])
    assert r["id_kinds"]["null_or_missing"] == 1 and r["id_kinds"]["empty_string"] == 1 and r["id_kinds"]["number"] == 1
    assert r["manager_kinds"]["no_manager_block"] == 1 and r["manager_kinds"]["name_empty"] == 1 and r["manager_kinds"]["name_not_a_string"] == 1
    assert r["totals"]["alerts"] == 6


def test_bad_lines_are_classified(tmp_path):
    _, r = scan(tmp_path, [b'{"id": "\xff"}\n', b"{not json}\n", b"[1]\n", b"\n", alert(10), b'{"partial'])
    t = r["totals"]
    assert (t["not_utf8"], t["malformed_json"], t["not_an_object"], t["blank"], t["partial_final_line"], t["alerts"]) == (1, 1, 1, 1, 1, 1)


def test_offset_and_clock_regressions_are_seen(tmp_path):
    _, r = scan(tmp_path, [alert(20), alert(10), alert(10), alert(30, seconds=SECONDS - 5)])
    assert r["totals"]["offset_fell"] == 1 and r["totals"]["offset_repeated"] == 1 and r["totals"]["seconds_fell"] == 1


def test_the_id_seconds_are_compared_with_the_timestamp(tmp_path):
    _, r = scan(tmp_path, [alert(10), alert(11, seconds=SECONDS + 1)])
    assert r["id_seconds_minus_timestamp_clamped"] == {0: 1, 1: 1}


def test_the_old_key_merges_are_counted_per_file(tmp_path):
    _, r = scan(tmp_path, [alert(10), alert(11), alert(12)])
    assert r["totals"]["legacy_key_groups"] == 1 and r["totals"]["legacy_key_alerts_merged"] == 2


def test_gzip_files_are_read(tmp_path):
    path = tmp_path / "x.json.gz"
    with gzip.open(path, "wb") as handle:
        handle.write((json.dumps(alert(10)) + "\n").encode())
    s = a.Scan()
    s.current = "x"
    assert s.add_file(path, "x")["alerts"] == 1


def test_no_payload_field_reaches_the_report(tmp_path):
    _, r = scan(tmp_path, [alert(10, full_log="SECRET-PAYLOAD", data={"src_ip": "10.9.8.7"})])
    text = json.dumps(r)
    assert "SECRET-PAYLOAD" not in text and "10.9.8.7" not in text

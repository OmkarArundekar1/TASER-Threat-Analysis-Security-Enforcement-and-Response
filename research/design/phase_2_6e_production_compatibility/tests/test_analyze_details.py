"""Focused tests for the detail analysis, on small synthetic files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_details as d  # noqa: E402

STAMP = "2026-03-11T00:00:00.000+0000"


def alert(identifier, firedtimes=1, **extra):
    row = {"timestamp": STAMP, "rule": {"id": "1", "firedtimes": firedtimes}, "agent": {"id": "001"}, "location": "l", "decoder": {"name": "json"}, "id": identifier, "data": {}}
    row.update(extra)
    return row


def run(tmp_path, rows):
    path = tmp_path / "f.json"
    with open(path, "wb") as handle:
        for row in rows:
            handle.write(row if isinstance(row, bytes) else (json.dumps(row) + "\n").encode())
    return d.analyze_file(path)


def test_merges_that_differ_only_by_id_and_counter_are_separated_from_those_differing_in_content(tmp_path):
    r = run(tmp_path, [alert("5.1"), alert("5.2", firedtimes=2), alert("5.3", data={"src_ip": "10.0.0.1"})])
    assert r["merged"] == 2 and r["merged_identical_content_apart_from_id"] == 1 and r["merged_uses_src_ip_spelling"] == 1 and r["merged_has_no_network_field"] == 1
    assert r["merged_decoders"] == {"json": 2}


def test_seconds_drops_and_peak_are_measured(tmp_path):
    r = run(tmp_path, [alert("10.1"), alert("10.2"), alert("7.3"), alert("8.4")])
    assert r["id_seconds_drops"] == [3] and r["busiest_second"] == 2


def test_bad_lines_report_only_their_shape(tmp_path):
    r = run(tmp_path, [b'{"secret": "value\n', b'{not json}\n', b"[broken"])
    assert [(b["ends_in_newline"], b["ends_in_brace"]) for b in r["bad_lines"]] == [(True, False), (True, True), (False, False)]
    assert "secret" not in json.dumps(r)

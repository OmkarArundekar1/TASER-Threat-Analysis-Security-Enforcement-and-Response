"""Focused tests for the D6 marker contract, against the disposable instance only, using the same scenarios as run_experiments.py with fewer repetitions.
A scenario's checks read database state directly from the server, never only client return values.

Runs outside the backend's test tree, so its guard doesn't apply. marker_lib's own guard does, and it's tested here.
Without a disposable instance these fail at import, rather than skip: a skipped validation would read as a passed one."""
import ast
import socket
from pathlib import Path

import pytest

import marker_lib as m
import scenarios

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ctx():
    context = m.Context("test")
    yield context
    context.close()


def _assert_all(rows):
    assert rows
    failures = [(r["experiment"], r["case"], name) for r in rows for name, passed in r["checks"].items() if not passed]
    assert failures == []


# ---------------------------------------------------------------- isolation

@pytest.mark.parametrize("target", [("127.0.0.1", 7687), ("localhost", 7687), ("127.0.0.1", 7474), ("192.0.2.1", 17710), ("127.0.0.1", 17712)])
def test_the_guard_refuses_anything_but_the_experiments_ports(target):
    sock = socket.socket()
    try:
        with pytest.raises(m.TargetRefused):
            sock.connect(target)
    finally:
        sock.close()


def test_a_driver_for_the_standard_port_cannot_reach_it():
    driver = m.GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "unused"), connection_timeout=2)
    try:
        with pytest.raises(BaseException) as caught:
            driver.verify_connectivity()
        assert isinstance(caught.value, m.TargetRefused) or "outside the experiment" in repr(caught.value.__context__) or "outside" in str(caught.value)
    finally:
        driver.close()


def test_the_target_is_positively_the_disposable_instance(ctx):
    found = m.confirm_environment()
    assert found["daemon_id"] == m.EXPECTED_DAEMON and found["platform"].startswith("Docker Desktop")
    assert found["container"] == "cyukti-d6-marker-v1" and found["target"] == "bolt://127.0.0.1:17710"
    assert ("volume", "cyukti-d6-marker-v1-data", "/data") in found["mounts"]
    assert m.CONN["upstream_port"] != 7687
    m.verify_disposable(ctx.direct)


def test_the_database_refuses_to_be_stamped_when_it_holds_anything_unknown(ctx, monkeypatch):
    monkeypatch.setitem(m.CONN, "marker", "a-token-this-database-has-never-held")
    with pytest.raises(SystemExit, match="not this experiment's token"):
        m.initialize(ctx.direct)


def test_the_token_check_refuses_a_database_without_this_token(ctx, monkeypatch):
    monkeypatch.setitem(m.CONN, "marker", "a-token-this-database-has-never-held")
    with pytest.raises(SystemExit, match="token"):
        m.verify_disposable(ctx.direct)


def test_no_credential_reaches_the_results():
    password = m.CONN["password"]
    for path in (ROOT / "results").glob("*"):
        if path.is_file():
            assert password not in path.read_text(encoding="utf-8", errors="replace"), path.name


def test_the_experiment_imports_nothing_production_as_a_module():
    """Real import statements, read from the syntax tree. scenarios.py loads the standalone ledger, on a temporary file, for E9 only, and it's not listed here."""
    forbidden = {"neo4j_client", "config", "dotenv", "realtime_socgraph", "neo4j_write_journal", "dedup_engine", "campaign_manager"}
    for name in ("marker_lib.py", "marker_worker.py", "perf.py", "run_experiments.py", "scenarios.py"):
        imported = set()
        for node in ast.walk(ast.parse((ROOT / name).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & forbidden, name


# ---------------------------------------------------------------- the lookup, without a server

def test_a_failed_lookup_is_never_absent_whatever_the_failure():
    class Fails:
        def session(self, **kwargs):
            raise RuntimeError("anything at all")

    assert m.lookup(Fails(), "k").status == m.UNKNOWN


def test_a_lookup_past_its_deadline_is_unknown():
    import time

    class Stalls:
        def session(self, **kwargs):
            time.sleep(5)

    assert m.lookup(Stalls(), "k", deadline=0.2) == m.Lookup(m.UNKNOWN, error_type="ClientDeadline")


# ---------------------------------------------------------------- E1 to E10

def test_e1_atomic_commit(ctx):
    _assert_all(scenarios.e1_atomic_commit(ctx, reps=5))


def test_e2_atomic_rollback(ctx):
    _assert_all(scenarios.e2_atomic_rollback(ctx, reps=5))


def test_e3_lost_commit_response_is_recovered_and_the_control_differs(ctx):
    _assert_all(scenarios.e3_lost_commit_response(ctx, reps=5))


def test_e4_failure_before_commit(ctx):
    _assert_all(scenarios.e4_failure_before_commit(ctx, reps=3))


def test_e5_duplicate_key(ctx):
    _assert_all(scenarios.e5_duplicate_key(ctx, reps=5))


def test_e6_concurrent_duplicates(ctx):
    _assert_all(scenarios.e6_concurrent_duplicates(ctx, contenders=8, rounds=6))


def test_e7_different_keys_same_payload(ctx):
    _assert_all(scenarios.e7_different_keys_same_payload(ctx, reps=5))


def test_e8_lookup_outcomes(ctx):
    _assert_all(scenarios.e8_lookup(ctx))


def test_absence_is_not_proof(ctx):
    _assert_all(scenarios.absence(ctx, reps=3))


def test_e9_payload_mismatch(ctx):
    rows = scenarios.e9_payload_mismatch(ctx)
    _assert_all(rows)
    assert rows[0]["marker_can_tell_the_payloads_apart"] is False, "the proposed marker alone can't: the fuller one is needed"


def test_e10_retention(ctx):
    _assert_all(scenarios.e10_retention(ctx))

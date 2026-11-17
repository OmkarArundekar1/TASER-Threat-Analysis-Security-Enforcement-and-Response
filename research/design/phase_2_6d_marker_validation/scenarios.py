"""E1 to E10, and the supporting cases the D6 design listed. Each function returns rows of evidence with explicit checks on database state, read
directly from the server. The same functions serve run_experiments.py (more repetitions, results kept) and tests/ (fewer, asserted)."""
from __future__ import annotations

import collections
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import marker_lib as m
from marker_lib import ABSENT, APPLIED, CONSTRAINT_FAILED, INCONSISTENT, UNKNOWN

WORKER = str(Path(__file__).resolve().parent / "marker_worker.py")


class Row(dict):
    def __init__(self, experiment, case, **details):
        super().__init__(experiment=experiment, case=case, checks={}, **details)

    def check(self, name, condition):
        self["checks"][name] = bool(condition)
        return self

    @property
    def ok(self):
        return all(self["checks"].values())


def _write(ctx, event_id, key, helper=None, **kwargs):
    return (helper or ctx.help).write_query("d6", m.MARKED_DUPLICATE_UPDATE, m.params(event_id, key, **kwargs))


def _spawn(*args, **kwargs):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.Popen([sys.executable, WORKER, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, **kwargs)


# ---------------------------------------------------------------- E1

def e1_atomic_commit(ctx, reps=20):
    rows = []
    row = Row("E1", "commit, unattributed event", reps=reps)
    outcomes = collections.Counter()
    good = 0
    for _ in range(reps):
        event, key = ctx.event(), ctx.key("e1")
        result = _write(ctx, event, key)
        outcomes[result.outcome.value] += 1
        good += result.outcome.value == "COMMITTED" and result.counters["nodes_created"] == 1 and ctx.markers(key) == 1 and ctx.occurrences(event) == 1
    row.update(outcomes=dict(outcomes), both_visible=good)
    rows.append(row.check("every commit shows both marker and mutation", good == reps))

    event, key = ctx.event(), ctx.key("e1")
    _write(ctx, event, key, tps=0)
    row = Row("E1", "the unattributed path stops before the Campaign")
    row.update(event_occurrences=ctx.occurrences(event), campaign_occurrences=ctx.campaign_occurrences(event))
    rows.append(row.check("event updated", ctx.occurrences(event) == 1).check("campaign untouched", ctx.campaign_occurrences(event) == 0))

    event, key = ctx.event(attributed=True), ctx.key("e1")
    _write(ctx, event, key, tps=5)
    row = Row("E1", "an attributed event updates all three, in one transaction")
    row.update(event=ctx.occurrences(event), technique=ctx.technique_occurrences(event), campaign=ctx.campaign_occurrences(event))
    rows.append(row.check("all three updated once, with one marker", (ctx.occurrences(event), ctx.technique_occurrences(event), ctx.campaign_occurrences(event), ctx.markers(key)) == (1, 1, 1, 1)))

    key = ctx.key("e1")
    result = ctx.help.write_query("d6", m.MARKED_DUPLICATE_UPDATE, m.params("no-such-event", key))
    row = Row("E1", "missing target: committed, with no effect and no marker", **m.describe(result))
    rows.append(row.check("committed", result.outcome.value == "COMMITTED").check("no node created", result.counters["nodes_created"] == 0).check("no marker", ctx.markers(key) == 0))

    key = ctx.key("e1")
    result = ctx.help.write_statements("d6", [(m.SEPARATE_MARKER, {"request_key": key}), (m.UNMARKED_DUPLICATE_UPDATE, m.params("no-such-event", key))])
    row = Row("E1", "negative control: a marker as its own statement, missing target", **m.describe(result))
    rows.append(row.check("the separate marker commits though nothing was applied, which is the false evidence D6 warned about", ctx.markers(key) == 1 and result.outcome.value == "COMMITTED"))
    return rows


# ---------------------------------------------------------------- E2

def e2_atomic_rollback(ctx, reps=20):
    rows = []
    for name, action in (("explicit rollback", "rollback"), ("a failing statement after the marker", "fail"), ("an exception in the client before commit", "raise")):
        clean = 0
        for _ in range(reps):
            event, key = ctx.event(), ctx.key("e2")
            with ctx.direct.session() as session:
                tx = session.begin_transaction()
                try:
                    tx.run(m.MARKED_DUPLICATE_UPDATE, m.params(event, key)).consume()
                    if action == "rollback":
                        tx.rollback()
                    elif action == "fail":
                        tx.run("RETURN 1/0").consume()
                    else:
                        raise RuntimeError("client failure before commit")
                except Exception:
                    pass
                finally:
                    tx.close()
            clean += ctx.markers(key) == 0 and ctx.occurrences(event) == 0
        rows.append(Row("E2", name, reps=reps, neither_remains=clean).check("neither marker nor mutation remains", clean == reps))
    clean = 0
    for _ in range(reps):
        event, key = ctx.event(), ctx.key("e2")
        result = ctx.help.write_statements("d6", [(m.MARKED_DUPLICATE_UPDATE, m.params(event, key)), ("RETURN 1/0", None)])
        clean += result.outcome.value == "NOT_COMMITTED" and ctx.markers(key) == 0 and ctx.occurrences(event) == 0
    rows.append(Row("E2", "through the helper, second statement fails", reps=reps, neither_remains=clean).check("NOT_COMMITTED, and neither remains", clean == reps))
    return rows


# ---------------------------------------------------------------- E3

def _through_proxy(ctx, key, event, action, msg, times=1):
    ctx.proxy.arm(key, action, msg, times)
    driver = m.proxied_driver()
    try:
        return _write(ctx, event, key, helper=m.helper(driver))
    finally:
        driver.close()


def e3_lost_commit_response(ctx, reps=20):
    rows = []
    summary = collections.Counter()
    for _ in range(reps):
        event, key = ctx.event(), ctx.key("e3")
        result = _through_proxy(ctx, key, event, "drop_reply", "COMMIT")
        time.sleep(0.2)
        names = ctx.proxy.timeline.names()
        seen_success = any("SUCCESS to COMMIT" in n and "not delivered" in n for n in names)
        fresh = m.direct_driver()
        found = m.lookup(fresh, key)
        retried = _write(ctx, event, key)
        again = m.lookup(fresh, key)
        fresh.close()
        summary[(result.outcome.value, result.exception_type, seen_success, found.status, ctx.occurrences(event), retried.outcome.value, retried.exception_code, again.status, ctx.occurrences(event))] += 1
    row = Row("E3", "the server committed and the reply was withheld, then an independent reconnect", reps=reps,
              observed=[{"client_outcome": k[0], "client_exception": k[1], "server_said_success_undelivered": k[2], "lookup": k[3], "occurrences": k[4],
                         "retry_outcome": k[5], "retry_code": k[6], "lookup_after_retry": k[7], "occurrences_after_retry": k[8], "times": v} for k, v in summary.items()])
    expected = all(k[0] == "AMBIGUOUS" and k[2] and k[3] == APPLIED and k[4] == 1 and k[5] == "NOT_COMMITTED" and k[6] == CONSTRAINT_FAILED and k[7] == APPLIED and k[8] == 1 for k in summary)
    rows.append(row.check("client AMBIGUOUS, server SUCCESS undelivered, marker found, mutation present, retry changes nothing", expected and sum(summary.values()) == reps))

    summary = collections.Counter()
    for _ in range(reps):
        event, key = ctx.event(), ctx.key("e3")
        result = _through_proxy(ctx, key, event, "drop_before", "COMMIT")
        absent = ctx.settled_absent(key, event)
        summary[(result.outcome.value, result.exception_type, absent)] += 1
    row = Row("E3", "control: COMMIT never reached the server", reps=reps, observed=[{"client_outcome": k[0], "client_exception": k[1], "stayed_absent": k[2], "times": v} for k, v in summary.items()])
    rows.append(row.check("the same AMBIGUOUS to the client, but neither marker nor mutation exists", all(k[0] == "AMBIGUOUS" and k[2] for k in summary)))
    return rows


# ---------------------------------------------------------------- E4

def _explicit_through_proxy(ctx, key, event, action, msg):
    """An explicit transaction, so no driver retry can hide the fault."""
    ctx.proxy.arm(key, action, msg)
    driver = m.proxied_driver()
    error = None
    try:
        with driver.session() as session:
            tx = session.begin_transaction()
            try:
                tx.run(m.MARKED_DUPLICATE_UPDATE, m.params(event, key)).consume()
                tx.commit()
            finally:
                tx.close()
    except Exception as caught:
        error = type(caught).__name__
    driver.close()
    return error


def e4_failure_before_commit(ctx, reps=10):
    rows = []
    for name, action, msg in (("connection closed instead of RUN", "drop_before", "RUN"), ("connection closed before PULL", "lose_after_run_reply", "PULL"), ("connection closed instead of COMMIT", "drop_before", "COMMIT")):
        clean, errors = 0, collections.Counter()
        for _ in range(reps):
            event, key = ctx.event(), ctx.key("e4")
            errors[_explicit_through_proxy(ctx, key, event, action, msg)] += 1
            clean += ctx.settled_absent(key, event, 1.5)
        rows.append(Row("E4", name, reps=reps, client_errors=dict(errors), stayed_absent=clean).check("neither present, and neither appears later", clean == reps))

    clean = 0
    for _ in range(max(3, reps // 2)):
        event, key = ctx.event(), ctx.key("e4")
        worker = _spawn("open", event, key)
        ready = json.loads(worker.stdout.readline())["open"]
        during = ctx.markers(key)
        worker.send_signal(signal.SIGKILL)
        worker.wait(10)
        worker.stdout.close()
        worker.stderr.close()
        clean += ready and during == 0 and ctx.settled_absent(key, event, 2.0)
    rows.append(Row("E4", "client process killed by SIGKILL with the transaction open", reps=max(3, reps // 2), stayed_absent=clean).check("neither present, and neither appears later", clean == max(3, reps // 2)))

    # The server's own termination, if Community allows it.
    event, key = ctx.event(), ctx.key("e4")
    ctx.proxy.arm(key, "hold", "COMMIT")
    driver = m.proxied_driver()
    holder = {}
    thread = threading.Thread(target=lambda: holder.update(result=_write(ctx, event, key, helper=m.helper(driver))), daemon=True)
    thread.start()
    held = ctx.proxy.holding.wait(10)
    outcome = {"held": held}
    try:
        with ctx.direct.session() as session:
            txs = [dict(r) for r in session.run("SHOW TRANSACTIONS YIELD transactionId, currentQuery, status RETURN transactionId, currentQuery, status")]
            mine = [t for t in txs if "AppliedWrite" in (t["currentQuery"] or "")]
            outcome["show_transactions_works"] = True
            outcome["open_transactions_seen"] = len(txs)
            ids = [t["transactionId"] for t in txs if "SHOW TRANSACTIONS" not in (t["currentQuery"] or "")]
            outcome["terminated"] = [dict(r)["message"] for r in session.run("TERMINATE TRANSACTIONS $ids YIELD message RETURN message", ids=ids)]
    except Exception as error:
        outcome["admin_error"] = f"{type(error).__name__} {getattr(error, 'code', '')}"
    ctx.proxy.release.set()
    thread.join(20)
    driver.close()
    result = holder.get("result")
    absent = ctx.settled_absent(key, event, 1.5)
    row = Row("E4", "the server terminates the open transaction, then the held COMMIT arrives", **outcome, helper=m.describe(result) if result else None, stayed_absent=absent)
    rows.append(row.check("neither present", absent).check("the helper classifies it as not committed", result is not None and result.outcome.value == "NOT_COMMITTED"))
    return rows


# ---------------------------------------------------------------- E5

def e5_duplicate_key(ctx, reps=20):
    rows = []
    summary = collections.Counter()
    applied_same = 0
    for _ in range(reps):
        event, key = ctx.event(), ctx.key("e5")
        first = _write(ctx, event, key)
        before = m.lookup(ctx.direct, key)
        other = m.direct_driver()
        second = _write(ctx, event, key, helper=m.helper(other))
        after = m.lookup(other, key)
        other.close()
        applied_same += before.marker["applied_at"] == after.marker["applied_at"]
        summary[(first.counters["nodes_created"], second.outcome.value, second.attempts, second.exception_code, after.status, ctx.markers(key), ctx.occurrences(event))] += 1
    row = Row("E5", "second run through a separate driver and session", reps=reps, marker_unchanged=applied_same,
              observed=[dict(zip(("first_nodes_created", "second_outcome", "second_attempts", "second_code", "lookup", "markers", "occurrences"), k), times=v) for k, v in summary.items()])
    rows.append(row.check("one marker, one effect, a recognizable error, and the marker untouched",
                          list(summary) == [(1, "NOT_COMMITTED", 1, CONSTRAINT_FAILED, APPLIED, 1, 1)] and applied_same == reps))

    event, key = ctx.event(), ctx.key("e5")
    _write(ctx, event, key)
    worker = _spawn("once", event, key)
    out, _ = worker.communicate(timeout=60)
    second = json.loads(out.strip().splitlines()[-1])
    row = Row("E5", "second run from a separate process", second=second, markers=ctx.markers(key), occurrences=ctx.occurrences(event))
    rows.append(row.check("one marker, one effect", (ctx.markers(key), ctx.occurrences(event)) == (1, 1)).check("the same error", second["code"] == CONSTRAINT_FAILED))

    event, key, uid = ctx.event(), ctx.key("e5"), f"uid-{ctx.prefix}"
    with ctx.direct.session() as session:
        session.run("CREATE (:OtherUnique {uid:$u})", u=uid).consume()
    result = ctx.help.write_statements("d6", [(m.MARKED_DUPLICATE_UPDATE, m.params(event, key)), ("CREATE (:OtherUnique {uid:$u})", {"u": uid})])
    found = m.lookup(ctx.direct, key)
    row = Row("E5", "the same error code from a different constraint", **m.describe(result), lookup=found.status, markers=ctx.markers(key), occurrences=ctx.occurrences(event))
    rows.append(row.check("the code alone doesn't mean already applied: the lookup says absent, and the whole transaction rolled back",
                          result.exception_code == CONSTRAINT_FAILED and found.status == ABSENT and ctx.occurrences(event) == 0))
    return rows


# ---------------------------------------------------------------- E6

def _race_threads(ctx, query, event, key, contenders):
    barrier = threading.Barrier(contenders)
    results = []
    lock = threading.Lock()

    def attempt():
        barrier.wait()
        result = ctx.help.write_query("d6", query, m.params(event, key))
        with lock:
            results.append(result)
    threads = [threading.Thread(target=attempt) for _ in range(contenders)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    return results


def e6_concurrent_duplicates(ctx, contenders=16, rounds=30):
    rows = []
    summary = collections.Counter()
    commits_total, markers_total, effects_total, wrong = 0, 0, 0, 0
    codes = collections.Counter()
    for _ in range(rounds):
        event, key = ctx.event(), ctx.key("e6")
        results = _race_threads(ctx, m.MARKED_DUPLICATE_UPDATE, event, key, contenders)
        commits = sum(r.outcome.value == "COMMITTED" for r in results)
        commits_total += commits
        markers_total += ctx.markers(key)
        effects_total += ctx.occurrences(event)
        wrong += not (commits == 1 and ctx.markers(key) == 1 and ctx.occurrences(event) == 1)
        for result in results:
            codes[f"{result.outcome.value} {result.exception_code} attempts={result.attempts}"] += 1
    row = Row("E6", "threads on one key", contenders=contenders, rounds=rounds, commits=commits_total, markers=markers_total, business_mutations=effects_total, rounds_wrong=wrong, outcomes=dict(codes))
    rows.append(row.check("each round: exactly one commit, one marker, one effect", wrong == 0))

    # Separate processes, so no client state is shared.
    process_rounds, wrong, commits_total, codes = max(3, rounds // 6), 0, 0, collections.Counter()
    for _ in range(process_rounds):
        event, key = ctx.event(), ctx.key("e6")
        with tempfile.TemporaryDirectory(prefix="cyukti-d6-race-") as directory:
            go = os.path.join(directory, "go")
            workers = [_spawn("race", event, key, go) for _ in range(4)]
            time.sleep(4)
            Path(go).touch()
            outcomes = [json.loads(w.communicate(timeout=60)[0].strip().splitlines()[-1]) for w in workers]
        commits = sum(o["outcome"] == "COMMITTED" for o in outcomes)
        commits_total += commits
        for outcome in outcomes:
            codes[f"{outcome['outcome']} {outcome['code']}"] += 1
        wrong += not (commits == 1 and ctx.markers(key) == 1 and ctx.occurrences(event) == 1)
    row = Row("E6", "separate processes on one key", contenders=4, rounds=process_rounds, commits=commits_total, rounds_wrong=wrong, outcomes=dict(codes))
    rows.append(row.check("each round: exactly one commit, one marker, one effect", wrong == 0))

    # The control: the same race with no constraint on the label.
    duplicates, effects, control_rounds = 0, 0, max(5, rounds // 3)
    for _ in range(control_rounds):
        event, key = ctx.event(), ctx.key("e6")
        _race_threads(ctx, m.UNCONSTRAINED_DUPLICATE_UPDATE, event, key, contenders)
        duplicates += ctx.markers(key, "AppliedWriteNoConstraint") > 1
        effects += ctx.occurrences(event) > 1
    row = Row("E6", "control: the same race with no constraint", contenders=contenders, rounds=control_rounds, rounds_with_duplicate_markers=duplicates, rounds_with_duplicate_effects=effects)
    rows.append(row.check("without the constraint, duplicates appear, so the constraint is what prevents them", duplicates == control_rounds))

    # Different keys on one event: nothing should be refused, and no increment should be lost.
    lost, attempted, committed_total = 0, 0, 0
    lost_rounds = max(5, rounds // 3)
    for _ in range(lost_rounds):
        event = ctx.event()
        keys = [ctx.key("e6") for _ in range(contenders)]
        barrier = threading.Barrier(contenders)
        results = []

        def attempt(key):
            barrier.wait()
            results.append(_write(ctx, event, key))
        threads = [threading.Thread(target=attempt, args=(k,)) for k in keys]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        committed = sum(r.outcome.value == "COMMITTED" for r in results)
        committed_total += committed
        attempted += contenders
        lost += ctx.occurrences(event) != committed
    row = Row("E6", "different keys on one event, concurrently", contenders=contenders, rounds=lost_rounds, attempted=attempted, committed=committed_total, rounds_where_count_differs_from_commits=lost)
    rows.append(row.check("the additive count equals the commits: no lost increment", lost == 0))
    return rows


# ---------------------------------------------------------------- E7

def e7_different_keys_same_payload(ctx, reps=10):
    good = 0
    for _ in range(reps):
        event = ctx.event()
        one, two = ctx.key("e7"), ctx.key("e7")
        first, second = _write(ctx, event, one), _write(ctx, event, two)
        good += (first.outcome.value, second.outcome.value, ctx.markers(one), ctx.markers(two), ctx.occurrences(event)) == ("COMMITTED", "COMMITTED", 1, 1, 2)
    return [Row("E7", "two keys, identical payload", reps=reps, both_applied=good).check("both count, each with its own marker", good == reps)]


# ---------------------------------------------------------------- E8

def e8_lookup(ctx):
    rows = []
    event, key = ctx.event(), ctx.key("e8")
    _write(ctx, event, key)
    missing = ctx.key("e8")
    found, absent = m.lookup(ctx.direct, key), m.lookup(ctx.direct, missing)
    rows.append(Row("E8", "marker exists", status=found.status).check("APPLIED, positive evidence", found.status == APPLIED and found.marker["request_key"] == key))
    rows.append(Row("E8", "marker doesn't exist", status=absent.status).check("ABSENT, from a read that succeeded", absent.status == ABSENT))

    # In every failure below the marker really EXISTS, so anything reading as ABSENT would be a false answer.
    ctx.proxy.arm(key, "drop_before", "RUN", times=99)
    driver = m.proxied_driver()
    interrupted = m.lookup(driver, key)
    driver.close()
    rows.append(Row("E8", "connection interrupted during lookup", status=interrupted.status, error=interrupted.error_type, marker_really_exists=ctx.markers(key)).check("UNKNOWN", interrupted.status == UNKNOWN))

    ctx.proxy.arm(key, "drop_reply", "RUN", times=99)
    driver = m.proxied_driver()
    lost = m.lookup(driver, key)
    driver.close()
    rows.append(Row("E8", "reply lost during lookup", status=lost.status, error=lost.error_type, marker_really_exists=ctx.markers(key)).check("UNKNOWN", lost.status == UNKNOWN))

    ctx.proxy.arm(key, "hold", "RUN", times=99)
    driver = m.proxied_driver()
    stalled = m.lookup(driver, key, deadline=2.0)
    ctx.proxy.release.set()
    driver.close()
    rows.append(Row("E8", "the query stalls past the client's deadline", status=stalled.status, error=stalled.error_type, marker_really_exists=ctx.markers(key)).check("UNKNOWN", stalled.status == UNKNOWN))

    slow = "UNWIND range(1, 200000000) AS x WITH count(x) AS c MATCH (m:AppliedWrite {request_key:$request_key}) RETURN m.request_key AS request_key, toString(m.applied_at) AS applied_at, m.operation AS operation, m.fingerprint AS fingerprint"
    timed = m.lookup(ctx.direct, key, timeout=0.2, deadline=15.0, query=slow)
    rows.append(Row("E8", "the server times the query out", status=timed.status, error=timed.error_type, code=timed.error_code, marker_really_exists=ctx.markers(key)).check("UNKNOWN", timed.status == UNKNOWN))

    bad = m.lookup(ctx.direct, key, query="MATCH (m:AppliedWrite {request_key:$request_key} RETURN m")
    rows.append(Row("E8", "the query returns an error", status=bad.status, error=bad.error_type, code=bad.error_code, marker_really_exists=ctx.markers(key)).check("UNKNOWN", bad.status == UNKNOWN))

    wrong = m.GraphDatabase.driver(m.CONN["uri"], auth=(m.CONN["user"], "not-the-password"), connection_timeout=5)
    denied = m.lookup(wrong, key)
    wrong.close()
    rows.append(Row("E8", "authentication fails", status=denied.status, error=denied.error_type, code=denied.error_code, marker_really_exists=ctx.markers(key)).check("UNKNOWN", denied.status == UNKNOWN))

    closed = socket_free_port_driver()
    unavailable = m.lookup(closed, key)
    closed.close()
    rows.append(Row("E8", "nothing is listening at the target", status=unavailable.status, error=unavailable.error_type, marker_really_exists=ctx.markers(key)).check("UNKNOWN", unavailable.status == UNKNOWN))
    return rows


def socket_free_port_driver():
    """The experiment's proxy port with nothing behind it: a temporary driver to a port closed for this call."""
    import socket as _socket
    probe = _socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    m.ALLOWED_PORTS.add(port)
    return m.GraphDatabase.driver(f"bolt://127.0.0.1:{port}", auth=(m.CONN["user"], m.CONN["password"]), connection_timeout=2)


# ---------------------------------------------------------------- absence

def absence(ctx, reps=5):
    """What a successful read that finds no marker really shows."""
    rows = []
    summary = collections.Counter()
    for _ in range(reps):
        event, key = ctx.event(), ctx.key("absence")
        ctx.proxy.arm(key, "hold", "COMMIT")
        driver = m.proxied_driver()
        box = {}
        thread = threading.Thread(target=lambda: box.update(result=_write(ctx, event, key, helper=m.helper(driver))), daemon=True)
        thread.start()
        ctx.proxy.holding.wait(10)
        during = m.lookup(ctx.direct, key)
        ctx.proxy.release.set()
        thread.join(15)
        driver.close()
        summary[(during.status, box["result"].outcome.value, m.lookup(ctx.direct, key).status, ctx.occurrences(event))] += 1
    row = Row("ABSENCE", "lookup during an open transaction whose COMMIT is in flight", reps=reps, observed=[dict(zip(("during", "client_outcome", "after", "occurrences"), k), times=v) for k, v in summary.items()])
    rows.append(row.check("ABSENT, and then the write commits: absence wasn't proof", list(summary) == [(ABSENT, "COMMITTED", APPLIED, 1)]))

    summary = collections.Counter()
    for _ in range(max(3, reps - 2)):
        event, key = ctx.event(), ctx.key("absence")
        ctx.proxy.arm(key, "hold", "COMMIT")
        worker = _spawn("through_proxy", event, key)
        worker.stdout.readline()
        ctx.proxy.holding.wait(15)
        worker.send_signal(signal.SIGKILL)
        worker.wait(10)
        worker.stdout.close()
        worker.stderr.close()
        during = m.lookup(ctx.direct, key)
        ctx.proxy.release.set()
        end = time.monotonic() + 5
        while time.monotonic() < end and m.lookup(ctx.direct, key).status != APPLIED:
            time.sleep(0.1)
        summary[(during.status, m.lookup(ctx.direct, key).status, ctx.occurrences(event))] += 1
    row = Row("ABSENCE", "the client is dead, its COMMIT still in flight in the network", reps=max(3, reps - 2), observed=[dict(zip(("after_client_death", "later", "occurrences"), k), times=v) for k, v in summary.items()])
    rows.append(row.check("the owner's death doesn't make absence safe: ABSENT after the client died, then APPLIED", list(summary) == [(ABSENT, APPLIED, 1)]))

    summary = collections.Counter()
    for _ in range(reps):
        event, key = ctx.event(), ctx.key("absence")
        ctx.proxy.arm(key, "hold", "COMMIT")
        driver = m.proxied_driver()
        helper = m.helper(driver)
        box = {}
        limits = helper.WriteLimits(transaction_timeout=1.0, max_retry_time=1.0, connection_acquisition_timeout=2.0)
        thread = threading.Thread(target=lambda: box.update(result=helper.write_query("d6", m.MARKED_DUPLICATE_UPDATE, m.params(event, key), limits=limits)), daemon=True)
        thread.start()
        ctx.proxy.holding.wait(10)
        time.sleep(3.0)
        ctx.proxy.release.set()
        thread.join(15)
        driver.close()
        absent = ctx.settled_absent(key, event, 1.5)
        result = box["result"]
        summary[(result.outcome.value, result.exception_code, absent)] += 1
    row = Row("ABSENCE", "COMMIT arrives after the transaction's timeout", reps=reps, transaction_timeout_s=1.0, held_s=3.0, observed=[dict(zip(("client_outcome", "code", "stayed_absent"), k), times=v) for k, v in summary.items()])
    rows.append(row.check("rejected, and neither ever appears", all(k[2] for k in summary)))
    return rows


# ---------------------------------------------------------------- E9

def _load_ledger():
    path = m.BACKEND / "neo4j_idempotency_ledger.py"
    spec = importlib.util.spec_from_file_location("ledger_for_d6_experiment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def e9_payload_mismatch(ctx):
    rows = []
    event, key = ctx.event(), ctx.key("e9")
    _write(ctx, event, key, payload='{"version":"A"}')
    before = m.lookup(ctx.direct, key)
    second = _write(ctx, event, key, payload='{"version":"B"}')
    after = m.lookup(ctx.direct, key)
    row = Row("E9", "the proposed marker: same key, different payload", second=m.describe(second), lookup=after.status, stored_payload=ctx.payload(event), occurrences=ctx.occurrences(event))
    row.check("nothing is overwritten", ctx.payload(event) == '{"version":"A"}' and ctx.occurrences(event) == 1 and before.marker == after.marker)
    row["marker_can_tell_the_payloads_apart"] = before.marker != after.marker or second.exception_code != CONSTRAINT_FAILED
    rows.append(row.check("LIMITATION: to the caller this looks exactly like a true retry", second.exception_code == CONSTRAINT_FAILED and after.status == APPLIED))

    event, key = ctx.event(), ctx.key("e9")
    second = ctx.help.write_query("d6", m.MARKED_DUPLICATE_UPDATE.replace("e.tps = coalesce(e.tps,0) + $tps", "e.tps = coalesce(e.tps,0) + $tps + 1"), m.params(event, key))
    third = _write(ctx, event, key)
    rows.append(Row("E9", "the proposed marker: same key, different operation", third=m.describe(third)).check("no conflict signal beyond the duplicate itself", third.exception_code == CONSTRAINT_FAILED and ctx.occurrences(event) == 1))

    ledger = _load_ledger()
    statements_a = [(m.UNMARKED_DUPLICATE_UPDATE, m.params("E", "K", payload='{"version":"A"}'))]
    statements_b = [(m.UNMARKED_DUPLICATE_UPDATE, m.params("E", "K", payload='{"version":"B"}'))]
    fingerprint_a = ledger.fingerprint_payload("duplicate_event.update", statements_a).value
    fingerprint_b = ledger.fingerprint_payload("duplicate_event.update", statements_b).value
    event, key = ctx.event(), ctx.key("e9")
    first = ctx.help.write_query("d6", m.FINGERPRINTED_DUPLICATE_UPDATE, m.params(event, key, payload='{"version":"A"}', operation="duplicate_event.update", fingerprint=fingerprint_a))
    second = ctx.help.write_query("d6", m.FINGERPRINTED_DUPLICATE_UPDATE, m.params(event, key, payload='{"version":"B"}', operation="duplicate_event.update", fingerprint=fingerprint_b))
    found = m.lookup(ctx.direct, key)
    verdict = "SAME" if found.marker["fingerprint"] == fingerprint_b and found.marker["operation"] == "duplicate_event.update" else "CONFLICT"
    row = Row("E9", "a fuller marker: fingerprint and operation stored immutably", first=m.describe(first), second=m.describe(second), verdict=verdict, occurrences=ctx.occurrences(event))
    rows.append(row.check("the mismatch is detected, nothing overwritten", verdict == "CONFLICT" and ctx.occurrences(event) == 1 and ctx.payload(event) == '{"version":"A"}'))
    third = ctx.help.write_query("d6", m.FINGERPRINTED_DUPLICATE_UPDATE, m.params(event, key, payload='{"version":"A"}', operation="duplicate_event.update", fingerprint=fingerprint_a))
    match = m.lookup(ctx.direct, key).marker["fingerprint"] == fingerprint_a
    rows.append(Row("E9", "a fuller marker: a true retry", third=m.describe(third)).check("recognized as the same request", match and third.exception_code == CONSTRAINT_FAILED))

    with tempfile.TemporaryDirectory(prefix="cyukti-d6-ledger-") as directory:
        store = ledger.IdempotencyLedger(os.path.join(directory, "ledger.db"))
        try:
            store.register("K", "duplicate_event.update", statements_a)
            refused = False
            try:
                store.register("K", "duplicate_event.update", statements_b)
            except ledger.IdempotencyConflictError:
                refused = True
        finally:
            store.close()
    rows.append(Row("E9", "the real ledger on a temporary file, before Neo4j is reached").check("IdempotencyConflictError, while the ledger is intact", refused))
    return rows


# ---------------------------------------------------------------- E10

def e10_retention(ctx):
    rows = []
    event, key = ctx.event(), ctx.key("e10")
    _write(ctx, event, key)
    with ctx.direct.session() as session:
        session.run("MATCH (m:AppliedWrite {request_key:$k}) DELETE m", k=key).consume()
    gone = m.lookup(ctx.direct, key)
    retry = _write(ctx, event, key)
    row = Row("E10", "marker deleted, then a retry of the applied request", lookup_after_delete=gone.status, retry=m.describe(retry), occurrences=ctx.occurrences(event))
    rows.append(row.check("the committed request looks unapplied, and the retry applies it a second time: this is the harm", gone.status == ABSENT and retry.outcome.value == "COMMITTED" and ctx.occurrences(event) == 2))
    return rows


def constraint_loss(ctx):
    """Runs last in a sequence: it drops and restores the constraint on the disposable database."""
    rows = []
    with ctx.direct.session() as session:
        session.run("DROP CONSTRAINT applied_write_key").consume()
    present = [c for c in m.constraints(ctx.direct) if c["name"] == "applied_write_key"]
    event, key = ctx.event(), ctx.key("constraint")
    first, second = _write(ctx, event, key), _write(ctx, event, key)
    found = m.lookup(ctx.direct, key)
    row = Row("E10", "the constraint is missing", first=m.describe(first), second=m.describe(second), markers=ctx.markers(key), occurrences=ctx.occurrences(event), lookup=found.status)
    row.check("a startup check can see it's missing", not present)
    rows.append(row.check("a repeat silently applies twice, and the lookup reports the inconsistency", ctx.markers(key) == 2 and ctx.occurrences(event) == 2 and found.status == INCONSISTENT))
    restore_failed = None
    try:
        with ctx.direct.session() as session:
            session.run("CREATE CONSTRAINT applied_write_key IF NOT EXISTS FOR (m:AppliedWrite) REQUIRE m.request_key IS UNIQUE").consume()
    except Exception as error:
        restore_failed = getattr(error, "code", type(error).__name__)
    with ctx.direct.session() as session:
        session.run("MATCH (m:AppliedWrite {request_key:$k}) DELETE m", k=key).consume()
        session.run("CREATE CONSTRAINT applied_write_key IF NOT EXISTS FOR (m:AppliedWrite) REQUIRE m.request_key IS UNIQUE").consume()
        session.run("CALL db.awaitIndexes(60)").consume()
    row = Row("E10", "restoring the constraint over duplicates", refused_with=restore_failed)
    rows.append(row.check("refused until the duplicates are removed, and then in place", restore_failed is not None and any(c["name"] == "applied_write_key" for c in m.constraints(ctx.direct))))
    return rows


# ---------------------------------------------------------------- a real server crash

def server_crash(ctx, writers=6, seconds=6):
    """Kills the server under load, restarts it on the same volume, and compares every attempted key with its own business effect."""
    rows = []
    stop = threading.Event()
    attempts, lock = [], threading.Lock()
    driver = m.direct_driver(max_connection_pool_size=writers + 2)
    helper = m.helper(driver)
    limits = helper.WriteLimits(transaction_timeout=4.0, max_retry_time=0.5, connection_acquisition_timeout=1.0)
    events = [[ctx.event() for _ in range(400)] for _ in range(writers)]

    def writer(mine):
        for event in mine:
            if stop.is_set():
                return
            key = f"crash|{event}"
            result = helper.write_query("d6", m.MARKED_DUPLICATE_UPDATE, m.params(event, key), limits=limits)
            with lock:
                attempts.append((event, key, result.outcome.value, result.exception_type))

    threads = [threading.Thread(target=writer, args=(mine,), daemon=True) for mine in events]
    for thread in threads:
        thread.start()
    time.sleep(seconds)
    m.confirm_environment()
    m._docker("kill", m.CONN["container"])
    killed_at = len(attempts)
    during = m.lookup(ctx.direct, f"crash|{attempts[0][0]}" if attempts else "none", deadline=4.0)
    stop.set()
    for thread in threads:
        thread.join(30)
    driver.close()
    m._docker("start", m.CONN["container"])
    end = time.monotonic() + 180
    ready = False
    while time.monotonic() < end and not ready:
        try:
            with ctx.direct.session() as session:
                session.run("RETURN 1").consume()
            ready = True
        except Exception:
            time.sleep(1)
    m.verify_disposable(ctx.direct)
    tally = collections.Counter()
    disagree = committed_missing = not_committed_present = 0
    for event, key, outcome, _ in attempts:
        marker, effect = ctx.markers(key), ctx.occurrences(event)
        disagree += (marker, effect) not in ((0, 0), (1, 1))
        committed_missing += outcome == "COMMITTED" and marker != 1
        not_committed_present += outcome == "NOT_COMMITTED" and marker != 0
        tally[f"{outcome} marker={marker} effect={effect}"] += 1
    row = Row("CRASH", "server killed under load, restarted on the same volume", writers=writers, attempts=len(attempts), completed_before_kill=killed_at, lookup_while_down=during.status, server_back=ready, tally=dict(tally),
              marker_effect_disagree=disagree, committed_without_marker=committed_missing, not_committed_with_marker=not_committed_present)
    row.check("while the server is down, a lookup is UNKNOWN", during.status == UNKNOWN)
    row.check("marker and effect agree for every attempted key", disagree == 0 and len(attempts) > 0)
    row.check("every confirmed commit survives", committed_missing == 0)
    rows.append(row.check("no confirmed non-commit has a marker", not_committed_present == 0))
    return rows


ALL = {"e1": e1_atomic_commit, "e2": e2_atomic_rollback, "e3": e3_lost_commit_response, "e4": e4_failure_before_commit, "e5": e5_duplicate_key, "e6": e6_concurrent_duplicates,
       "e7": e7_different_keys_same_payload, "e8": e8_lookup, "absence": absence, "e9": e9_payload_mismatch, "e10": e10_retention, "constraint": constraint_loss, "crash": server_crash}

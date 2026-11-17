"""
Tests for neo4j_client.write_query and write_statements.

These run the driver's real retry loop: _DriverUnderTest builds a real neo4j Session, so
Session.execute_write and _run_transaction are the installed driver's own code. Only the
transaction and the server are fakes. _FakeServer holds what has really committed, so each
test can compare the reported outcome with what the server did. Exceptions are the driver's
real classes.

No test here opens a socket. Behaviour against a real server was checked separately with a
fault proxy and a disposable instance, because a fake can't prove what a server does.
"""

from __future__ import annotations

import logging
import math
import sys
from types import SimpleNamespace

import pytest
from neo4j._conf import SessionConfig
from neo4j._sync.work.session import Session
from neo4j._sync.work.transaction import ManagedTransaction
from neo4j.exceptions import (
    ClientError,
    ConstraintError,
    CypherSyntaxError,
    DatabaseError,
    DriverError,
    IncompleteCommit,
    Neo4jError,
    ResultError,
    ServiceUnavailable,
    SessionExpired,
    TransactionError,
    TransientError,
)

import neo4j_client
from neo4j_client import (
    DEFAULT_WRITE_LIMITS,
    WriteLimits,
    WriteOutcome,
    WriteResult,
    write_query,
    write_query_rows,
    write_statements,
)

SECRET = "s3cret-parameter-value"
ADDRESS = "bolt://user:s3cret-password@10.1.2.3:7687"


def _server_error(code, message="server message"):
    return Neo4jError._hydrate_neo4j(code=code, message=message)


def _deadlock():
    return _server_error("Neo.TransientError.Transaction.DeadlockDetected")


class _FakeServer:
    """What has really committed, however the client's view turns out."""

    def __init__(self):
        self.committed = []
        self.commits_received = 0


class _Attempt:
    """Scripts one attempt. Leave everything None for a clean success."""

    def __init__(self, on_run=None, on_consume=None, on_commit=None, commit_applies=False, on_data=None):
        self.on_run = on_run
        self.on_consume = on_consume
        self.on_data = on_data
        self.on_commit = on_commit
        self.commit_applies = commit_applies


class _FakeResult:
    def __init__(self, tx, counters, on_consume, rows=(), on_data=None):
        self.tx, self.counters, self.on_consume, self.rows, self.on_data = tx, counters, on_consume, rows, on_data

    def data(self):
        self.tx.data_reads += 1
        if self.on_data:
            raise self.on_data
        return [dict(row) for row in self.rows]

    def consume(self):
        self.tx.consumed += 1
        if self.on_consume:
            raise self.on_consume
        return SimpleNamespace(counters=SimpleNamespace(**self.counters))


class _FakeTransaction(ManagedTransaction):
    """The driver's retry loop checks isinstance(tx, ManagedTransaction)."""

    def __init__(self, server, attempt, counters, session):  # no super(): no connection
        self.server, self.attempt, self.counters, self.session = server, attempt, counters, session
        self.pending = []
        self.closed = False
        self.committed_ok = False
        self.consumed = 0
        self.data_reads = 0

    def run(self, query, parameters=None, **kwargs):
        error = self.attempt.on_run
        if error:
            raise error
        self.pending.append((query, parameters))
        return _FakeResult(self, self.counters, self.attempt.on_consume, self.session.parent.rows, self.attempt.on_data)

    def _commit(self):
        self.server.commits_received += 1
        self.session.transactions_seen_committing += 1
        if self.attempt.on_commit and not self.attempt.commit_applies:
            self.closed = True
            raise self.attempt.on_commit
        self.server.committed.extend(self.pending)
        self.closed = True
        if self.attempt.on_commit:
            raise self.attempt.on_commit
        self.committed_ok = True

    def _close(self):
        self.closed = True


class _SessionUnderTest(Session):
    def __init__(self, parent, requested):
        config = SessionConfig()
        config.initial_retry_delay = 0.001
        config.retry_delay_jitter_factor = 0.0
        # Keeps the real retry loop fast. What was requested is recorded on the driver.
        config.max_transaction_retry_time = min(parent.retry_budget, requested.get("max_transaction_retry_time", 30.0))
        super().__init__(SimpleNamespace(), config)
        self.parent = parent
        self.transactions_seen_committing = 0

    def _open_transaction(self, **kwargs):
        self.parent.begin_kwargs.append(kwargs)
        if self.parent.begin_error:
            raise self.parent.begin_error
        index = min(len(self.parent.transactions), len(self.parent.script) - 1)
        tx = _FakeTransaction(self.parent.server, self.parent.script[index], self.parent.counters, self)
        self.parent.transactions.append(tx)
        self._transaction = tx

    def close(self):
        self.parent.sessions_closed += 1
        if self.parent.close_error:
            raise self.parent.close_error
        super().close()


class _DriverUnderTest:
    def __init__(self, *script, counters=None, begin_error=None, session_error=None,
                 close_error=None, retry_budget=0.05, rows=()):
        self.rows = rows
        self.begin_kwargs = []
        self.session_configs = []
        self.script = list(script) or [_Attempt()]
        self.server = _FakeServer()
        self.counters = counters or {}
        self.begin_error = begin_error
        self.session_error = session_error
        self.close_error = close_error
        self.retry_budget = retry_budget
        self.transactions = []
        self.sessions_opened = 0
        self.sessions_closed = 0

    def session(self, **config):
        self.session_configs.append(config)
        if self.session_error:
            raise self.session_error
        self.sessions_opened += 1
        return _SessionUnderTest(self, config)


@pytest.fixture
def use(monkeypatch):
    def install(*script, **kwargs):
        driver = _DriverUnderTest(*script, **kwargs)
        monkeypatch.setattr(neo4j_client, "driver", driver)
        return driver
    return install


# ---------------------------------------------------------------- 1. confirmed commit

def test_success_is_committed_only_after_the_commit(use):
    driver = use()

    result = write_query("op", "CREATE (n)", {"a": 1})

    assert result.outcome is WriteOutcome.COMMITTED
    assert result.committed is True
    assert result.attempts == 1
    assert result.exception is None
    assert driver.server.committed == [("CREATE (n)", {"a": 1})]
    assert driver.transactions[0].committed_ok is True


def test_result_carries_the_operation_and_duration(use):
    use()
    result = write_query("flush_duplicates", "CREATE (n)")
    assert result.operation == "flush_duplicates"
    assert result.duration_seconds >= 0


def test_the_result_is_consumed_before_the_commit(use):
    driver = use()
    write_query("op", "CREATE (n)")
    assert driver.transactions[0].consumed == 1


def test_parameters_are_passed_to_the_query_unchanged_and_default_to_empty(use):
    driver = use()
    write_query("op", "A", {"query": "a parameter named query", "records": [1, 2]})
    write_query("op", "B")
    assert driver.server.committed[0] == ("A", {"query": "a parameter named query", "records": [1, 2]})
    assert driver.server.committed[1] == ("B", {})


# ---------------------------------------------------------------- 2. failure before commit

@pytest.mark.parametrize("error_type, code", [
    (ClientError, "Neo.ClientError.Statement.ArithmeticError"),
    (ConstraintError, "Neo.ClientError.Schema.ConstraintValidationFailed"),
    (CypherSyntaxError, "Neo.ClientError.Statement.SyntaxError"),
    (DatabaseError, "Neo.DatabaseError.General.UnknownError"),
])
def test_server_failure_while_running_is_not_committed(use, error_type, code):
    error = _server_error(code)
    assert isinstance(error, error_type)
    driver = use(_Attempt(on_run=error))

    result = write_query("op", "CREATE (n)")

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.exception is error
    assert result.exception_code == code
    assert result.attempts == 1
    assert driver.server.committed == []
    assert driver.server.commits_received == 0


def test_failure_opening_a_session_is_not_committed(use):
    error = ServiceUnavailable("no connection")
    driver = use(session_error=error)

    result = write_query("op", "CREATE (n)")

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts == 0
    assert result.exception is error
    assert driver.server.commits_received == 0


def test_authentication_failure_is_not_committed(use):
    error = _server_error("Neo.ClientError.Security.Unauthorized")
    use(begin_error=error)
    result = write_query("op", "CREATE (n)")
    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts == 0


# ---------------------------------------------------------------- 3. failure during consumption

def test_server_failure_during_consume_is_not_committed(use):
    """The case the discard pattern hides: RUN succeeded and PULL failed."""
    error = _server_error("Neo.ClientError.Transaction.Terminated")
    driver = use(_Attempt(on_consume=error))

    result = write_query("op", "CREATE (n)")

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.exception is error
    assert driver.server.committed == []
    assert driver.server.commits_received == 0


def test_connection_lost_during_consume_is_retried_then_commits(use):
    driver = use(_Attempt(on_consume=ServiceUnavailable("lost")), _Attempt())

    result = write_query("op", "CREATE (n)")

    assert result.outcome is WriteOutcome.COMMITTED
    assert result.attempts == 2
    assert len(driver.server.committed) == 1


# ---------------------------------------------------------------- 4. IncompleteCommit

@pytest.mark.parametrize("commit_applies", [True, False], ids=["server committed", "server did not"])
def test_incomplete_commit_is_ambiguous_on_both_sides_of_the_commit(use, commit_applies):
    error = IncompleteCommit("connection lost")
    driver = use(_Attempt(on_commit=error, commit_applies=commit_applies))

    result = write_query("op", "CREATE (n)")

    assert result.outcome is WriteOutcome.AMBIGUOUS
    assert result.exception is error
    assert result.exception_type == "IncompleteCommit"
    assert len(driver.server.committed) == (1 if commit_applies else 0)


def test_incomplete_commit_is_never_retried(use):
    driver = use(_Attempt(on_commit=IncompleteCommit("lost"), commit_applies=True), _Attempt())

    result = write_query("op", "CREATE (n)")

    assert result.attempts == 1
    assert len(driver.transactions) == 1
    assert len(driver.server.committed) == 1


def test_incomplete_commit_is_recognised_before_its_parent_class():
    """IncompleteCommit subclasses ServiceUnavailable, which alone means something else."""
    assert issubclass(IncompleteCommit, ServiceUnavailable)
    assert IncompleteCommit("x").is_retryable() is False
    outcome, _ = neo4j_client._classify_write_failure(IncompleteCommit("x"), commit_attempted=False)
    assert outcome is WriteOutcome.AMBIGUOUS


# ---------------------------------------------------------------- 5. other connection failures

@pytest.mark.parametrize("make", [lambda: ServiceUnavailable("x"), lambda: SessionExpired("x")])
def test_connection_failures_before_any_commit_exhaust_retries_and_are_not_committed(use, make):
    driver = use(_Attempt(on_run=make()))

    result = write_query("op", "CREATE (n)")

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts >= 2
    assert driver.server.commits_received == 0


def test_a_plain_connection_error_after_a_commit_attempt_is_ambiguous(use):
    """Never treated as proof of rollback. The driver would retry this, so the helper
    refuses the re-execution, and the write isn't applied a second time."""
    driver = use(_Attempt(on_commit=ServiceUnavailable("lost"), commit_applies=True), _Attempt())

    result = write_query("op", "CREATE (n)")

    assert result.outcome is WriteOutcome.AMBIGUOUS
    assert result.attempts == 1
    assert len(driver.server.committed) == 1


def test_a_read_of_a_generic_connection_error_alone_never_says_not_committed_after_commit():
    outcome, _ = neo4j_client._classify_write_failure(ServiceUnavailable("x"), commit_attempted=True)
    assert outcome is WriteOutcome.AMBIGUOUS


# ---------------------------------------------------------------- reply to COMMIT

def test_verified_commit_rejection_is_not_committed(use):
    error = _server_error("Neo.ClientError.Transaction.Terminated")
    driver = use(_Attempt(on_commit=error))

    result = write_query("op", "CREATE (n)")

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert driver.server.committed == []


@pytest.mark.parametrize("code", [
    "Neo.DatabaseError.Transaction.TransactionCommitFailed",
    "Neo.ClientError.Transaction.SomethingNotYetVerified",
])
def test_unverified_reply_to_commit_is_ambiguous(use, code):
    error = _server_error(code)
    use(_Attempt(on_commit=error))

    result = write_query("op", "CREATE (n)")

    assert result.outcome is WriteOutcome.AMBIGUOUS
    assert result.exception is error


def test_retryable_reply_to_commit_is_not_re_executed(use):
    """The driver would re-run the work after a TransientError on COMMIT. That this is
    safe hasn't been verified, so the helper stops it."""
    driver = use(_Attempt(on_commit=_deadlock()), _Attempt())

    result = write_query("op", "CREATE (n)")

    assert result.outcome is WriteOutcome.AMBIGUOUS
    assert result.attempts == 1
    assert driver.server.commits_received == 1
    assert driver.server.committed == []
    assert "refused" in result.diagnostic


# ---------------------------------------------------------------- 6-7. retries and side effects

def test_retryable_failure_before_commit_is_retried_and_counted(use):
    driver = use(_Attempt(on_run=_deadlock()), _Attempt(on_consume=ServiceUnavailable("x")), _Attempt())

    result = write_query("op", "CREATE (n)")

    assert result.outcome is WriteOutcome.COMMITTED
    assert result.attempts == 3
    assert driver.server.commits_received == 1
    assert len(driver.server.committed) == 1


def test_non_retryable_failure_is_not_retried(use):
    driver = use(_Attempt(on_run=_server_error("Neo.ClientError.Statement.SyntaxError")), _Attempt())

    result = write_query("op", "CREATE (n)")

    assert result.attempts == 1
    assert len(driver.transactions) == 1


def test_failed_attempts_leave_nothing_committed(use):
    """A retried attempt's statements run again, but only the committing attempt's count."""
    driver = use(
        _Attempt(on_consume=ServiceUnavailable("x")),
        _Attempt(on_consume=ServiceUnavailable("x")),
        _Attempt(),
    )

    write_statements("op", [("A", None), ("B", None)])

    assert [query for query, _ in driver.server.committed] == ["A", "B"]


def test_the_api_takes_no_callback_so_no_side_effect_can_join_the_retried_work():
    import inspect
    assert list(inspect.signature(write_query).parameters) == ["operation", "query", "parameters", "limits"]
    assert list(inspect.signature(write_query_rows).parameters) == ["operation", "query", "parameters", "limits"]
    assert list(inspect.signature(write_statements).parameters) == ["operation", "statements", "limits"]
    for function in (write_query, write_query_rows, write_statements):
        assert inspect.signature(function).parameters["limits"].kind is inspect.Parameter.KEYWORD_ONLY


def test_a_parameter_iterator_is_not_exhausted_by_the_first_attempt(use):
    """Statements are materialised once, so a retry doesn't run an empty list."""
    driver = use(_Attempt(on_consume=ServiceUnavailable("x")), _Attempt())

    result = write_statements("op", iter([("A", None)]))

    assert result.committed
    assert [query for query, _ in driver.server.committed] == ["A"]


def test_nothing_is_logged_once_per_retry_by_the_helper(use, caplog):
    use(_Attempt(on_consume=ServiceUnavailable("x")), _Attempt())
    with caplog.at_level(logging.DEBUG, logger="neo4j_client"):
        write_query("op", "CREATE (n)")
    assert len([r for r in caplog.records if r.name == "neo4j_client"]) == 1


# ---------------------------------------------------------------- 8. counters

def test_summary_counters_are_extracted(use):
    use(counters={"nodes_created": 2, "properties_set": 5, "relationships_created": 1})

    result = write_query("op", "CREATE (n)")

    assert result.counters["nodes_created"] == 2
    assert result.counters["properties_set"] == 5
    assert result.counters["relationships_created"] == 1
    assert result.counters["nodes_deleted"] == 0


def test_counters_are_summed_across_statements(use):
    use(counters={"properties_set": 3})
    result = write_statements("op", [("A", None), ("B", None)])
    assert result.counters["properties_set"] == 6


def test_counters_come_only_from_the_committing_attempt(use):
    use(_Attempt(on_consume=ServiceUnavailable("x")), _Attempt(), counters={"nodes_created": 1})
    assert write_query("op", "CREATE (n)").counters["nodes_created"] == 1


@pytest.mark.parametrize("attempt", [
    _Attempt(on_run=_server_error("Neo.ClientError.Statement.SyntaxError")),
    _Attempt(on_commit=IncompleteCommit("x"), commit_applies=True),
])
def test_no_counters_unless_committed(use, attempt):
    use(attempt, counters={"nodes_created": 1})
    assert write_query("op", "CREATE (n)").counters is None


# ---------------------------------------------------------------- 9. empty results

def test_a_write_that_changes_nothing_is_committed_with_zero_counters(use):
    """A MATCH that finds nothing commits. The counters are how a caller can tell."""
    use(counters={})
    result = write_query("op", "MATCH (n:Missing) SET n.x = 1")
    assert result.outcome is WriteOutcome.COMMITTED
    assert all(value == 0 for value in result.counters.values())


# ---------------------------------------------------------------- 10. batches

def test_a_batch_is_one_statement_with_all_its_rows(use):
    driver = use(counters={"properties_set": 200})
    rows = [{"id": i} for i in range(200)]

    result = write_query("flush", "UNWIND $records AS r SET r.x = 1", {"records": rows})

    assert result.committed
    assert result.counters["properties_set"] == 200
    assert driver.server.committed == [("UNWIND $records AS r SET r.x = 1", {"records": rows})]


def test_several_statements_commit_together(use):
    driver = use()
    write_statements("op", [("A", {"a": 1}), ("B", {"b": 2})])
    assert driver.server.committed == [("A", {"a": 1}), ("B", {"b": 2})]
    assert len(driver.transactions) == 1


def test_a_later_statement_failing_commits_none_of_them(use):
    class _FailsOnSecond(_Attempt):
        pass

    driver = use()
    original_run = _FakeTransaction.run
    error = _server_error("Neo.ClientError.Statement.ArithmeticError")

    def run(self, query, parameters=None, **kwargs):
        if query == "B":
            raise error
        return original_run(self, query, parameters, **kwargs)

    _FakeTransaction.run = run
    try:
        result = write_statements("op", [("A", None), ("B", None)])
    finally:
        _FakeTransaction.run = original_run

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert driver.server.committed == []


@pytest.mark.parametrize("statements", [[], (), iter([]), (s for s in [])],
                         ids=["list", "tuple", "iterator", "generator"])
def test_no_statements_raises_value_error_and_never_touches_the_driver(use, statements):
    """A programming error, and the contract callers were given. It raises rather than
    returning NOT_COMMITTED."""
    driver = use()
    with pytest.raises(ValueError):
        write_statements("op", statements)
    assert driver.session_configs == []
    assert driver.transactions == []


@pytest.mark.parametrize("error", [ValueError("x"), TypeError("x"), OverflowError("x")],
                         ids=["ValueError", "TypeError", "OverflowError"])
def test_only_an_empty_batch_raises_a_failing_generator_is_still_reported(use, error):
    """The empty-batch exception must not change how other invalid input is reported."""
    def failing():
        yield ("Q", None)
        raise error
    driver = use()
    result = write_statements("op", failing())
    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts == 0
    assert result.exception is error
    assert driver.session_configs == []


def test_a_non_iterable_batch_is_still_reported_not_raised(use):
    driver = use()
    result = write_statements("op", 7)
    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert isinstance(result.exception, TypeError)
    assert driver.session_configs == []


# ---------------------------------------------------------------- 11. resources

@pytest.mark.parametrize("script", [
    [_Attempt()],
    [_Attempt(on_run=_server_error("Neo.ClientError.Statement.SyntaxError"))],
    [_Attempt(on_commit=IncompleteCommit("x"))],
    [_Attempt(on_consume=ServiceUnavailable("x"))],
], ids=["success", "server failure", "incomplete commit", "retries exhausted"])
def test_session_and_transactions_are_closed_on_every_path(use, script):
    driver = use(*script)

    write_query("op", "CREATE (n)")

    assert driver.sessions_opened == 1
    assert driver.sessions_closed == 1
    assert all(tx.closed or tx.committed_ok for tx in driver.transactions)


def test_a_failure_closing_the_session_after_commit_is_still_committed(use):
    driver = use(close_error=ServiceUnavailable("close failed"))

    result = write_query("op", "CREATE (n)")

    assert result.outcome is WriteOutcome.COMMITTED
    assert len(driver.server.committed) == 1


def test_the_session_is_closed_when_a_base_exception_passes_through(use):
    driver = use(_Attempt(on_run=KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        write_query("op", "CREATE (n)")

    assert driver.sessions_closed == 1


# ---------------------------------------------------------------- 12. exceptions preserved

def test_the_original_exception_is_preserved_and_can_be_raised(use):
    error = _server_error("Neo.ClientError.Statement.SyntaxError")
    use(_Attempt(on_run=error))

    result = write_query("op", "CREATE (n)")

    assert result.exception is error
    assert result.exception_type == type(error).__name__
    with pytest.raises(CypherSyntaxError) as caught:
        result.raise_if_not_committed()
    assert caught.value is error


def test_raise_if_not_committed_does_nothing_after_a_commit(use):
    use()
    assert write_query("op", "CREATE (n)").raise_if_not_committed() is None


def test_ambiguous_can_be_raised_too(use):
    error = IncompleteCommit("x")
    use(_Attempt(on_commit=error))
    with pytest.raises(IncompleteCommit):
        write_query("op", "CREATE (n)").raise_if_not_committed()


def test_no_exception_escapes_as_an_unreported_failure(use):
    for error in (ResultError(None, "x"), TransactionError(None, "x"), ServiceUnavailable("x")):
        use(_Attempt(on_run=error))
        result = write_query("op", "CREATE (n)")
        assert result.outcome is WriteOutcome.NOT_COMMITTED
        assert result.exception is error


# ---------------------------------------------------------------- 13. sensitive data

def _leaky_error():
    return _server_error(
        "Neo.ClientError.Schema.ConstraintValidationFailed",
        f"Node already exists with property value = '{SECRET}' at {ADDRESS}",
    )


@pytest.mark.parametrize("attempt", [
    _Attempt(on_run=_leaky_error()),
    _Attempt(on_commit=IncompleteCommit(f"lost {ADDRESS} {SECRET}"), commit_applies=True),
], ids=["not committed", "ambiguous"])
def test_parameters_and_messages_are_absent_from_logs_and_safe_fields(use, caplog, attempt):
    use(attempt)

    with caplog.at_level(logging.DEBUG):
        result = write_query("op", f"CREATE (n {{x:'{SECRET}'}})", {"password": SECRET})

    assert result.outcome is not WriteOutcome.COMMITTED
    assert caplog.records, "a failed write must be logged"
    for record in caplog.records:
        assert SECRET not in record.getMessage()
        assert "s3cret-password" not in record.getMessage()
    assert SECRET not in result.diagnostic
    assert SECRET not in repr(result)
    assert "s3cret-password" not in repr(result)
    assert SECRET in str(result.exception), "the original must still hold the full detail"


def test_a_committed_write_does_not_log_parameters_either(use, caplog):
    use()
    with caplog.at_level(logging.DEBUG, logger="neo4j_client"):
        write_query("op", "CREATE (n {x:$x})", {"x": SECRET})
    assert all(SECRET not in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------- 14. unknown exceptions

@pytest.mark.parametrize("error", [RuntimeError("?"), ValueError("?"), OSError("?"), TypeError("?")],
                         ids=lambda e: type(e).__name__)
def test_unknown_exception_before_commit_is_ambiguous_not_a_clean_failure(use, error):
    use(_Attempt(on_run=error))

    result = write_query("op", "CREATE (n)")

    assert result.outcome is WriteOutcome.AMBIGUOUS
    assert result.exception is error
    assert "unrecognised" in result.diagnostic


@pytest.mark.parametrize("error", [RuntimeError("?"), OSError("?")], ids=lambda e: type(e).__name__)
def test_unknown_exception_during_commit_is_ambiguous(use, error):
    use(_Attempt(on_commit=error))
    assert write_query("op", "CREATE (n)").outcome is WriteOutcome.AMBIGUOUS


def test_every_failure_maps_to_a_real_outcome():
    for error in (RuntimeError("x"), IncompleteCommit("x"), ServiceUnavailable("x"), _deadlock()):
        for attempted in (True, False):
            outcome, phase = neo4j_client._classify_write_failure(error, attempted)
            assert isinstance(outcome, WriteOutcome)
            assert outcome is not WriteOutcome.COMMITTED
            assert phase


# ---------------------------------------------------------------- contract

def test_the_outcomes_are_exactly_the_three_in_the_contract():
    assert {o.value for o in WriteOutcome} == {"COMMITTED", "NOT_COMMITTED", "AMBIGUOUS"}


def test_the_result_cannot_be_changed_after_the_fact():
    from dataclasses import FrozenInstanceError
    result = WriteResult("op", WriteOutcome.COMMITTED, 1, 0.0)
    with pytest.raises(FrozenInstanceError):
        result.outcome = WriteOutcome.AMBIGUOUS


def test_classification_matches_the_installed_driver():
    """If a driver upgrade changes any of these, the classification needs re-verifying."""
    assert issubclass(IncompleteCommit, DriverError)
    assert IncompleteCommit("x").is_retryable() is False
    assert ServiceUnavailable("x").is_retryable() is True
    assert SessionExpired("x").is_retryable() is True
    assert isinstance(_deadlock(), TransientError) and _deadlock().is_retryable() is True
    terminated = _server_error("Neo.ClientError.Transaction.Terminated")
    assert terminated.is_retryable() is False
    assert terminated.code in neo4j_client._VERIFIED_COMMIT_REJECTIONS


def test_the_helper_uses_the_module_driver_and_creates_no_other():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(neo4j_client))
    created = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and getattr(n.func, "attr", "") == "driver"
               and getattr(n.func.value, "id", "") == "GraphDatabase"]
    assert len(created) == 1
    assert "neo4j_client" in sys.modules


# ================================================================ A. row-returning writes

ROWS = [{"event_id": "e1"}, {"event_id": "e2"}, {"event_id": "e3"}]


def test_rows_one_row_after_a_confirmed_commit(use):
    driver = use(rows=ROWS[:1], counters={"nodes_created": 1})

    result = write_query_rows("op", "CREATE (n) RETURN n.id AS event_id")

    assert result.outcome is WriteOutcome.COMMITTED
    assert result.records == [{"event_id": "e1"}]
    assert result.counters["nodes_created"] == 1
    assert driver.transactions[0].committed_ok is True


def test_rows_several_rows_are_materialised_in_order(use):
    use(rows=ROWS)
    result = write_query_rows("op", "Q")
    assert result.records == ROWS
    assert isinstance(result.records, list) and all(isinstance(row, dict) for row in result.records)


def test_rows_zero_rows_is_still_a_commit_and_is_an_empty_list(use):
    driver = use(rows=[], counters={"properties_set": 2})

    result = write_query_rows("op", "MATCH (n) SET n.x = 1 RETURN n")

    assert result.outcome is WriteOutcome.COMMITTED
    assert result.records == []
    assert result.counters["properties_set"] == 2
    assert len(driver.server.committed) == 1


def test_rows_are_read_and_the_result_consumed_inside_the_transaction(use):
    driver = use(rows=ROWS)
    write_query_rows("op", "Q")
    tx = driver.transactions[0]
    assert tx.data_reads == 1 and tx.consumed == 1


def test_rows_not_exposed_when_the_failure_comes_before_commit(use):
    error = _server_error("Neo.ClientError.Transaction.Terminated")
    driver = use(_Attempt(on_consume=error), rows=ROWS)

    result = write_query_rows("op", "Q")

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.records is None
    assert result.counters is None
    assert driver.server.committed == []


@pytest.mark.parametrize("commit_applies", [True, False], ids=["server committed", "server did not"])
def test_rows_not_exposed_when_the_commit_is_ambiguous(use, commit_applies):
    use(_Attempt(on_commit=IncompleteCommit("lost"), commit_applies=commit_applies), rows=ROWS)

    result = write_query_rows("op", "Q")

    assert result.outcome is WriteOutcome.AMBIGUOUS
    assert result.records is None
    assert result.counters is None


def test_rows_not_exposed_when_commit_is_rejected(use):
    use(_Attempt(on_commit=_server_error("Neo.ClientError.Transaction.Terminated")), rows=ROWS)
    result = write_query_rows("op", "Q")
    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.records is None


def test_rows_come_from_the_attempt_that_committed(use):
    driver = use(_Attempt(on_consume=ServiceUnavailable("x")), _Attempt(), rows=ROWS)
    result = write_query_rows("op", "Q")
    assert result.attempts == 2
    assert result.records == ROWS
    assert len(driver.server.committed) == 1


def test_rows_a_failure_while_reading_them_is_handled_like_any_other(use):
    driver = use(_Attempt(on_data=_server_error("Neo.ClientError.Statement.ArithmeticError")), rows=ROWS)
    result = write_query_rows("op", "Q")
    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.records is None
    assert driver.server.commits_received == 0


def test_rows_are_not_in_the_repr_because_they_can_hold_sensitive_values(use):
    use(rows=[{"password": SECRET}])
    result = write_query_rows("op", "Q")
    assert result.records == [{"password": SECRET}]
    assert SECRET not in repr(result)


def test_counters_only_functions_keep_returning_no_records(use):
    use(rows=ROWS, counters={"nodes_created": 1})
    assert write_query("op", "Q").records is None
    assert write_statements("op", [("Q", None)]).records is None
    assert write_query("op", "Q").counters["nodes_created"] == 1


def test_counters_only_functions_never_read_rows(use):
    driver = use(rows=ROWS)
    write_query("op", "Q")
    assert driver.transactions[0].data_reads == 0


def test_every_public_function_runs_through_one_engine(monkeypatch):
    calls = []
    monkeypatch.setattr(neo4j_client, "_execute_write", lambda *a, **k: calls.append((a, k)) or "result")
    assert write_query("a", "Q", {"p": 1}) == "result"
    assert write_query_rows("b", "Q") == "result"
    assert write_statements("c", [("Q", None)]) == "result"
    assert [call[1]["return_rows"] for call in calls] == [False, True, False]


def test_write_result_can_still_be_built_as_before():
    result = WriteResult("op", WriteOutcome.COMMITTED, 1, 0.0, {"nodes_created": 1})
    assert result.records is None and result.exception is None and result.committed


# ================================================================ B. classification by stage

@pytest.mark.parametrize("call", [
    lambda: write_statements("op", [("Q",)]),
    lambda: write_statements("op", ["Q"]),
    lambda: write_statements("op", 5),
    lambda: write_query("op", ""),
    lambda: write_query("op", "   "),
    lambda: write_query("op", None),
    lambda: write_query("op", "Q", "not a dict"),
    lambda: write_query("op", "Q", limits="not limits"),
], ids=["short statement", "bare string", "not iterable", "empty query", "blank query",
        "no query", "parameters not a dict", "limits wrong type"])
def test_invalid_arguments_never_reach_the_driver(use, call):
    driver = use()

    result = call()

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts == 0
    assert isinstance(result.exception, (TypeError, ValueError))
    assert "nothing submitted" in result.diagnostic
    assert driver.session_configs == []
    assert driver.transactions == []


REJECTED_PARAMETERS = {
    "non-string map key": {"a": {1: "x"}},
    "set": {"a": {1, 2}},
    "frozenset": {"a": frozenset({1})},
    "complex": {"a": 1j},
    "plain object": {"a": object()},
    "integer too large": {"a": 2 ** 63},
    "integer too small": {"a": -(2 ** 63) - 1},
    "nested in a list": {"a": [{"b": {1, 2}}]},
}


@pytest.mark.parametrize("parameters", REJECTED_PARAMETERS.values(), ids=REJECTED_PARAMETERS.keys())
def test_parameters_the_driver_always_rejects_are_stopped_first(use, parameters):
    driver = use()

    result = write_query("op", "Q", parameters)

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts == 0
    assert driver.session_configs == []


@pytest.mark.parametrize("parameters", REJECTED_PARAMETERS.values(), ids=REJECTED_PARAMETERS.keys())
def test_everything_rejected_locally_is_really_rejected_by_the_installed_packer(parameters):
    """A local rejection claims the driver could never have sent this. Checked against the
    driver's own packer, so a driver upgrade that starts accepting a type fails here."""
    from neo4j._codec.packstream.v1 import PackableBuffer, Packer

    buffer = PackableBuffer()
    with pytest.raises((TypeError, ValueError, OverflowError)):
        with buffer.tmp_buffer():
            Packer(buffer).pack(parameters)


def test_a_reference_cycle_is_rejected_rather_than_recursing_forever(use):
    cycle = []
    cycle.append(cycle)
    driver = use()
    result = write_query("op", "Q", {"a": cycle})
    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert driver.session_configs == []


@pytest.mark.parametrize("parameters", [
    {"a": 1, "b": 1.5, "c": "s", "d": True, "e": None, "f": b"x"},
    {"a": [1, [2, {"b": 3}]], "c": (1, 2)},
    {"a": 2 ** 63 - 1, "b": -(2 ** 63)},
    {"when": __import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc)},
    {"custom": type("Unknown", (), {})()},
], ids=["scalars", "nested", "integer limits", "datetime", "type not recognised"])
def test_values_that_may_be_valid_are_never_refused_locally(use, parameters):
    driver = use()
    result = write_query("op", "Q", parameters)
    assert result.outcome is WriteOutcome.COMMITTED
    assert driver.server.committed == [("Q", parameters)]


def test_messages_from_local_rejection_show_no_values(use, caplog):
    use()
    with caplog.at_level(logging.DEBUG, logger="neo4j_client"):
        result = write_query("op", SECRET and "Q", {"a": {SECRET: {1, 2}}, "b": {1: SECRET}})
    assert SECRET not in str(result.exception)
    assert SECRET not in result.diagnostic
    assert all(SECRET not in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("error", [TypeError("x"), ValueError("x"), OverflowError("x")], ids=lambda e: type(e).__name__)
@pytest.mark.parametrize("stage", ["on_run", "on_consume"])
def test_the_same_exception_types_after_submission_stay_ambiguous(use, error, stage):
    """Identical classes to local rejection. What differs is that a query may have been sent."""
    use(_Attempt(**{stage: error}))

    result = write_query("op", "Q")

    assert result.outcome is WriteOutcome.AMBIGUOUS
    assert result.attempts == 1
    assert result.exception is error


def test_an_unknown_failure_before_any_attempt_is_not_committed(use):
    """It came from opening the session, so no query can have been sent."""
    driver = use(session_error=RuntimeError("x"))
    result = write_query("op", "Q")
    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts == 0
    assert driver.server.commits_received == 0


def test_an_unknown_failure_opening_a_transaction_is_not_committed(use):
    use(begin_error=RuntimeError("x"))
    result = write_query("op", "Q")
    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts == 0


def test_an_unknown_failure_on_a_later_attempt_stays_ambiguous(use):
    """An earlier attempt submitted a query, so attempts == 0 can't be the evidence here."""
    use(_Attempt(on_consume=ServiceUnavailable("x")), _Attempt(on_run=RuntimeError("x")))
    result = write_query("op", "Q")
    assert result.outcome is WriteOutcome.AMBIGUOUS
    assert result.attempts == 2


def test_classification_is_not_decided_by_class_name_alone():
    classify = neo4j_client._classify_write_failure
    assert classify(ServiceUnavailable("x"), False)[0] is WriteOutcome.NOT_COMMITTED
    assert classify(ServiceUnavailable("x"), True)[0] is WriteOutcome.AMBIGUOUS
    assert classify(RuntimeError("x"), False, attempts=0)[0] is WriteOutcome.NOT_COMMITTED
    assert classify(RuntimeError("x"), False, attempts=1)[0] is WriteOutcome.AMBIGUOUS
    assert classify(RuntimeError("x"), True, attempts=0)[0] is WriteOutcome.AMBIGUOUS


@pytest.mark.parametrize("call", [write_query, write_query_rows])
def test_ambiguous_outcomes_are_never_replayed_by_any_row_or_counter_path(use, call):
    driver = use(_Attempt(on_commit=IncompleteCommit("x"), commit_applies=True), _Attempt())
    result = call("op", "Q")
    assert result.outcome is WriteOutcome.AMBIGUOUS
    assert result.attempts == 1
    assert len(driver.transactions) == 1
    assert len(driver.server.committed) == 1


# ================================================================ C. limits

def _bound_of(limits):
    return limits.max_retry_time + limits.connection_acquisition_timeout + limits.transaction_timeout


def test_defaults_are_finite_and_leave_time_for_shutdown():
    import dedup_engine

    DEFAULT_WRITE_LIMITS.validate()
    assert _bound_of(DEFAULT_WRITE_LIMITS) <= dedup_engine.FINAL_FLUSH_TIMEOUT * 0.6
    assert neo4j_client._WRITE_LIMIT_CEILING_SECONDS == dedup_engine.FINAL_FLUSH_TIMEOUT


def test_defaults_are_passed_to_the_real_driver_options(use):
    driver = use()

    write_query("op", "Q")

    assert driver.session_configs == [{
        "max_transaction_retry_time": DEFAULT_WRITE_LIMITS.max_retry_time,
        "connection_acquisition_timeout": DEFAULT_WRITE_LIMITS.connection_acquisition_timeout,
    }]
    assert driver.begin_kwargs[0]["timeout"] == DEFAULT_WRITE_LIMITS.transaction_timeout


def test_per_call_limits_override_the_defaults_for_that_call_only(use):
    driver = use()
    limits = WriteLimits(transaction_timeout=2.5, max_retry_time=1.5, connection_acquisition_timeout=0.5)

    write_query("op", "Q", limits=limits)
    write_query("op", "Q")

    assert driver.session_configs[0]["max_transaction_retry_time"] == 1.5
    assert driver.session_configs[0]["connection_acquisition_timeout"] == 0.5
    assert driver.begin_kwargs[0]["timeout"] == 2.5
    assert driver.session_configs[1]["max_transaction_retry_time"] == DEFAULT_WRITE_LIMITS.max_retry_time
    assert driver.begin_kwargs[1]["timeout"] == DEFAULT_WRITE_LIMITS.transaction_timeout


def test_the_real_driver_accepts_these_options_on_a_session_without_connecting():
    from neo4j import GraphDatabase
    real = GraphDatabase.driver("bolt://127.0.0.1:1", auth=("a", "b"))
    try:
        session = real.session(max_transaction_retry_time=1.5, connection_acquisition_timeout=0.5)
        assert session._config.max_transaction_retry_time == 1.5
        assert session._config.connection_acquisition_timeout == 0.5
        session.close()
    finally:
        real.close()


def test_global_driver_configuration_is_left_alone():
    from neo4j import GraphDatabase
    real = GraphDatabase.driver("bolt://127.0.0.1:1", auth=("a", "b"))
    try:
        session = real.session()
        assert session._config.max_transaction_retry_time == 30.0
        session.close()
    finally:
        real.close()


@pytest.mark.parametrize("field_name", ["transaction_timeout", "max_retry_time", "connection_acquisition_timeout"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 30.1, "5", None, True],
                         ids=["zero", "negative", "nan", "infinity", "above ceiling", "string", "none", "bool"])
def test_invalid_limits_are_rejected_before_the_driver(use, field_name, value):
    driver = use()

    result = write_query("op", "Q", limits=WriteLimits(**{field_name: value}))

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts == 0
    assert isinstance(result.exception, (TypeError, ValueError))
    assert driver.session_configs == []


def test_zero_timeout_is_rejected_because_the_server_reads_it_as_no_timeout():
    with pytest.raises(ValueError):
        WriteLimits(transaction_timeout=0).validate()


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(neo4j_client, "time", fake)
    return fake


class _TimedFailure(_Attempt):
    """Each attempt takes `seconds` of the fake clock, then fails retryably."""

    def __init__(self, fake_clock, seconds, error):
        super().__init__()
        self.fake_clock, self.seconds, self.error = fake_clock, seconds, error

    @property
    def on_run(self):
        self.fake_clock.now += self.seconds
        return self.error

    @on_run.setter
    def on_run(self, value):
        pass


def test_a_pre_commit_failure_is_retried_within_the_budget(use, clock):
    driver = use(_TimedFailure(clock, 1.0, ServiceUnavailable("x")), _Attempt(), retry_budget=30.0)

    result = write_query("op", "Q", limits=WriteLimits(max_retry_time=5.0))

    assert result.outcome is WriteOutcome.COMMITTED
    assert result.attempts == 2
    assert len(driver.server.committed) == 1


def test_no_new_attempt_starts_after_the_budget(use, clock):
    """Even with the driver willing to go on, because its own budget is far larger."""
    error = ServiceUnavailable("x")
    driver = use(_TimedFailure(clock, 2.0, error), retry_budget=30.0)

    result = write_query("op", "Q", limits=WriteLimits(max_retry_time=5.0))

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts == 3  # started at 0, 2 and 4 s; the attempt at 6 s is refused
    assert result.exception is error, "the real failure is reported, not the internal stop"
    assert "retry budget" in result.diagnostic
    assert driver.server.commits_received == 0


def test_repeated_transient_failures_end_within_the_driver_budget_too(use):
    driver = use(_Attempt(on_run=_deadlock()), retry_budget=0.05)
    result = write_query("op", "Q")
    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts >= 2
    assert driver.server.commits_received == 0


def test_a_slow_first_attempt_still_gets_its_one_run(use, clock):
    driver = use(_TimedFailure(clock, 100.0, _server_error("Neo.ClientError.Statement.SyntaxError")), retry_budget=30.0)
    result = write_query("op", "Q", limits=WriteLimits(max_retry_time=5.0))
    assert result.attempts == 1
    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert len(driver.transactions) == 1


@pytest.mark.parametrize("code", [
    "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration",
    "Neo.ClientError.Transaction.LockClientStopped",
], ids=["computing", "blocked on a lock"])
def test_a_server_side_timeout_before_commit_is_not_committed_and_not_retried(use, code):
    """The codes a real server sent (helper_revalidation_v1, B3 to B6, with COMMIT never sent)."""
    error = _server_error(code)
    driver = use(_Attempt(on_consume=error), _Attempt())

    result = write_query("op", "Q")

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts == 1
    assert driver.server.commits_received == 0


@pytest.mark.parametrize("error", [
    _server_error("Neo.ClientError.Transaction.TransactionTimedOut"),
    TimeoutError("client side"),
    ServiceUnavailable("timed out"),
], ids=["server timeout", "client timeout", "connection timeout"])
def test_a_timeout_at_commit_is_ambiguous_and_never_replayed(use, error):
    driver = use(_Attempt(on_commit=error, commit_applies=True), _Attempt())

    result = write_query("op", "Q")

    assert result.outcome is WriteOutcome.AMBIGUOUS
    assert result.attempts == 1
    assert len(driver.server.committed) == 1


def test_commit_stage_failure_is_not_retried_even_with_budget_left(use):
    driver = use(_Attempt(on_commit=_deadlock()), _Attempt(), retry_budget=30.0)
    result = write_query("op", "Q", limits=WriteLimits(max_retry_time=14.0))
    assert result.outcome is WriteOutcome.AMBIGUOUS
    assert result.attempts == 1
    assert driver.server.commits_received == 1


def test_calls_without_limits_behave_as_before(use):
    use(counters={"nodes_created": 1})
    result = write_query("op", "Q", {"a": 1})
    assert result.committed
    assert result.attempts == 1
    assert result.counters["nodes_created"] == 1
    statements = write_statements("op", [("A", None), ("B", {"b": 1})])
    assert statements.committed


# ================================================================ D. aggregate limit

AGGREGATE_ERROR = "aggregate ceiling"


def test_the_aggregate_ceiling_leaves_a_margin_inside_the_flush_budget():
    import dedup_engine

    assert neo4j_client._WRITE_LIMITS_TOTAL_CEILING_SECONDS == 25.0
    assert neo4j_client._WRITE_LIMITS_TOTAL_CEILING_SECONDS < dedup_engine.FINAL_FLUSH_TIMEOUT


def test_the_defaults_are_unchanged_and_within_the_aggregate_ceiling():
    assert DEFAULT_WRITE_LIMITS == WriteLimits(8.0, 5.0, 3.0)
    DEFAULT_WRITE_LIMITS.validate()
    assert _bound_of(DEFAULT_WRITE_LIMITS) == 16.0


@pytest.mark.parametrize("values", [
    (5.0, 5.0, 5.0),
    (10.0, 10.0, 5.0),
    (10, 10, 5),
    (23.0, 1.0, 1.0),
    (1.0, 23.0, 1.0),
    (8.5, 8.5, 8.0),
    (24.998, 0.001, 0.001),
    (0.1, 17.1, 7.8),
], ids=["below", "exactly 25", "exactly 25, ints", "mostly transaction", "mostly retry", "mostly acquisition",
        "one field near 25", "exactly 25, where plain addition gives 25.000000000000004"])
def test_a_total_up_to_the_ceiling_is_accepted_and_used_exactly(use, values):
    driver = use()
    limits = WriteLimits(*values)

    limits.validate()
    result = write_query("op", "Q", limits=limits)

    assert result.outcome is WriteOutcome.COMMITTED
    assert driver.begin_kwargs[0]["timeout"] == values[0]
    assert driver.session_configs[0] == {"max_transaction_retry_time": values[1], "connection_acquisition_timeout": values[2]}


def test_the_float_boundary_case_really_does_overshoot_with_plain_addition():
    assert 0.1 + 17.1 + 7.8 > 25.0


@pytest.mark.parametrize("values", [
    (10.0, 10.0, 5.001),
    (20.0, 5.0, 1.0),
    (5.0, 20.0, 1.0),
    (16.0, 2.0, 8.0),
    (25.0, 1.0, 1.0),
    (10, 10, 6),
    (30.0, 30.0, 8.0),
], ids=["just above", "transaction", "retry", "acquisition at its maximum", "one field at 25", "ints", "other fields at their own limit"])
def test_a_total_above_the_ceiling_is_rejected_with_a_clear_message(values):
    with pytest.raises(ValueError, match=AGGREGATE_ERROR) as caught:
        WriteLimits(*values).validate()
    message = str(caught.value)
    assert "transaction_timeout + max_retry_time + connection_acquisition_timeout" in message
    assert "25.0" in message and "lower at least one" in message


@pytest.mark.parametrize("call", [
    lambda limits: write_query("op", "Q", limits=limits),
    lambda limits: write_query_rows("op", "Q", limits=limits),
    lambda limits: write_statements("op", [("Q", None)], limits=limits),
], ids=["write_query", "write_query_rows", "write_statements"])
def test_every_public_function_rejects_it_before_the_driver_and_reports_it(use, call):
    driver = use()

    result = call(WriteLimits(transaction_timeout=20.0, max_retry_time=5.0, connection_acquisition_timeout=1.0))

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts == 0
    assert isinstance(result.exception, ValueError)
    assert AGGREGATE_ERROR in str(result.exception)
    assert driver.session_configs == []
    assert driver.server.commits_received == 0


@pytest.mark.parametrize("field_name", ["transaction_timeout", "max_retry_time", "connection_acquisition_timeout"])
def test_a_field_above_its_own_limit_gets_the_per_field_error(field_name):
    with pytest.raises(ValueError, match=f"^{field_name} must be above 0 and at most 30.0 seconds$"):
        WriteLimits(**{field_name: 30.1}).validate()


@pytest.mark.parametrize("field_name", ["transaction_timeout", "max_retry_time"])
def test_a_field_at_its_own_limit_passes_that_check_and_fails_only_the_total(field_name):
    with pytest.raises(ValueError, match=AGGREGATE_ERROR):
        WriteLimits(**{field_name: 30.0}).validate()


@pytest.mark.parametrize("field_name", ["transaction_timeout", "max_retry_time", "connection_acquisition_timeout"])
@pytest.mark.parametrize("value, error", [
    (0, ValueError), (-1, ValueError), (float("nan"), ValueError), (float("inf"), ValueError),
    ("5", TypeError), (None, TypeError), (True, TypeError), (10 ** 400, OverflowError),
], ids=["zero", "negative", "nan", "infinity", "string", "none", "bool", "too large for a float"])
def test_invalid_values_keep_their_own_errors_rather_than_the_aggregate_one(field_name, value, error):
    with pytest.raises(error) as caught:
        WriteLimits(**{field_name: value}).validate()
    assert AGGREGATE_ERROR not in str(caught.value)


def test_a_negative_field_cannot_buy_room_for_the_others():
    with pytest.raises(ValueError, match="must be above 0"):
        WriteLimits(transaction_timeout=30.0, max_retry_time=-10.0, connection_acquisition_timeout=1.0).validate()


def test_invalid_limits_are_refused_not_rewritten(use):
    driver = use()
    limits = WriteLimits(20.0, 5.0, 1.0)

    write_query("op", "Q", limits=limits)

    assert limits == WriteLimits(transaction_timeout=20.0, max_retry_time=5.0, connection_acquisition_timeout=1.0)
    assert driver.begin_kwargs == []


def test_limits_are_still_checked_at_use_not_at_construction():
    """The call reports invalid limits in its result, so building them must not raise."""
    WriteLimits(30.0, 30.0, 30.0)


# ================================================================ E. the documented timings

class _StallingConnectionSession(_SessionUnderTest):
    """The real retry loop with the driver's own default delays, unjittered. Each stalled
    connection attempt spends the whole acquisition limit on the fake clock, then fails as it
    did against a real driver (helper_revalidation_v1, exp_c2)."""

    def __init__(self, parent, requested):
        super().__init__(parent, requested)
        self._config.initial_retry_delay = SessionConfig.initial_retry_delay
        self._config.retry_delay_multiplier = SessionConfig.retry_delay_multiplier
        self.acquisition = requested["connection_acquisition_timeout"]

    def _open_transaction(self, **kwargs):
        self.parent.connection_attempts += 1
        durations = self.parent.failure_durations
        if durations is not None:
            if self.parent.connection_attempts <= len(durations):
                self.parent.clock.now += durations[self.parent.connection_attempts - 1]
                raise ServiceUnavailable("connection failed")
        elif self.parent.connection_attempts <= self.parent.stalled_attempts:
            self.parent.clock.now += self.acquisition
            raise ServiceUnavailable("stalled connection")
        super()._open_transaction(**kwargs)


class _StallingConnectionDriver(_DriverUnderTest):
    def __init__(self, *script, clock, stalled_attempts, failure_durations=None, **kwargs):
        super().__init__(*script, retry_budget=30.0, **kwargs)
        self.clock, self.stalled_attempts, self.failure_durations = clock, stalled_attempts, failure_durations
        self.connection_attempts = 0

    def session(self, **config):
        self.session_configs.append(config)
        return _StallingConnectionSession(self, config)


@pytest.fixture
def stalling(monkeypatch, clock):
    """Puts the driver's retry loop on the same fake clock as the helper."""
    import neo4j._sync.work.session as driver_session

    def advance(seconds):
        clock.now += seconds

    monkeypatch.setattr(driver_session, "monotonic", clock.monotonic)
    monkeypatch.setattr(driver_session, "sleep", advance)

    def install(*script, stalled_attempts=0, failure_durations=None):
        driver = _StallingConnectionDriver(*script, clock=clock, stalled_attempts=stalled_attempts,
                                           failure_durations=failure_durations)
        monkeypatch.setattr(neo4j_client, "driver", driver)
        return driver
    return install


def test_the_measured_12_s_when_every_connection_stalls_under_the_defaults(stalling):
    """3 stalled, 1 delay, 3 stalled, 2 delay, 3 stalled: real runs took 11.7 to 12.3 s."""
    driver = stalling(stalled_attempts=99)

    result = write_query("op", "Q")

    assert result.duration_seconds == 12.0
    assert driver.connection_attempts == 3
    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts == 0
    assert driver.transactions == []


def test_the_documented_18_s_worst_case_follows_from_the_defaults(stalling, clock):
    """3 stalled, 1 delay, 3 stalled, 2 delay, then an 8 s transaction plus about 1 s of server
    lag. The lag is the measured figure; the rest comes from the defaults and the driver."""
    limits = DEFAULT_WRITE_LIMITS
    timed_out = _server_error("Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration")
    driver = stalling(_TimedFailure(clock, limits.transaction_timeout + 1.0, timed_out), stalled_attempts=2)

    result = write_query("op", "Q")

    assert result.duration_seconds == 18.0
    assert driver.connection_attempts == 3
    assert result.attempts == 1
    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert "about 18 s" in WriteLimits.__doc__
    assert "about 16 s" not in WriteLimits.__doc__


def test_the_retry_budget_starts_after_the_first_failed_attempt(stalling):
    """A deadline would have stopped at 3 s. A budget of 1 s still allows a second attempt,
    as real runs showed: 7.06 s with these limits."""
    driver = stalling(stalled_attempts=99)

    result = write_query("op", "Q", limits=WriteLimits(max_retry_time=1.0))

    assert driver.connection_attempts == 2
    assert result.duration_seconds == 7.0


def test_a_valid_total_is_not_a_deadline_for_the_call(stalling):
    """Pins what the docstring says: within the ceiling, yet the call takes longer than it."""
    limits = WriteLimits(transaction_timeout=1.0, max_retry_time=16.0, connection_acquisition_timeout=8.0)
    limits.validate()
    stalling(stalled_attempts=99)

    result = write_query("op", "Q", limits=limits)

    assert result.duration_seconds == 27.0  # 8 stalled, 1 delay, 8 stalled, 2 delay, 8 stalled
    assert result.duration_seconds > neo4j_client._WRITE_LIMITS_TOTAL_CEILING_SECONDS


def test_the_docstring_keeps_the_silent_read_limitation_and_makes_no_deadline_claim():
    doc = " ".join(WriteLimits.__doc__.split())
    assert "A silent stalled read has no guaranteed client-side timeout" in doc
    assert "can exceed all of these figures" in doc
    assert "Neither any one of them nor their total is a deadline" in doc
    assert "promise nothing about elapsed time" in doc
    assert "That isn't the longest case" in doc
    assert "smaller of this and the driver's connection_timeout" in doc


# ================================================================ F. behaviour that must not move

COMMIT_TIMEOUT = "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration"


def test_only_the_verified_commit_rejection_means_not_committed():
    """Changing this list needs separate evidence and authorization. 7 of 7 real runs of a
    COMMIT rejected with COMMIT_TIMEOUT held no records, which doesn't prove it can't commit."""
    assert neo4j_client._VERIFIED_COMMIT_REJECTIONS == frozenset({"Neo.ClientError.Transaction.Terminated"})
    assert COMMIT_TIMEOUT not in neo4j_client._VERIFIED_COMMIT_REJECTIONS


@pytest.mark.parametrize("applied", [False, True], ids=["not applied", "applied"])
@pytest.mark.parametrize("call", [
    lambda: write_query("op", "Q"),
    lambda: write_query_rows("op", "Q"),
    lambda: write_statements("op", [("Q", None)]),
], ids=["write_query", "write_query_rows", "write_statements"])
def test_a_commit_timeout_from_the_real_server_stays_ambiguous_and_is_not_replayed(use, call, applied):
    driver = use(_Attempt(on_commit=_server_error(COMMIT_TIMEOUT), commit_applies=applied), _Attempt(), retry_budget=30.0)

    result = call()

    assert result.outcome is WriteOutcome.AMBIGUOUS
    assert result.exception_code == COMMIT_TIMEOUT
    assert result.attempts == 1
    assert len(driver.transactions) == 1
    assert driver.server.commits_received == 1
    assert result.counters is None and result.records is None


def test_a_terminated_commit_is_still_not_committed(use):
    use(_Attempt(on_commit=_server_error("Neo.ClientError.Transaction.Terminated")))
    assert write_query("op", "Q").outcome is WriteOutcome.NOT_COMMITTED


def test_an_empty_batch_raises_whatever_the_limits(use):
    driver = use()
    with pytest.raises(ValueError, match="at least one statement"):
        write_statements("op", [], limits=WriteLimits(30.0, 30.0, 30.0))
    assert driver.session_configs == []


# ================================================================ G. acquisition ceiling

ACQUISITION_ERROR = "above its maximum of 8.0 seconds"


def test_the_acquisition_ceiling_and_everything_it_sits_beside():
    assert neo4j_client._WRITE_ACQUISITION_CEILING_SECONDS == 8.0
    assert neo4j_client._WRITE_LIMITS_TOTAL_CEILING_SECONDS == 25.0
    assert neo4j_client._WRITE_LIMIT_CEILING_SECONDS == 30.0
    assert DEFAULT_WRITE_LIMITS.connection_acquisition_timeout == 3.0
    assert DEFAULT_WRITE_LIMITS.transaction_timeout == 8.0
    assert DEFAULT_WRITE_LIMITS.max_retry_time == 5.0


@pytest.mark.parametrize("acquisition", [0.001, 3.0, 7.999, 8, 8.0, math.nextafter(8.0, 0.0)],
                         ids=["tiny", "default", "just below", "8 as an int", "exactly 8", "one float below 8"])
def test_acquisition_up_to_8_is_accepted_and_used_exactly(use, acquisition):
    driver = use()

    result = write_query("op", "Q", limits=WriteLimits(connection_acquisition_timeout=acquisition))

    assert result.outcome is WriteOutcome.COMMITTED
    assert driver.session_configs[0]["connection_acquisition_timeout"] == acquisition


def test_acquisition_of_8_is_accepted_at_exactly_the_aggregate_ceiling():
    WriteLimits(transaction_timeout=8.5, max_retry_time=8.5, connection_acquisition_timeout=8.0).validate()


@pytest.mark.parametrize("acquisition", [math.nextafter(8.0, 9.0), 8.001, 8.5, 9, 20.0, 30, 30.0],
                         ids=["one float above 8", "8.001", "8.5", "9 as an int", "20", "30 as an int", "at its own limit"])
def test_acquisition_above_8_gets_the_specific_error_naming_field_value_and_maximum(acquisition):
    with pytest.raises(ValueError) as caught:
        WriteLimits(transaction_timeout=1.0, max_retry_time=1.0, connection_acquisition_timeout=acquisition).validate()
    message = str(caught.value)
    assert message.startswith("connection_acquisition_timeout is ")
    assert str(acquisition) in message
    assert ACQUISITION_ERROR in message
    assert AGGREGATE_ERROR not in message


def test_when_both_rules_fail_the_acquisition_error_comes_first():
    """The one precedence change: this total used to get the aggregate error."""
    with pytest.raises(ValueError, match=ACQUISITION_ERROR):
        WriteLimits(transaction_timeout=20.0, max_retry_time=5.0, connection_acquisition_timeout=9.0).validate()


def test_a_total_above_the_ceiling_with_a_valid_acquisition_still_gets_the_aggregate_error():
    with pytest.raises(ValueError, match=AGGREGATE_ERROR):
        WriteLimits(transaction_timeout=9.0, max_retry_time=9.0, connection_acquisition_timeout=8.0).validate()


@pytest.mark.parametrize("acquisition", [30.1, 1000.0], ids=["just above 30", "far above 30"])
def test_acquisition_above_the_per_field_limit_keeps_the_existing_error(acquisition):
    with pytest.raises(ValueError, match="^connection_acquisition_timeout must be above 0 and at most 30.0 seconds$"):
        WriteLimits(connection_acquisition_timeout=acquisition).validate()


@pytest.mark.parametrize("value, error", [
    (0, ValueError), (-1, ValueError), (float("nan"), ValueError), (float("inf"), ValueError),
    ("9", TypeError), (None, TypeError), (True, TypeError), (10 ** 400, OverflowError),
], ids=["zero", "negative", "nan", "infinity", "string", "none", "bool", "too large for a float"])
def test_invalid_acquisition_values_keep_their_existing_errors(value, error):
    with pytest.raises(error) as caught:
        WriteLimits(connection_acquisition_timeout=value).validate()
    assert ACQUISITION_ERROR not in str(caught.value)


def test_an_invalid_other_field_is_reported_ahead_of_the_acquisition_ceiling():
    with pytest.raises(TypeError, match="transaction_timeout must be a number"):
        WriteLimits(transaction_timeout="5", connection_acquisition_timeout=9.0).validate()
    with pytest.raises(ValueError, match="max_retry_time must be above 0"):
        WriteLimits(max_retry_time=float("nan"), connection_acquisition_timeout=9.0).validate()


@pytest.mark.parametrize("call", [
    lambda limits: write_statements("op", [("Q", None)], limits=limits),
    lambda limits: write_query_rows("op", "Q", limits=limits),
    lambda limits: write_query("op", "Q", limits=limits),
], ids=["write_statements", "write_query_rows", "write_query"])
@pytest.mark.parametrize("acquisition", [8.001, 20.0], ids=["just above", "far above"])
def test_an_invalid_acquisition_never_reaches_the_driver(use, call, acquisition):
    driver = use()
    limits = WriteLimits(transaction_timeout=1.0, max_retry_time=1.0, connection_acquisition_timeout=acquisition)

    result = call(limits)

    assert result.outcome is WriteOutcome.NOT_COMMITTED
    assert result.attempts == 0
    assert isinstance(result.exception, ValueError)
    assert ACQUISITION_ERROR in str(result.exception)
    assert result.counters is None and result.records is None
    assert driver.session_configs == []
    assert driver.sessions_opened == 0
    assert driver.transactions == []
    assert driver.begin_kwargs == []
    assert driver.server.commits_received == 0
    assert limits.connection_acquisition_timeout == acquisition, "refused, not rewritten"


# ================================================================ H. what the ceiling doesn't bound

def _transaction_timeout_after(clock, seconds):
    return _TimedFailure(clock, seconds, _server_error(COMMIT_TIMEOUT))


def test_connections_that_fail_fast_make_a_default_call_longer_than_18_s(stalling, clock):
    """3 stalled, then 1, 2 and 4 of delay around two instant failures, then a connection taking
    its full 3 s and the 8 s transaction with 1 s of lag. A restarting server could do this."""
    driver = stalling(_transaction_timeout_after(clock, 3.0 + 8.0 + 1.0), failure_durations=[3.0, 0.0, 0.0])

    result = write_query("op", "Q")

    assert result.duration_seconds == 22.0
    assert driver.connection_attempts == 4
    assert result.attempts == 1


def test_the_acquisition_ceiling_does_not_bring_a_valid_call_inside_the_flush_budget(stalling, clock):
    """Within every rule, and still about 50 s, because max_retry_time allows a 16 s delay. Any
    caller that needs a bound has to enforce it, as with a silent stalled read."""
    import dedup_engine

    limits = WriteLimits(transaction_timeout=2.0, max_retry_time=15.0, connection_acquisition_timeout=8.0)
    limits.validate()
    stalling(_transaction_timeout_after(clock, 8.0 + 2.0 + 1.0), failure_durations=[8.0, 0.0, 0.0, 0.0, 0.0])

    result = write_query("op", "Q", limits=limits)

    assert result.duration_seconds == 50.0
    assert result.duration_seconds > dedup_engine.FINAL_FLUSH_TIMEOUT

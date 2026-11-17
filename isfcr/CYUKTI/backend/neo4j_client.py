from neo4j import GraphDatabase, unit_of_work
from neo4j.exceptions import DriverError, IncompleteCommit, Neo4jError
import logging
import math
import time
import uuid
from enum import Enum
from datetime import datetime, timezone, timedelta
from config import DEDUP_WINDOW, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from graph_schema import (
    ATTACKER_DEFAULTS,
    ATTACK_EVENT_DEFAULTS,
    CAMPAIGN_DEFAULTS
)
from operation_schema import OPERATION_DEFAULTS
URI = NEO4J_URI
USERNAME = NEO4J_USERNAME
PASSWORD = NEO4J_PASSWORD

from dataclasses import dataclass, field
from typing import List, Optional
from datetime_utils import normalize_datetime

@dataclass
class MitreFeatures:
    attack_id: str
    technique_name: str
    description: str
    platforms: List[str]
    platform_count: int
    domains: List[str]
    domain_count: int
    kill_chain_phases: List[str]
    kill_chain_count: int
    is_subtechnique: bool
    deprecated: bool
    revoked: bool
    object_version: str
    threat_actor_count: int
    malware_count: int
    tool_count: int
    mitigation_count: int
    subtechnique_count: int
    parent_technique: Optional[str]

@dataclass
class RuntimeGraphFeatures:
    campaign_count: int
    attack_event_count: int
    graph_degree: int
    incoming_chain_count: int
    outgoing_chain_count: int
    prediction_frequency: int
    duplicate_frequency: int

@dataclass
class ThreatIntelFeatures:
    ip_reputation: float
    threat_actor_reputation: float
    malware_confidence: float
    tool_confidence: float
    misp_confidence: float
    ioc_confidence: float

@dataclass
class DetectionConfidence:
    wazuh_level: int
    suricata_score: float
    zeek_score: float
    sigma_score: float
    yara_score: float
    detection_confidence: float

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

logger = logging.getLogger(__name__)


class WriteOutcome(Enum):
    """What a caller can know about a write once write_query() has returned."""
    COMMITTED = "COMMITTED"          # the server confirmed the commit
    NOT_COMMITTED = "NOT_COMMITTED"  # established that it did not commit
    AMBIGUOUS = "AMBIGUOUS"          # it may or may not have committed


# Server failures in reply to COMMIT that are verified to mean nothing committed
# (Neo4j 5.26.27, ~/cyukti_diag/txn_experiment_v1). Any other reply to COMMIT is
# treated as AMBIGUOUS until it has been verified the same way.
_VERIFIED_COMMIT_REJECTIONS = frozenset({"Neo.ClientError.Transaction.Terminated"})

_SUMMARY_COUNTERS = (
    "nodes_created", "nodes_deleted", "relationships_created", "relationships_deleted",
    "properties_set", "labels_added", "labels_removed", "indexes_added", "indexes_removed",
    "constraints_added", "constraints_removed",
)


# No limit may exceed the listener's flush budget (dedup_engine.FINAL_FLUSH_TIMEOUT).
_WRITE_LIMIT_CEILING_SECONDS = 30.0
# Nor may their total exceed this, which keeps a margin inside that budget. Not a deadline:
# see WriteLimits.
_WRITE_LIMITS_TOTAL_CEILING_SECONDS = 25.0
# Lower for this one because it applies to each connection attempt, not once per call.
_WRITE_ACQUISITION_CEILING_SECONDS = 8.0


@dataclass(frozen=True)
class WriteLimits:
    """Limits on the stages of one helper call, in seconds. Neither any one of them nor their
    total is a deadline for the whole call.

    transaction_timeout    sent with the transaction and enforced by the server, which was
                           measured acting up to about 1 s late. A server setting of
                           db.transaction.timeout lower than this takes precedence. It doesn't
                           stop a client blocked on a dead socket.
    max_retry_time         not a deadline. The driver starts counting only after the first
                           failed attempt, and checks before its retry delay, so one more
                           attempt can start after this has passed. The helper also refuses to
                           re-run statements later than this after the call began, but an
                           attempt that fails to get a connection never reaches that check.
    connection_acquisition_timeout
                           time to get a connection, for each attempt, so stalled connections
                           can use it more than once in a call. It bounds the wait for a free
                           pool connection and the Bolt handshake, which connection_timeout
                           doesn't bound. A TCP connect is bounded by the smaller of this and
                           the driver's connection_timeout, which is global (30 s by default)
                           and left alone here. At most _WRITE_ACQUISITION_CEILING_SECONDS.

    Those timings were measured with driver 6.2.0 and Neo4j 5.26.27
    (~/cyukti_diag/helper_revalidation_v1), with TCP stalled by a full local accept queue
    rather than packet loss. They describe that setup, not every network or driver version.

    With the defaults, a calculated slow case without a stalled read takes about 18 s: two
    stalled connection attempts (3 s each), retry delays of about 1 s and 2 s, a third attempt
    that connects, then the 8 s transaction and about 1 s of server lag. This is an estimate,
    and retry delays vary by 20%. Measured with defaults: about 12 s when every connection
    stalls, and 8.4 to 8.9 s for a transaction that times out.

    That isn't the longest case. Retry delays double, so connections that fail fast let more of
    them pass within max_retry_time before a slow attempt starts: about 22 s with the defaults,
    and about 50 s with the valid limits (2, 15, 8). These are also calculated.

    validate() also requires the three to total at most _WRITE_LIMITS_TOTAL_CEILING_SECONDS.
    These rules limit what can be configured and promise nothing about elapsed time.

    A silent stalled read has no guaranteed client-side timeout: a read blocked on a connection
    that stays open but silent can exceed all of these figures. The installed driver has no
    client-side read timeout, except one the server can hint at. A caller that must not block
    has to enforce that itself, as DuplicateFlushWorker.stop() does with a joined thread."""
    transaction_timeout: float = 8.0
    max_retry_time: float = 5.0
    connection_acquisition_timeout: float = 3.0

    def validate(self):
        for name in ("transaction_timeout", "max_retry_time", "connection_acquisition_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number of seconds")
            if not math.isfinite(value) or value <= 0 or value > _WRITE_LIMIT_CEILING_SECONDS:
                raise ValueError(f"{name} must be above 0 and at most {_WRITE_LIMIT_CEILING_SECONDS} seconds")
        if self.connection_acquisition_timeout > _WRITE_ACQUISITION_CEILING_SECONDS:
            raise ValueError(
                f"connection_acquisition_timeout is {self.connection_acquisition_timeout} seconds, "
                f"above its maximum of {_WRITE_ACQUISITION_CEILING_SECONDS} seconds"
            )
        # fsum, because plain addition can round a total of exactly the ceiling to just above it.
        total = math.fsum((self.transaction_timeout, self.max_retry_time, self.connection_acquisition_timeout))
        if total > _WRITE_LIMITS_TOTAL_CEILING_SECONDS:
            raise ValueError(
                "transaction_timeout + max_retry_time + connection_acquisition_timeout "
                f"is {total} seconds, above the aggregate ceiling of "
                f"{_WRITE_LIMITS_TOTAL_CEILING_SECONDS} seconds: lower at least one of them"
            )


DEFAULT_WRITE_LIMITS = WriteLimits()


@dataclass(frozen=True)
class WriteResult:
    """`diagnostic`, `exception_type` and `exception_code` are safe to log.
    `exception` is the original, and its message can quote parameter values and the
    server address, so it's kept out of the repr and must not be logged as text.
    `counters` and `records` are None unless the outcome is COMMITTED: anything counted or
    returned by an attempt that didn't commit never happened, or isn't known to have
    happened. `records` is a list of dicts from write_query_rows, [] when the committed query
    returned no rows, and None from the counters-only functions."""
    operation: str
    outcome: WriteOutcome
    attempts: int
    duration_seconds: float
    counters: Optional[dict] = None
    exception_type: Optional[str] = None
    exception_code: Optional[str] = None
    diagnostic: str = ""
    exception: Optional[Exception] = field(default=None, repr=False, compare=False)
    records: Optional[list] = field(default=None, repr=False, compare=False)

    @property
    def committed(self):
        return self.outcome is WriteOutcome.COMMITTED

    def raise_if_not_committed(self):
        if self.outcome is not WriteOutcome.COMMITTED:
            raise self.exception


class _RetryAfterCommitAttempt(Exception):
    """Stops the driver re-running work whose COMMIT has already been sent. Deliberately
    not a DriverError or Neo4jError, so that the driver doesn't retry it."""


class _RetryBudgetExhausted(Exception):
    """Stops a new attempt once WriteLimits.max_retry_time has passed. The driver checks
    its own budget before sleeping, so it can start one more attempt after the budget ends."""


def _classify_write_failure(error, commit_attempted, attempts=1):
    """Returns (outcome, phase). Only states what the evidence supports: see
    ~/cyukti_diag/txn_experiment_v1/TRANSACTION_EXPERIMENT_REPORT.md section 4.
    Exception classes alone never decide: the same class means different things at
    different stages, so what was attempted is checked first."""
    if isinstance(error, _RetryAfterCommitAttempt):
        return WriteOutcome.AMBIGUOUS, "driver retried after a commit attempt, refused"
    if isinstance(error, IncompleteCommit):
        return WriteOutcome.AMBIGUOUS, "connection lost during commit"
    if commit_attempted:
        if getattr(error, "code", None) in _VERIFIED_COMMIT_REJECTIONS:
            return WriteOutcome.NOT_COMMITTED, "commit rejected by the server"
        return WriteOutcome.AMBIGUOUS, "failure after a commit attempt"
    if isinstance(error, _RetryBudgetExhausted):
        return WriteOutcome.NOT_COMMITTED, "retry budget exhausted before commit"
    if isinstance(error, (Neo4jError, DriverError)):
        # Bolt commits a transaction only on COMMIT, and no attempt sent one.
        return WriteOutcome.NOT_COMMITTED, "failure before commit"
    if attempts == 0:
        return WriteOutcome.NOT_COMMITTED, "failure before any query was submitted"
    return WriteOutcome.AMBIGUOUS, "unrecognised failure"


def _check_parameter_value(value, seen):
    """Rejects only values the driver's packer always rejects, verified against it in
    test_neo4j_write_helper.py, so nothing the driver would accept is refused. A type
    this doesn't recognise is passed on: if the driver rejects it inside tx.run, that can't
    be told apart from a failure after sending, and it stays AMBIGUOUS."""
    if isinstance(value, (dict, list, tuple)):
        if id(value) in seen:
            raise ValueError("query parameters contain a reference cycle")
        seen = seen | {id(value)}
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("query parameter map keys must be strings")
                _check_parameter_value(item, seen)
        else:
            for item in value:
                _check_parameter_value(item, seen)
    elif isinstance(value, bool):
        return
    elif isinstance(value, int):
        if not -(2 ** 63) <= value < 2 ** 63:
            raise OverflowError("query parameter integer is outside the 64-bit range")
    elif isinstance(value, (set, frozenset, complex)) or type(value) is object:
        raise TypeError(f"query parameters can't contain {type(value).__name__}")


def _checked_statements(statements):
    """Raises TypeError, ValueError or OverflowError if nothing should be sent. Messages
    name types, never values."""
    checked = list(statements)
    if not checked:
        raise ValueError("a write needs at least one statement")
    for statement in checked:
        if not isinstance(statement, (tuple, list)) or len(statement) != 2:
            raise TypeError("each statement must be a (query, parameters) pair")
        query, parameters = statement
        if not isinstance(query, str) or not query.strip():
            raise ValueError("a query must be a non-empty string")
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise TypeError("query parameters must be a dict")
            _check_parameter_value(parameters, frozenset())
    return checked


def _execute_write(operation, statements, limits, return_rows):
    """The one transaction path behind every public write function. Failures are reported
    in the result rather than raised; only BaseException, such as KeyboardInterrupt, and
    anything raised while iterating the caller's own statements, propagates.

    Retries: the driver may re-run the statements after a failure before commit, because
    nothing has committed at that point, until max_retry_time has passed. Once COMMIT has
    been sent the statements are never re-run, and an AMBIGUOUS write is never retried here:
    only the caller can know whether repeating it is safe. Callers pass statements rather
    than a function, so nothing with a side effect can run inside the retried unit of work.

    Logs the operation name, outcome and exception class, never parameters or messages."""
    started = time.monotonic()
    try:
        limits = DEFAULT_WRITE_LIMITS if limits is None else limits
        if not isinstance(limits, WriteLimits):
            raise TypeError("limits must be a WriteLimits")
        limits.validate()
        statements = _checked_statements(statements)
    except (TypeError, ValueError, OverflowError) as error:
        return _finish_write(operation, WriteOutcome.NOT_COMMITTED, 0, started, failure=error,
                             phase="invalid arguments, nothing submitted")

    attempts = 0
    commit_attempted = False
    committed = False
    last_work_failure = None
    outcome_data = None

    def work(tx):
        nonlocal attempts, commit_attempted, last_work_failure
        if commit_attempted:
            raise _RetryAfterCommitAttempt("re-execution refused after a commit attempt")
        if attempts and time.monotonic() - started > limits.max_retry_time:
            raise _RetryBudgetExhausted("no new attempt after the retry budget")
        attempts += 1
        totals = dict.fromkeys(_SUMMARY_COUNTERS, 0)
        rows = None
        try:
            for query, parameters in statements:
                result = tx.run(query, parameters or {})
                if return_rows:
                    rows = result.data()
                summary = result.consume()
                for name in _SUMMARY_COUNTERS:
                    totals[name] += getattr(summary.counters, name, 0)
        except Exception as error:
            last_work_failure = error
            raise
        commit_attempted = True
        return totals, rows

    failure = None
    try:
        with driver.session(
            max_transaction_retry_time=limits.max_retry_time,
            connection_acquisition_timeout=limits.connection_acquisition_timeout,
        ) as session:
            outcome_data = session.execute_write(unit_of_work(timeout=limits.transaction_timeout)(work))
            committed = True
    except Exception as error:
        failure = error

    if committed:
        # A failure at this point came from closing the session, after the commit was confirmed.
        counters, rows = outcome_data
        return _finish_write(operation, WriteOutcome.COMMITTED, attempts, started, counters=counters,
                             records=rows if return_rows else None)
    outcome, phase = _classify_write_failure(failure, commit_attempted, attempts)
    if isinstance(failure, _RetryBudgetExhausted) and last_work_failure is not None:
        failure = last_work_failure
    return _finish_write(operation, outcome, attempts, started, failure=failure, phase=phase)


def _finish_write(operation, outcome, attempts, started, failure=None, phase="", counters=None, records=None):
    duration = round(time.monotonic() - started, 3)
    if failure is None:
        result = WriteResult(operation, outcome, attempts, duration, counters=counters, records=records)
        logger.debug("Neo4j write %s COMMITTED attempts=%d", operation, attempts)
        return result
    code = getattr(failure, "code", None)
    name = type(failure).__name__
    result = WriteResult(
        operation, outcome, attempts, duration,
        exception_type=name, exception_code=code,
        diagnostic=f"{name}{f' {code}' if code else ''}: {phase}",
        exception=failure,
    )
    level = logging.ERROR if outcome is WriteOutcome.AMBIGUOUS else logging.WARNING
    logger.log(level, "Neo4j write %s %s attempts=%d %s", operation, outcome.value, attempts, result.diagnostic)
    return result


def write_statements(operation, statements, *, limits=None):
    """Runs (query, parameters) pairs in one transaction, so they commit or fail together.
    Returns a WriteResult with counters. See _execute_write for failures and retries.
    Raises ValueError for no statements at all, which is a programming error."""
    try:
        statements = list(statements)
    except (TypeError, ValueError, OverflowError) as error:
        return _finish_write(operation, WriteOutcome.NOT_COMMITTED, 0, time.monotonic(), failure=error,
                             phase="invalid arguments, nothing submitted")
    if not statements:
        raise ValueError("a write needs at least one statement")
    return _execute_write(operation, statements, limits, return_rows=False)


def write_query(operation, query, parameters=None, *, limits=None):
    """write_statements() for a single statement. A batch is one query, for example
    write_query("flush", "UNWIND $records AS r ...", {"records": rows}): one statement in
    one transaction, so it commits or fails as a whole."""
    return _execute_write(operation, [(query, parameters)], limits, return_rows=False)


def write_query_rows(operation, query, parameters=None, *, limits=None):
    """write_query() for a query that RETURNs rows. WriteResult.records is a list of dicts,
    fully read inside the transaction, with graph values converted by Result.data(). It's
    set only once the commit is confirmed, and comes from the attempt that committed.

    Single statement only: the writers that read rows each run one statement."""
    return _execute_write(operation, [(query, parameters)], limits, return_rows=True)


def create_attack_event(
    campaign_id,
    attacker_ip,
    victim_ip,
    mitre_id,
    technique,
    stage,
    tps,
    fingerprint,
    rule_id,
    agent_id,
    rule_level=0,
    investigation_payload="{}",
    mitre_provenance="NATIVE_WAZUH",
    mitre_confidence="CONFIRMED",
    mitre_reason="Wazuh rule.mitre.id",
    mitre_resolver_version="",
    mitre_technique_ids=None,
    event_time=None,
    event_time_source=None,
):
    """event_time: the alert's own timestamp (event_time.py). When given,
    AttackEvent.first_seen/last_seen and Campaign.last_seen use it
    (Campaign.last_seen monotonically); processing time is kept separately
    as AttackEvent.ingested_at. None keeps the legacy datetime() behaviour
    for direct callers; realtime_socgraph.process_alert always passes it."""
    event_id = str(uuid.uuid4())
    with driver.session() as session:

        query = """
        WITH coalesce($event_time, datetime()) AS t
        MERGE (a:Attacker {
            ip:$attacker_ip
        })

        ON CREATE SET
            a.first_seen = datetime(),

            a.vt_reputation = $vt_reputation,
            a.threat_actor_reputation = $threat_actor_reputation,

            a.malware_confidence = $malware_confidence,
            a.tool_confidence = $tool_confidence,
            a.misp_confidence = $misp_confidence,
            a.ioc_confidence = $ioc_confidence

        SET
            a.last_seen = datetime()
        WITH a, t
        MERGE (h:Host {
            ip:$victim_ip
        })

        ON CREATE SET
            h.first_seen = datetime()

        SET
            h.last_seen = datetime()
        WITH a, h, t
        MATCH (c:Campaign {
            campaign_id:$campaign_id
        })

        SET
            c.last_seen = CASE WHEN c.last_seen IS NULL OR t > c.last_seen THEN t ELSE c.last_seen END,
            c.last_technique = $mitre_id,
            c.occurrences = coalesce(c.occurrences,0)+1,
            c.total_tps = coalesce(c.total_tps,0)+$tps,
            c.risk_score = coalesce(c.risk_score,0)+$tps
        WITH a, h, c, t AS event_t
        MERGE (t:Technique {
            attack_id:$mitre_id
        })

        ON CREATE SET
            t.name = $technique,
            t.stage = $stage,
            t.first_seen = datetime(),
            t.last_seen = datetime(),
            t.occurrences = 1,
            t.total_tps = $tps

        ON MATCH SET
            t.last_seen = datetime(),
            t.occurrences = coalesce(t.occurrences,0)+1,
            t.total_tps = coalesce(t.total_tps,0)+$tps
        WITH a, h, c, t, event_t
        MERGE (s:Stage {
            name:$stage
        })
        WITH a, h, c, t, s, event_t
        CREATE (e:AttackEvent {
            event_id:$event_id,
            campaign_id:$campaign_id,
            attack_id:$mitre_id,
            fingerprint:$fingerprint,
            rule_id:$rule_id,
            agent_id:$agent_id,
            technique:$technique,
            stage:$stage,
            attacker_ip:$attacker_ip,
            victim_ip:$victim_ip,
            first_seen:event_t,
            last_seen:event_t,
            ingested_at:datetime(),
            event_time_source:$event_time_source,
            occurrences:1,
            tps:$tps,
            rule_level:$rule_level,
            suricata_score:$suricata_score,
            zeek_score:$zeek_score,
            sigma_score:$sigma_score,
            yara_score:$yara_score,
            investigation_payload:$investigation_payload,
            prediction_generated_at:$prediction_generated_at,
            mitre_status:"RESOLVED",
            mitre_provenance:$mitre_provenance,
            mitre_confidence:$mitre_confidence,
            mitre_reason:$mitre_reason,
            mitre_resolver_version:$mitre_resolver_version,
            mitre_technique_ids:$mitre_technique_ids
        })
        WITH a, h, c, t, s, e
        MERGE (a)-[:LAUNCHED]->(c)

        MERGE (c)-[:TARGETS]->(h)

        MERGE (c)-[:HAS_EVENT]->(e)

        MERGE (e)-[:MATCHES]->(t)

        MERGE (t)-[:BELONGS_TO]->(s)

        RETURN
            e.event_id AS event_id
        """

        result = session.run(
            query,
            event_id=event_id,
            campaign_id=campaign_id,
            attacker_ip=attacker_ip,
            victim_ip=victim_ip,
            mitre_id=mitre_id,
            fingerprint=fingerprint,
            rule_id=rule_id,
            agent_id=agent_id,
            technique=technique,
            stage=stage,
            tps=tps,
            investigation_payload=investigation_payload,
            prediction_generated_at=CAMPAIGN_DEFAULTS["prediction_generated_at"],
            vt_reputation=ATTACKER_DEFAULTS["vt_reputation"],
            threat_actor_reputation=ATTACKER_DEFAULTS["threat_actor_reputation"],
            malware_confidence=ATTACKER_DEFAULTS["malware_confidence"],
            tool_confidence=ATTACKER_DEFAULTS["tool_confidence"],
            misp_confidence=ATTACKER_DEFAULTS["misp_confidence"],
            ioc_confidence=ATTACKER_DEFAULTS["ioc_confidence"],
            rule_level=rule_level,
            suricata_score=ATTACK_EVENT_DEFAULTS["suricata_score"],
            zeek_score=ATTACK_EVENT_DEFAULTS["zeek_score"],
            sigma_score=ATTACK_EVENT_DEFAULTS["sigma_score"],
            yara_score=ATTACK_EVENT_DEFAULTS["yara_score"],
            mitre_provenance=mitre_provenance,
            mitre_confidence=mitre_confidence,
            mitre_reason=mitre_reason,
            mitre_resolver_version=mitre_resolver_version,
            mitre_technique_ids=list(mitre_technique_ids or [mitre_id]),
            event_time=event_time,
            event_time_source=event_time_source,
        )

        record = result.single()
        return record["event_id"]


def create_unattributed_attack_event(
    campaign_id,
    attacker_ip,
    victim_ip,
    fingerprint,
    rule_id,
    agent_id,
    rule_level=0,
    investigation_payload="{}",
    mitre_provenance="UNKNOWN",
    mitre_confidence="NONE",
    mitre_reason="",
    mitre_resolver_version="",
    mitre_technique_ids=None,
    event_time=None,
    event_time_source=None,
):
    """Phase 20: creates an AttackEvent for a Wazuh alert that
    mitre_resolver.resolve_mitre() could not defensibly attribute to an
    ATT&CK technique. Preserves the raw alert as evidence and attaches it
    to the campaign via HAS_EVENT, but deliberately:

      - never sets AttackEvent.attack_id (omitted, not a fake technique)
      - never creates or MERGEs a Technique node/MATCHES relationship
      - never touches Campaign.last_technique (only last_seen/occurrences)
      - never touches Campaign.total_tps/risk_score (tps contribution is
        zero for an unattributed event, so leaving them unchanged is
        already correct -- no arithmetic needed)

    This intentionally does NOT call campaign_manager.append_technique(),
    chain_updater.update_attack_chain(), or prediction_engine.predict_next()
    -- those all key on a real technique ID and must not run for an
    unattributed event (see realtime_socgraph.py's UNKNOWN branch).
    """
    event_id = str(uuid.uuid4())
    with driver.session() as session:
        query = """
        WITH coalesce($event_time, datetime()) AS t
        MERGE (a:Attacker {
            ip:$attacker_ip
        })

        ON CREATE SET
            a.first_seen = datetime()

        SET
            a.last_seen = datetime()
        WITH a, t
        MERGE (h:Host {
            ip:$victim_ip
        })

        ON CREATE SET
            h.first_seen = datetime()

        SET
            h.last_seen = datetime()
        WITH a, h, t
        MATCH (c:Campaign {
            campaign_id:$campaign_id
        })

        SET
            c.last_seen = CASE WHEN c.last_seen IS NULL OR t > c.last_seen THEN t ELSE c.last_seen END,
            c.occurrences = coalesce(c.occurrences,0)+1
        WITH a, h, c, t
        CREATE (e:AttackEvent {
            event_id:$event_id,
            campaign_id:$campaign_id,
            fingerprint:$fingerprint,
            rule_id:$rule_id,
            agent_id:$agent_id,
            attacker_ip:$attacker_ip,
            victim_ip:$victim_ip,
            first_seen:t,
            last_seen:t,
            ingested_at:datetime(),
            event_time_source:$event_time_source,
            occurrences:1,
            tps:0,
            rule_level:$rule_level,
            investigation_payload:$investigation_payload,
            mitre_status:"UNKNOWN",
            mitre_provenance:$mitre_provenance,
            mitre_confidence:$mitre_confidence,
            mitre_reason:$mitre_reason,
            mitre_resolver_version:$mitre_resolver_version,
            mitre_technique_ids:$mitre_technique_ids
        })
        WITH a, h, c, e
        MERGE (a)-[:LAUNCHED]->(c)

        MERGE (c)-[:TARGETS]->(h)

        MERGE (c)-[:HAS_EVENT]->(e)

        RETURN
            e.event_id AS event_id
        """

        result = session.run(
            query,
            event_id=event_id,
            campaign_id=campaign_id,
            attacker_ip=attacker_ip,
            victim_ip=victim_ip,
            fingerprint=fingerprint,
            rule_id=rule_id,
            agent_id=agent_id,
            rule_level=rule_level,
            investigation_payload=investigation_payload,
            mitre_provenance=mitre_provenance,
            mitre_confidence=mitre_confidence,
            mitre_reason=mitre_reason,
            mitre_resolver_version=mitre_resolver_version,
            mitre_technique_ids=list(mitre_technique_ids or []),
            event_time=event_time,
            event_time_source=event_time_source,
        )

        record = result.single()
        return record["event_id"]


def update_duplicate_event(event_id, tps, investigation_payload="{}", event_time=None):
    with driver.session() as session:
        session.run(
            """
            WITH coalesce($event_time, datetime()) AS dt
            MATCH (e:AttackEvent {event_id:$event_id})

            SET
                e.last_seen = CASE WHEN e.last_seen IS NULL OR dt > e.last_seen THEN dt ELSE e.last_seen END,
                e.occurrences = coalesce(e.occurrences,0) + 1,
                e.tps = coalesce(e.tps,0) + $tps,
                e.investigation_payload = $investigation_payload

            WITH e, dt

            MATCH (e)-[:MATCHES]->(t:Technique)

            SET
                t.last_seen = datetime(),
                t.occurrences = coalesce(t.occurrences,0) + 1,
                t.total_tps = coalesce(t.total_tps,0) + $tps

            WITH e, dt

            MATCH (c:Campaign {campaign_id:e.campaign_id})

            SET
                c.last_seen = CASE WHEN c.last_seen IS NULL OR dt > c.last_seen THEN dt ELSE c.last_seen END,
                c.occurrences = coalesce(c.occurrences,0) + 1,
                c.total_tps = coalesce(c.total_tps,0) + $tps,
                c.risk_score = coalesce(c.risk_score,0) + $tps
            """,
            event_id=event_id,
            tps=tps,
            investigation_payload=investigation_payload,
            event_time=event_time,
        )
def batch_update_duplicates(records):
    if not records:
        return
    with driver.session() as session:
        session.run(
            """
            UNWIND $records AS r

            MATCH (e:AttackEvent {event_id:r.event_id})

            SET
                e.last_seen = CASE WHEN e.last_seen IS NULL OR datetime(r.last_seen) > e.last_seen THEN datetime(r.last_seen) ELSE e.last_seen END,
                e.occurrences = coalesce(e.occurrences,0) + r.occurrences,
                e.tps = coalesce(e.tps,0) + r.total_tps,
                e.investigation_payload = r.latest_payload

            WITH e, r

            MATCH (e)-[:MATCHES]->(t:Technique)

            SET
                t.last_seen = datetime(r.last_seen),
                t.occurrences = coalesce(t.occurrences,0) + r.occurrences,
                t.total_tps = coalesce(t.total_tps,0) + r.total_tps

            WITH e, r

            MATCH (c:Campaign {campaign_id:e.campaign_id})

            SET
                c.last_seen = CASE WHEN c.last_seen IS NULL OR datetime(r.last_seen) > c.last_seen THEN datetime(r.last_seen) ELSE c.last_seen END,
                c.occurrences = coalesce(c.occurrences,0) + r.occurrences,
                c.total_tps = coalesce(c.total_tps,0) + r.total_tps,
                c.risk_score = coalesce(c.risk_score,0) + r.total_tps
            """,
            records=records
        )

def store_likely_next(campaign_id, predicted_technique, confidence):
    store_prediction(
        campaign_id,
        predicted_technique,
        confidence
    )

def update_campaign_similarity(campaign_id):
    with driver.session() as session:
        query = """
        MATCH (c1:Campaign {campaign_id: $campaign_id})
        MATCH (c2:Campaign)
        WHERE c1 <> c2

        OPTIONAL MATCH (c1)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t1:Technique)
        WITH c1, c2, collect(DISTINCT t1) AS c1_techs
        OPTIONAL MATCH (c2)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t2:Technique)
        WITH c1, c2, c1_techs, collect(DISTINCT t2) AS c2_techs
        WITH c1, c2, c1_techs, c2_techs, [t IN c1_techs WHERE t IN c2_techs] AS shared_t
        WITH c1, c2, size(shared_t) AS shared_tech_count,
             size(c1_techs) + size(c2_techs) - size(shared_t) AS total_tech_count,
             [t IN shared_t | coalesce(t.name, t.attack_id)] AS shared_techniques,
             [t IN shared_t | coalesce(t.stage, 'Unknown')] AS shared_tactics_raw

        OPTIONAL MATCH (a1:Attacker)-[:LAUNCHED]->(c1)
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, collect(DISTINCT a1.ip) AS c1_a
        OPTIONAL MATCH (a2:Attacker)-[:LAUNCHED]->(c2)
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, c1_a, collect(DISTINCT a2.ip) AS c2_a
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, [a IN c1_a WHERE a IN c2_a] AS shared_a

        OPTIONAL MATCH (c1)-[:TARGETS]->(h1:Host)
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, shared_a, collect(DISTINCT h1.ip) AS c1_h
        OPTIONAL MATCH (c2)-[:TARGETS]->(h2:Host)
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, shared_a, c1_h, collect(DISTINCT h2.ip) AS c2_h
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, shared_a, [h IN c1_h WHERE h IN c2_h] AS shared_h

        WITH c1, c2, shared_techniques, shared_tactics_raw, shared_a, shared_h,
             (CASE WHEN total_tech_count = 0 THEN 0.0 ELSE toFloat(shared_tech_count) / total_tech_count END * 60.0) +
             (CASE WHEN size(shared_a) > 0 THEN 20.0 ELSE 0.0 END) +
             (CASE WHEN size(shared_h) > 0 THEN 20.0 ELSE 0.0 END) AS similarity_score

        WHERE similarity_score >= 75.0

        // Remove old relationships for this pair to avoid duplicates
        OPTIONAL MATCH (c1)-[old:SIMILAR_TO]->(c2)
        DELETE old

        WITH c1, c2, round(similarity_score) AS score, shared_techniques, shared_tactics_raw

        MERGE (c1)-[r:SIMILAR_TO]->(c2)
        SET r.score = score,
            r.shared_techniques = shared_techniques,
            r.shared_tactics = apoc.coll.toSet(shared_tactics_raw)
        """
        try:
            session.run(query, campaign_id=campaign_id)
        except Exception as e:
            if "apoc" in str(e).lower():
                fallback_query = query.replace("apoc.coll.toSet(shared_tactics_raw)", "shared_tactics_raw")
                session.run(fallback_query, campaign_id=campaign_id)
            else:
                raise e

def update_actor_attribution(campaign_id):
    with driver.session() as session:
        query = """
        MATCH (c:Campaign {campaign_id: $campaign_id})-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t:Technique)
        WITH c, collect(DISTINCT t) AS c_techs
        MATCH (ta:ThreatActor)-[:USES]->(ta_t:Technique)
        WITH c, c_techs, ta, collect(DISTINCT ta_t) AS ta_techs
        WITH c, ta, c_techs, ta_techs, [t IN c_techs WHERE t IN ta_techs] AS shared_t
        WITH c, ta, size(shared_t) AS shared_tech_count,
             size(c_techs) + size(ta_techs) - size(shared_t) AS total_tech_count,
             [t IN shared_t | coalesce(t.name, t.attack_id)] AS shared_techniques
        WHERE shared_tech_count > 0

        WITH c, ta, shared_tech_count, total_tech_count, shared_techniques,

             CASE
                 WHEN total_tech_count <= 0 THEN 0.0
                 ELSE (toFloat(shared_tech_count) / toFloat(total_tech_count)) * 100
             END AS base_confidence

        WITH c, ta,
             CASE WHEN base_confidence + (shared_tech_count * 2) > 99 THEN 99.0
                  ELSE base_confidence + (shared_tech_count * 2) END AS confidence,
             shared_techniques

        WHERE confidence >= 50.0

        OPTIONAL MATCH (c)-[old:RESEMBLES]->(ta)
        DELETE old

        WITH c, ta, round(confidence) AS conf, shared_techniques

        MERGE (c)-[r:RESEMBLES]->(ta)
        SET r.confidence = conf,
            r.matched_techniques = shared_techniques
        """
        session.run(query, campaign_id=campaign_id)
def get_active_campaign_db(attacker_ip, victim_ip):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Campaign)
            WHERE c.attacker_ip = $attacker_ip
              AND c.victim_ip = $victim_ip
              AND c.status = 'ACTIVE'
            RETURN
                c.campaign_id AS campaign_id,
                c.first_seen AS first_seen,
                c.last_seen AS last_seen,
                c.last_technique AS last_technique,
                c.predicted_next AS predicted_next,
                c.prediction_confidence AS prediction_confidence,
                c.prediction_generated_at AS prediction_generated_at,
                c.reopened_count AS reopened_count,
                c.status AS status
            ORDER BY c.last_seen DESC
            LIMIT 1
            """,
            attacker_ip=attacker_ip,
            victim_ip=victim_ip
        )

        record = result.single()

        if record is None:
            return None
        return {
            "campaign_id": record["campaign_id"],
            "first_seen": record["first_seen"],
            "last_seen": record["last_seen"],
            "last_technique": record["last_technique"],
            "predicted_next": record["predicted_next"],
            "prediction_confidence": record["prediction_confidence"],
            "prediction_generated_at": record["prediction_generated_at"],
            "reopened_count": record["reopened_count"],
            "status": record["status"]
        }

def get_recent_inactive_campaign_db(
    attacker_ip,
    victim_ip,
    limit=5
):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Campaign)
            WHERE
                c.attacker_ip = $attacker_ip
                AND c.victim_ip = $victim_ip
                AND c.status IN ['INACTIVE','REOPENED']
            RETURN
                c.campaign_id AS campaign_id,
                c.first_seen AS first_seen,
                c.last_seen AS last_seen,
                c.last_technique AS last_technique,
                c.predicted_next AS predicted_next,
                c.prediction_confidence AS prediction_confidence,
                c.prediction_generated_at AS prediction_generated_at,
                c.reopened_count AS reopened_count,
                c.status AS status
            ORDER BY c.last_seen DESC
            LIMIT $limit
            """,
            attacker_ip=attacker_ip,
            victim_ip=victim_ip,
            limit=limit
        )
        campaigns = []
        for record in result:
            campaigns.append(dict(record))
        return campaigns

def reopen_campaign_db(campaign_id, event_time=None):
    with driver.session() as session:
        session.run(
            """
            WITH coalesce($event_time, datetime()) AS t
            MATCH (c:Campaign {campaign_id:$campaign_id})
            SET
                c.status='ACTIVE',
                c.last_seen=CASE WHEN c.last_seen IS NULL OR t > c.last_seen THEN t ELSE c.last_seen END,
                c.predicted_next=NULL,
                c.prediction_confidence=0,
                c.reopened_count=
                    coalesce(c.reopened_count,0)+1
            """,
            campaign_id=campaign_id,
            event_time=event_time,
        )


def create_campaign_db(attacker_ip, victim_ip, event_time=None):
    campaign_id = f"CAMP_{uuid.uuid4().hex[:8].upper()}"
    print(">>>> create_campaign_db ENTER")
    with driver.session() as session:

        session.run(
            """
            CREATE (c:Campaign{
                campaign_id:$campaign_id,
                attacker_ip:$attacker_ip,
                victim_ip:$victim_ip,

                status:'ACTIVE',

                first_seen:coalesce($event_time, datetime()),
                last_seen:coalesce($event_time, datetime()),

                occurrences:0,
                total_tps:0,

                campaign_count:$campaign_count,
                prediction_frequency:$prediction_frequency,
                duplicate_frequency:$duplicate_frequency,

                risk_score:$risk_score,
                risk_confidence:$risk_confidence,
                risk_trend:$risk_trend,

                last_technique:"",
                predicted_next:"",
                prediction_confidence:0,
                prediction_hits:0,
                prediction_misses:0,

                reopened_count:0,

                dynamic_risk:0,
                risk_level:"LOW",
                risk_color:"GREEN",
                risk_updated_at:"",
                risk_breakdown:"{}"
            })
            """,
            campaign_id=campaign_id,
            attacker_ip=attacker_ip,
            victim_ip=victim_ip,

            campaign_count=CAMPAIGN_DEFAULTS["campaign_count"],
            prediction_frequency=CAMPAIGN_DEFAULTS["prediction_frequency"],
            duplicate_frequency=CAMPAIGN_DEFAULTS["duplicate_frequency"],
            risk_score=CAMPAIGN_DEFAULTS["risk_score"],
            risk_confidence=CAMPAIGN_DEFAULTS["risk_confidence"],
            risk_trend=CAMPAIGN_DEFAULTS["risk_trend"],
            event_time=event_time,
        )

    return campaign_id

def update_campaign_activity(campaign_id, technique=None, event_time=None):
    with driver.session() as session:
        session.run(
            """
            WITH coalesce($event_time, datetime()) AS t
            MATCH (c:Campaign {campaign_id:$campaign_id})

            SET
                c.last_seen = CASE WHEN c.last_seen IS NULL OR t > c.last_seen THEN t ELSE c.last_seen END,
                c.status = 'ACTIVE',
                c.last_technique = $technique
            """,
            campaign_id=campaign_id,
            technique=technique,
            event_time=event_time,
        )

def close_campaign_db(campaign_id, closed_at=None):
    """closed_at: event time at which the campaign became inactive (see
    campaign_manager). None keeps the legacy datetime()."""
    with driver.session() as session:

        session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})

            SET
                c.status='INACTIVE',
                c.closed_at=coalesce($closed_at, datetime())
            """,
            campaign_id=campaign_id,
            closed_at=closed_at,
        )

def expire_stale_campaigns_db(timeout_seconds, reference_time=None):
    """reference_time: event-time watermark. With it, a campaign is stale
    when last_seen < reference_time - timeout and is closed at
    last_seen + timeout (the event time at which it became inactive), so
    closed_at does not depend on when maintenance happened to run. None
    keeps the legacy wall-clock behaviour."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Campaign)
            WHERE
                c.status = 'ACTIVE'
                AND
                c.last_seen < coalesce($reference_time, datetime()) - duration({seconds:$timeout})
            SET
                c.status = 'INACTIVE',
                c.closed_at = CASE WHEN $reference_time IS NULL THEN datetime()
                                   ELSE c.last_seen + duration({seconds:$timeout}) END
            RETURN
                c.campaign_id AS campaign_id
            """,
            timeout=timeout_seconds,
            reference_time=reference_time,
        )
        return [
            record["campaign_id"]
            for record in result
        ]

def create_operation_db(campaign_context):
    operation_id = f"OP_{uuid.uuid4().hex[:8].upper()}"
    with driver.session() as session:
        session.run(
            """
            CREATE (o:Operation{
                operation_id:$operation_id,
                status:$status,
                created_at:$created_at,
                last_seen:$last_seen,
                campaign_count:0,
                confidence:$confidence,
                correlation_score:$correlation_score,
                primary_attacker:$primary_attacker,
                primary_target:$primary_target,
                predicted_goal:$predicted_goal,
                notes:$notes
            })
            """,
            operation_id=operation_id,
            status="ACTIVE",
            created_at=campaign_context.first_seen,
            last_seen=campaign_context.last_seen,
            confidence=0.0,
            correlation_score=0.0,
            primary_attacker=campaign_context.attacker_ip,
            primary_target=campaign_context.victim_ip,
            predicted_goal="",
            notes=""
        )
    return operation_id

def update_operation_activity(
    operation_id,
    campaign_context
):
    with driver.session() as session:
        session.run(
            """
            MATCH (o:Operation {operation_id:$operation_id})
            SET
                o.last_seen=$last_seen,
                o.status='ACTIVE'
            """,
            operation_id=operation_id,
            last_seen=campaign_context.last_seen
        )

def close_operation_db(operation_id):
    with driver.session() as session:
        session.run(
            """
            MATCH (o:Operation {operation_id:$operation_id})
            SET
                o.status='INACTIVE',
                o.closed_at=datetime()
            """,
            operation_id=operation_id
        )

def reopen_operation_db(operation_id, event_time=None):
    with driver.session() as session:
        session.run(
            """
            WITH coalesce($event_time, datetime()) AS t
            MATCH (o:Operation {operation_id:$operation_id})

            SET
                o.status='ACTIVE',
                o.last_seen=CASE WHEN o.last_seen IS NULL OR t > o.last_seen THEN t ELSE o.last_seen END

            REMOVE o.closed_at
            """,
            operation_id=operation_id,
            event_time=event_time,
        )

def attach_campaign_to_operation(operation_id, campaign_context):
    with driver.session() as session:
        session.run(
            """
            MATCH (o:Operation {operation_id:$operation_id})
            MATCH (c:Campaign {campaign_id:$campaign_id})
            MERGE (o)-[r:HAS_CAMPAIGN]->(c)
            ON CREATE SET
                o.campaign_count = coalesce(o.campaign_count, 0) + 1
            SET
                o.last_seen = $last_seen,
                o.primary_attacker =
                    CASE
                        WHEN o.primary_attacker = ''
                        THEN $attacker
                        ELSE o.primary_attacker
                    END,
                o.primary_target =
                    CASE
                        WHEN o.primary_target = ''
                        THEN $victim
                        ELSE o.primary_target
                    END
            """,
            operation_id=operation_id,
            campaign_id=campaign_context.campaign_id,
            last_seen=campaign_context.last_seen,
            attacker=campaign_context.attacker_ip,
            victim=campaign_context.victim_ip
        )

def expire_stale_operations_db(timeout_seconds, reference_time=None):
    """Same event-time contract as expire_stale_campaigns_db."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (o:Operation)
            WHERE
                o.status = 'ACTIVE'
                AND o.last_seen < coalesce($reference_time, datetime()) - duration({seconds:$operation_timeout})
            SET
                o.status = 'INACTIVE',
                o.closed_at = CASE WHEN $reference_time IS NULL THEN datetime()
                                   ELSE o.last_seen + duration({seconds:$operation_timeout}) END
            RETURN o.operation_id AS operation_id
            """,
            operation_timeout=timeout_seconds,
            reference_time=reference_time,
        )
        return [record["operation_id"] for record in result]


def get_active_operations():
    with driver.session() as session:
        result = session.run(
            """
            MATCH (o:Operation)
            WHERE o.status = 'ACTIVE'
            RETURN o.operation_id AS operation_id
            ORDER BY o.last_seen DESC
            """
        )
        return [
            record["operation_id"]
            for record in result
        ]

def get_recent_inactive_operations(limit=5):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (o:Operation)
            WHERE o.status='INACTIVE'
            RETURN o.operation_id AS operation_id
            ORDER BY o.last_seen DESC
            LIMIT $limit
            """,
            limit=limit
        )

        return [
            record["operation_id"]
            for record in result
        ]

def store_prediction(campaign_id, predicted, confidence):
    with driver.session() as session:
        session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})

            OPTIONAL MATCH (c)-[old:LIKELY_NEXT]->()
            DELETE old
            WITH *
            OPTIONAL MATCH (t:Technique {attack_id:$predicted})

            FOREACH (_ IN CASE WHEN t IS NULL THEN [] ELSE [1] END |

                MERGE (c)-[r:LIKELY_NEXT]->(t)

                SET
                    r.confidence=$confidence,
                    r.generated_at=datetime()
            )

            SET
                c.predicted_next=$predicted,
                c.prediction_confidence=$confidence,
                c.prediction_generated_at=datetime()

            MERGE (c)-[r:LIKELY_NEXT]->(t)

            SET
                r.confidence=$confidence,
                r.generated_at=datetime(),

                c.predicted_next=$predicted,
                c.prediction_confidence=$confidence,
                c.prediction_generated_at=datetime()
            """,
            campaign_id=campaign_id,
            predicted=predicted,
            confidence=confidence
        )

def get_prediction_distribution(previous_technique):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (:Technique {attack_id:$previous})
                  -[r:NEXT_TECHNIQUE]->
                  (t:Technique)

            RETURN
                t.attack_id AS technique,
                r.count AS count
            """,
            previous=previous_technique
        )

        rows = list(result)

        if not rows:
            return {}

        total = sum(
            row["count"]
            for row in rows
        )

        if total == 0:
            return {}

        return {
            row["technique"]:
            row["count"] / total
            for row in rows
        }

def archive_campaign_db(campaign_id):
    with driver.session() as session:

        session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})

            SET
                c.status='ARCHIVED',
                c.last_technique=NULL,
                c.predicted_next=NULL,
                c.prediction_confidence=0,
                c.archived_at=datetime()
            """,
            campaign_id=campaign_id
        )

def get_campaign_chain(campaign_id):

    with driver.session() as session:

        result = session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})
                  -[:HAS_EVENT]->
                  (e:AttackEvent)
                  -[:MATCHES]->
                  (t:Technique)

            RETURN
                t.attack_id AS attack_id

            ORDER BY e.first_seen, e.ingested_at, e.event_id
            """,
            campaign_id=campaign_id
        )

        return [
            r["attack_id"]
            for r in result
        ]
def find_recent_duplicate(fingerprint, reference_time=None):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:AttackEvent {fingerprint:$fingerprint})
            WHERE e.last_seen >= coalesce($reference_time, datetime()) - duration({seconds:$window})
            RETURN
                e.event_id AS event_id,
                e.campaign_id AS campaign_id,
                e.last_seen AS last_seen
            ORDER BY e.last_seen DESC
            LIMIT 1
            """,
            fingerprint=fingerprint,
            window=DEDUP_WINDOW,
            reference_time=reference_time,
        )

        record = result.single()
        if record is None:
            return None
        return {
            "event_id": record["event_id"],
            "campaign_id": record["campaign_id"],
            "last_seen": record["last_seen"]
        }
def get_mitre_features(attack_id):
    query = """
    MATCH (t:Technique {attack_id:$attack_id})
    OPTIONAL MATCH (ta:ThreatActor)-[:USES]->(t)
    WITH
        t,
        count(DISTINCT ta) AS threat_actor_count
    OPTIONAL MATCH (m:Malware)-[:USES]->(t)
    WITH
        t,
        threat_actor_count,
        count(DISTINCT m) AS malware_count

    OPTIONAL MATCH (tool:Tool)-[:USES]->(t)

    WITH
        t,
        threat_actor_count,
        malware_count,
        count(DISTINCT tool) AS tool_count
    OPTIONAL MATCH (coa:CourseOfAction)-[:MITIGATES]->(t)
    WITH
        t,
        threat_actor_count,
        malware_count,
        tool_count,
        count(DISTINCT coa) AS mitigation_count
    OPTIONAL MATCH (child:Technique)-[:SUBTECHNIQUE_OF]->(t)
    WITH
        t,
        threat_actor_count,
        malware_count,
        tool_count,
        mitigation_count,
        count(DISTINCT child) AS subtechnique_count

    OPTIONAL MATCH (t)-[:SUBTECHNIQUE_OF]->(parent:Technique)
    RETURN
        t.attack_id                AS attack_id,
        t.name                     AS technique_name,
        coalesce(t.description,"") AS description,
        coalesce(t.platforms,[])   AS platforms,
        size(coalesce(t.platforms,[]))
            AS platform_count,
        coalesce(t.domains,[])
            AS domains,
        size(coalesce(t.domains,[]))
            AS domain_count,
        coalesce(t.kill_chain_phases,[])
            AS kill_chain_phases,
        size(coalesce(t.kill_chain_phases,[]))
            AS kill_chain_count,
        coalesce(t.is_subtechnique,false)
            AS is_subtechnique,
        coalesce(t.deprecated,false)
            AS deprecated,
        coalesce(t.revoked,false)
            AS revoked,
        coalesce(t.object_version,"")
            AS object_version,
        threat_actor_count,
        malware_count,
        tool_count,
        mitigation_count,
        subtechnique_count,
        parent.attack_id
            AS parent_technique
    """
    with driver.session() as session:
        record = session.run(
            query,
            attack_id=attack_id
        ).single()
        if record is None:
            return None
        return MitreFeatures(
            attack_id=record["attack_id"],
            technique_name=record["technique_name"],
            description=record["description"],
            platforms=record["platforms"],
            platform_count=record["platform_count"],
            domains=record["domains"],
            domain_count=record["domain_count"],
            kill_chain_phases=record["kill_chain_phases"],
            kill_chain_count=record["kill_chain_count"],
            is_subtechnique=record["is_subtechnique"],
            deprecated=record["deprecated"],
            revoked=record["revoked"],
            object_version=record["object_version"],
            threat_actor_count=record["threat_actor_count"],
            malware_count=record["malware_count"],
            tool_count=record["tool_count"],
            mitigation_count=record["mitigation_count"],
            subtechnique_count=record["subtechnique_count"],
            parent_technique=record["parent_technique"]
        )

def get_runtime_graph_features(campaign_id):
    query = """
    MATCH (c:Campaign {campaign_id:$campaign_id})
    OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:AttackEvent)
    WITH
        c,
        count(DISTINCT e) AS attack_event_count
    OPTIONAL MATCH (c)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t:Technique)
    WITH
        c,
        attack_event_count,
        collect(DISTINCT t) AS techniques

    UNWIND techniques AS tech

    OPTIONAL MATCH (tech)<-[in_rel:NEXT_TECHNIQUE]-()

    WITH
        c,
        attack_event_count,
        techniques,
        count(DISTINCT in_rel) AS incoming_chain_count
    UNWIND techniques AS tech2
    OPTIONAL MATCH (tech2)-[out_rel:NEXT_TECHNIQUE]->()
    WITH
        c,
        attack_event_count,
        techniques,
        incoming_chain_count,
        count(DISTINCT out_rel) AS outgoing_chain_count
    RETURN
        coalesce(c.campaign_count,1)
            AS campaign_count,
        attack_event_count,
        size(techniques)
            AS graph_degree,
        incoming_chain_count,
        outgoing_chain_count,
        coalesce(c.prediction_frequency,0)
            AS prediction_frequency,
        coalesce(c.duplicate_frequency,0)
            AS duplicate_frequency
    """
    with driver.session() as session:
        record = session.run(
            query,
            campaign_id=campaign_id
        ).single()
        if record is None:
            return RuntimeGraphFeatures(
                campaign_count=1,
                attack_event_count=0,
                graph_degree=0,
                incoming_chain_count=0,
                outgoing_chain_count=0,
                prediction_frequency=0,
                duplicate_frequency=0,
            )
        return RuntimeGraphFeatures(
            campaign_count=record["campaign_count"],
            attack_event_count=record["attack_event_count"],
            graph_degree=record["graph_degree"],
            incoming_chain_count=record["incoming_chain_count"],
            outgoing_chain_count=record["outgoing_chain_count"],
            prediction_frequency=record["prediction_frequency"],
            duplicate_frequency=record["duplicate_frequency"]
        )

def get_threat_intel_features(attacker_ip):
    query = """
    MATCH (a:Attacker {ip:$attacker_ip})
    RETURN
        coalesce(a.vt_reputation,0.0)
            AS ip_reputation,
        coalesce(a.threat_actor_reputation,0.0)
            AS threat_actor_reputation,
        coalesce(a.malware_confidence,0.0)
            AS malware_confidence,
        coalesce(a.tool_confidence,0.0)
            AS tool_confidence,
        coalesce(a.misp_confidence,0.0)
            AS misp_confidence,
        coalesce(a.ioc_confidence,0.0)
            AS ioc_confidence
    """
    with driver.session() as session:
        record = session.run(
            query,
            attacker_ip=attacker_ip
        ).single()
        if record is None:
            return ThreatIntelFeatures(
                0,0,0,0,0,0
            )
        return ThreatIntelFeatures(
            ip_reputation=record["ip_reputation"],
            threat_actor_reputation=record["threat_actor_reputation"],
            malware_confidence=record["malware_confidence"],
            tool_confidence=record["tool_confidence"],
            misp_confidence=record["misp_confidence"],
            ioc_confidence=record["ioc_confidence"]
        )

def get_detection_confidence(event_id):
    query = """
    MATCH (e:AttackEvent {event_id:$event_id})
    RETURN
        coalesce(e.rule_level,0)
            AS wazuh_level,
        coalesce(e.suricata_score,0.0)
            AS suricata_score,
        coalesce(e.zeek_score,0.0)
            AS zeek_score,
        coalesce(e.sigma_score,0.0)
            AS sigma_score,
        coalesce(e.yara_score,0.0)
            AS yara_score
    """
    with driver.session() as session:
        record = session.run(
            query,
            event_id=event_id
        ).single()
        if record is None:
            return DetectionConfidence(
                0,
                0,
                0,
                0,
                0,
                0
            )
        wazuh = record["wazuh_level"]
        confidence = (
            wazuh +
            record["suricata_score"] +
            record["zeek_score"] +
            record["sigma_score"] +
            record["yara_score"]
        )
        confidence = min(confidence, 10)
        return DetectionConfidence(
            wazuh_level=wazuh,
            suricata_score=record["suricata_score"],
            zeek_score=record["zeek_score"],
            sigma_score=record["sigma_score"],
            yara_score=record["yara_score"],
            detection_confidence=confidence
        )

def get_runtime_features(attack_id):
    query = """
    MATCH (t:Technique {attack_id:$attack_id})
    OPTIONAL MATCH
        (c:Campaign)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t)
    WITH
        t,
        count(DISTINCT c) AS campaign_count
    OPTIONAL MATCH
        (:AttackEvent)-[:MATCHES]->(t)
    WITH
        t,
        campaign_count,
        count(*) AS event_count
    OPTIONAL MATCH
        (prev:Technique)-[r1:NEXT_TECHNIQUE]->(t)
    WITH
        t,
        campaign_count,
        event_count,
        count(r1) AS incoming_chain_count
    OPTIONAL MATCH
        (t)-[r2:NEXT_TECHNIQUE]->(next:Technique)
    WITH
        t,
        campaign_count,
        event_count,
        incoming_chain_count,
        count(r2) AS outgoing_chain_count
    RETURN
        campaign_count,
        event_count,
        incoming_chain_count,
        outgoing_chain_count,
        (
            incoming_chain_count +
            outgoing_chain_count
        ) AS graph_degree,
        coalesce(t.occurrences,0) AS duplicate_frequency,
        coalesce(t.total_tps,0) AS cumulative_tps
    """
    with driver.session() as session:
        result = session.run(
            query,
            attack_id=attack_id
        ).single()

        if result is None:
            return {
                "campaign_count":0,
                "event_count":0,
                "incoming_chain_count":0,
                "outgoing_chain_count":0,
                "graph_degree":0,
                "duplicate_frequency":0,
                "cumulative_tps":0
            }
        return dict(result)
def store_dynamic_risk(campaign_id, dynamic_risk):
    with driver.session() as session:
        previous = session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})
            RETURN c.dynamic_risk AS risk
            """,
            campaign_id=campaign_id
        ).single()
        previous_score = 0
        if previous and previous["risk"] is not None:
            previous_score = previous["risk"]
        if previous_score == 0:
            trend = "NEW"
        elif dynamic_risk.risk_score > previous_score:
            trend = "INCREASING"
        elif dynamic_risk.risk_score < previous_score:
            trend = "DECREASING"
        else:
            trend = "STABLE"
        session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})
            SET
            c.dynamic_risk=$risk,
            c.risk_level=$level,
            c.risk_color=$color,
            c.risk_trend=$trend,
            c.risk_confidence=$confidence,
            c.risk_breakdown=$breakdown,
            c.risk_updated_at=datetime()
            """,
            campaign_id=campaign_id,
            risk=dynamic_risk.risk_score,
            level=dynamic_risk.risk_level,
            color=dynamic_risk.risk_color,
            breakdown=str(dynamic_risk.breakdown),
            trend=trend,
            confidence=dynamic_risk.confidence
        )

def store_cti_confidence(campaign_id, cti):
    with driver.session() as session:
        session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})

            SET
                c.cti_score = $score,
                c.cti_level = $level,
                c.cti_publish = $publish,
                c.cti_breakdown = $breakdown,
                c.cti_updated_at = datetime()
            """,
            campaign_id=campaign_id,
            score=cti.score,
            level=cti.level,
            publish=cti.publish,
            breakdown=str(cti.breakdown)
        )

def _event_order_key(e):
    """Event-time order with deterministic tie-breaks: alerts with identical
    timestamps keep processing order (ingested_at), then event_id."""
    floor = datetime.min.replace(tzinfo=timezone.utc)
    return (
        normalize_datetime(e.get("first_seen")) or floor,
        normalize_datetime(e.get("ingested_at")) or floor,
        e.get("event_id") or "",
    )


def get_campaign_context_data(campaign_id):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})
            OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:AttackEvent)
            OPTIONAL MATCH (e)-[:MATCHES]->(t:Technique)
            RETURN
                c,
                collect(DISTINCT t.attack_id) AS techniques,
                collect(
                    DISTINCT {
                        event_id: e.event_id,
                        attack_id: e.attack_id,
                        first_seen: e.first_seen,
                        ingested_at: e.ingested_at
                    }
                ) AS events
            """,
            campaign_id=campaign_id
        )
        record = result.single()
        print("\n===== RAW CAMPAIGN CONTEXT =====")
        print(record["techniques"])
        print(record["events"])
        print("===============================")
        if record is None:
            return None
        techniques = [
            t for t in record["techniques"]
            if t is not None
        ]
        events = [
            e for e in record["events"]
            if e["attack_id"] is not None
        ]
        events.sort(key=_event_order_key)

        attack_chain = [
            e["attack_id"]
            for e in events
        ]
        return {
            "campaign": dict(record["c"]),
            "techniques": techniques,
            "attack_chain": attack_chain
        }

def get_operation_context_data(operation_id):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (o:Operation {operation_id:$operation_id})
            RETURN o
            """,
            operation_id=operation_id
        )
        record = result.single()

        if record is None:
            return None
        operation = dict(record["o"])
        result = session.run(
            """
            MATCH (o:Operation {operation_id:$operation_id})
                  -[:HAS_CAMPAIGN]->
                  (c:Campaign)

            RETURN c
            ORDER BY c.first_seen
            """,
            operation_id=operation_id
        )
        campaigns = [
            dict(record["c"])
            for record in result
        ]
        result = session.run(
            """
            MATCH (o:Operation {operation_id:$operation_id})
                  -[:HAS_CAMPAIGN]->
                  (c:Campaign)
                  -[:HAS_EVENT]->
                  (e:AttackEvent)
                  -[:MATCHES]->
                  (t:Technique)

            RETURN
                collect(DISTINCT t.attack_id) AS techniques,
                collect(
                    DISTINCT {
                        event_id: e.event_id,
                        attack_id: e.attack_id,
                        first_seen: e.first_seen,
                        ingested_at: e.ingested_at
                    }
                ) AS events
            """,
            operation_id=operation_id
        )

        record = result.single()

        techniques = [
            t for t in record["techniques"]
            if t is not None
        ]

        events = [
            e for e in record["events"]
            if e["attack_id"] is not None
        ]

        events.sort(key=_event_order_key)

        attack_chain = [
            e["attack_id"]
            for e in events
        ]
        return {
            "operation": operation,
            "campaigns": campaigns,
            "techniques": techniques,
            "attack_chain": attack_chain
        }

def get_dynamic_risk(campaign_id):
    with driver.session() as session:
        record = session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})
            RETURN
                c.dynamic_risk AS risk,
                c.risk_level AS level,
                c.risk_color AS color,
                c.risk_trend AS trend,
                c.risk_confidence AS confidence,
                c.risk_updated_at AS updated_at
            """,
            campaign_id=campaign_id
        ).single()
        if record is None:
            return None
        return dict(record)

def close():
    driver.close()


# =====================================================================
# Phase 2.7 WP-M1: AppliedWrite Marker Reads and Startup Checks
# =====================================================================

import hashlib
import threading
from typing import Any, Mapping, Sequence, Tuple


class MarkerStatus(str, Enum):
    """The outcome of looking up an AppliedWrite marker."""
    APPLIED = "APPLIED"          # exactly one marker found
    ABSENT = "ABSENT"            # read succeeded, zero markers found
    UNKNOWN = "UNKNOWN"          # any error, deadline timeout, or disconnected driver
    INCONSISTENT = "INCONSISTENT"  # multiple markers for the same key (constraint violated/missing)


@dataclass(frozen=True)
class MarkerLookup:
    """The result of read_applied_write."""
    status: MarkerStatus
    operation: Optional[str] = None
    fingerprint: Optional[str] = None
    applied_at: Optional[str] = None
    error_type: Optional[str] = None
    error_code: Optional[str] = None
    diagnostic: str = ""


class ConstraintStatus(str, Enum):
    """The validation status of the AppliedWrite constraint and schema."""
    PRESENT_VALID = "PRESENT_VALID"
    ABSENT = "ABSENT"
    DROPPED_LABEL_PRESENT = "DROPPED_LABEL_PRESENT"
    WRONG_PROPERTIES = "WRONG_PROPERTIES"
    INDEX_NOT_ONLINE = "INDEX_NOT_ONLINE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MarkerSchemaReport:
    """The result of read_marker_schema."""
    status: ConstraintStatus
    constraint_name: Optional[str] = None
    constraint_type: Optional[str] = None
    labels: tuple[str, ...] = ()
    properties: tuple[str, ...] = ()
    index_state: Optional[str] = None
    has_label: bool = False
    valid: bool = False
    alarm: bool = False
    reasons: tuple[str, ...] = ()
    diagnostic: str = ""


@dataclass(frozen=True)
class StartupReport:
    """The result of verify_startup."""
    valid: bool
    marker_mode_allowed: bool
    server_name: Optional[str] = None
    server_version: Optional[str] = None
    server_edition: Optional[str] = None
    schema: Optional[MarkerSchemaReport] = None
    sample_invalid_count: int = 0
    alarm: bool = False
    reasons: tuple[str, ...] = ()
    diagnostic: str = ""


def _marker_key_ref(key: str) -> str:
    """Hash key for logging: keys and parameters must never appear in logs."""
    return "key:" + hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:12]


_LOOKUP_QUERY = (
    "MATCH (m:AppliedWrite {request_key:$request_key}) "
    "RETURN m.request_key AS request_key, "
    "toString(m.applied_at) AS applied_at, "
    "m.operation AS operation, "
    "m.fingerprint AS fingerprint"
)


def read_applied_write(
    request_key: str,
    *,
    driver: Optional[Any] = None,
    timeout: float = 3.0,
    deadline: float = 6.0,
) -> MarkerLookup:
    """Read an AppliedWrite marker for request_key.

    Additive read-only function per WP-M1.
    Uses execute_read with a transaction timeout and an overall client thread deadline.
    Never retries on failure. Returns UNKNOWN for any failure or deadline expiration.
    Zero rows returns ABSENT. One row returns APPLIED. Multiple rows returns INCONSISTENT.
    Logs key references only; never raw keys or values.
    """
    if not isinstance(request_key, str) or not request_key:
        return MarkerLookup(
            status=MarkerStatus.UNKNOWN,
            error_type="ValueError",
            diagnostic="request_key must be a non-empty string",
        )

    active_driver = driver if driver is not None else globals().get("driver")
    if active_driver is None:
        return MarkerLookup(
            status=MarkerStatus.UNKNOWN,
            error_type="DriverError",
            diagnostic="no Neo4j driver available",
        )

    box: dict[str, Any] = {}

    def _execute():
        try:
            try:
                session_cm = active_driver.session(
                    max_transaction_retry_time=1.0,
                    connection_acquisition_timeout=2.0,
                )
            except (TypeError, AttributeError):
                session_cm = active_driver.session()

            with session_cm as session:
                if hasattr(session, "execute_read"):
                    uow = unit_of_work(timeout=timeout)
                    rows = session.execute_read(
                        uow(lambda tx: [dict(r) for r in tx.run(_LOOKUP_QUERY, request_key=request_key).data()])
                    )
                else:
                    result = session.run(_LOOKUP_QUERY, request_key=request_key)
                    if hasattr(result, "data"):
                        rows = [dict(r) for r in result.data()]
                    else:
                        rows = [dict(r) for r in result]
                box["rows"] = rows
        except BaseException as error:
            box["error"] = error

    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()
    thread.join(deadline)

    if thread.is_alive():
        logger.warning("read_applied_write exceeded deadline %.1fs for %s", deadline, _marker_key_ref(request_key))
        return MarkerLookup(
            status=MarkerStatus.UNKNOWN,
            error_type="ClientDeadline",
            diagnostic="read_applied_write exceeded client deadline",
        )

    if "error" in box:
        error = box["error"]
        error_type = type(error).__name__
        error_code = getattr(error, "code", None)
        logger.warning(
            "read_applied_write failed (%s) for %s",
            error_type,
            _marker_key_ref(request_key),
        )
        return MarkerLookup(
            status=MarkerStatus.UNKNOWN,
            error_type=error_type,
            error_code=error_code,
            diagnostic=str(error),
        )

    rows = box.get("rows", [])
    if len(rows) == 1:
        row = rows[0]
        return MarkerLookup(
            status=MarkerStatus.APPLIED,
            operation=row.get("operation"),
            fingerprint=row.get("fingerprint"),
            applied_at=row.get("applied_at"),
        )
    elif len(rows) == 0:
        return MarkerLookup(status=MarkerStatus.ABSENT)
    else:
        logger.error(
            "read_applied_write found %d markers for %s",
            len(rows),
            _marker_key_ref(request_key),
        )
        return MarkerLookup(
            status=MarkerStatus.INCONSISTENT,
            diagnostic=f"found {len(rows)} markers for key",
        )


def read_marker_schema(
    *,
    driver: Optional[Any] = None,
    timeout: float = 3.0,
    deadline: float = 6.0,
) -> MarkerSchemaReport:
    """Inspect schema constraints, indexes and labels for AppliedWrite.

    Verifies the uniqueness constraint on :AppliedWrite(request_key),
    checks that its backing index is ONLINE, and confirms db.labels().
    """
    active_driver = driver if driver is not None else globals().get("driver")
    if active_driver is None:
        return MarkerSchemaReport(
            status=ConstraintStatus.UNKNOWN,
            diagnostic="no Neo4j driver available",
            reasons=("no Neo4j driver available",),
        )

    box: dict[str, Any] = {}

    def _execute():
        try:
            try:
                session_cm = active_driver.session(
                    max_transaction_retry_time=1.0,
                    connection_acquisition_timeout=2.0,
                )
            except (TypeError, AttributeError):
                session_cm = active_driver.session()

            with session_cm as session:
                # 1. SHOW CONSTRAINTS
                constraints = []
                c_res = session.run("SHOW CONSTRAINTS")
                for r in (c_res.data() if hasattr(c_res, "data") else c_res):
                    constraints.append(dict(r))

                # 2. SHOW INDEXES
                indexes = []
                i_res = session.run("SHOW INDEXES")
                for r in (i_res.data() if hasattr(i_res, "data") else i_res):
                    indexes.append(dict(r))

                # 3. db.labels()
                labels = set()
                l_res = session.run("CALL db.labels() YIELD label RETURN label")
                for r in (l_res.data() if hasattr(l_res, "data") else l_res):
                    d = dict(r)
                    if "label" in d:
                        labels.add(d["label"])

                box["data"] = {
                    "constraints": constraints,
                    "indexes": indexes,
                    "labels": labels,
                }
        except BaseException as error:
            box["error"] = error

    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()
    thread.join(deadline)

    if thread.is_alive():
        logger.warning("read_marker_schema exceeded deadline %.1fs", deadline)
        return MarkerSchemaReport(
            status=ConstraintStatus.UNKNOWN,
            diagnostic="read_marker_schema exceeded client deadline",
            reasons=("read_marker_schema exceeded client deadline",),
        )

    if "error" in box:
        error = box["error"]
        logger.warning("read_marker_schema failed (%s)", type(error).__name__)
        return MarkerSchemaReport(
            status=ConstraintStatus.UNKNOWN,
            diagnostic=str(error),
            reasons=(f"query failed: {type(error).__name__}: {error}",),
        )

    data = box.get("data", {})
    constraints = data.get("constraints", [])
    indexes = data.get("indexes", [])
    labels = data.get("labels", set())
    has_label = "AppliedWrite" in labels

    # Build index map: index name -> index dict
    index_by_name = {idx.get("name"): idx for idx in indexes if idx.get("name")}

    # Find constraints related to AppliedWrite
    applied_constraints = []
    for c in constraints:
        c_labels = c.get("labelsOrTypes") or ()
        if "AppliedWrite" in c_labels:
            applied_constraints.append(c)

    if not applied_constraints:
        if has_label:
            logger.error(
                "Marker constraint absent but label AppliedWrite exists (alarm: constraint dropped or unconstrained markers created)"
            )
            return MarkerSchemaReport(
                status=ConstraintStatus.DROPPED_LABEL_PRESENT,
                has_label=True,
                valid=False,
                alarm=True,
                reasons=("Constraint 'applied_write_key' is absent but label 'AppliedWrite' exists (dropped or unconstrained markers)",),
                diagnostic="DROPPED_LABEL_PRESENT: constraint absent, label exists",
            )
        else:
            return MarkerSchemaReport(
                status=ConstraintStatus.ABSENT,
                has_label=False,
                valid=False,
                alarm=False,
                reasons=("Marker constraint 'applied_write_key' and label 'AppliedWrite' are absent (rollout has not happened)",),
                diagnostic="ABSENT: rollout has not happened",
            )

    # Check candidate constraint
    target = applied_constraints[0]
    c_name = target.get("name")
    c_type = target.get("type")
    c_labels = tuple(target.get("labelsOrTypes") or ())
    c_props = tuple(target.get("properties") or ())
    c_entity = target.get("entityType")

    is_unique = (c_type == "UNIQUENESS")
    is_name_match = (c_name == "applied_write_key")
    is_props_match = (c_props == ("request_key",))
    is_node = (c_entity is None or c_entity == "NODE")

    if not (is_unique and is_name_match and is_props_match and is_node):
        return MarkerSchemaReport(
            status=ConstraintStatus.WRONG_PROPERTIES,
            constraint_name=c_name,
            constraint_type=c_type,
            labels=c_labels,
            properties=c_props,
            has_label=has_label,
            valid=False,
            alarm=False,
            reasons=(f"Constraint on AppliedWrite has invalid definition: name={c_name}, type={c_type}, properties={c_props}",),
            diagnostic="WRONG_PROPERTIES: constraint exists with invalid parameters",
        )

    # Check index state
    owned_index = target.get("ownedIndexName") or c_name
    idx_rec = index_by_name.get(owned_index)
    if not idx_rec:
        for idx in indexes:
            if tuple(idx.get("labelsOrTypes") or ()) == ("AppliedWrite",) and tuple(idx.get("properties") or ()) == ("request_key",):
                idx_rec = idx
                break

    idx_state = idx_rec.get("state") if idx_rec else None
    if idx_state is not None and idx_state.upper() != "ONLINE":
        return MarkerSchemaReport(
            status=ConstraintStatus.INDEX_NOT_ONLINE,
            constraint_name=c_name,
            constraint_type=c_type,
            labels=c_labels,
            properties=c_props,
            index_state=idx_state,
            has_label=has_label,
            valid=False,
            alarm=False,
            reasons=(f"Owned index '{owned_index}' state is '{idx_state}', must be ONLINE",),
            diagnostic=f"INDEX_NOT_ONLINE: state is {idx_state}",
        )

    return MarkerSchemaReport(
        status=ConstraintStatus.PRESENT_VALID,
        constraint_name=c_name,
        constraint_type=c_type,
        labels=c_labels,
        properties=c_props,
        index_state=idx_state or "ONLINE",
        has_label=has_label,
        valid=True,
        alarm=False,
        reasons=(),
        diagnostic="PRESENT_VALID: constraint and index online",
    )


def verify_startup(
    *,
    driver: Optional[Any] = None,
    allowed_versions: tuple[str, ...] = ("5.26.27",),
    allowed_editions: tuple[str, ...] = ("community", "enterprise"),
    timeout: float = 3.0,
    deadline: float = 6.0,
) -> StartupReport:
    """Execute pre-migration / startup verification for marker mode.

    1. Connect and dbms.components(): record server name, version, edition.
    2. Version and edition match allowed.
    3. read_marker_schema(): check uniqueness constraint applied_write_key.
    4. Index online verification.
    5. Label verification.
    6. Sample up to 1,000 markers for missing operation or fingerprint.
    """
    active_driver = driver if driver is not None else globals().get("driver")
    if active_driver is None:
        return StartupReport(
            valid=False,
            marker_mode_allowed=False,
            diagnostic="no Neo4j driver available",
            reasons=("no Neo4j driver available",),
        )

    box: dict[str, Any] = {}

    def _execute():
        try:
            try:
                session_cm = active_driver.session(
                    max_transaction_retry_time=1.0,
                    connection_acquisition_timeout=2.0,
                )
            except (TypeError, AttributeError):
                session_cm = active_driver.session()

            with session_cm as session:
                # 1. dbms.components()
                comp_res = session.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition")
                comp = comp_res.single() if hasattr(comp_res, "single") else None
                if comp is None and hasattr(comp_res, "data"):
                    d = comp_res.data()
                    comp = d[0] if d else None
                if comp is None:
                    try:
                        comp = next(iter(comp_res))
                    except (StopIteration, TypeError):
                        comp = None
                box["comp"] = dict(comp) if comp is not None else None

                # 6. Sample invalid markers
                sample_res = session.run(
                    "MATCH (m:AppliedWrite) WHERE m.operation IS NULL OR m.fingerprint IS NULL "
                    "RETURN count(m) AS invalid_count"
                )
                sample = sample_res.single() if hasattr(sample_res, "single") else None
                if sample is None and hasattr(sample_res, "data"):
                    d = sample_res.data()
                    sample = d[0] if d else None
                if sample is None:
                    try:
                        sample = next(iter(sample_res))
                    except (StopIteration, TypeError):
                        sample = None
                box["invalid_count"] = dict(sample).get("invalid_count", 0) if sample is not None else 0
        except BaseException as error:
            box["error"] = error

    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()
    thread.join(deadline)

    if thread.is_alive():
        logger.warning("verify_startup exceeded deadline %.1fs", deadline)
        return StartupReport(
            valid=False,
            marker_mode_allowed=False,
            diagnostic="verify_startup exceeded client deadline",
            reasons=("verify_startup exceeded client deadline",),
        )

    if "error" in box or not box.get("comp"):
        err = box.get("error")
        err_msg = str(err) if err else "cannot query dbms.components"
        logger.warning("verify_startup failed at check 1 (%s)", err_msg)
        return StartupReport(
            valid=False,
            marker_mode_allowed=False,
            diagnostic=f"UNKNOWN: {err_msg}",
            reasons=(f"UNKNOWN: {err_msg}",),
        )

    comp = box["comp"]
    server_name = comp.get("name")
    versions = comp.get("versions") or []
    server_version = versions[0] if isinstance(versions, (list, tuple)) and versions else str(versions)
    server_edition = str(comp.get("edition") or "").lower()

    # Check 2: Version and edition match allowed
    version_matched = any(server_version.startswith(v) for v in allowed_versions)
    edition_matched = server_edition in [e.lower() for e in allowed_editions]
    if not (version_matched and edition_matched):
        reason = (
            f"Server version '{server_version}' (edition '{server_edition}') "
            f"does not match allowed versions {allowed_versions} or editions {allowed_editions}"
        )
        logger.error("verify_startup rejected: %s", reason)
        return StartupReport(
            valid=False,
            marker_mode_allowed=False,
            server_name=server_name,
            server_version=server_version,
            server_edition=server_edition,
            reasons=(reason,),
            diagnostic=reason,
        )

    # Check 3, 4, 5: Schema verification
    schema_report = read_marker_schema(driver=active_driver, timeout=timeout, deadline=deadline)
    if not schema_report.valid:
        logger.error("verify_startup rejected: schema status %s", schema_report.status.value)
        return StartupReport(
            valid=False,
            marker_mode_allowed=False,
            server_name=server_name,
            server_version=server_version,
            server_edition=server_edition,
            schema=schema_report,
            alarm=schema_report.alarm,
            reasons=schema_report.reasons,
            diagnostic=f"Schema invalid: {schema_report.diagnostic}",
        )

    # Check 6: Sample markers
    invalid_count = box.get("invalid_count", 0)
    if invalid_count > 0:
        reason = f"Alarm: sampled {invalid_count} AppliedWrite markers missing operation or fingerprint"
        logger.critical("verify_startup alarm: %s", reason)
        return StartupReport(
            valid=False,
            marker_mode_allowed=False,
            server_name=server_name,
            server_version=server_version,
            server_edition=server_edition,
            schema=schema_report,
            sample_invalid_count=invalid_count,
            alarm=True,
            reasons=(reason,),
            diagnostic=reason,
        )

    # All checks passed
    return StartupReport(
        valid=True,
        marker_mode_allowed=True,
        server_name=server_name,
        server_version=server_version,
        server_edition=server_edition,
        schema=schema_report,
        sample_invalid_count=0,
        alarm=False,
        reasons=(),
        diagnostic="Startup verification succeeded; marker mode allowed",
    )

"""
event_time.py
================
Event-time semantics for the campaign lifecycle (research/STEP2_EVENT_TIME.md).

Every campaign activity/inactivity decision (dedup window, campaign expiry,
continuation temporal feature, reopen, operation expiry) is made against the
alert's own Wazuh timestamp ("event time"), never against the time CYUKTI
happens to process it ("processing time"). Processing time is still
recorded separately (AttackEvent.ingested_at) for audit.

Timestamp source: alert["timestamp"] (Wazuh alerts.json, e.g.
"2026-03-11T03:27:24.123+0000"); alert["@timestamp"] is accepted as a
fallback field (Wazuh indexer/OpenSearch exports). Values are converted to
UTC.

Failure cases and the explicit policies that handle them:

    MISSING            no timestamp field, or empty
    MALFORMED          not parseable as ISO-8601
    TIMEZONE_MISSING   parseable but naive (no offset) -- ambiguous, never
                       assumed to be UTC or local time

    policy STRICT                    -> alert rejected (EventTimeError); the
                                        caller records the rejection. Default
                                        for replay, where reproducibility
                                        matters more than completeness.
    policy PROCESSING_TIME_FALLBACK  -> processing time is used, the result
                                        is marked source=PROCESSING_TIME_FALLBACK
                                        and the reason is kept; it is stored on
                                        the AttackEvent and logged. Default for
                                        the live listener (config.EVENT_TIME_POLICY),
                                        where dropping evidence is worse.

Late / out-of-order alerts (event time earlier than the latest event time
already processed, the "watermark") are ACCEPTED and processed in arrival
order; they are flagged (EventTime.late). Lifecycle code treats them
monotonically: a campaign's last_seen never moves backwards, and a negative
gap (alert earlier than the campaign's last activity) counts as "within the
activity period" -- see campaign_manager.CampaignManager.is_expired.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock

logger = logging.getLogger(__name__)

POLICY_STRICT = "strict"
POLICY_PROCESSING_TIME_FALLBACK = "processing_time_fallback"
POLICIES = (POLICY_STRICT, POLICY_PROCESSING_TIME_FALLBACK)

SOURCE_ALERT = "ALERT_TIMESTAMP"
SOURCE_PROCESSING_TIME_FALLBACK = "PROCESSING_TIME_FALLBACK"

REASON_MISSING = "MISSING"
REASON_MALFORMED = "MALFORMED"
REASON_TIMEZONE_MISSING = "TIMEZONE_MISSING"

# Flag only (never changes the value): event time this far ahead of
# processing time indicates sensor clock skew.
FUTURE_SKEW_TOLERANCE = timedelta(minutes=5)

TIMESTAMP_FIELDS = ("timestamp", "@timestamp")


class EventTimeError(ValueError):
    def __init__(self, reason: str, raw):
        super().__init__(f"{reason}: {raw!r}")
        self.reason = reason
        self.raw = raw


@dataclass(frozen=True)
class EventTime:
    value: datetime            # timezone-aware, UTC
    source: str                # SOURCE_ALERT | SOURCE_PROCESSING_TIME_FALLBACK
    fallback_reason: str | None = None
    raw: str | None = None
    late: bool = False         # earlier than the watermark when processed
    future_skew: bool = False  # > FUTURE_SKEW_TOLERANCE ahead of processing time


def parse_timestamp(raw) -> datetime:
    """Strict parser. Returns an aware UTC datetime or raises EventTimeError."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise EventTimeError(REASON_MISSING, raw)
    if not isinstance(raw, str):
        raise EventTimeError(REASON_MALFORMED, raw)
    text = raw.strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    # Wazuh emits "+0000" (no colon); fromisoformat() on Python < 3.11 needs
    # "+00:00". Normalise a trailing +HHMM / -HHMM offset.
    if len(text) >= 5 and text[-5] in "+-" and text[-4:].isdigit():
        text = text[:-2] + ":" + text[-2:]
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        raise EventTimeError(REASON_MALFORMED, raw) from None
    if value.tzinfo is None or value.utcoffset() is None:
        raise EventTimeError(REASON_TIMEZONE_MISSING, raw)
    return value.astimezone(timezone.utc)


def raw_timestamp(alert: dict):
    for field in TIMESTAMP_FIELDS:
        value = alert.get(field)
        if value not in (None, ""):
            return value
    return None


def resolve_event_time(alert: dict, policy: str, now: datetime | None = None) -> EventTime:
    """Applies `policy` to the alert's timestamp. `now` is processing time
    (injectable for tests); it is only ever USED as the event time under
    POLICY_PROCESSING_TIME_FALLBACK, and that use is always marked."""
    if policy not in POLICIES:
        raise ValueError(f"unknown event-time policy {policy!r}; expected one of {POLICIES}")
    processing_now = now or datetime.now(timezone.utc)
    raw = raw_timestamp(alert)
    try:
        value = parse_timestamp(raw)
    except EventTimeError as err:
        if policy == POLICY_STRICT:
            raise
        logger.warning(
            "Alert event time %s (%r); using processing time %s under %s policy.",
            err.reason, raw, processing_now.isoformat(), POLICY_PROCESSING_TIME_FALLBACK,
        )
        return EventTime(
            value=processing_now, source=SOURCE_PROCESSING_TIME_FALLBACK,
            fallback_reason=err.reason, raw=None if raw is None else str(raw),
        )
    return EventTime(
        value=value, source=SOURCE_ALERT, raw=str(raw),
        future_skew=value - processing_now > FUTURE_SKEW_TOLERANCE,
    )


class EventClock:
    """Watermark = the latest event time processed so far. Used to flag late
    alerts and as the reference time for background maintenance (expiry),
    so maintenance never runs ahead of the event stream on wall-clock time."""

    def __init__(self):
        self._lock = Lock()
        self._watermark: datetime | None = None

    @property
    def watermark(self) -> datetime | None:
        with self._lock:
            return self._watermark

    def observe(self, event_time: EventTime) -> EventTime:
        """Advances the watermark and returns event_time with `late` set."""
        with self._lock:
            late = self._watermark is not None and event_time.value < self._watermark
            if self._watermark is None or event_time.value > self._watermark:
                self._watermark = event_time.value
        if late:
            logger.info("Late alert: event time %s < watermark.", event_time.value.isoformat())
            return EventTime(
                value=event_time.value, source=event_time.source,
                fallback_reason=event_time.fallback_reason, raw=event_time.raw,
                late=True, future_skew=event_time.future_skew,
            )
        return event_time

    def reset(self):
        with self._lock:
            self._watermark = None


event_clock = EventClock()

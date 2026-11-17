"""
soar/shuffle_client.py
=========================
Thin adapter over Shuffle's two real, documented integration
mechanisms:

1. Webhook trigger (`SHUFFLE_WEBHOOK`) -- a per-workflow URL Shuffle
   generates for a "Webhook" trigger node. POSTing JSON to it starts
   that workflow. If the workflow's webhook trigger is configured to
   "wait for response", Shuffle returns the workflow's final output
   synchronously in the POST response body; otherwise it returns
   immediately with just an acknowledgement and the workflow keeps
   running server-side. This is the ONLY Shuffle mechanism this
   environment has ever had configured (`config.SHUFFLE_WEBHOOK`,
   read by the pre-existing but non-functional integration attempt in
   realtime_socgraph.py -- see SOAR_PLAYBOOK_INTEGRATION.md for that
   finding).

2. REST API polling (`SHUFFLE_BASE_URL` + `SHUFFLE_API_KEY`) -- Shuffle
   also exposes `/api/v1/workflows/<id>/executions/<execution_id>` for
   retrieving a previously started execution's status/result
   asynchronously. This requires credentials this environment does not
   have configured, so `get_execution_status()` honestly reports
   `not_configured` rather than fabricating a response.

Never hardcodes a URL or key -- both come from config.py/.env. Never
logs the API key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests

logger = logging.getLogger(__name__)


class ShuffleTriggerOutcome(str, Enum):
    TRIGGERED = "triggered"          # webhook accepted the request
    SYNCHRONOUS_RESULT = "synchronous_result"  # webhook returned the workflow's final output inline
    NOT_CONFIGURED = "not_configured"          # no SHUFFLE_WEBHOOK set
    AUTH_FAILED = "auth_failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ShuffleTriggerResult:
    outcome: ShuffleTriggerOutcome
    http_status: int | None = None
    shuffle_execution_id: str | None = None
    output: dict[str, Any] | str | None = None
    error: str | None = None


@dataclass
class ShuffleStatusResult:
    outcome: str  # "success" | "running" | "failed" | "not_configured" | "error"
    raw: dict[str, Any] | None = None
    error: str | None = None


class ShuffleClient:
    def __init__(self, webhook_url: str, base_url: str = "", api_key: str = "", timeout: int = 30):
        self.webhook_url = webhook_url
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.timeout = timeout

    def trigger(self, payload: dict[str, Any]) -> ShuffleTriggerResult:
        """POST payload to the configured webhook. Never raises --
        every failure mode (missing config, auth failure, timeout,
        network error) becomes a typed ShuffleTriggerResult so callers
        (soar/execution_service.py) never have to guess what happened."""
        if not self.webhook_url:
            return ShuffleTriggerResult(outcome=ShuffleTriggerOutcome.NOT_CONFIGURED)

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=self.timeout)
        except requests.exceptions.Timeout:
            return ShuffleTriggerResult(outcome=ShuffleTriggerOutcome.TIMEOUT)
        except requests.exceptions.RequestException as e:
            return ShuffleTriggerResult(outcome=ShuffleTriggerOutcome.ERROR, error=str(e))

        if response.status_code in (401, 403):
            return ShuffleTriggerResult(
                outcome=ShuffleTriggerOutcome.AUTH_FAILED, http_status=response.status_code
            )
        if response.status_code >= 400:
            return ShuffleTriggerResult(
                outcome=ShuffleTriggerOutcome.ERROR, http_status=response.status_code,
                error=f"Shuffle returned HTTP {response.status_code}",
            )

        body: dict[str, Any] | str
        try:
            body = response.json()
        except ValueError:
            body = response.text

        shuffle_execution_id = body.get("execution_id") if isinstance(body, dict) else None

        # A workflow with "wait for response" enabled returns its real
        # final output inline -- treat a dict body with actual content
        # as a synchronous result; an empty/ack-only body just confirms
        # the trigger fired and the workflow is running server-side.
        if isinstance(body, dict) and any(k not in ("execution_id", "success") for k in body):
            return ShuffleTriggerResult(
                outcome=ShuffleTriggerOutcome.SYNCHRONOUS_RESULT,
                http_status=response.status_code,
                shuffle_execution_id=shuffle_execution_id,
                output=body,
            )

        return ShuffleTriggerResult(
            outcome=ShuffleTriggerOutcome.TRIGGERED,
            http_status=response.status_code,
            shuffle_execution_id=shuffle_execution_id,
            output=body,
        )

    def get_execution_status(self, workflow_id: str, execution_id: str) -> ShuffleStatusResult:
        """Poll Shuffle's REST API for a previously triggered
        execution's status. Only usable when SHUFFLE_BASE_URL and
        SHUFFLE_API_KEY are both configured -- not the case in this
        environment (see SOAR_PLAYBOOK_INTEGRATION.md's
        SHUFFLE_INFRASTRUCTURE_BLOCKED note)."""
        if not self.base_url or not self.api_key:
            return ShuffleStatusResult(outcome="not_configured")

        url = f"{self.base_url}/api/v1/workflows/{workflow_id}/executions/{execution_id}"
        try:
            response = requests.get(
                url, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout
            )
        except requests.exceptions.RequestException as e:
            return ShuffleStatusResult(outcome="error", error=str(e))

        if response.status_code in (401, 403):
            return ShuffleStatusResult(outcome="error", error="Authentication failed")
        if response.status_code >= 400:
            return ShuffleStatusResult(outcome="error", error=f"HTTP {response.status_code}")

        try:
            data = response.json()
        except ValueError:
            return ShuffleStatusResult(outcome="error", error="Non-JSON response from Shuffle")

        shuffle_status = str(data.get("status", "")).upper()
        if shuffle_status in ("FINISHED", "SUCCESS"):
            outcome = "success"
        elif shuffle_status in ("EXECUTING", "RUNNING", "WAITING"):
            outcome = "running"
        elif shuffle_status:
            outcome = "failed"
        else:
            outcome = "error"

        return ShuffleStatusResult(outcome=outcome, raw=data)

    def health_check(self) -> bool:
        """Read-only reachability probe -- never triggers a workflow.
        Only meaningful when SHUFFLE_BASE_URL is configured; a
        webhook-only configuration has no equivalent unauthenticated
        endpoint to probe, so this reports False rather than guessing."""
        if not self.base_url:
            return False
        try:
            response = requests.get(f"{self.base_url}/api/v1/ping", timeout=5)
            return response.status_code < 400
        except requests.exceptions.RequestException:
            return False


def build_default_client() -> ShuffleClient:
    import config
    return ShuffleClient(
        webhook_url=config.SHUFFLE_WEBHOOK,
        base_url=config.SHUFFLE_BASE_URL,
        api_key=config.SHUFFLE_API_KEY,
    )

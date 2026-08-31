from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


class CTIPublisher:
    DEFAULT_TIMEOUT = 30

    def __init__(
        self,
        misp_url: str,
        api_key: str,
        verify_ssl: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ):

        self.url = misp_url.rstrip("/")
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.timeout = timeout

        self.session = requests.Session()

        retries = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1.5,
            status_forcelist=[
                429,
                500,
                502,
                503,
                504,
            ],
            allowed_methods=[
                "GET",
                "POST",
                "PUT",
            ],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount(
            "http://",
            adapter,
        )
        self.session.mount(
            "https://",
            adapter,
        )

        logger.info(
            "Initialized CTIPublisher -> %s",
            self.url,
        )

    def _headers(self) -> dict:
        return {
            "Authorization": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[dict] = None,
    ) -> dict:
        url = f"{self.url}{endpoint}"
        start = time.perf_counter()
        try:
            response: Response = self.session.request(
                method=method,
                url=url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            latency = round(
                time.perf_counter() - start,
                3,
            )
            logger.info(
                "%s %s -> %s (%.3fs)",
                method,
                endpoint,
                response.status_code,
                latency,
            )
            try:
                body = response.json()
            except Exception:
                body = {
                    "raw": response.text
                }
            return {
                "success": response.ok,
                "status": response.status_code,
                "response": body,
                "latency": latency,
            }
        except requests.exceptions.RequestException as exc:
            logger.exception(
                "MISP request failed: %s",
                exc,
            )

            return {
                "success": False,
                "status": None,
                "response": {
                    "error": str(exc)
                },
                "latency": None,
            }

    def create_event(
        self,
        event: dict,
    ) -> dict:
        result = self._request(
            method="POST",
            endpoint="/events/add",
            payload=event,
        )

        if not result["success"]:
            return result
        event_id = None
        try:
            event_id = int(
                result["response"]["Event"]["id"]
            )
        except Exception:
            pass
        result["event_id"] = event_id
        if event_id is not None:
            logger.info(
                "Created MISP Event [%s]",
                event_id,
            )
        return result

    def update_event(
        self,
        event_id: int,
        event: dict,
    ) -> dict:
        result = self._request(
            method="POST",
            endpoint=f"/events/edit/{event_id}",
            payload=event,
        )
        if result["success"]:
            logger.info(
                "Updated MISP Event [%s]",
                event_id,
            )
        return result

    def get_event(
        self,
        event_id: int,
    ) -> dict:
        return self._request(
            method="GET",
            endpoint=f"/events/view/{event_id}",
        )

    def search_campaign(
        self,
        campaign_id: str,
    ) -> Optional[int]:
        payload = {
            "returnFormat": "json",
            "value": f"CYUKTI Campaign {campaign_id}",
            "searchall": True,
        }
        result = self._request(
            method="POST",
            endpoint="/events/restSearch",
            payload=payload,
        )
        if not result["success"]:
            return None
        response = result["response"]
        if not isinstance(response, list):
            return None
        if len(response) == 0:
            return None
        try:
            event_id = int(
                response[0]["Event"]["id"]
            )
            logger.info(
                "Campaign [%s] already exists as Event [%d]",
                campaign_id,
                event_id,
            )
            return event_id

        except Exception:
            return None

    def event_exists(
        self,
        event_id: int,
    ) -> bool:
        result = self.get_event(event_id)
        return result["success"]

    def delete_event(
        self,
        event_id: int,
    ) -> dict:
        logger.warning(
            "Deleting MISP Event [%s]",
            event_id,
        )

        return self._request(
            method="POST",
            endpoint=f"/events/delete/{event_id}",
        )

    def health_check(self) -> bool:
        try:
            result = self._request(
                method="GET",
                endpoint="/servers/getVersion"
            )
            if result["success"]:
                logger.info(
                    "Connected to MISP successfully."
                )
                return True

            logger.error(
                "MISP health check failed (%s)",
                result["status"],
            )
            return False
        except Exception:
            logger.exception(
                "Unable to communicate with MISP."
            )
            return False

    def publish(
        self,
        event: dict,
    ) -> dict:
        return self.create_event(event)

    def close(self) -> None:
        try:
            self.session.close()
            logger.info(
                "Closed MISP session."
            )
        except Exception:
            logger.exception(
                "Failed closing MISP session."
            )

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):
        self.close()
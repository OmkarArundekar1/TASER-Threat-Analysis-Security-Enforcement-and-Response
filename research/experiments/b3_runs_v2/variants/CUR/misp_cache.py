from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MISPCache:
    def __init__(self, cache_file: str = "misp_cache.json"):
        self.cache_file = Path(cache_file)
        self._lock = threading.Lock()
        self._cache: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_file.exists():
            logger.info(
                "MISP cache does not exist. Creating a new cache."
            )
            self._cache = {}
            return
        try:
            with self.cache_file.open(
                "r",
                encoding="utf-8"
            ) as fp:
                data = json.load(fp)
                self._cache = {
                    str(k): int(v)
                    for k, v in data.items()
                }
            logger.info(
                "Loaded %d cached campaign mappings.",
                len(self._cache),
            )
        except Exception as exc:
            logger.exception(
                "Failed to load MISP cache: %s",
                exc,
            )
            self._cache = {}

    def _save(self) -> None:
        try:
            with self.cache_file.open(
                "w",
                encoding="utf-8"
            ) as fp:
                json.dump(
                    self._cache,
                    fp,
                    indent=4,
                    sort_keys=True,
                )
        except Exception as exc:
            logger.exception(
                "Failed saving MISP cache: %s",
                exc,
            )

    def get_event_id(
        self,
        campaign_id: str,
    ) -> Optional[int]:
        with self._lock:
            return self._cache.get(campaign_id)

    def set_event_id(
        self,
        campaign_id: str,
        event_id: int,
    ) -> None:
        with self._lock:
            self._cache[campaign_id] = int(event_id)
            self._save()
        logger.info(
            "Mapped Campaign [%s] -> MISP Event [%d]",
            campaign_id,
            event_id,
        )

    def remove(
        self,
        campaign_id: str,
    ) -> None:
        with self._lock:
            if campaign_id in self._cache:
                del self._cache[campaign_id]
                self._save()
                logger.info(
                    "Removed campaign [%s] from cache.",
                    campaign_id,
                )

    def contains(
        self,
        campaign_id: str,
    ) -> bool:
        with self._lock:
            return campaign_id in self._cache

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._save()
        logger.info("Cleared MISP cache.")

    def all(self) -> dict[str, int]:
        with self._lock:
            return dict(self._cache)

cache = MISPCache()
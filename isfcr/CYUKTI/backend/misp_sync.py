from __future__ import annotations
import logging
from misp_cache import cache
from misp_event_generator import engine as misp_generator
from cti_publisher import CTIPublisher
logger = logging.getLogger(__name__)

class MISPSync:
    def __init__(
        self,
        publisher: CTIPublisher,
    ):
        self.publisher = publisher

    def _generate_event(
        self,
        incident,
    ) -> dict:
        logger.info(
            "Generating MISP Event for campaign [%s]",
            incident.campaign_id,
        )
        return misp_generator.generate(
            incident
        )

    def _cached_event(
        self,
        campaign_id: str,
    ):
        event_id = cache.get_event_id(
            campaign_id
        )
        if event_id is not None:
            logger.info(
                "Cache Hit -> Campaign [%s] Event [%s]",
                campaign_id,
                event_id,
            )
        else:
            logger.info(
                "Cache Miss -> %s",
                campaign_id,
            )

        return event_id

    def _cache_event(
        self,
        campaign_id: str,
        event_id: int,
    ):
        cache.set_event_id(
            campaign_id,
            event_id,
        )

        logger.info(
            "Campaign [%s] mapped to Event [%s]",
            campaign_id,
            event_id,
        )

    def _search_existing(
        self,
        campaign_id: str,
    ):
        logger.info(
            "Searching MISP for Campaign [%s]",
            campaign_id,
        )
        return self.publisher.search_campaign(
            campaign_id
        )

    def _create(
        self,
        campaign_id: str,
        event: dict,
    ) -> dict:
        logger.info(
            "Creating new MISP Event for Campaign [%s]",
            campaign_id,
        )
        result = self.publisher.create_event(event)
        if not result["success"]:
            logger.error(
                "Failed creating MISP Event."
            )
            return result
        event_id = result.get("event_id")
        if event_id is not None:
            self._cache_event(
                campaign_id,
                event_id,
            )
        return result

    def _update(
        self,
        event_id: int,
        event: dict,
    ) -> dict:
        logger.info(
            "Updating existing MISP Event [%s]",
            event_id,
        )
        return self.publisher.update_event(
            event_id,
            event,
        )

    def synchronize(
        self,
        incident,
    ) -> dict:
        campaign_id = incident.campaign_id
        logger.info(
            "Starting synchronization for [%s]",
            campaign_id,
        )

        event = self._generate_event(
            incident
        )
        event_id = self._cached_event(
            campaign_id
        )

        if event_id is not None:
            if self.publisher.event_exists(
                event_id
            ):

                logger.info(
                    "Updating cached Event [%s]",
                    event_id,
                )

                return self._update(
                    event_id,
                    event,
                )
            logger.warning(
                "Cached Event [%s] no longer exists.",
                event_id,
            )
            cache.remove(
                campaign_id
            )

        event_id = self._search_existing(
            campaign_id
        )

        if event_id is not None:
            logger.info(
                "Campaign already exists in MISP."
            )
            self._cache_event(
                campaign_id,
                event_id,
            )
            return self._update(
                event_id,
                event,
            )
        logger.info(
            "Campaign not found. Creating event."
        )

        return self._create(
            campaign_id,
            event,
        )

    def should_publish(
        self,
        incident,
    ) -> bool:
        publish = bool(
            getattr(
                incident.cti,
                "publish",
                False,
            )
        )
        if publish:
            logger.info(
                "Campaign [%s] approved for publication.",
                incident.campaign_id,
            )
        else:
            logger.info(
                "Campaign [%s] not published (CTI policy).",
                incident.campaign_id,
            )
        return publish

    def publish_campaign(
        self,
        incident,
    ) -> dict:
        if not self.should_publish(
            incident
        ):
            return {
                "success": False,
                "action": "skipped",
                "reason": "CTI policy",
            }

        result = self.synchronize(
            incident
        )
        if result["success"]:
            action = (
                "updated"
                if result.get("event_id") is None
                else "created"
            )
            logger.info(
                "Campaign [%s] synchronized successfully.",
                incident.campaign_id,
            )
            result["action"] = action
        return result
        
    def close_campaign(
        self,
        campaign_id: str,
    ) -> bool:
        event_id = cache.get_event_id(
            campaign_id
        )
        if event_id is None:
            return False
        cache.remove(
            campaign_id
        )
        logger.info(
            "Campaign [%s] closed.",
            campaign_id,
        )
        return True

    def clear_cache(self):
        cache.clear()

        logger.info(
            "MISP cache cleared."
        )

    def statistics(self):
        mappings = cache.all()
        return {
            "cached_campaigns": len(
                mappings
            ),
            "campaigns": mappings,
        }

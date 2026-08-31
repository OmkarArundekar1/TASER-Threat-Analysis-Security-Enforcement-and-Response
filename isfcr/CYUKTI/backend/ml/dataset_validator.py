import math
from dataclasses import fields
from feature_schema import CampaignDatasetRecord

class DatasetValidator:
    VALID_SEVERITIES = {
        "Low",
        "Medium",
        "High",
        "Critical"
    }

    @staticmethod
    def _safe_number(value):
        if value is None:
            return 0

        if isinstance(value, float):
            if math.isnan(value):
                return 0.0
            if math.isinf(value):
                return 0.0
        return value

    @staticmethod
    def _clamp_probability(value):
        value = DatasetValidator._safe_number(value)
        return max(
            0.0,
            min(
                float(value),
                1.0
            )
        )

    def validate(
        self,
        record: CampaignDatasetRecord
    ) -> CampaignDatasetRecord:
        for field in fields(record):
            value = getattr(
                record,
                field.name
            )

            if isinstance(value, (int, float)):
                value = self._safe_number(value)
                setattr(
                    record,
                    field.name,
                    value
                )
        probability_fields = [
            "prediction_similarity",
            "chain_similarity",
            "temporal_similarity",
            "attacker_similarity",
            "duplicate_similarity",
            "graph_similarity",
            "runtime_similarity",
            "ip_reputation",
            "threat_actor_reputation",
            "malware_confidence",
            "tool_confidence",
            "misp_confidence",
            "ioc_confidence",
            "detection_confidence"
        ]

        for name in probability_fields:
            setattr(
                record,
                name,
                self._clamp_probability(
                    getattr(record, name)
                )
            )
        if record.severity not in self.VALID_SEVERITIES:
            record.severity = "Low"
        integer_fields = [
            "campaign_size",
            "unique_techniques",
            "node_count",
            "edge_count",
            "campaign_count",
            "attack_event_count",
            "prediction_frequency",
            "duplicate_frequency",
            "community_count",
            "largest_community",
            "attribution_correct"
        ]

        for name in integer_fields:
            value = getattr(
                record,
                name
            )
            setattr(
                record,
                name,
                max(
                    0,
                    int(value)
                )
            )

        # prediction_correct is NOT_APPLICABLE-aware: None means "no
        # observed transition or no learned prediction to check against"
        # (see ml/label_generator.py) and must be preserved as-is, not
        # coerced to 0 -- silently turning NA into "incorrect" would
        # misrepresent every single-event campaign as a wrong prediction.
        if record.prediction_correct is not None:
            record.prediction_correct = max(0, int(record.prediction_correct))
        return record

validator = DatasetValidator()
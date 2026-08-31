from feature_extractors import extractor
from label_generator import generator
from dataset_validator import validator
from dataset_writer import writer

class CampaignDatasetBuilder:
    def build(
        self,
        campaign_context,
        final_attack_id,
        attacker_ip,
        event_id,
        attributed_actor,
        technique_sequence=None,
        actual_actor=None
    ):
        # technique_sequence is the campaign's real chronological technique
        # order (oldest -> newest); defaults to [final_attack_id] when the
        # caller doesn't have it, which correctly yields NOT_APPLICABLE
        # next_technique/prediction_correct labels rather than fabricating
        # a transition out of a single known point.
        if technique_sequence is None:
            technique_sequence = [final_attack_id]
        labels = generator.generate(
            campaign_context=campaign_context,
            technique_sequence=technique_sequence,
            final_attack_id=final_attack_id,
            attributed_actor=attributed_actor,
            actual_actor=actual_actor
        )
        record = extractor.extract(
            campaign_context=campaign_context,
            current_attack_id=final_attack_id,
            attacker_ip=attacker_ip,
            event_id=event_id,
            severity=labels.severity,
            attributed_actor=labels.attributed_actor,
            prediction_correct=labels.prediction_correct,
            attribution_correct=labels.attribution_correct,
            next_technique=labels.next_technique
        )
        record = validator.validate(record)
        writer.append(record)
        return record

    def export_parquet(self):
        writer.export_parquet()

    def dataset_size(self):
        return writer.size()

    def load_dataset(self):
        return writer.load()

builder = CampaignDatasetBuilder()
from neo4j_client import get_mitre_features

class MitreFeatureEngine:
    def extract_features(self, attack_id):
        return get_mitre_features(attack_id)
engine = MitreFeatureEngine()
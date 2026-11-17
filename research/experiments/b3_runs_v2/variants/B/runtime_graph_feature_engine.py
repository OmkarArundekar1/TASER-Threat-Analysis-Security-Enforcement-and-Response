from neo4j_client import get_runtime_graph_features

class RuntimeGraphFeatureEngine:
    def extract_features(self, campaign_id):
        return get_runtime_graph_features(
            campaign_id
        )
engine = RuntimeGraphFeatureEngine()
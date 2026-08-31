from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from collections import OrderedDict
import networkx as nx
from neo4j_client import driver
logger = logging.getLogger(__name__)
from datetime_utils import normalize_datetime

@dataclass
class GraphFeatures:
    campaign_size: int = 0
    unique_techniques: int = 0
    campaign_duration: float = 0.0
    node_count: int = 0
    edge_count: int = 0
    graph_density: float = 0.0
    graph_connectivity: float = 0.0
    average_degree: float = 0.0
    attacker_degree: int = 0
    victim_degree: int = 0
    technique_degree: int = 0
    attack_chain_depth: int = 0
    average_path_length: float = 0.0
    graph_diameter: int = 0
    branching_factor: float = 0.0
    average_clustering: float = 0.0
    average_betweenness: float = 0.0
    average_closeness: float = 0.0
    pagerank: Dict[str, float] = field(default_factory=dict)
    betweenness: Dict[str, float] = field(default_factory=dict)
    closeness: Dict[str, float] = field(default_factory=dict)
    eigenvector: Dict[str, float] = field(default_factory=dict)
    community_count: int = 0
    largest_community: int = 0
    campaign_complexity: float = 0.0
    structural_risk: float = 0.0
    evolution_rate: float = 0.0

@dataclass
class GraphSnapshot:
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)

class GraphSnapshotLoader:
    def load_campaign_graph(
        self,
        campaign_id: str
    ) -> GraphSnapshot:
        snapshot = GraphSnapshot()
        query = """
        MATCH (c:Campaign {campaign_id:$campaign_id})
        OPTIONAL MATCH (a)-[r1]->(c)
        OPTIONAL MATCH (c)-[r2]->(e:AttackEvent)
        OPTIONAL MATCH (e)-[r3]->(t)
        OPTIONAL MATCH (c)-[r4]->(h)

        WITH
            collect(DISTINCT a) +
            collect(DISTINCT c) +
            collect(DISTINCT e) +
            collect(DISTINCT t) +
            collect(DISTINCT h) AS nodes,

            collect(DISTINCT r1) +
            collect(DISTINCT r2) +
            collect(DISTINCT r3) +
            collect(DISTINCT r4) AS rels

        RETURN nodes, rels
        """
        try:
            with driver.session() as session:
                record = session.run(
                    query,
                    campaign_id=campaign_id
                ).single()
                if record is None:
                    return snapshot
                for node in record["nodes"]:
                    if node is None:
                        continue
                    snapshot.nodes.append(
                        {
                            "id": node.element_id,
                            "labels": list(node.labels),
                            "properties": dict(node)
                        }
                    )
                for rel in record["rels"]:
                    if rel is None:
                        continue
                    snapshot.relationships.append(
                        {
                            "id": rel.element_id,
                            "type": rel.type,
                            "start": rel.start_node.element_id,
                            "end": rel.end_node.element_id,
                            "properties": dict(rel)
                        }
                    )
        except Exception as e:
            logger.exception(
                "GraphSnapshotLoader failed: %s",
                e
            )
        return snapshot

class GraphBuilder:
    def build(
        self,
        snapshot: GraphSnapshot
    ) -> nx.DiGraph:
        G = nx.DiGraph()
        for node in snapshot.nodes:
            G.add_node(
                node["id"],
                labels=node["labels"],
                **node["properties"]
            )
        for rel in snapshot.relationships:
            G.add_edge(
                rel["start"],
                rel["end"],
                relationship=rel["type"],
                **rel["properties"]
            )
        return G

class GraphAnalytics:
    def __init__(self):
        self.loader = GraphSnapshotLoader()
        self.builder = GraphBuilder()
        self.graph_cache = OrderedDict()
        self.MAX_GRAPH_CACHE = 500

    def clear_cache(self):
        self.graph_cache.clear()

    def invalidate_campaign(
        self,
        campaign_id: str
    ):
        self.graph_cache.pop(campaign_id, None)
        
    def load_graph(
        self,
        campaign_id: str,
        force_reload: bool = False
    ) -> nx.DiGraph:
    
        if (
            not force_reload and
            campaign_id in self.graph_cache
        ):
            self.graph_cache.move_to_end(campaign_id)
            return self.graph_cache[campaign_id]
    
        snapshot = self.loader.load_campaign_graph(
            campaign_id
        )    
        graph = self.builder.build(snapshot)
        self.graph_cache[campaign_id] = graph
        self.graph_cache.move_to_end(campaign_id)
        if len(self.graph_cache) > self.MAX_GRAPH_CACHE:
            self.graph_cache.popitem(last=False)
        return graph

    def extract_features(
        self,
        campaign_id: str,
        force_reload: bool = False
    ) -> GraphFeatures:
        G = self.load_graph(
            campaign_id,
            force_reload
        )
        UG = G.to_undirected()        
        self.undirected_graph = UG
        if UG.number_of_nodes():
            self.largest_component = self.largest_connected_component(G)
        else:
            self.largest_component = UG
            
        self.label_index = {}
        for node, data in G.nodes(data=True):
            for label in data.get("labels", []):
                self.label_index.setdefault(
                    label,
                    []
                ).append((node, data))
        self.degree_cache = dict(G.degree())
        try:
            self.communities = list(
                nx.community.greedy_modularity_communities(
                    self.undirected_graph
                )
            )
        except Exception:
            self.communities = []
        features = GraphFeatures()
        features.campaign_size = self.campaign_size(G)
        features.unique_techniques = self.unique_techniques(G)
        features.campaign_duration = self.campaign_duration(G)
        features.node_count = self.node_count(G)
        features.edge_count = self.edge_count(G)
        features.graph_density = self.graph_density(G)
        features.graph_connectivity = self.graph_connectivity(G)
        features.average_degree = self.average_degree(G)
        features.attacker_degree = self.attacker_degree(G)
        features.victim_degree = self.victim_degree(G)
        features.technique_degree = self.technique_degree(G)
        features.attack_chain_depth = self.attack_chain_depth(G)
        features.branching_factor = self.branching_factor(G)
        features.graph_diameter = self.graph_diameter(G)
        features.average_path_length = (
            self.average_path_length(G)
        )
        self.cached_pagerank = self.pagerank(G)
        self.cached_betweenness = self.betweenness(G)
        self.cached_closeness = self.closeness(G)
        self.cached_eigenvector = self.eigenvector(G)
        features.pagerank = self.cached_pagerank
        features.betweenness = self.cached_betweenness
        features.closeness = self.cached_closeness
        features.eigenvector = self.cached_eigenvector
        features.average_betweenness = self.safe_average(
            list(self.cached_betweenness.values())
        )
        
        features.average_closeness = self.safe_average(
            list(self.cached_closeness.values())
        )
        
        features.average_clustering = self.average_clustering(G)
        features.community_count = len(self.communities)
        features.largest_community = (
            self.largest_community(G)
        )
        features.campaign_complexity = (
            self.campaign_complexity(features)
        )
        features.structural_risk = (
            self.structural_risk(features)
        )
        features.evolution_rate = (
            self.evolution_rate(features)
        )
        return features

    def largest_connected_component(
        self,
        G: nx.DiGraph
    ) -> nx.Graph:
        if G.number_of_nodes() == 0:
            return G.to_undirected()
        UG = G.to_undirected()
        if nx.is_connected(UG):
            return UG

        component = max(
            nx.connected_components(UG),
            key=len
        )
        return UG.subgraph(component).copy()

    def campaign_node(
        self,
        G
    ) -> Optional[Dict]:
        for _, data in G.nodes(data=True):
            if "Campaign" in data.get(
                "labels",
                []
            ):
                return data
        return None

    def nodes_by_label(
        self,
        G,
        label
    ):
        return self.label_index.get(
            label,
            []
        )
        
    def safe_average(
        self,
        values
    ):
        if not values:
            return 0.0
        return round(
            sum(values) / len(values),
            4
        )
        
    def safe_max(
        self,
        values
    ):
        if not values:
            return 0
        return max(values)

    def campaign_size(self, G):
        return len(
            self.nodes_by_label(
                G,
                "AttackEvent"
            )
        )

    def unique_techniques(self, G):
        techniques = set()
        for _, data in self.nodes_by_label(
            G,
            "Technique"
        ):
            tid = (
                data.get("technique_id")
                or data.get("attack_id")
                or data.get("external_id")
                or data.get("id")
                or data.get("name")
            )
            if tid:
                techniques.add(tid)
        return len(techniques)

    def campaign_duration(self, G):
        campaign = self.campaign_node(G)
        if campaign is None:
            return 0.0
        first = campaign.get("first_seen")
        last = campaign.get("last_seen")
        if first is None or last is None:
            return 0.0
        try:
            first = normalize_datetime(first)
            last = normalize_datetime(last)
            
            if first is None or last is None:
                return 0.0
            
            try:
                return (last - first).total_seconds()
            except Exception:
                return 0.0
            if isinstance(first, datetime):
                return (
                    last - first
                ).total_seconds()
            return float(last - first)
        except Exception:
            return 0.0

    def node_count(self, G):
        return G.number_of_nodes()

    def edge_count(self, G):
        return G.number_of_edges()

    def graph_density(self, G):
        if G.number_of_nodes() <= 1:
            return 0.0
        return round(
            nx.density(G),
            4
        )

    def graph_connectivity(self, G):
        if G.number_of_nodes() == 0:
            return 0.0
        UG = G.to_undirected()
        connected = sum(
            len(component)
            for component in nx.connected_components(
                UG
            )
        )
        return round(
            connected /
            G.number_of_nodes(),
            4
        )

    def average_degree(self, G):
        if G.number_of_nodes() == 0:
            return 0.0
        degrees = list(
            self.degree_cache.values()
        )
        return round(
            sum(degrees) /
            len(degrees),
            2
        )

    def attacker_degree(self, G):
        values = [
            self.degree_cache[node]
            for node, _ in self.nodes_by_label(
                G,
                "Attacker"
            )
        ]
        return self.safe_max(values)

    def victim_degree(self, G):
        values = []
        for node, data in G.nodes(data=True):
            labels = data.get(
                "labels",
                []
            )
            if (
                "Host" in labels
                or
                "Victim" in labels
            ):
                values.append(
                    self.degree_cache[node]
                )
        return self.safe_max(values)

    def technique_degree(self, G):
        values = [
            self.degree_cache[node]

            for node, _ in self.nodes_by_label(
                G,
                "Technique"
            )
        ]
        return self.safe_max(values)

    def attack_chain_depth(self, G):
        if G.number_of_nodes() == 0:
            return 0
        try:
            if nx.is_directed_acyclic_graph(G):
                return nx.dag_longest_path_length(
                    G
                )
            return 0
        except Exception:
            return 0

    def branching_factor(self, G):
        if G.number_of_nodes() == 0:
            return 0.0
        values = [
            G.out_degree(node)
            for node in G.nodes()
        ]
        return round(
            sum(values) /
            len(values),
            2
        )

    def graph_diameter(self, G):
        if G.number_of_nodes() <= 1:
            return 0
        try:
            largest = self.largest_component
            return nx.diameter(
                largest
            )
        except Exception:
            return 0

    def average_path_length(self, G):
        if G.number_of_nodes() <= 1:
            return 0.0
        try:
            largest = self.largest_component
            return round(
                nx.average_shortest_path_length(
                    largest
                ),
                3
            )
        except Exception:
            return 0.0

    def pagerank(self, G):
        if G.number_of_nodes() <= 1:
            return {}
        try:
            return nx.pagerank(
                G,
                alpha=0.85
            )

        except Exception:
            logger.exception(
                "PageRank calculation failed."
            )
            return {}

    def betweenness(self, G):
        if G.number_of_nodes() <= 1:
            return {}
        try:
            return nx.betweenness_centrality(
                G,
                normalized=True
            )
        except Exception:
            logger.exception(
                "Betweenness calculation failed."
            )
            return {}

    def closeness(self, G):
        if G.number_of_nodes() <= 1:
            return {}
        try:
            return nx.closeness_centrality(
                G
            )
        except Exception:
            logger.exception(
                "Closeness calculation failed."
            )
            return {}

    def eigenvector(self, G):
        if G.number_of_nodes() <= 1:
            return {}
        try:
            UG = self.largest_component
            values = nx.eigenvector_centrality_numpy(
                UG
            )
            return dict(values)
        except Exception:
            logger.exception(
                "Eigenvector calculation failed."
            )
            return {}

    def average_clustering(self, G):
        if G.number_of_nodes() <= 1:
            return 0.0
        try:
            return round(
                nx.average_clustering(
                    G.to_undirected()
                ),
                4
            )
        except Exception:
            return 0.0
            
    def average_betweenness(self, G):
        values = self.betweenness(G)
        if not values:
            return 0.0
        return round(
            sum(values.values()) /
            len(values),
            4
        )

    def average_closeness(self, G):
        values = self.closeness(G)
        if not values:
            return 0.0
        return round(
            sum(values.values()) /
            len(values),
            4
        )

    def top_pagerank(
        self,
        top_n=5
    ):
        return sorted(
            self.cached_pagerank.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

    def top_betweenness(
        self,
        top_n=5
    ):
        values = self.cached_betweenness
        return sorted(
            values.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

    def top_closeness(
        self,
        top_n=5
    ):
        values = self.cached_closeness
        return sorted(
            values.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

    def top_eigenvector(
        self,
        top_n=5
    ):
        values = self.cached_eigenvector
        return sorted(
            values.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

    def attack_chain(self, G):
        if G.number_of_nodes() == 0:
            return []
        try:
            if nx.is_directed_acyclic_graph(G):
                return nx.dag_longest_path(G)
            return []
        except Exception:
            return []

    def chain_length(self, G):
        chain = self.attack_chain(G)
        return len(chain)

    def terminal_nodes(self, G):
        terminals = []
        for node in G.nodes():
            if G.out_degree(node) == 0:
                terminals.append(node)
        return terminals

    def entry_nodes(self, G):
        entries = []
        for node in G.nodes():
            if G.in_degree(node) == 0:
                entries.append(node)
        return entries

    def branching_nodes(self, G):
        nodes = []
        for node in G.nodes():
            if G.out_degree(node) > 1:
                nodes.append(node)
        return nodes

    def dead_end_nodes(self, G):
        return self.terminal_nodes(G)

    def chain_complexity(self, G):
        length = self.chain_length(G)
        branches = len(
            self.branching_nodes(G)
        )
        terminals = len(
            self.terminal_nodes(G)
        )
        return round(
            length * 2 +
            branches * 3 +
            terminals,
            2
        )

    def community_count(self, G):
        if G.number_of_nodes() <= 1:
            return 0
        try:
            communities = self.communities
            return len(communities)
        except Exception:
            return 0
            
    def largest_community(self, G):
        try:
            communities = self.communities
            return max(
                len(c)
                for c in communities
            )
        except Exception:
            return 0

    def community_sizes(self, G):
        try:
            communities = self.communities
            return [
                len(c)
                for c in communities
            ]
        except Exception:
            return []

    def modularity_score(self, G):
        try:
            communities = self.communities
            return round(
                nx.community.modularity(
                    G.to_undirected(),
                    communities
                ),
                4
            )
        except Exception:
            return 0.0

    def bridge_nodes(self, G):
        try:
            UG = G.to_undirected()
            return list(
                nx.articulation_points(UG)
            )
        except Exception:
            return []

    def bridge_count(self, G):
        return len(
            self.bridge_nodes(G)
        )

    def connected_components(self, G):
        try:
            UG = G.to_undirected()
            return nx.number_connected_components(
                UG
            )
        except Exception:
            return 0

    def campaign_complexity(self, features):
        score = (
            features.graph_density * 25 +
            features.average_degree * 2 +
            features.average_clustering * 15 +
            features.attack_chain_depth * 2 +
            features.unique_techniques +
            features.community_count * 2 +
            features.branching_factor
        )
        return round(score, 2)

    def structural_risk(self, features):
        risk = (
            features.graph_density * 20 +
            features.average_degree * 3 +
            features.attack_chain_depth * 4 +
            features.unique_techniques * 2 +
            features.campaign_complexity
        )
        return round(risk, 2)

    def evolution_rate(self, features):
        if features.campaign_duration <= 0:
            return 0.0
        return round(
            features.campaign_size /
            max(features.campaign_duration, 1),
            4
        )

    def feature_summary(
        self,
        features: GraphFeatures
    ):
        return {
            "campaign_size":
                features.campaign_size,
            "unique_techniques":
                features.unique_techniques,
            "campaign_duration":
                features.campaign_duration,
            "node_count":
                features.node_count,
            "edge_count":
                features.edge_count,
            "graph_density":
                features.graph_density,
            "graph_connectivity":
                features.graph_connectivity,
            "average_degree":
                features.average_degree,
            "attack_chain_depth":
                features.attack_chain_depth,
            "graph_diameter":
                features.graph_diameter,
            "average_path_length":
                features.average_path_length,
            "branching_factor":
                features.branching_factor,
            "average_clustering":
                features.average_clustering,
            "average_betweenness":
                features.average_betweenness,
            "average_closeness":
                features.average_closeness,
            "community_count":
                features.community_count,
            "largest_community":
                features.largest_community,
            "campaign_complexity":
                features.campaign_complexity,
            "structural_risk":
                features.structural_risk,
            "evolution_rate":
                features.evolution_rate
        }

    def refresh_campaign(
        self,
        campaign_id: str
    ):
        self.invalidate_campaign(
            campaign_id
        )
        return self.extract_features(
            campaign_id,
            force_reload=True
        )

    def remove_campaign(
        self,
        campaign_id: str
    ):
        self.graph_cache.pop(
            campaign_id,
            None
        )

    def cache_size(self):
        return len(
            self.graph_cache
        )

    def cached_campaigns(self):
        return list(
            self.graph_cache.keys()
        )

    def graph_statistics(
        self,
        campaign_id: str
    ):
        G = self.load_graph(
            campaign_id
        )
        return {
            "nodes":
                G.number_of_nodes(),
            "edges":
                G.number_of_edges(),
            "weak_components":
                nx.number_weakly_connected_components(G),
            "is_dag":
                nx.is_directed_acyclic_graph(G),
            "is_empty":
                G.number_of_nodes() == 0,
            "density":
                self.graph_density(G)
        }
    def print_summary(
        self,
        campaign_id
    ):
        features = self.extract_features(
            campaign_id
        )
        print("\n===== Graph Feature Summary =====")
        for key, value in self.feature_summary(
            features
        ).items():
            print(f"{key:<30}{value}")
        print("===============================\n")
graph_analytics = GraphAnalytics()
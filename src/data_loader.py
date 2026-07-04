"""
Data Loading Module – Optimized for Low Memory
"""

import numpy as np
import networkx as nx
from typing import Tuple, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpinionsDataLoader:
    ""Loading the Epinions dataset with optimized memory management""
    
    def __init__(self, file_path: str, max_nodes: Optional[int] = None):
        self.file_path = file_path
        self.max_nodes = max_nodes
        self.graph = None
        self.adj_matrix = None
        
    def load_graph(self) -> nx.DiGraph:
        """Loading a directed graph from a file"""
        logger.info(f"Loading graph from {self.file_path}")
        
        # Reading file lines
        edges = []
        node_set = set()
        
        with open(self.file_path, 'r') as f:
            for line in tqdm(f, desc="Reading edges"):
                line = line.strip()
                if line.startswith('#'):
                    continue
                try:
                    src, dst = map(int, line.split())
                    edges.append((src, dst))
                    node_set.add(src)
                    node_set.add(dst)
                    
                    # Limiting the number of nodes, if necessary.
                    if self.max_nodes and len(node_set) >= self.max_nodes:
                        break
                except:
                    continue
        
        # Constructing a directed graph
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edges)
        
        logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def get_stats(self) -> dict:
        """Calculating Graph Statistics"""
        if self.graph is None:
            raise ValueError("Graph not loaded. Call load_graph() first.")
        
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': 2 * self.graph.number_of_edges() / self.graph.number_of_nodes(),
        }
        
        # Calculating the expansion coefficient (via sampling for large graphs)
        if self.graph.number_of_nodes() > 10000:
            stats['clustering'] = nx.average_clustering(
                self.graph.to_undirected(), trials=1000
            )
        else:
            stats['clustering'] = nx.average_clustering(self.graph.to_undirected())
        
        return stats


class GraphPreprocessor:
    """Efficiency-Preserving Graph Preprocessing"""
    
    def __init__(self, config: dict):
        self.config = config
        self.graph = None
        
    def preprocess(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Applying the necessary preprocessing steps"""
        logger.info("Preprocessing graph...")
        
        # Delete self-loops
        if self.config.get('remove_self_loops', True):
            edges_to_remove = [(u, v) for u, v in graph.edges() if u == v]
            graph.remove_edges_from(edges_to_remove)
            logger.info(f"Removed {len(edges_to_remove)} self-loops")
        
        # Removing isolated nodes
        if self.config.get('remove_isolated', True):
            isolated = [n for n in graph.nodes() if graph.degree(n) == 0]
            graph.remove_nodes_from(isolated)
            logger.info(f"Removed {len(isolated)} isolated nodes")
        
        # Using the largest connected component
        if self.config.get('use_lcc', True):
            undirected = graph.to_undirected()
            largest_cc = max(nx.connected_components(undirected), key=len)
            subgraph = graph.subgraph(largest_cc).copy()
            logger.info(f"Using LCC: {subgraph.number_of_nodes()} nodes")
            self.graph = subgraph
        else:
            self.graph = graph
        
        return self.graph
    
    def sample_subgraph(self, num_nodes: int) -> nx.DiGraph:
        """Graph sampling for memory reduction"""
        if num_nodes >= self.graph.number_of_nodes():
            return self.graph
        
        # Using random walk for sampling (preserving structure)
        nodes = list(self.graph.nodes())
        start_node = np.random.choice(nodes)
        
        # BFS Restricted
        sampled_nodes = set([start_node])
        frontier = [start_node]
        
        while len(sampled_nodes) < num_nodes and frontier:
            current = frontier.pop(0)
            neighbors = list(self.graph.neighbors(current))
            np.random.shuffle(neighbors)
            
            for neighbor in neighbors[:5]:  # Limiting the degree
                if neighbor not in sampled_nodes:
                    sampled_nodes.add(neighbor)
                    frontier.append(neighbor)
                if len(sampled_nodes) >= num_nodes:
                    break
        
        # Adding random nodes if necessary
        if len(sampled_nodes) < num_nodes:
            remaining = set(nodes) - sampled_nodes
            sampled_nodes.update(np.random.choice(list(remaining), 
                                                   num_nodes - len(sampled_nodes), 
                                                   replace=False))
        
        return self.graph.subgraph(sampled_nodes).copy()

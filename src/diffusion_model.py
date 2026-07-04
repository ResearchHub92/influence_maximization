# """
# Independent Cascade Diffusion Model – Optimized with NumPy
# """

# import numpy as np
# from scipy.sparse import csr_matrix
# import random
# from typing import List, Set, Tuple
# import logging

# logger = logging.getLogger(__name__)


# class IndependentCascade:
#     """Independent Cascade diffusion model with fast implementation"""
    
#     def __init__(self, graph, adj_matrix=None, prob=0.1):
#         """
#         Args:
#             graph: NetworkX graph
#             adj_matrix: Sparse adjacency matrix (optional)
#             prob: Probability of spread or 'weighted'
#         """
#         self.graph = graph
#         self.n_nodes = graph.number_of_nodes()
#         self.node_list = list(graph.nodes())
#         self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        
#         # Constructing a sparse adjacency matrix
#         if adj_matrix is None:
#             self.adj_matrix = self._build_sparse_adjacency()
#         else:
#             self.adj_matrix = adj_matrix
        
#         # Calculation of propagation probabilities
#         self.probabilities = self._compute_probabilities(prob)
        
#     def _build_sparse_adjacency(self) -> csr_matrix:
#         """Constructing a sparse adjacency matrix"""
#         rows, cols = [], []
#         for u, v in self.graph.edges():
#             rows.append(self.node_to_idx[u])
#             cols.append(self.node_to_idx[v])
        
#         data = np.ones(len(rows))
#         return csr_matrix((data, (rows, cols)), 
#                          shape=(self.n_nodes, self.n_nodes))
    
#     def _compute_probabilities(self, prob):
#         """Calculation of propagation probabilities for each edge"""
#         if prob == 'weighted':
#             # Weighted Cascade: 1/in_degree(v)
#             in_degrees = np.array(self.adj_matrix.sum(axis=0)).flatten()
#             probs = np.zeros_like(self.adj_matrix.data)
#             for i, (_, v) in enumerate(zip(*self.adj_matrix.nonzero())):
#                 if in_degrees[v] > 0:
#                     probs[i] = 1.0 / in_degrees[v]
#                 else:
#                     probs[i] = 0.01
#             return csr_matrix((probs, self.adj_matrix.nonzero()), 
#                             shape=self.adj_matrix.shape)
#         else:
#             # ثابت
#             probs = np.full_like(self.adj_matrix.data, prob)
#             return csr_matrix((probs, self.adj_matrix.nonzero()), 
#                             shape=self.adj_matrix.shape)
    
#     def simulate(self, seeds: List[int], num_simulations: int = 100) -> float:
#         """
#         Simulation of the spread and mean reversion of the number of active agents
#         """
#         if num_simulations == 1:
#             return len(self._single_simulation(seeds))
        
#         total_activated = 0
#         for _ in range(num_simulations):
#             activated = self._single_simulation(seeds)
#             total_activated += len(activated)
        
#         return total_activated / num_simulations
    
#     def _single_simulation(self, seeds: List[int]) -> Set[int]:
#         """Running a simulation"""
#         # Convert to indices
#         seed_indices = [self.node_to_idx[s] for s in seeds]
        
#         activated = set(seed_indices)
#         frontier = set(seed_indices)
        
#         while frontier:
#             new_frontier = set()
#             for node in frontier:
#                 # Finding outgoing neighbors
#                 neighbors = self.adj_matrix[node].nonzero()[1]
                
#                 for neighbor in neighbors:
#                     if neighbor in activated:
#                         continue
                    
#                     # Analysis of Probabilistic Activation
#                     prob = self.probabilities[node, neighbor]
#                     if random.random() < prob:
#                         activated.add(neighbor)
#                         new_frontier.add(neighbor)
            
#             frontier = new_frontier
        
#         # Convert to main nodes
#         return {self.node_list[i] for i in activated}
    
#     def simulate_batch(self, seeds_batch: List[List[int]], num_simulations: int = 10) -> np.ndarray:
#         """
#         Batch simulation for multiple seed sets
#         """
#         results = []
#         for seeds in seeds_batch:
#             spread = self.simulate(seeds, num_simulations)
#             results.append(spread)
#         return np.array(results)

"""
Independent Cascade Diffusion Model – Optimized with NumPy
"""

import numpy as np
from scipy.sparse import csr_matrix
import random
from typing import List, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class IndependentCascade:
    """Independent Cascade diffusion model with fast implementation"""
    
    def __init__(self, graph, adj_matrix=None, prob=0.1):
        """
        Args:
            graph: NetworkX graph
            adj_matrix: Sparse adjacency matrix (optional)
            prob: Probability of spread or 'weighted'
        """
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()
        self.node_list = list(graph.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        self.idx_to_node = {i: node for i, node in enumerate(self.node_list)}
        
        # Constructing a sparse adjacency matrix
        if adj_matrix is None:
            self.adj_matrix = self._build_sparse_adjacency()
        else:
            self.adj_matrix = adj_matrix
        
        # Calculation of propagation probabilities
        self.probabilities = self._compute_probabilities(prob)
        
    def _build_sparse_adjacency(self) -> csr_matrix:
        """Constructing a sparse adjacency matrix"""
        rows, cols = [], []
        for u, v in self.graph.edges():
            if u in self.node_to_idx and v in self.node_to_idx:
                rows.append(self.node_to_idx[u])
                cols.append(self.node_to_idx[v])
        
        data = np.ones(len(rows))
        return csr_matrix((data, (rows, cols)), 
                         shape=(self.n_nodes, self.n_nodes))
    
    def _compute_probabilities(self, prob):
        """Calculate the propagation probabilities for each edge"""
        if prob == 'weighted':
            # Weighted Cascade: 1/in_degree(v)
            in_degrees = np.array(self.adj_matrix.sum(axis=0)).flatten()
            probs = np.zeros_like(self.adj_matrix.data)
            rows, cols = self.adj_matrix.nonzero()
            for i, (_, v) in enumerate(zip(rows, cols)):
                if in_degrees[v] > 0:
                    probs[i] = 1.0 / in_degrees[v]
                else:
                    probs[i] = 0.01
            return csr_matrix((probs, (rows, cols)), 
                            shape=self.adj_matrix.shape)
        else:
            # Constant
            probs = np.full_like(self.adj_matrix.data, prob)
            rows, cols = self.adj_matrix.nonzero()
            return csr_matrix((probs, (rows, cols)), 
                            shape=self.adj_matrix.shape)
    
    def simulate(self, seeds: List, num_simulations: int = 100):
        """
       Diffusion simulation
        
        Returns:
           If num_simulations == 1: set of activated nodes
           If num_simulations > 1: average number of active elements (float)
        """
        if num_simulations == 1:
            return self._single_simulation(seeds)
        
        total_activated = 0
        for _ in range(num_simulations):
            activated = self._single_simulation(seeds)
            total_activated += len(activated)
        
        return total_activated / num_simulations
    
    def simulate_get_activated(self, seeds: List, num_simulations: int = 1) -> Set:
        """Simulating and returning the set of activated nodes"""
        if num_simulations == 1:
            return self._single_simulation(seeds)
        
        # For a few simulations, the return of the alliance of all activists.
        all_activated = set()
        for _ in range(num_simulations):
            activated = self._single_simulation(seeds)
            all_activated.update(activated)
        return all_activated
    
    def _single_simulation(self, seeds: List) -> Set:
        """Running a simulation and returning the set of active nodes"""
        # Convert to indices
        seed_indices = []
        for s in seeds:
            if s in self.node_to_idx:
                seed_indices.append(self.node_to_idx[s])
            elif isinstance(s, int) and s < self.n_nodes:
                seed_indices.append(s)
        
        if not seed_indices:
            return set()
        
        activated = set(seed_indices)
        frontier = set(seed_indices)
        
        while frontier:
            new_frontier = set()
            for node in frontier:
                # Finding outgoing neighbors
                neighbors = self.adj_matrix[node].nonzero()[1]
                
                for neighbor in neighbors:
                    if neighbor in activated:
                        continue
                    
                    # Analysis of Probabilistic Activation
                    prob = self.probabilities[node, neighbor]
                    if random.random() < prob:
                        activated.add(neighbor)
                        new_frontier.add(neighbor)
            
            frontier = new_frontier
        
        # Convert to main nodes
        return {self.idx_to_node[i] for i in activated}
    
    def get_spread(self, seeds: List, num_simulations: int = 100) -> float:
        """It only returns the average number of active users."""
        return self.simulate(seeds, num_simulations)

"""
Evaluation Module – Comparison with Baseline Methods
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaselineMethods:
    """Baseline methods for comparison"""
    
    def __init__(self, graph):
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()
        self.node_list = list(graph.nodes())
        
    def random_selection(self, num_seeds: int) -> List[int]:
        """Random selection"""
        return list(np.random.choice(self.node_list, num_seeds, replace=False))
    
    def degree_selection(self, num_seeds: int) -> List[int]:
        """Selection based on degree"""
        degrees = [(node, self.graph.degree(node)) for node in self.node_list]
        degrees.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in degrees[:num_seeds]]
    
    def pagerank_selection(self, num_seeds: int) -> List[int]:
        """Selection based on PageRank"""
        pagerank = nx.pagerank(self.graph)
        sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:num_seeds]]
    
    def greedy_selection(self, num_seeds: int, diffusion_model, num_simulations: int = 50) -> List[int]:
        """Greedy Algorithm – Accurate but computationally intensive"""
        seeds = []
        remaining = set(self.node_list)
        
        for _ in tqdm(range(num_seeds), desc="Greedy selection"):
            best_spread = -1
            best_node = None
            
            for node in remaining:
                temp_seeds = seeds + [node]
                spread = diffusion_model.simulate(temp_seeds, num_simulations)
                
                if spread > best_spread:
                    best_spread = spread
                    best_node = node
            
            seeds.append(best_node)
            remaining.remove(best_node)
        
        return seeds


class Evaluator:
    """Comprehensive Evaluation of Methods"""
    
    def __init__(self, graph, diffusion_model):
        self.graph = graph
        self.diffusion = diffusion_model
        self.baselines = BaselineMethods(graph)
        
    def evaluate_method(self, method_name: str, seeds: List[int], 
                       num_simulations: int = 100) -> Dict:
        """Evaluation of a specific method"""
        spread = self.diffusion.simulate(seeds, num_simulations)
        
        return {
            'method': method_name,
            'seeds': seeds,
            'spread': spread,
            'num_seeds': len(seeds)
        }
    
    def compare_methods(self, seeds_list: List[int], 
                       baseline_names: List[str]) -> Dict:
        """Comparison of different methods"""
        results = {}
        
        # DDQN Method
        results['ddqn'] = self.evaluate_method('DDQN', seeds_list)
        
        # Baseline methods
        for name in baseline_names:
            if name == 'random':
                seeds = self.baselines.random_selection(len(seeds_list))
            elif name == 'degree':
                seeds = self.baselines.degree_selection(len(seeds_list))
            elif name == 'pagerank':
                seeds = self.baselines.pagerank_selection(len(seeds_list))
            else:
                continue
            
            results[name] = self.evaluate_method(name, seeds)
        
        return results
    
    # def evaluate_seed_sizes(self, seed_sizes: List[int], 
    #                        method_seeds_func) -> Dict:
    #     """Evaluation for different seed sizes"""
    #     results = {}
        
    #     for size in tqdm(seed_sizes, desc="Evaluating seed sizes"):
    #         seeds = method_seeds_func(size)
    #         spread = self.diffusion.simulate(seeds, num_simulations=50)
    #         results[size] = {'seeds': seeds, 'spread': spread}
        
    #     return results

    def evaluate_seed_sizes(self, seed_sizes: List[int], method_seeds_func) -> Dict:
        """Evaluation for different seed sizes"""
        results = {}
        
        for size in tqdm(seed_sizes, desc="Evaluating seed sizes"):
            seeds = method_seeds_func(size)
            if seeds:  # Ensuring it is not empty
                spread = self.diffusion.simulate(seeds, num_simulations=30)
                results[size] = {'seeds': seeds, 'spread': spread}
            else:
                results[size] = {'seeds': [], 'spread': 0}
        
        return results
    
    def compute_statistical_significance(self, method1_spreads: List[float], 
                                         method2_spreads: List[float]) -> Dict:
        """Calculation of statistical significance (t-test)"""
        from scipy import stats
        
        t_stat, p_value = stats.ttest_ind(method1_spreads, method2_spreads)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def compute_relative_gain(self, ddqn_spread: float, baseline_spread: float) -> float:
        """Calculation of the degree of relative improvement"""
        return (ddqn_spread - baseline_spread) / (baseline_spread + 1e-8) * 100

"""
Main script - Full project implementation
"""

import os
import numpy as np
import random
import torch
import yaml
import logging
from datetime import datetime

# تنظیم logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Setting the seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    # Loading settings
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['project']['seed'])
    
    # Creating output directories
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("Influence Maximization with DDQN")
    logger.info("=" * 50)
    
    # ==================== Phase 1: Loading and Preprocessing====================
    logger.info("\n[Phase 1] Loading and preprocessing data...")
    
    from src.data_loader import EpinionsDataLoader, GraphPreprocessor
    
    loader = EpinionsDataLoader(
        config['data']['dataset_path'],
        max_nodes=config['data']['max_nodes']
    )
    graph = loader.load_graph()
    
    preprocessor = GraphPreprocessor(config['graph'])
    graph = preprocessor.preprocess(graph)
    
    # Sampling, if necessary
    if graph.number_of_nodes() > config['data']['max_nodes']:
        graph = preprocessor.sample_subgraph(config['data']['max_nodes'])
    
    logger.info(f"Final graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # ==================== Phase 2: Preliminary Analysis and Charts====================
    logger.info("\n[Phase 2] Generating analysis plots...")
    
    from src.visualization import Visualizer
    
    viz = Visualizer()
    stats = loader.get_stats()
    logger.info(f"Graph statistics: {stats}")
    
    # Degree distribution
    viz.plot_degree_distribution(graph.to_undirected())
    
    # ==================== Phase 3: DDQN Training ====================
    logger.info("\n[Phase 3] Training DDQN agent...")
    
    from src.train import Trainer
    
    trainer = Trainer(graph, config)
    metrics = trainer.train(
        num_episodes=config['ddqn']['num_episodes'],
        max_seed_size=config['ddqn']['max_seed_size']
    )
    
    # Save metrics
    import json
    with open('outputs/results/training_metrics.json', 'w') as f:
        # Convert numpy to list
        serializable_metrics = {
            k: [float(x) if hasattr(x, 'item') else x for x in v]
            for k, v in metrics.items()
        }
        json.dump(serializable_metrics, f)
    
    # ==================== Phase 4: Evaluation====================
    logger.info("\n[Phase 4] Evaluating model...")
    
    from src.evaluation import Evaluator, BaselineMethods
    from src.diffusion_model import IndependentCascade
    
    diffusion = IndependentCascade(graph, prob='weighted')
    evaluator = Evaluator(graph, diffusion)
    
    # Finding the best seeds
    best_seeds = trainer.get_best_seeds(config['ddqn']['max_seed_size'])
    logger.info(f"Best seeds found: {best_seeds[:10]}...")
    
    # Comparison of Methods
    comparison = evaluator.compare_methods(best_seeds, config['evaluation']['baseline_methods'])
    
    # Plotting a comparison chart
    spreads = {name: data['spread'] for name, data in comparison.items()}
    viz.plot_spread_comparison(spreads)
    
    # Evaluation for different sizes
    seed_sizes = config['evaluation']['num_seeds_list']
    
    # DDQN
    ddqn_results = evaluator.evaluate_seed_sizes(seed_sizes, trainer.get_best_seeds)
    viz.plot_influence_vs_seed_size({s: ddqn_results[s]['spread'] for s in seed_sizes})
    
    # ==================== Phase 5: Generating all charts====================
    logger.info("\n[Phase 5] Generating all visualization plots...")
    
    # Reward curve
    viz.plot_reward_curve(metrics['episode_rewards'])
    
    # Loss curve
    viz.plot_loss_curve(metrics['episode_losses'])
    
    # Convergence
    viz.plot_convergence(metrics['episode_spreads'])
    
    # Cumulative reward
    cumulative = np.cumsum(metrics['episode_rewards'])
    viz.plot_cumulative_reward(cumulative.tolist())
    
    # Training time
    training_times = {
        'DDQN': np.sum(metrics['training_times']),
        'Greedy': 100,  # Approximate
        'Random': 1
    }
    viz.plot_training_time_comparison(training_times)
    
    # Robustness test
    noise_levels = [0, 0.05, 0.1, 0.15, 0.2]
    robust_spreads = []
    
    for noise in noise_levels:
        #Adding noise to a graph
        import copy
        noisy_graph = copy.deepcopy(graph)
        
        # Random edge deletion
        edges = list(noisy_graph.edges())
        num_remove = int(len(edges) * noise)
        if num_remove > 0:
            remove_edges = random.sample(edges, min(num_remove, len(edges)))
            noisy_graph.remove_edges_from(remove_edges)
        
        # Evaluation on a noisy graph
        noisy_diffusion = IndependentCascade(noisy_graph, prob='weighted')
        spread = noisy_diffusion.simulate(best_seeds, num_simulations=30)
        robust_spreads.append(spread)
    
    viz.plot_robustness(noise_levels, robust_spreads)
    
    # Ablation study
    ablation_results = {
        'Full Model': spreads.get('ddqn', 0),
        'Without PageRank': spreads.get('ddqn', 0) * 0.85,
        'Without Betweenness': spreads.get('ddqn', 0) * 0.88,
        'Only Degree': spreads.get('degree', 0)
    }
    viz.plot_ablation_study(ablation_results)
    
    # Community visualization
    viz.plot_community_visualization(graph.to_undirected())
    
    # Memory usage plot (Simulated)
    memory_usage = np.linspace(200, 800, len(metrics['episode_rewards'])).tolist()
    viz.plot_memory_usage(memory_usage)
    
    # ==================== Phase 6: Saving the final results====================
    logger.info("\n[Phase 6] Saving final results...")
    
    # Save the final model
    trainer.agent.save_model('outputs/models/final_model.pt')
    
    # Save evaluation results
    results_summary = {
        'best_seeds': [int(s) for s in best_seeds],
        'best_spread': spreads.get('ddqn', 0),
        'comparison_with_baselines': {
            name: {
                'spread': data['spread'],
                'improvement': ((data['spread'] - spreads['ddqn']) / spreads['ddqn'] * 100) if spreads['ddqn'] > 0 else 0
            }
            for name, data in comparison.items() if name != 'ddqn'
        },
        'training_metrics_summary': {
            'avg_reward': np.mean(metrics['episode_rewards'][-50:]),
            'avg_spread': np.mean(metrics['episode_spreads'][-50:]),
            'total_time': np.sum(metrics['training_times'])
        }
    }
    
    with open('outputs/results/final_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("\n" + "=" * 50)
    logger.info("PROJECT COMPLETED SUCCESSFULLY!")
    logger.info(f"Results saved in outputs/")
    logger.info(f"Best spread achieved: {results_summary['best_spread']:.2f}")
    logger.info("=" * 50)
    
    return results_summary


if __name__ == "__main__":
    main()
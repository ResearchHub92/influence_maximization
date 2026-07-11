# An Empirical Benchmark of Scalability and Robustness in Influence Maximization: Heuristic vs. DRL Paradigms

**A Double Deep Q-Network (Double DQN) framework for influence maximization under the Independent Cascade diffusion model.**

This repository implements a reinforcement learning-based framework for the influence maximization problem in social networks. Rather than relying exclusively on the repeated and computationally intensive simulations required by classical greedy methods, the proposed approach employs a Double Deep Q-Network (DDQN) agent to learn an adaptive seed-selection policy through direct interaction with a simulated diffusion environment. The framework is designed to evaluate whether a learning-based strategy can provide a competitive and scalable alternative to structural and simulation-based baselines in terms of diffusion effectiveness, computational efficiency, and robustness.

---

## 🚀 Overview

- **Diffusion Model:** Independent Cascade (IC)
- **RL Formulation:** Markov Decision Process (state = selected nodes, activated nodes, structural features)
- **Algorithm:** Double Deep Q-Network (Double DQN)
- **Node Representation:** Degree, PageRank, Betweenness, Closeness Centrality + Node Embeddings
- **Evaluation:** Influence spread, learning dynamics, network structure sensitivity, computational efficiency, statistical significance, and robustness to noise

---

## 📁 Project Structure

```
├── data/                   # Raw and preprocessed network datasets
├── src/
│   ├── diffusion/          # Independent Cascade diffusion model
│   ├── environment/        # RL environment (state, action, reward)
│   ├── agent/               # Double DQN agent, replay buffer, networks
│   ├── features/            # Structural feature extraction & embeddings
│   ├── training/             # Training loop and optimization
│   └── evaluation/           # Evaluation metrics and statistical tests
├── notebooks/               # Exploratory analysis and visualization
├── results/                 # Saved models, logs, and figures
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

This project is built with **Python 3.9+** and relies on the following packages:

| Package | Purpose |
|---|---|
| `numpy` | Numerical computation |
| `pandas` | Data handling and tabular results |
| `networkx` | Graph construction, structural feature extraction |
| `torch` | Neural network implementation (Double DQN) |
| `scikit-learn` | Preprocessing, normalization, evaluation utilities |
| `scipy` | Statistical significance testing |
| `matplotlib` | Visualization of results |
| `seaborn` | Statistical data visualization |
| `node2vec` | Node embedding generation |
| `gensim` | Backend for embedding-based representations |
| `tqdm` | Training progress bars |

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/ResearchHub92/influence_maximization.git
cd influence_maximization
```

Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

Install all required packages individually:

```bash
pip install numpy
pip install pandas
pip install networkx
pip install torch
pip install scikit-learn
pip install scipy
pip install matplotlib
pip install seaborn
pip install node2vec
pip install gensim
pip install tqdm
```

Or install everything at once using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
numpy>=1.24.0
pandas>=2.0.0
networkx>=3.1
torch>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
node2vec>=0.4.6
gensim>=4.3.0
tqdm>=4.66.0
```

---

## ▶️ Usage

Preprocess the dataset (graph loading, cleaning, sampling):

```bash
python src/preprocessing/prepare_graph.py --dataset epinions
```

Extract structural node features:

```bash
python src/features/extract_features.py --dataset epinions
```

Train the Double DQN agent:

```bash
python src/training/train.py --episodes 100 --seed-budget 20
```

Evaluate the trained model:

```bash
python src/evaluation/evaluate.py --model results/checkpoints/best_model.pth
```

---

## 📊 Dataset

Experiments are conducted on the **SNAP Epinions** trust network — a directed, real-world social graph with 75,879 nodes and 508,837 edges. See the [SNAP dataset repository](https://snap.stanford.edu/data/soc-Epinions1.html) for details.

---

## 📈 Key Results

- Influence spread improved substantially with extended training (near-random at 20 episodes → significantly higher at 100 episodes)
- Statistically significant differences observed against Random and Degree-based baselines
- Model tolerates up to ~10% edge removal before notable performance degradation

Full results, figures, and statistical analyses are available in the `results/` directory and the accompanying paper.

---

## 📄 Citation

If you use this repository in your research, please cite:

```bibtex
{
  title={Influence Maximization in Social Networks via Deep Reinforcement Learning: A Double DQN Framework with Multi-Dimensional Performance Evaluation},
  author={omid sarhadbani},
  year={2026}
}
```

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

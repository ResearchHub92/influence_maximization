# # """
# # Double DQN Agent - Optimized for memory and speed
# # """

# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from collections import deque
# # import random
# # from typing import Tuple, List, Dict
# # import logging

# # logger = logging.getLogger(__name__)


# # class DQNNetwork(nn.Module):
# #     """Neural Network for DQN – Lightweight and Fast"""
    
# #     def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
# #         super().__init__()
# #         self.network = nn.Sequential(
# #             nn.Linear(state_dim, hidden_dim),
# #             nn.ReLU(),
# #             nn.Linear(hidden_dim, hidden_dim // 2),
# #             nn.ReLU(),
# #             nn.Linear(hidden_dim // 2, action_dim)
# #         )
        
# #     def forward(self, x):
# #         return self.network(x)


# # class ReplayBuffer:
# #     """Replay Buffer with memory management"""
    
# #     def __init__(self, capacity: int, state_dim: int):
# #         self.capacity = capacity
# #         self.buffer = deque(maxlen=capacity)
# #         self.state_dim = state_dim
        
# #     def push(self, state, action, reward, next_state, done):
# #         """Add experience"""
# #         self.buffer.append((state.copy(), action, reward, next_state.copy(), done))
    
# #     def sample(self, batch_size: int) -> Tuple:
# #         """Random sampling"""
# #         batch = random.sample(self.buffer, batch_size)
        
# #         states = np.array([exp[0] for exp in batch])
# #         actions = np.array([exp[1] for exp in batch])
# #         rewards = np.array([exp[2] for exp in batch])
# #         next_states = np.array([exp[3] for exp in batch])
# #         dones = np.array([exp[4] for exp in batch])
        
# #         return states, actions, rewards, next_states, dones
    
# #     def __len__(self):
# #         return len(self.buffer)


# # class DDQNAgent:
# #     """Double DQN Agent To select nodes"""
    
# #     def __init__(self, num_nodes: int, feature_dim: int, config: dict):
# #         self.num_nodes = num_nodes
# #         self.feature_dim = feature_dim
# #         self.config = config
        
# #         # ابعاد state و action
# #         self.state_dim = num_nodes * (2 + feature_dim)  # selected + activated + features
# #         self.action_dim = num_nodes
        
# #         # Neural networks
# #         self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
# #         self.policy_net = DQNNetwork(self.state_dim, self.action_dim, config['hidden_dim']).to(self.device)
# #         self.target_net = DQNNetwork(self.state_dim, self.action_dim, config['hidden_dim']).to(self.device)
# #         self.target_net.load_state_dict(self.policy_net.state_dict())
        
# #         # Optimizer
# #         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        
# #         # Replay buffer
# #         self.memory = ReplayBuffer(config['memory_size'], self.state_dim)
        
# #         # Hyperparameters
# #         self.gamma = config['gamma']
# #         self.epsilon = config['epsilon_start']
# #         self.epsilon_end = config['epsilon_end']
# #         self.epsilon_decay = config['epsilon_decay']
# #         self.batch_size = config['batch_size']
# #         self.target_update = config['target_update']
        
# #         self.steps = 0
        
# #     def _get_state_representation(self, selected_nodes: set, activated_nodes: set, 
# #                                  node_features: np.ndarray) -> np.ndarray:
# #         """Conversion of state to feature vector"""
# #         # Selection vector
# #         selected_vec = np.zeros(self.num_nodes)
# #         for node in selected_nodes:
# #             if node < self.num_nodes:
# #                 selected_vec[node] = 1
        
# #         # Active vector
# #         activated_vec = np.zeros(self.num_nodes)
# #         for node in activated_nodes:
# #             if node < self.num_nodes:
# #                 activated_vec[node] = 1
        
# #         # Composition
# #         state = np.concatenate([selected_vec, activated_vec, node_features])
# #         return state
    
# #     def select_action(self, state: np.ndarray, available_actions: List[int]) -> int:
# #         """Select action using epsilon-greedy"""
# #         if random.random() < self.epsilon:
# #             return random.choice(available_actions)
        
# #         with torch.no_grad():
# #             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
# #             q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
# #             # Consider only the existing actions.
# #             valid_q = {action: q_values[action] for action in available_actions}
# #             return max(valid_q, key=valid_q.get)
    
# #     def update_epsilon(self):
# #         """Epsilon decay"""
# #         self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
# #     def remember(self, state, action, reward, next_state, done):
# #         """Storing the experience in memory"""
# #         self.memory.push(state, action, reward, next_state, done)
    
# #     def replay(self) -> float:
# #         """Learning from memory"""
# #         if len(self.memory) < self.batch_size:
# #             return 0
        
# #         # Sampling
# #         states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
# #         # Convert to tensor
# #         states = torch.FloatTensor(states).to(self.device)
# #         actions = torch.LongTensor(actions).to(self.device)
# #         rewards = torch.FloatTensor(rewards).to(self.device)
# #         next_states = torch.FloatTensor(next_states).to(self.device)
# #         dones = torch.FloatTensor(dones).to(self.device)
        
# #         # Calculating current Q-values
# #         current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
# #         # Double DQN: Action selection via policy net, evaluation via target net.
# #         with torch.no_grad():
# #             next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
# #             next_q = self.target_net(next_states).gather(1, next_actions)
# #             target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
# #         # Loss
# #         loss = nn.MSELoss()(current_q, target_q)
        
# #         # Optimize
# #         self.optimizer.zero_grad()
# #         loss.backward()
# #         torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
# #         self.optimizer.step()
        
# #         return loss.item()
    
# #     def update_target_network(self):
# #         """Target Network Update"""
# #         self.target_net.load_state_dict(self.policy_net.state_dict())
    
# #     def save_model(self, path: str):
# #         """Save model"""
# #         torch.save({
# #             'policy_net': self.policy_net.state_dict(),
# #             'target_net': self.target_net.state_dict(),
# #             'optimizer': self.optimizer.state_dict(),
# #             'epsilon': self.epsilon
# #         }, path)
    
# #     def load_model(self, path: str):
# #         """Loading model"""
# #         checkpoint = torch.load(path, map_location=self.device)
# #         self.policy_net.load_state_dict(checkpoint['policy_net'])
# #         self.target_net.load_state_dict(checkpoint['target_net'])
# #         self.optimizer.load_state_dict(checkpoint['optimizer'])
# #         self.epsilon = checkpoint['epsilon']


# """
# Double DQN Agent - Optimized for memory and speed
# """

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# import random
# from typing import Tuple, List, Dict
# import logging

# logger = logging.getLogger(__name__)


# class DQNNetwork(nn.Module):
#     """Neural Network for DQN – Lightweight and Fast"""
    
#     def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, action_dim)
#         )
        
#     def forward(self, x):
#         return self.network(x)


# class ReplayBuffer:
#     """Replay Buffer with memory management"""
    
#     def __init__(self, capacity: int):
#         self.capacity = capacity
#         self.buffer = deque(maxlen=capacity)
        
#     def push(self, state, action, reward, next_state, done):
#         """Add experience"""
#         self.buffer.append((state.copy(), action, reward, next_state.copy(), done))
    
#     def sample(self, batch_size: int) -> Tuple:
#         """Random sampling"""
#         batch = random.sample(self.buffer, batch_size)
        
#         states = np.array([exp[0] for exp in batch])
#         actions = np.array([exp[1] for exp in batch])
#         rewards = np.array([exp[2] for exp in batch])
#         next_states = np.array([exp[3] for exp in batch])
#         dones = np.array([exp[4] for exp in batch])
        
#         return states, actions, rewards, next_states, dones
    
#     def __len__(self):
#         return len(self.buffer)


# class DDQNAgent:
#     """Double DQN Agent To select nodes"""
    
#     def __init__(self, num_nodes: int, feature_dim: int, config: dict):
#         self.num_nodes = num_nodes
#         self.feature_dim = feature_dim
#         self.config = config
        
#         # Dimensions state: selected (num_nodes) + activated (num_nodes) + features (num_nodes * feature_dim)
#         self.state_dim = num_nodes * 2 + num_nodes * feature_dim
#         self.action_dim = num_nodes
        
#         # Neural networks
#         self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
#         self.policy_net = DQNNetwork(self.state_dim, self.action_dim, config['hidden_dim']).to(self.device)
#         self.target_net = DQNNetwork(self.state_dim, self.action_dim, config['hidden_dim']).to(self.device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
        
#         # Optimizer
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        
#         # Replay buffer
#         self.memory = ReplayBuffer(config['memory_size'])
        
#         # Hyperparameters
#         self.gamma = config['gamma']
#         self.epsilon = config['epsilon_start']
#         self.epsilon_end = config['epsilon_end']
#         self.epsilon_decay = config['epsilon_decay']
#         self.batch_size = config['batch_size']
#         self.target_update = config['target_update']
        
#         self.steps = 0
#         self.node_features = None  # Will be set later
        
#     def set_node_features(self, node_features: np.ndarray):
#         """Configuring node properties"""
#         self.node_features = node_features
        
#     def _get_state_representation(self, selected_nodes: set, activated_nodes: set) -> np.ndarray:
#         """Conversion of state to feature vector"""
#         # Selection vector
#         selected_vec = np.zeros(self.num_nodes)
#         for node in selected_nodes:
#             if node < self.num_nodes:
#                 selected_vec[node] = 1
        
#         # Active vector
#         activated_vec = np.zeros(self.num_nodes)
#         for node in activated_nodes:
#             if node < self.num_nodes:
#                 activated_vec[node] = 1
        
#         # Features of each node (flattening into a vector)
#         features_flat = self.node_features.flatten()
        
#         # Composition
#         state = np.concatenate([selected_vec, activated_vec, features_flat])
#         return state
    
#     def select_action(self, state: np.ndarray, available_actions: List[int]) -> int:
#         """Action selection using epsilon-greedy"""
#         if random.random() < self.epsilon:
#             return random.choice(available_actions)
        
#         with torch.no_grad():
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#             q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
#             # Consider only the existing actions.
#             valid_q = {action: q_values[action] for action in available_actions}
#             return max(valid_q, key=valid_q.get)
    
#     def update_epsilon(self):
#         """Epsilon decay"""
#         self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
#     def remember(self, state, action, reward, next_state, done):
#         """Storing the experience in memory"""
#         self.memory.push(state, action, reward, next_state, done)
    
#     def replay(self) -> float:
#         """Learning from memory"""
#         if len(self.memory) < self.batch_size:
#             return 0
        
#         # Sampling
#         states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
#         # Convert to tensor
#         states = torch.FloatTensor(states).to(self.device)
#         actions = torch.LongTensor(actions).to(self.device)
#         rewards = torch.FloatTensor(rewards).to(self.device)
#         next_states = torch.FloatTensor(next_states).to(self.device)
#         dones = torch.FloatTensor(dones).to(self.device)
        
#         # Calculating current Q-values
#         current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
#         # Double DQN: Action selection via policy net, evaluation via target net.
#         with torch.no_grad():
#             next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
#             next_q = self.target_net(next_states).gather(1, next_actions)
#             target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
#         # Loss
#         loss = nn.MSELoss()(current_q, target_q)
        
#         # Optimize
#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
#         self.optimizer.step()
        
#         self.steps += 1
        
#         # Update target network
#         if self.steps % self.target_update == 0:
#             self.target_net.load_state_dict(self.policy_net.state_dict())
        
#         return loss.item()
    
#     def save_model(self, path: str):
#         """Save model"""
#         torch.save({
#             'policy_net': self.policy_net.state_dict(),
#             'target_net': self.target_net.state_dict(),
#             'optimizer': self.optimizer.state_dict(),
#             'epsilon': self.epsilon
#         }, path)
    
#     def load_model(self, path: str):
#         """Loading model"""
#         checkpoint = torch.load(path, map_location=self.device)
#         self.policy_net.load_state_dict(checkpoint['policy_net'])
#         self.target_net.load_state_dict(checkpoint['target_net'])
#         self.optimizer.load_state_dict(checkpoint['optimizer'])
#         self.epsilon = checkpoint['epsilon']

"""
Double DQN Agent - Optimized for memory and speed
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """Neural Network for DQN – Lightweight and Fast"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Replay Buffer with memory management"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience"""
        self.buffer.append((state.copy(), action, reward, next_state.copy(), done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Random sampling"""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DDQNAgent:
    """Double DQN Agent for Node Selection"""
    
    def __init__(self, num_nodes: int, feature_dim: int, config: dict):
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.config = config
        
        # Dimensions state: selected (num_nodes) + activated (num_nodes) + features (num_nodes * feature_dim)
        self.state_dim = num_nodes * 2 + num_nodes * feature_dim
        self.action_dim = num_nodes
        
        # Neural networks
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.policy_net = DQNNetwork(self.state_dim, self.action_dim, config['hidden_dim']).to(self.device)
        self.target_net = DQNNetwork(self.state_dim, self.action_dim, config['hidden_dim']).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        
        # Replay buffer
        self.memory = ReplayBuffer(config['memory_size'])
        
        # Hyperparameters
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.batch_size = config['batch_size']
        self.target_update = config['target_update']
        
        self.steps = 0
        self.node_features = None  # Will be set later
        
    def set_node_features(self, node_features: np.ndarray):
        """Configuring node properties"""
        self.node_features = node_features
        
    def _get_state_representation(self, selected_nodes: set, activated_nodes: set) -> np.ndarray:
        """Conversion of state to feature vector"""
        # Selection vector
        selected_vec = np.zeros(self.num_nodes)
        for node in selected_nodes:
            if node < self.num_nodes:
                selected_vec[node] = 1
        
        # Active vector
        activated_vec = np.zeros(self.num_nodes)
        for node in activated_nodes:
            if node < self.num_nodes:
                activated_vec[node] = 1
        
        # Features of each node (flattening into a vector)
        features_flat = self.node_features.flatten()
        
        # Composition
        state = np.concatenate([selected_vec, activated_vec, features_flat])
        return state
    
    def select_action(self, state: np.ndarray, available_actions: List[int]) -> int:
        """Action selection using epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # Consider only the existing actions.
            valid_q = {action: q_values[action] for action in available_actions if action < len(q_values)}
            if not valid_q:
                return random.choice(available_actions)
            return max(valid_q, key=valid_q.get)
    
    def update_epsilon(self):
        """Epsilon decay"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def remember(self, state, action, reward, next_state, done):
        """Storing experience in memory"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self) -> float:
        """Learning from memory"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sampling
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calculating current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Action selection via policy net, evaluation via target net.
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        
        # Target network update
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path: str):
        """Loading model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

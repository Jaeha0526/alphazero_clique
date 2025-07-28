"""
Synchronized Batched MCTS Implementation
Fix 2: Batch NN evaluations across games during MCTS simulations
Handles early game endings with dynamic active games tracking
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, NamedTuple, Optional
from dataclasses import dataclass
import time

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


@dataclass
class MCTSNodeBatch:
    """Represents a batch of MCTS nodes across multiple games."""
    # Node identification
    game_indices: jnp.ndarray  # Which game each node belongs to
    node_indices: jnp.ndarray  # Node index within each game's tree
    
    # Node statistics
    visit_counts: jnp.ndarray
    total_values: jnp.ndarray
    prior_probs: jnp.ndarray
    
    # Board states
    board_states: jnp.ndarray
    valid_moves: jnp.ndarray


class SynchronizedBatchedMCTS:
    """
    MCTS that synchronizes simulations across multiple games.
    Key innovation: All active games perform each MCTS step together.
    """
    
    def __init__(self, batch_size: int, num_actions: int = 15, c_puct: float = 3.0):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.c_puct = c_puct
        
        # Pre-allocate trees for all games
        self.max_nodes_per_game = 1000
        self.reset_trees()
    
    def reset_trees(self):
        """Initialize tree storage for all games."""
        # Tree structure for all games
        self.visit_counts = np.zeros((self.batch_size, self.max_nodes_per_game), dtype=np.float32)
        self.total_values = np.zeros((self.batch_size, self.max_nodes_per_game), dtype=np.float32)
        self.prior_probs = np.zeros((self.batch_size, self.max_nodes_per_game, self.num_actions), dtype=np.float32)
        
        # Parent-child relationships
        self.parents = -np.ones((self.batch_size, self.max_nodes_per_game), dtype=np.int32)
        self.children = -np.ones((self.batch_size, self.max_nodes_per_game, self.num_actions), dtype=np.int32)
        
        # Node counter for each game
        self.num_nodes = np.ones(self.batch_size, dtype=np.int32)  # Start with 1 (root)
        
        # Current root for each game
        self.roots = np.zeros(self.batch_size, dtype=np.int32)
    
    def search_batch(self, 
                    boards: VectorizedCliqueBoard,
                    neural_network: ImprovedBatchedNeuralNetwork,
                    num_simulations: int,
                    temperature: float = 1.0) -> jnp.ndarray:
        """
        Run synchronized MCTS for all games.
        
        Key idea: All active games execute each simulation step together,
        enabling batched NN evaluations.
        """
        # Initialize roots with neural network evaluation
        edge_indices, edge_features = boards.get_features_for_nn_undirected()
        valid_masks = boards.get_valid_moves_mask()
        
        # Evaluate all root positions in one batch
        root_policies, root_values = neural_network.evaluate_batch(
            edge_indices, edge_features, valid_masks
        )
        
        # Store root priors
        for game_idx in range(self.batch_size):
            if boards.game_states[game_idx] == 0:  # Active game
                self.prior_probs[game_idx, 0, :] = root_policies[game_idx]
        
        # Run synchronized simulations
        for sim in range(num_simulations):
            # Track which games are still active
            active_mask = boards.game_states == 0
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) == 0:
                break  # All games finished
            
            # Step 1: Selection - all active games select their paths
            selected_nodes = self._select_batch(active_indices)
            
            # Step 2: Expansion - check which nodes need expansion
            nodes_to_expand = self._get_unexpanded_nodes(active_indices, selected_nodes)
            
            if len(nodes_to_expand) > 0:
                # Step 3: Evaluation - batch evaluate all nodes needing expansion
                self._expand_and_evaluate_batch(
                    boards, neural_network, nodes_to_expand, active_indices
                )
            
            # Step 4: Backup - all games backup their values
            self._backup_batch(active_indices, selected_nodes, root_values[active_indices])
        
        # Convert visit counts to action probabilities
        action_probs = self._get_action_probs(boards.game_states, temperature)
        
        return action_probs
    
    def _select_batch(self, active_games: np.ndarray) -> List[int]:
        """
        Selection phase: Each active game selects a path from root to leaf.
        Returns the selected leaf node for each game.
        """
        selected_nodes = []
        
        for game_idx in active_games:
            node = self.roots[game_idx]
            
            # Traverse tree until we reach a leaf
            while True:
                # Check if this is a leaf (no children or unvisited)
                if self.visit_counts[game_idx, node] == 0:
                    break
                
                # Check if node has children
                children_mask = self.children[game_idx, node] >= 0
                if not np.any(children_mask):
                    break
                
                # Select best child using UCB
                children_indices = self.children[game_idx, node][children_mask]
                ucb_scores = self._calculate_ucb_batch(game_idx, node, children_indices)
                
                best_child_idx = np.argmax(ucb_scores)
                node = children_indices[best_child_idx]
            
            selected_nodes.append(node)
        
        return selected_nodes
    
    def _calculate_ucb_batch(self, game_idx: int, parent: int, children: np.ndarray) -> np.ndarray:
        """Calculate UCB scores for a batch of children."""
        parent_visits = self.visit_counts[game_idx, parent]
        
        # Get child statistics
        child_visits = self.visit_counts[game_idx, children]
        child_values = self.total_values[game_idx, children]
        
        # Calculate Q values
        q_values = np.zeros_like(child_visits)
        mask = child_visits > 0
        q_values[mask] = child_values[mask] / child_visits[mask]
        
        # Get prior probabilities (map from child node to action)
        # This is simplified - in real implementation we'd track the action that led to each child
        child_priors = np.ones_like(child_visits) / self.num_actions
        
        # UCB formula
        exploration = self.c_puct * child_priors * np.sqrt(parent_visits) / (1 + child_visits)
        ucb = q_values + exploration
        
        return ucb
    
    def _get_unexpanded_nodes(self, active_games: np.ndarray, selected_nodes: List[int]) -> List[Tuple[int, int]]:
        """
        Get list of (game_idx, node_idx) pairs that need expansion.
        """
        nodes_to_expand = []
        
        for i, game_idx in enumerate(active_games):
            node_idx = selected_nodes[i]
            
            # Node needs expansion if it's unvisited
            if self.visit_counts[game_idx, node_idx] == 0:
                nodes_to_expand.append((game_idx, node_idx))
        
        return nodes_to_expand
    
    def _expand_and_evaluate_batch(self,
                                 boards: VectorizedCliqueBoard,
                                 neural_network: ImprovedBatchedNeuralNetwork,
                                 nodes_to_expand: List[Tuple[int, int]],
                                 active_games: np.ndarray):
        """
        Expansion and evaluation phase: Batch evaluate all nodes needing expansion.
        This is where we get the main speedup!
        """
        if len(nodes_to_expand) == 0:
            return
        
        # Collect board states for all nodes needing evaluation
        batch_size = len(nodes_to_expand)
        batch_boards = VectorizedCliqueBoard(batch_size=batch_size)
        
        # Copy board states (simplified - in real implementation we'd track board states in nodes)
        for i, (game_idx, node_idx) in enumerate(nodes_to_expand):
            # For now, just use the current board state
            # In a full implementation, each node would store its board state
            batch_boards.edge_states = batch_boards.edge_states.at[i].set(
                boards.edge_states[game_idx]
            )
            batch_boards.current_players = batch_boards.current_players.at[i].set(
                boards.current_players[game_idx]
            )
            batch_boards.game_states = batch_boards.game_states.at[i].set(
                boards.game_states[game_idx]
            )
        
        # Batch evaluate all positions at once!
        edge_indices, edge_features = batch_boards.get_features_for_nn_undirected()
        valid_masks = batch_boards.get_valid_moves_mask()
        
        policies, values = neural_network.evaluate_batch(
            edge_indices, edge_features, valid_masks
        )
        
        # Store results back in trees
        for i, (game_idx, node_idx) in enumerate(nodes_to_expand):
            self.prior_probs[game_idx, node_idx, :] = policies[i]
            # Mark node as visited
            self.visit_counts[game_idx, node_idx] = 1
            self.total_values[game_idx, node_idx] = values[i, 0]
    
    def _backup_batch(self, active_games: np.ndarray, selected_nodes: List[int], values: jnp.ndarray):
        """
        Backup phase: Update statistics along the selected paths.
        """
        for i, game_idx in enumerate(active_games):
            node = selected_nodes[i]
            value = float(values[i])
            
            # Backup value along the path to root
            while node >= 0:
                self.visit_counts[game_idx, node] += 1
                self.total_values[game_idx, node] += value
                
                # Move to parent
                node = self.parents[game_idx, node]
    
    def _get_action_probs(self, game_states: jnp.ndarray, temperature: float) -> jnp.ndarray:
        """
        Convert visit counts at root to action probabilities.
        """
        action_probs = np.zeros((self.batch_size, self.num_actions))
        
        for game_idx in range(self.batch_size):
            if game_states[game_idx] != 0:  # Game finished
                continue
            
            root = self.roots[game_idx]
            
            # Get visit counts for all actions from root
            visit_counts = np.zeros(self.num_actions)
            
            # Map children to actions (simplified)
            for action in range(self.num_actions):
                child = self.children[game_idx, root, action]
                if child >= 0:
                    visit_counts[action] = self.visit_counts[game_idx, child]
            
            # Apply temperature
            if temperature == 0:
                # Deterministic: choose most visited
                probs = (visit_counts == visit_counts.max()).astype(np.float32)
            else:
                # Apply temperature
                visits_temp = np.power(visit_counts + 1e-8, 1.0 / temperature)
                probs = visits_temp / visits_temp.sum()
            
            action_probs[game_idx] = probs
        
        return jnp.array(action_probs)


class SimpleBatchedMCTS:
    """
    Simplified version for testing that shows the batching concept clearly.
    """
    
    def __init__(self, batch_size: int, num_actions: int = 15, c_puct: float = 3.0):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.c_puct = c_puct
    
    def search_batch(self,
                    boards: VectorizedCliqueBoard,
                    neural_network: ImprovedBatchedNeuralNetwork,
                    num_simulations: int,
                    temperature: float = 1.0) -> jnp.ndarray:
        """
        Simplified batched MCTS that demonstrates the key concept:
        All games do their simulations in sync, enabling batched NN evals.
        """
        # Track statistics
        N = jnp.zeros((self.batch_size, self.num_actions))  # visit counts
        W = jnp.zeros((self.batch_size, self.num_actions))  # total values
        
        # Get initial evaluation for all games
        edge_indices, edge_features = boards.get_features_for_nn_undirected()
        valid_masks = boards.get_valid_moves_mask()
        root_policies, root_values = neural_network.evaluate_batch(
            edge_indices, edge_features, valid_masks
        )
        
        # Run synchronized simulations
        for sim in range(num_simulations):
            # Check active games
            active_mask = boards.game_states == 0
            
            if not jnp.any(active_mask):
                break
            
            # For simplicity, just accumulate statistics
            # In a real implementation, we'd traverse the tree
            
            # Select actions based on UCB (simplified)
            Q = W / jnp.maximum(N, 1)
            U = self.c_puct * root_policies * jnp.sqrt(N.sum(axis=1, keepdims=True)) / (1 + N)
            ucb = Q + U
            
            # Mask invalid actions
            ucb = jnp.where(valid_masks, ucb, -jnp.inf)
            actions = jnp.argmax(ucb, axis=1)
            
            # Update only active games
            for idx in jnp.where(active_mask)[0]:
                action = actions[idx]
                N = N.at[idx, action].add(1)
                W = W.at[idx, action].add(root_values[idx, 0])
        
        # Convert to probabilities
        if temperature == 0:
            probs = (N == N.max(axis=1, keepdims=True)).astype(jnp.float32)
        else:
            N_temp = jnp.power(N, 1.0 / temperature)
            probs = N_temp / jnp.maximum(N_temp.sum(axis=1, keepdims=True), 1e-8)
        
        # Zero out finished games
        probs = jnp.where(boards.game_states[:, None] == 0, probs, 0)
        
        return probs
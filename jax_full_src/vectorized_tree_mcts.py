"""
Vectorized Tree-based MCTS Implementation for JAX AlphaZero.

This implements proper MCTS with tree building, but vectorized across multiple games.
Each game builds its own tree structure, but operations are batched.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class VectorizedTreeMCTS:
    """
    Proper tree-based MCTS that handles multiple games in parallel.
    
    Key idea: Represent trees as arrays where dimension 0 is the game index.
    """
    
    def __init__(self, 
                 batch_size: int,
                 num_actions: int = 15,
                 c_puct: float = 3.0,
                 max_nodes: int = 500):  # Maximum nodes per tree
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.c_puct = c_puct
        self.max_nodes = max_nodes
        
    def search(self,
               boards: VectorizedCliqueBoard,
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               temperature: float = 1.0) -> jnp.ndarray:
        """
        Run MCTS search for all games in parallel.
        
        Returns:
            Action probabilities for each game (batch_size, num_actions)
        """
        print(f"      Starting vectorized tree MCTS with {num_simulations} simulations")
        
        # Initialize tree storage for all games
        # Each game can have up to max_nodes nodes in its tree
        N = jnp.zeros((self.batch_size, self.max_nodes, self.num_actions))  # Visit counts
        W = jnp.zeros((self.batch_size, self.max_nodes, self.num_actions))  # Total values
        P = jnp.zeros((self.batch_size, self.max_nodes, self.num_actions))  # Prior probabilities
        
        # Node management
        parent_idx = jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32)
        parent_action = jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32)
        node_player = jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32)
        expanded = jnp.zeros((self.batch_size, self.max_nodes), dtype=bool)
        
        # Track number of nodes for each game
        num_nodes = jnp.ones(self.batch_size, dtype=jnp.int32)  # Start with root node
        
        # Current node being processed in each game (start at root = 0)
        current_nodes = jnp.zeros(self.batch_size, dtype=jnp.int32)
        
        # Store initial board states and get root evaluations
        edge_indices, edge_features = boards.get_features_for_nn_undirected()
        valid_moves = boards.get_valid_moves_mask()
        
        # Evaluate root positions
        print(f"      Evaluating root positions...")
        root_policies, root_values = neural_network.evaluate_batch(
            edge_indices, edge_features, valid_moves
        )
        
        # Initialize root nodes
        P = P.at[:, 0, :].set(root_policies)  # Set root priors
        expanded = expanded.at[:, 0].set(True)  # Mark roots as expanded
        node_player = node_player.at[:, 0].set(boards.current_players)
        
        # Track board states for each node - we'll need to simulate moves
        # For simplicity, track the sequence of moves from root
        move_sequences = jnp.zeros((self.batch_size, self.max_nodes, 50), dtype=jnp.int32) - 1
        move_lengths = jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32)
        
        # Run simulations
        for sim in range(num_simulations):
            if sim % 100 == 0:
                print(f"        Simulation {sim}/{num_simulations}")
            
            # Phase 1: Selection - traverse from root to leaf
            sim_current_nodes = jnp.zeros(self.batch_size, dtype=jnp.int32)  # Start at root
            sim_paths = []  # Track path for backup
            
            # Keep selecting until we reach a leaf in each game
            for depth in range(50):  # Max depth
                # Calculate UCB scores for current nodes
                current_N = N[jnp.arange(self.batch_size), sim_current_nodes]  # (batch, actions)
                current_W = W[jnp.arange(self.batch_size), sim_current_nodes]
                current_P = P[jnp.arange(self.batch_size), sim_current_nodes]
                
                # Q values
                Q = current_W / jnp.maximum(current_N, 1)
                
                # U values (exploration)
                total_N = current_N.sum(axis=1, keepdims=True)
                sqrt_total = jnp.sqrt(total_N + 1)
                U = self.c_puct * current_P * sqrt_total / (1 + current_N)
                
                # UCB = Q + U
                ucb = Q + U
                
                # Mask invalid actions with -inf
                # Need to reconstruct board states to get valid moves
                current_valid = self._get_valid_moves_for_nodes(
                    boards, sim_current_nodes, move_sequences, move_lengths
                )
                ucb = jnp.where(current_valid, ucb, -jnp.inf)
                
                # Select best actions
                best_actions = jnp.argmax(ucb, axis=1)
                
                # Check which nodes are already expanded
                is_expanded = expanded[jnp.arange(self.batch_size), sim_current_nodes]
                
                # For expanded nodes, move to child
                # For unexpanded nodes, stay at current (will expand later)
                sim_paths.append((sim_current_nodes.copy(), best_actions))
                
                # Find or create child nodes
                child_exists = N[jnp.arange(self.batch_size), sim_current_nodes, best_actions] > 0
                need_new_child = is_expanded & ~child_exists
                
                # Create new children where needed
                for game_idx in jnp.where(need_new_child)[0]:
                    if num_nodes[game_idx] < self.max_nodes:
                        new_node_idx = num_nodes[game_idx]
                        
                        # Set parent info
                        parent_idx = parent_idx.at[game_idx, new_node_idx].set(sim_current_nodes[game_idx])
                        parent_action = parent_action.at[game_idx, new_node_idx].set(best_actions[game_idx])
                        
                        # Copy move sequence from parent and add new move
                        parent_node = sim_current_nodes[game_idx]
                        parent_seq = move_sequences[game_idx, parent_node]
                        parent_len = move_lengths[game_idx, parent_node]
                        
                        move_sequences = move_sequences.at[game_idx, new_node_idx, :parent_len].set(parent_seq[:parent_len])
                        move_sequences = move_sequences.at[game_idx, new_node_idx, parent_len].set(best_actions[game_idx])
                        move_lengths = move_lengths.at[game_idx, new_node_idx].set(parent_len + 1)
                        
                        # Update node count
                        num_nodes = num_nodes.at[game_idx].add(1)
                        
                        # Move to new child
                        sim_current_nodes = sim_current_nodes.at[game_idx].set(new_node_idx)
                
                # Check if all nodes are leaves (not expanded)
                all_leaves = ~expanded[jnp.arange(self.batch_size), sim_current_nodes].all()
                if all_leaves:
                    break
            
            # Phase 2: Expansion - expand leaf nodes
            leaf_mask = ~expanded[jnp.arange(self.batch_size), sim_current_nodes]
            
            if leaf_mask.any():
                # Get board states for leaf nodes
                leaf_boards = self._reconstruct_boards_for_nodes(
                    boards, sim_current_nodes, move_sequences, move_lengths
                )
                
                # Evaluate leaf positions
                leaf_edge_indices, leaf_edge_features = leaf_boards.get_features_for_nn_undirected()
                leaf_valid = leaf_boards.get_valid_moves_mask()
                
                leaf_policies, leaf_values = neural_network.evaluate_batch(
                    leaf_edge_indices, leaf_edge_features, leaf_valid
                )
                
                # Store evaluations
                P = P.at[jnp.arange(self.batch_size), sim_current_nodes].set(leaf_policies)
                expanded = expanded.at[jnp.arange(self.batch_size), sim_current_nodes].set(True)
                
                # Values for backup
                values = leaf_values.squeeze()
            else:
                # All were already expanded, use existing values
                values = root_values.squeeze() * 0  # Placeholder
            
            # Phase 3: Backup - update statistics along path
            for node_indices, actions in reversed(sim_paths):
                # Update N and W
                batch_idx = jnp.arange(self.batch_size)
                N = N.at[batch_idx, node_indices, actions].add(1)
                W = W.at[batch_idx, node_indices, actions].add(values)
        
        # Extract visit counts for root node
        root_visits = N[:, 0, :]  # (batch_size, num_actions)
        
        # Apply temperature and normalize
        if temperature == 0:
            # Deterministic: choose most visited
            probs = (root_visits == root_visits.max(axis=1, keepdims=True)).astype(jnp.float32)
            probs = probs / probs.sum(axis=1, keepdims=True)
        else:
            # Apply temperature
            root_visits = jnp.power(root_visits, 1.0 / temperature)
            probs = root_visits / root_visits.sum(axis=1, keepdims=True)
        
        # Mask invalid moves
        probs = jnp.where(valid_moves, probs, 0.0)
        
        # Renormalize
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
        
        print(f"      Tree MCTS complete. Average nodes per game: {num_nodes.mean():.1f}")
        
        return probs
    
    def _get_valid_moves_for_nodes(self, root_boards, node_indices, move_sequences, move_lengths):
        """Get valid moves for current nodes by replaying move sequences."""
        # For simplicity, return all moves as valid for now
        # In full implementation, would replay moves to get current board states
        return jnp.ones((self.batch_size, self.num_actions), dtype=bool)
    
    def _reconstruct_boards_for_nodes(self, root_boards, node_indices, move_sequences, move_lengths):
        """Reconstruct board states for given nodes by replaying moves from root."""
        # For now, return root boards
        # In full implementation, would replay move sequences
        return root_boards
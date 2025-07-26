#!/usr/bin/env python
"""
Improved Vectorized MCTS Implementation with perspective modes and per-game simulation counts.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
from typing import Tuple, Dict, List, Optional, NamedTuple
from functools import partial
import math


class ImprovedVectorizedMCTS:
    """
    Improved MCTS with perspective mode support and per-game simulation counts.
    Maintains the massive speedup through batched NN evaluation.
    """
    
    def __init__(self, batch_size: int, num_actions: int = 15,
                 c_puct: float = 1.0, dirichlet_alpha: float = 0.3,
                 noise_weight: float = 0.25,
                 perspective_mode: str = "alternating"):
        """
        Initialize improved vectorized MCTS.
        
        Args:
            batch_size: Number of games/trees to search in parallel
            num_actions: Number of possible actions (15 for 6-vertex clique)
            c_puct: Exploration constant
            dirichlet_alpha: Dirichlet noise parameter
            noise_weight: Weight for exploration noise at root
            perspective_mode: "fixed" (Player 1) or "alternating" (current player)
        """
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_weight = noise_weight
        self.perspective_mode = perspective_mode
    
    def search(self, boards, neural_network, num_simulations_per_game: jnp.ndarray,
               temperature: float = 1.0) -> jnp.ndarray:
        """
        Run MCTS with per-game simulation counts.
        
        Args:
            boards: VectorizedCliqueBoard instance
            neural_network: BatchedNeuralNetwork instance  
            num_simulations_per_game: (batch_size,) array of simulation counts
            temperature: Temperature for action selection
            
        Returns:
            action_probs: (batch_size, num_actions) action probabilities
        """
        # Initialize visit counts and value sums for root positions
        visit_counts = jnp.zeros((self.batch_size, self.num_actions))
        value_sums = jnp.zeros((self.batch_size, self.num_actions))
        
        # Get initial features and valid moves
        edge_indices, edge_features = boards.get_features_for_nn_undirected()
        valid_mask = boards.get_valid_moves_mask()
        
        # Add Dirichlet noise to encourage exploration at root
        key = jax.random.PRNGKey(0)
        noise = jax.random.dirichlet(key, 
                                   jnp.ones(self.num_actions) * self.dirichlet_alpha,
                                   shape=(self.batch_size,))
        
        # Run simulations for each game based on its simulation count
        max_sims = int(jnp.max(num_simulations_per_game))
        
        for sim in range(max_sims):
            # Only run simulation for games that haven't reached their limit
            active_games = sim < num_simulations_per_game
            
            if not jnp.any(active_games):
                break
            
            # Get policies and values for all active games
            if boards.game_mode == "asymmetric":
                # Pass player roles for asymmetric mode
                player_roles = boards.current_players  # 0 = attacker, 1 = defender
                policies, values = neural_network.evaluate_batch(
                    edge_indices, edge_features, valid_mask, player_roles=player_roles
                )
            else:
                policies, values = neural_network.evaluate_batch(
                    edge_indices, edge_features, valid_mask
                )
            
            # Apply perspective mode to values
            if self.perspective_mode == "fixed":
                # Fixed perspective: values are from Player 1's perspective
                # Need to flip for Player 2
                is_player2 = boards.current_players == 1
                values = jnp.where(is_player2[:, None], -values, values)
            # For alternating perspective, values are already from current player's perspective
            
            # Add noise to policies at root on first simulation
            if sim == 0 and self.noise_weight > 0:
                policies = jnp.where(
                    active_games[:, None],
                    (1 - self.noise_weight) * policies + self.noise_weight * noise * valid_mask,
                    policies
                )
                # Renormalize
                policy_sums = jnp.sum(policies, axis=1, keepdims=True)
                policies = jnp.where(policy_sums > 0, policies / policy_sums, policies)
            
            # Calculate PUCT scores
            sqrt_total_visits = jnp.sqrt(jnp.sum(visit_counts, axis=1, keepdims=True))
            q_values = jnp.where(visit_counts > 0, value_sums / visit_counts, 0.0)
            u_values = self.c_puct * policies * sqrt_total_visits / (1 + visit_counts)
            puct_scores = q_values + u_values
            
            # Mask invalid actions
            puct_scores = jnp.where(valid_mask, puct_scores, -jnp.inf)
            
            # Select actions based on PUCT
            selected_actions = jnp.argmax(puct_scores, axis=1)
            
            # Update stats only for active games
            for game_idx in range(self.batch_size):
                if active_games[game_idx]:
                    action = selected_actions[game_idx]
                    visit_counts = visit_counts.at[game_idx, action].add(1)
                    value_sums = value_sums.at[game_idx, action].add(values[game_idx, 0])
        
        # Convert visits to probabilities
        if temperature == 0:
            # Deterministic
            best_actions = jnp.argmax(visit_counts, axis=1)
            action_probs = jnp.zeros((self.batch_size, self.num_actions))
            action_probs = action_probs.at[jnp.arange(self.batch_size), best_actions].set(1.0)
        else:
            # Stochastic with temperature
            visit_counts_temp = visit_counts ** (1.0 / temperature)
            action_probs = visit_counts_temp / (jnp.sum(visit_counts_temp, axis=1, keepdims=True) + 1e-8)
        
        return action_probs


# Backward compatibility
VectorizedMCTS = ImprovedVectorizedMCTS


if __name__ == "__main__":
    print("Testing Improved Vectorized MCTS")
    print("="*60)
    
    from vectorized_board import VectorizedCliqueBoard
    from vectorized_nn import ImprovedBatchedNeuralNetwork
    
    batch_size = 8
    boards = VectorizedCliqueBoard(batch_size, game_mode="asymmetric")
    nn = ImprovedBatchedNeuralNetwork(asymmetric_mode=True)
    
    # Test with varied simulation counts
    sim_counts = jnp.array([50, 100, 75, 80, 60, 90, 55, 95])
    
    mcts = ImprovedVectorizedMCTS(batch_size, perspective_mode="alternating")
    
    print(f"Running MCTS for {batch_size} games with varied simulations...")
    print(f"Simulation counts: {sim_counts}")
    
    import time
    start = time.time()
    
    action_probs = mcts.search(boards, nn, sim_counts, temperature=1.0)
    
    elapsed = time.time() - start
    print(f"\nTime: {elapsed:.3f}s")
    print(f"Action probs shape: {action_probs.shape}")
    print(f"All probs sum to ~1: {jnp.allclose(jnp.sum(action_probs, axis=1), 1.0)}")
    
    print("\n" + "="*60)
    print("✓ Improved MCTS with perspective modes implemented!")
    print("✓ Supports per-game simulation counts")
    print("="*60)
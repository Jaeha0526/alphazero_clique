"""
Proper vectorized MCTS using JAX's vmap.
This is how it SHOULD be done in JAX - functional and vectorized!
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
import numpy as np
from functools import partial
from typing import Tuple, NamedTuple

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class MCTSState(NamedTuple):
    """State for vectorized MCTS."""
    # Tree structure (batch_size, max_nodes, ...)
    visit_counts: jnp.ndarray  # (batch_size, max_nodes)
    value_sums: jnp.ndarray    # (batch_size, max_nodes)
    prior_probs: jnp.ndarray   # (batch_size, max_nodes, num_actions)
    children: jnp.ndarray      # (batch_size, max_nodes, num_actions) - child node indices
    parents: jnp.ndarray       # (batch_size, max_nodes) - parent node index
    actions: jnp.ndarray       # (batch_size, max_nodes) - action that led to this node
    
    # Board states for each node
    board_states: jnp.ndarray  # (batch_size, max_nodes, board_representation_size)
    current_players: jnp.ndarray  # (batch_size, max_nodes)
    game_over: jnp.ndarray     # (batch_size, max_nodes)
    
    # Current root for each game
    roots: jnp.ndarray         # (batch_size,)
    num_nodes: jnp.ndarray     # (batch_size,) - number of nodes created


class VectorizedMCTSProper:
    """Properly vectorized MCTS using JAX primitives."""
    
    def __init__(self, 
                 batch_size: int,
                 num_actions: int = 15,
                 max_nodes: int = 1000,
                 c_puct: float = 3.0):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.max_nodes = max_nodes
        self.c_puct = c_puct
    
    def search_vectorized(self, 
                         boards: VectorizedCliqueBoard,
                         neural_network: ImprovedBatchedNeuralNetwork,
                         num_simulations: int,
                         temperature: float = 1.0) -> jnp.ndarray:
        """
        Run MCTS for all games in parallel using vmap.
        
        This is a simplified version that shows the concept.
        A full implementation would need more sophisticated tree handling.
        """
        
        # For simplicity, we'll do a flat Monte Carlo search
        # (evaluating random rollouts in parallel)
        # A full tree-based version would require more complex state management
        
        # Get initial features for all games
        edge_indices, edge_features = boards.get_features_for_nn_undirected()
        valid_masks = boards.get_valid_moves_mask()
        
        # Get initial policy and value for all games at once
        policies, values = neural_network.evaluate_batch(edge_indices, edge_features, valid_masks)
        
        # Initialize visit counts and value sums
        visit_counts = jnp.zeros((self.batch_size, self.num_actions))
        value_sums = jnp.zeros((self.batch_size, self.num_actions))
        
        # Vectorized simulation function
        @jit
        def run_simulation_batch(carry, _):
            """Run one simulation for all games in parallel."""
            boards_state, visit_counts, value_sums, key = carry
            
            # Sample actions for all games based on current policy
            # Add exploration noise
            key, subkey = jax.random.split(key)
            noise = jax.random.dirichlet(subkey, alpha=jnp.ones(self.num_actions), shape=(self.batch_size,))
            
            # Combine policy with noise
            exploration_probs = 0.75 * policies + 0.25 * noise
            
            # Sample actions
            key, subkey = jax.random.split(key)
            actions = jax.random.categorical(subkey, jnp.log(exploration_probs + 1e-8), axis=1)
            
            # For now, just use the neural network value directly
            # A full implementation would simulate to the end
            action_values = values.squeeze()
            
            # Update statistics using scatter operations
            action_indices = jnp.arange(self.batch_size)
            visit_counts = visit_counts.at[action_indices, actions].add(1)
            value_sums = value_sums.at[action_indices, actions].add(action_values)
            
            return (boards_state, visit_counts, value_sums, key), None
        
        # Run simulations in parallel for all games
        key = jax.random.PRNGKey(0)
        init_carry = (boards.edge_states, visit_counts, value_sums, key)
        (_, final_visits, final_values, _), _ = lax.scan(
            run_simulation_batch, init_carry, None, length=num_simulations
        )
        
        # Convert visit counts to probabilities
        if temperature == 0:
            # Deterministic: choose most visited
            action_probs = (final_visits == final_visits.max(axis=1, keepdims=True)).astype(jnp.float32)
        else:
            # Apply temperature
            visit_temp = jnp.power(final_visits, 1.0 / temperature)
            action_probs = visit_temp / visit_temp.sum(axis=1, keepdims=True)
        
        return action_probs


class SimplifiedVectorizedMCTS:
    """
    Simplified but truly parallel MCTS.
    Instead of building a tree, we do parallel rollouts.
    """
    
    def __init__(self, batch_size: int, num_actions: int = 15, c_puct: float = 3.0):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.c_puct = c_puct
        
        # Pre-compile the search function
        self._search_fn = None
    
    def search_parallel(self,
                       boards: VectorizedCliqueBoard,
                       neural_network: ImprovedBatchedNeuralNetwork,
                       num_simulations: jnp.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        """Run parallel MCTS-style search for all games."""
        
        # Get features and evaluate all positions at once
        edge_indices, edge_features = boards.get_features_for_nn_undirected()
        valid_masks = boards.get_valid_moves_mask()
        
        # Single batch evaluation for all games!
        root_policies, root_values = neural_network.evaluate_batch(
            edge_indices, edge_features, valid_masks
        )
        
        # Initialize statistics
        N = jnp.zeros((self.batch_size, self.num_actions))  # visit counts
        W = jnp.zeros((self.batch_size, self.num_actions))  # total values
        Q = jnp.zeros((self.batch_size, self.num_actions))  # average values
        
        # Add Dirichlet noise to root
        key = jax.random.PRNGKey(42)
        noise = jax.random.dirichlet(key, jnp.ones(self.num_actions), shape=(self.batch_size,))
        root_policies = 0.75 * root_policies + 0.25 * noise
        
        # Simplified MCTS: just accumulate statistics from policy
        # For each simulation, we select based on UCB and update
        def simulation_step(carry, _):
            N, W, Q, key = carry
            
            # Calculate UCB scores
            sqrt_total = jnp.sqrt(N.sum(axis=1, keepdims=True))
            ucb = Q + self.c_puct * root_policies * sqrt_total / (1 + N)
            
            # Mask invalid actions
            ucb = jnp.where(valid_masks, ucb, -jnp.inf)
            
            # Select actions
            actions = jnp.argmax(ucb, axis=1)
            
            # Get values (simplified - just use root value)
            values = root_values.squeeze()
            
            # Update statistics
            idx = (jnp.arange(self.batch_size), actions)
            N = N.at[idx].add(1)
            W = W.at[idx].add(values)
            Q = W / jnp.maximum(N, 1)
            
            key = jax.random.split(key)[0]
            return (N, W, Q, key), None
        
        # Run simulations
        # Use the minimum number of simulations across all games
        min_sims = int(jnp.min(num_simulations))
        (N_final, _, _, _), _ = lax.scan(
            simulation_step, (N, W, Q, key), None, length=min_sims
        )
        
        # Convert to probabilities
        if temperature == 0:
            # Deterministic
            probs = (N_final == N_final.max(axis=1, keepdims=True)).astype(jnp.float32)
        else:
            # With temperature
            counts_temp = jnp.power(N_final, 1.0 / temperature)
            probs = counts_temp / counts_temp.sum(axis=1, keepdims=True)
        
        # Mask invalid actions
        probs = jnp.where(valid_masks, probs, 0.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs
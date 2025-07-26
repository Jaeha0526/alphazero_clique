"""
JIT-optimized Vectorized MCTS for improved performance.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
from typing import Tuple, Dict, List, Optional, NamedTuple
from functools import partial
import math

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class MCTSState(NamedTuple):
    """State for vectorized MCTS."""
    visit_counts: jnp.ndarray  # [batch_size, num_actions]
    value_sums: jnp.ndarray    # [batch_size, num_actions]
    policies: jnp.ndarray      # [batch_size, num_actions]
    values: jnp.ndarray        # [batch_size, 1]


class JITVectorizedMCTS:
    """JIT-optimized Vectorized MCTS implementation."""
    
    def __init__(self, batch_size: int, num_actions: int, 
                 c_puct: float = 3.0, noise_weight: float = 0.25,
                 perspective_mode: str = "alternating"):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.c_puct = c_puct
        self.noise_weight = noise_weight
        self.perspective_mode = perspective_mode
        
        # Pre-compile the key functions
        self._simulate_step = jit(self._simulate_step_impl)
        self._compute_action_probs = jit(self._compute_action_probs_impl)
    
    def search(self, boards: VectorizedCliqueBoard, nn_model: ImprovedBatchedNeuralNetwork,
               num_simulations: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        """
        Perform MCTS search with JIT optimization.
        
        Args:
            boards: Current board states
            nn_model: Neural network for evaluation
            num_simulations: Number of simulations per game
            temperature: Temperature for action selection
            
        Returns:
            Action probabilities [batch_size, num_actions]
        """
        # Initialize MCTS state
        state = MCTSState(
            visit_counts=jnp.zeros((self.batch_size, self.num_actions)),
            value_sums=jnp.zeros((self.batch_size, self.num_actions)),
            policies=jnp.zeros((self.batch_size, self.num_actions)),
            values=jnp.zeros((self.batch_size, 1))
        )
        
        # Generate noise for root exploration
        rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
        noise = jax.random.dirichlet(rng, alpha=jnp.ones(self.num_actions), shape=(self.batch_size,))
        
        # Get initial board features
        edge_indices, edge_features = boards.get_features_for_nn_undirected()
        valid_mask = boards.get_valid_moves_mask()
        active_games = boards.game_states == 0
        
        # Get player roles for asymmetric games
        player_roles = boards.current_players if self.perspective_mode == "alternating" else jnp.zeros(self.batch_size)
        
        # Evaluate root position with valid moves mask
        policies, values = nn_model.evaluate_batch(edge_indices, edge_features, valid_mask, player_roles)
        
        # Update state with root evaluation
        state = state._replace(policies=policies, values=values)
        
        # Run simulations using lax.fori_loop for JIT compilation
        max_sims = int(jnp.max(num_simulations))
        
        def simulation_step(sim, carry):
            state, active_games, valid_mask, noise = carry
            
            # Add noise on first simulation
            policies_with_noise = lax.cond(
                sim == 0,
                lambda p: jnp.where(
                    active_games[:, None] & (self.noise_weight > 0),
                    (1 - self.noise_weight) * p + self.noise_weight * noise * valid_mask,
                    p
                ),
                lambda p: p,
                state.policies
            )
            
            # Renormalize after noise
            policy_sums = jnp.sum(policies_with_noise, axis=1, keepdims=True)
            policies_with_noise = jnp.where(policy_sums > 0, policies_with_noise / policy_sums, policies_with_noise)
            
            # Update state for this simulation
            new_state = self._simulate_step(
                state._replace(policies=policies_with_noise),
                valid_mask,
                active_games & (sim < num_simulations[:, None]).squeeze()
            )
            
            return (new_state, active_games, valid_mask, noise)
        
        # Run all simulations
        (final_state, _, _, _) = lax.fori_loop(
            0, max_sims,
            simulation_step,
            (state, active_games, valid_mask, noise)
        )
        
        # Compute final action probabilities
        action_probs = self._compute_action_probs(final_state.visit_counts, temperature)
        
        return action_probs
    
    def _simulate_step_impl(self, state: MCTSState, valid_mask: jnp.ndarray, 
                            active_mask: jnp.ndarray) -> MCTSState:
        """Single MCTS simulation step (JIT-compiled)."""
        # Calculate PUCT scores
        sqrt_total_visits = jnp.sqrt(jnp.sum(state.visit_counts, axis=1, keepdims=True))
        q_values = jnp.where(state.visit_counts > 0, 
                            state.value_sums / state.visit_counts, 
                            0.0)
        u_values = self.c_puct * state.policies * sqrt_total_visits / (1 + state.visit_counts)
        puct_scores = q_values + u_values
        
        # Mask invalid actions
        puct_scores = jnp.where(valid_mask, puct_scores, -jnp.inf)
        
        # Select actions
        selected_actions = jnp.argmax(puct_scores, axis=1)
        
        # Create update masks
        action_mask = jax.nn.one_hot(selected_actions, self.num_actions)
        update_mask = active_mask[:, None] * action_mask
        
        # Update counts and values
        new_visit_counts = state.visit_counts + update_mask
        new_value_sums = state.value_sums + update_mask * state.values
        
        return state._replace(
            visit_counts=new_visit_counts,
            value_sums=new_value_sums
        )
    
    def _compute_action_probs_impl(self, visit_counts: jnp.ndarray, temperature: float) -> jnp.ndarray:
        """Compute action probabilities from visit counts (JIT-compiled)."""
        return lax.cond(
            temperature == 0,
            # Deterministic: one-hot at max
            lambda vc: jax.nn.one_hot(jnp.argmax(vc, axis=1), self.num_actions),
            # Stochastic: softmax with temperature
            lambda vc: jax.nn.softmax(jnp.log(vc + 1e-8) / temperature, axis=1),
            visit_counts
        )
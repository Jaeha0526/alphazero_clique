"""
Fix 3: JIT-compiled MCTS operations
Key idea: Compile the core MCTS operations using JAX's JIT
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple, NamedTuple

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class MCTSState(NamedTuple):
    """State for vectorized MCTS across multiple games."""
    # Visit statistics (batch_size, num_actions)
    visit_counts: jnp.ndarray
    total_values: jnp.ndarray
    prior_probs: jnp.ndarray
    
    # Root values for backup
    root_values: jnp.ndarray
    
    # Valid actions mask
    valid_actions: jnp.ndarray


class JITCompiledMCTS:
    """
    MCTS with JIT-compiled operations.
    
    Key optimizations:
    1. UCB calculation is JIT compiled
    2. Action selection is JIT compiled  
    3. Statistics update is JIT compiled
    4. Full simulation loop can be JIT compiled
    """
    
    def __init__(self, batch_size: int, num_actions: int = 15, c_puct: float = 3.0):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.c_puct = c_puct
        
        # Pre-compile core functions
        self._jit_select_actions = jax.jit(self._select_actions)
        self._jit_update_statistics = jax.jit(self._update_statistics)
        self._jit_calculate_ucb = jax.jit(self._calculate_ucb)
        
        # Compile the full simulation step
        self._jit_simulation_step = jax.jit(self._simulation_step)
        
        # Compile the search function with proper static arguments
        self._search_jit = jax.jit(self.search_jit, static_argnums=(4,))
    
    @partial(jax.jit, static_argnums=(0,))
    def _calculate_ucb(self, 
                      visit_counts: jnp.ndarray,
                      total_values: jnp.ndarray, 
                      prior_probs: jnp.ndarray,
                      parent_visits: jnp.ndarray,
                      valid_actions: jnp.ndarray) -> jnp.ndarray:
        """
        JIT-compiled UCB calculation for all games and actions.
        
        Args:
            visit_counts: (batch_size, num_actions)
            total_values: (batch_size, num_actions)
            prior_probs: (batch_size, num_actions)
            parent_visits: (batch_size,)
            valid_actions: (batch_size, num_actions)
        
        Returns:
            UCB scores: (batch_size, num_actions)
        """
        # Q-values: average value per visit
        q_values = jnp.where(
            visit_counts > 0,
            total_values / visit_counts,
            0.0
        )
        
        # Exploration term
        sqrt_parent = jnp.sqrt(parent_visits[:, None])
        exploration = self.c_puct * prior_probs * sqrt_parent / (1 + visit_counts)
        
        # UCB = Q + U
        ucb = q_values + exploration
        
        # Mask invalid actions
        ucb = jnp.where(valid_actions, ucb, -jnp.inf)
        
        return ucb
    
    @partial(jax.jit, static_argnums=(0,))
    def _select_actions(self, state: MCTSState) -> jnp.ndarray:
        """
        JIT-compiled action selection for all games.
        
        Returns:
            Selected actions: (batch_size,)
        """
        # Calculate UCB scores
        parent_visits = jnp.sum(state.visit_counts, axis=1)
        ucb_scores = self._calculate_ucb(
            state.visit_counts,
            state.total_values,
            state.prior_probs,
            parent_visits,
            state.valid_actions
        )
        
        # Select best actions
        actions = jnp.argmax(ucb_scores, axis=1)
        
        return actions
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_statistics(self,
                          state: MCTSState,
                          actions: jnp.ndarray,
                          values: jnp.ndarray) -> MCTSState:
        """
        JIT-compiled statistics update.
        
        Args:
            state: Current MCTS state
            actions: Selected actions for each game
            values: Values to backup
        
        Returns:
            Updated state
        """
        # Create indices for updating
        batch_indices = jnp.arange(self.batch_size)
        
        # Update visit counts
        new_visit_counts = state.visit_counts.at[batch_indices, actions].add(1)
        
        # Update total values
        new_total_values = state.total_values.at[batch_indices, actions].add(values)
        
        return state._replace(
            visit_counts=new_visit_counts,
            total_values=new_total_values
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _simulation_step(self, state: MCTSState, _) -> Tuple[MCTSState, None]:
        """
        JIT-compiled single simulation step.
        
        This function is designed to be used with lax.scan for the full loop.
        """
        # Select actions based on UCB
        actions = self._select_actions(state)
        
        # For now, use root values as the backup values
        # In a full implementation, we'd evaluate the selected positions
        values = state.root_values
        
        # Update statistics
        new_state = self._update_statistics(state, actions, values)
        
        return new_state, None
    
    def search_jit(self,
                   initial_policies: jnp.ndarray,
                   initial_values: jnp.ndarray,
                   valid_actions: jnp.ndarray,
                   num_simulations: int,
                   temperature: float = 1.0) -> jnp.ndarray:
        """
        Fully JIT-compiled MCTS search.
        
        Args:
            initial_policies: Prior probabilities from NN (batch_size, num_actions)
            initial_values: Values from NN (batch_size,)
            valid_actions: Valid actions mask (batch_size, num_actions)
            num_simulations: Number of simulations to run
            temperature: Temperature for final action selection
        
        Returns:
            Action probabilities (batch_size, num_actions)
        """
        # Initialize state
        initial_state = MCTSState(
            visit_counts=jnp.zeros((self.batch_size, self.num_actions)),
            total_values=jnp.zeros((self.batch_size, self.num_actions)),
            prior_probs=initial_policies,
            root_values=initial_values,
            valid_actions=valid_actions
        )
        
        # Run simulations using lax.scan (fully JIT-able)
        final_state, _ = jax.lax.scan(
            self._simulation_step,
            initial_state,
            None,
            length=num_simulations
        )
        
        # Convert visit counts to probabilities
        action_probs = self._get_action_probs(
            final_state.visit_counts,
            temperature
        )
        
        return action_probs
    
    def _get_action_probs(self,
                         visit_counts: jnp.ndarray,
                         temperature: float) -> jnp.ndarray:
        """
        Convert visit counts to action probabilities.
        """
        # Use JAX's conditional to handle temperature
        def deterministic_probs():
            probs = (visit_counts == visit_counts.max(axis=1, keepdims=True))
            return probs.astype(jnp.float32)
        
        def temperature_probs():
            counts_temp = jnp.power(visit_counts + 1e-8, 1.0 / temperature)
            return counts_temp / jnp.maximum(
                counts_temp.sum(axis=1, keepdims=True), 1e-8
            )
        
        # Use lax.cond for JIT-compatible conditional
        return jax.lax.cond(
            temperature == 0.0,
            deterministic_probs,
            temperature_probs
        )
    
    def search(self,
               boards: VectorizedCliqueBoard,
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               temperature: float = 1.0) -> jnp.ndarray:
        """
        High-level search interface that handles board features.
        """
        # Get neural network evaluation
        edge_indices, edge_features = boards.get_features_for_nn_undirected()
        valid_masks = boards.get_valid_moves_mask()
        
        policies, values = neural_network.evaluate_batch(
            edge_indices, edge_features, valid_masks
        )
        
        # Run JIT-compiled search
        action_probs = self._search_jit(
            policies,
            values.squeeze(),
            valid_masks,
            num_simulations,
            temperature
        )
        
        # Mask finished games
        active_mask = boards.game_states == 0
        action_probs = jnp.where(
            active_mask[:, None],
            action_probs,
            0.0
        )
        
        return action_probs


class FullyJITMCTS:
    """
    Fully JIT-compiled MCTS that includes neural network in the loop.
    This is the ultimate optimization but requires pure JAX functions.
    """
    
    def __init__(self, batch_size: int, num_actions: int = 15, c_puct: float = 3.0):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.c_puct = c_puct
    
    @partial(jax.jit, static_argnums=(0, 4, 5))
    def search_full_jit(self,
                       edge_indices: jnp.ndarray,
                       edge_features: jnp.ndarray,
                       nn_params: dict,
                       nn_apply_fn,
                       num_simulations: int,
                       temperature: float = 1.0) -> jnp.ndarray:
        """
        Fully JIT-compiled MCTS including NN evaluation.
        
        This compiles the entire MCTS + NN pipeline into a single XLA computation.
        """
        # Initial NN evaluation
        policies, values = nn_apply_fn(nn_params, edge_indices, edge_features)
        
        # Initialize statistics
        N = jnp.zeros((self.batch_size, self.num_actions))
        W = jnp.zeros((self.batch_size, self.num_actions))
        
        def sim_step(carry, _):
            N, W = carry
            
            # UCB calculation
            Q = W / jnp.maximum(N, 1)
            sqrt_sum = jnp.sqrt(N.sum(axis=1, keepdims=True))
            ucb = Q + self.c_puct * policies * sqrt_sum / (1 + N)
            
            # Action selection
            actions = jnp.argmax(ucb, axis=1)
            
            # Update statistics
            batch_idx = jnp.arange(self.batch_size)
            N = N.at[batch_idx, actions].add(1)
            W = W.at[batch_idx, actions].add(values.squeeze())
            
            return (N, W), None
        
        # Run simulations
        (N_final, W_final), _ = jax.lax.scan(
            sim_step, (N, W), None, length=num_simulations
        )
        
        # Convert to probabilities
        if temperature == 0:
            action_probs = (N_final == N_final.max(axis=1, keepdims=True))
            action_probs = action_probs.astype(jnp.float32)
        else:
            N_temp = jnp.power(N_final + 1e-8, 1.0 / temperature)
            action_probs = N_temp / N_temp.sum(axis=1, keepdims=True)
        
        return action_probs
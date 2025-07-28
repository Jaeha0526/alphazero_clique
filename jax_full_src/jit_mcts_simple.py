"""
Fix 3: Simplified JIT-compiled MCTS
Focus on JIT-compiling the hot paths without complex static arguments
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple
import time

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class SimpleJITMCTS:
    """
    Simplified JIT MCTS that focuses on compiling the hot paths.
    """
    
    def __init__(self, batch_size: int, num_actions: int = 15, c_puct: float = 3.0):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.c_puct = c_puct
        
        # JIT compile the core operations
        self._ucb_and_select = jax.jit(self._calculate_ucb_and_select_actions)
        self._update_stats = jax.jit(self._update_statistics)
        self._to_probs = jax.jit(self._visit_counts_to_probs)
    
    def _calculate_ucb_and_select_actions(self,
                                         N: jnp.ndarray,
                                         W: jnp.ndarray,
                                         P: jnp.ndarray,
                                         valid: jnp.ndarray) -> jnp.ndarray:
        """
        JIT-compiled UCB calculation and action selection.
        
        Args:
            N: Visit counts (batch_size, num_actions)
            W: Total values (batch_size, num_actions)
            P: Prior probabilities (batch_size, num_actions)
            valid: Valid actions mask (batch_size, num_actions)
        
        Returns:
            Selected actions (batch_size,)
        """
        # Q-values
        Q = W / jnp.maximum(N, 1)
        
        # Exploration term
        sqrt_parent = jnp.sqrt(N.sum(axis=1, keepdims=True))
        U = self.c_puct * P * sqrt_parent / (1 + N)
        
        # UCB scores
        ucb = Q + U
        
        # Mask invalid actions
        ucb = jnp.where(valid, ucb, -jnp.inf)
        
        # Select best actions
        actions = jnp.argmax(ucb, axis=1)
        
        return actions
    
    def _update_statistics(self,
                          N: jnp.ndarray,
                          W: jnp.ndarray,
                          actions: jnp.ndarray,
                          values: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        JIT-compiled statistics update.
        """
        batch_idx = jnp.arange(self.batch_size)
        
        # Update counts and values
        N_new = N.at[batch_idx, actions].add(1)
        W_new = W.at[batch_idx, actions].add(values)
        
        return N_new, W_new
    
    def _visit_counts_to_probs(self,
                              N: jnp.ndarray,
                              temperature: float) -> jnp.ndarray:
        """
        JIT-compiled conversion of visit counts to probabilities.
        """
        eps = 1e-8
        temp_safe = jnp.maximum(temperature, eps)
        
        # Temperature-based probabilities
        N_temp = jnp.power(N + eps, 1.0 / temp_safe)
        temp_probs = N_temp / N_temp.sum(axis=1, keepdims=True)
        
        # Deterministic probabilities
        det_probs = (N == N.max(axis=1, keepdims=True)).astype(jnp.float32)
        det_probs = det_probs / det_probs.sum(axis=1, keepdims=True)
        
        # Select based on temperature
        return jnp.where(temperature > eps, temp_probs, det_probs)
    
    def search(self,
               boards: VectorizedCliqueBoard,
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               temperature: float = 1.0) -> jnp.ndarray:
        """
        Run MCTS search with JIT-compiled hot paths.
        """
        # Get initial evaluation
        edge_indices, edge_features = boards.get_features_for_nn_undirected()
        valid_masks = boards.get_valid_moves_mask()
        
        policies, values = neural_network.evaluate_batch(
            edge_indices, edge_features, valid_masks
        )
        
        # Initialize statistics
        N = jnp.zeros((self.batch_size, self.num_actions))
        W = jnp.zeros((self.batch_size, self.num_actions))
        
        # Run simulations (Python loop, but operations are JIT)
        for _ in range(num_simulations):
            # JIT-compiled action selection
            actions = self._ucb_and_select(N, W, policies, valid_masks)
            
            # JIT-compiled statistics update
            N, W = self._update_stats(N, W, actions, values.squeeze())
        
        # JIT-compiled conversion to probabilities
        action_probs = self._to_probs(N, temperature)
        
        # Mask finished games
        active_mask = boards.game_states == 0
        action_probs = jnp.where(
            active_mask[:, None],
            action_probs,
            0.0
        )
        
        return action_probs


class VectorizedJITMCTS:
    """
    Fully vectorized MCTS using JAX's functional style.
    This version uses fori_loop for better JIT compilation.
    """
    
    def __init__(self, batch_size: int, num_actions: int = 15, c_puct: float = 3.0):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.c_puct = c_puct
    
    @partial(jax.jit, static_argnums=(0, 4))
    def search_vectorized(self,
                         policies: jnp.ndarray,
                         values: jnp.ndarray,
                         valid_masks: jnp.ndarray,
                         num_simulations: int,
                         temperature: float) -> jnp.ndarray:
        """
        Fully JIT-compiled MCTS search using fori_loop.
        
        This compiles the entire MCTS loop into XLA.
        """
        # Initialize statistics
        N = jnp.zeros((self.batch_size, self.num_actions))
        W = jnp.zeros((self.batch_size, self.num_actions))
        
        def body_fn(i, state):
            N, W = state
            
            # Calculate UCB
            Q = W / jnp.maximum(N, 1)
            sqrt_parent = jnp.sqrt(N.sum(axis=1, keepdims=True))
            # Ensure shapes match for broadcasting
            sqrt_parent_broadcast = jnp.broadcast_to(sqrt_parent, policies.shape)
            U = self.c_puct * policies * sqrt_parent_broadcast / (1 + N)
            ucb = Q + U
            
            # Mask and select
            ucb = jnp.where(valid_masks, ucb, -jnp.inf)
            actions = jnp.argmax(ucb, axis=1)
            
            # Update
            batch_idx = jnp.arange(self.batch_size)
            N = N.at[batch_idx, actions].add(1)
            W = W.at[batch_idx, actions].add(values)
            
            return (N, W)
        
        # Run simulations with fori_loop (JIT-friendly)
        N_final, W_final = jax.lax.fori_loop(
            0, num_simulations, body_fn, (N, W)
        )
        
        # Convert to probabilities
        # Handle temperature=0 case properly
        eps = 1e-8
        temp_safe = jnp.maximum(temperature, eps)
        
        # Always compute temperature-based probabilities
        N_temp = jnp.power(N_final + eps, 1.0 / temp_safe)
        temp_probs = N_temp / N_temp.sum(axis=1, keepdims=True)
        
        # Compute deterministic probabilities
        det_probs = (N_final == N_final.max(axis=1, keepdims=True)).astype(jnp.float32)
        det_probs = det_probs / det_probs.sum(axis=1, keepdims=True)
        
        # Select based on temperature
        probs = jnp.where(temperature > eps, temp_probs, det_probs)
        
        return probs
    
    def search(self,
               boards: VectorizedCliqueBoard,
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               temperature: float = 1.0) -> jnp.ndarray:
        """
        High-level search interface.
        """
        # Get initial evaluation
        print(f"      Getting NN features...")
        edge_indices, edge_features = boards.get_features_for_nn_undirected()
        valid_masks = boards.get_valid_moves_mask()
        
        print(f"      Evaluating NN batch...")
        eval_start = time.time()
        policies, values = neural_network.evaluate_batch(
            edge_indices, edge_features, valid_masks
        )
        print(f"      NN evaluation took {time.time() - eval_start:.3f}s")
        
        # Run JIT-compiled search
        print(f"      Running {num_simulations} MCTS simulations...")
        mcts_start = time.time()
        action_probs = self.search_vectorized(
            policies,
            values.squeeze(),
            valid_masks,
            num_simulations,
            temperature
        )
        print(f"      MCTS simulations took {time.time() - mcts_start:.3f}s")
        
        # Mask finished games
        active_mask = boards.game_states == 0
        action_probs = jnp.where(
            active_mask[:, None],
            action_probs,
            0.0
        )
        
        return action_probs
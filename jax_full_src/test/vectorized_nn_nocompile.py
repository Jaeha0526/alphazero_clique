#!/usr/bin/env python
"""
Version of neural network without JIT pre-compilation
"""

import sys
sys.path.append('/workspace/alphazero_clique/jax_full_src')

# First, import everything except the problematic class
from vectorized_nn import (
    EdgeAwareGNNBlock, EdgeBlock, ImprovedVectorizedCliqueGNN,
    create_improved_model
)

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, Optional


class ImprovedBatchedNeuralNetwork:
    """
    Fixed wrapper without JIT pre-compilation in __init__.
    """
    
    def __init__(self, num_vertices: int = 6, hidden_dim: int = 64, 
                 num_layers: int = 2, asymmetric_mode: bool = False):
        self.num_vertices = num_vertices
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.asymmetric_mode = asymmetric_mode
        self.num_actions = num_vertices * (num_vertices - 1) // 2
        
        # Create model and params
        self.model, self.params = create_improved_model(
            num_vertices, hidden_dim, num_layers, asymmetric_mode
        )
        
        # Don't pre-compile - we'll compile on first use
        self._compiled_eval = None
    
    def evaluate_batch(self, edge_indices: jnp.ndarray, edge_features: jnp.ndarray,
                      valid_moves_mask: Optional[jnp.ndarray] = None,
                      player_roles: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate a batch of positions."""
        
        # Simple forward pass without mutable state for now
        if self.asymmetric_mode and player_roles is not None:
            policies, values = self.model.apply(
                self.params, edge_indices, edge_features, player_roles,
                deterministic=True
            )
        else:
            policies, values = self.model.apply(
                self.params, edge_indices, edge_features,
                deterministic=True
            )
        
        # Apply valid moves mask if provided
        if valid_moves_mask is not None:
            policies = policies * valid_moves_mask
            policies = policies / (jnp.sum(policies, axis=1, keepdims=True) + 1e-8)
        
        return policies, values
    
    def evaluate_single(self, edge_index: jnp.ndarray, edge_features: jnp.ndarray,
                       player_role: Optional[int] = None) -> Tuple[jnp.ndarray, float]:
        """Evaluate a single position."""
        edge_indices = edge_index[None, :, :]
        edge_features_batch = edge_features[None, :, :]
        player_roles = jnp.array([player_role if player_role is not None else 0]) if self.asymmetric_mode else None
        
        policies, values = self.evaluate_batch(edge_indices, edge_features_batch, player_roles=player_roles)
        
        return policies[0], float(values[0, 0])


if __name__ == "__main__":
    print("Testing neural network without pre-compilation...")
    
    import time
    start = time.time()
    model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=32, num_layers=1)
    print(f"Model created in {time.time()-start:.3f}s")
    
    # Test with dummy data
    from vectorized_board import VectorizedCliqueBoard
    board = VectorizedCliqueBoard(batch_size=2)
    edge_indices, edge_features = board.get_features_for_nn_undirected()
    
    start = time.time()
    policies, values = model.evaluate_batch(edge_indices, edge_features)
    print(f"Evaluation completed in {time.time()-start:.3f}s")
    print(f"Policies shape: {policies.shape}, Values shape: {values.shape}")
    print("Success!")
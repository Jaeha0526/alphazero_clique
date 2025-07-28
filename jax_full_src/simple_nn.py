#!/usr/bin/env python
"""
Simplified neural network without JIT pre-compilation in __init__
"""

import jax
import jax.numpy as jnp
from jax import jit
import flax.linen as nn
from typing import Tuple, Optional, Dict, Any

# Import the existing model architecture
from vectorized_nn import ImprovedVectorizedCliqueGNN, create_improved_model


class SimpleNeuralNetwork:
    """
    Simplified wrapper without JIT pre-compilation during init.
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
        
        # Don't pre-compile anything - let it compile on first use
        self._forward_fn = None
    
    def _get_forward_fn(self):
        """Lazily create and compile the forward function."""
        if self._forward_fn is None:
            if self.asymmetric_mode:
                def forward(params, edge_indices, edge_features, player_roles):
                    return self.model.apply(params, edge_indices, edge_features, player_roles, deterministic=True)
            else:
                def forward(params, edge_indices, edge_features):
                    return self.model.apply(params, edge_indices, edge_features, deterministic=True)
            
            self._forward_fn = jit(forward)
        return self._forward_fn
    
    def evaluate_batch(self, edge_indices: jnp.ndarray, edge_features: jnp.ndarray,
                      valid_moves_mask: Optional[jnp.ndarray] = None,
                      player_roles: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate a batch of positions."""
        forward_fn = self._get_forward_fn()
        
        if self.asymmetric_mode:
            if player_roles is None:
                player_roles = jnp.zeros((edge_indices.shape[0],), dtype=jnp.int32)
            policies, values = forward_fn(self.params, edge_indices, edge_features, player_roles)
        else:
            policies, values = forward_fn(self.params, edge_indices, edge_features)
        
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


# Use this as the main neural network
ImprovedBatchedNeuralNetwork = SimpleNeuralNetwork


if __name__ == "__main__":
    print("Testing Simple Neural Network...")
    
    # Test creation
    import time
    start = time.time()
    model = SimpleNeuralNetwork(num_vertices=6, hidden_dim=32, num_layers=2)
    print(f"Model created in {time.time()-start:.3f}s")
    
    # Test evaluation
    from vectorized_board import VectorizedCliqueBoard
    board = VectorizedCliqueBoard(batch_size=2)
    edge_indices, edge_features = board.get_features_for_nn_undirected()
    
    start = time.time()
    policies, values = model.evaluate_batch(edge_indices, edge_features)
    print(f"First evaluation (with JIT compilation): {time.time()-start:.3f}s")
    
    start = time.time()
    policies, values = model.evaluate_batch(edge_indices, edge_features)
    print(f"Second evaluation (cached): {time.time()-start:.3f}s")
    
    print(f"Policies shape: {policies.shape}, Values shape: {values.shape}")
    print("Simple neural network working correctly!")
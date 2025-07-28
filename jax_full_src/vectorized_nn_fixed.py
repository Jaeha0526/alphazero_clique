#!/usr/bin/env python
"""
Fixed version of the vectorized neural network with proper JIT compilation.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import flax.linen as nn
from flax import struct
import numpy as np
from typing import Tuple, Dict, Any, Optional
from functools import partial

# Import the existing GNN architecture
from vectorized_nn import (
    EdgeAwareGNNBlock, EdgeBlock, ImprovedVectorizedCliqueGNN,
    create_improved_model
)


class FixedImprovedBatchedNeuralNetwork:
    """
    Fixed wrapper for the improved neural network with proper JIT compilation.
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
        
        # Initialize batch stats if they exist
        if 'batch_stats' not in self.params:
            self.params['batch_stats'] = {}
        
        # Create separate functions for symmetric and asymmetric modes
        if asymmetric_mode:
            # Define the function outside of lambda for better JIT compilation
            def _eval_asymmetric(params, batch_stats, edge_indices, edge_features, player_roles):
                variables = {'params': params, 'batch_stats': batch_stats}
                output = self.model.apply(
                    variables, 
                    edge_indices, 
                    edge_features, 
                    player_roles,
                    training=False  # Use running averages for evaluation
                )
                return output
            
            self._batch_eval = jit(_eval_asymmetric)
        else:
            # Define the function outside of lambda for better JIT compilation
            def _eval_symmetric(params, batch_stats, edge_indices, edge_features):
                variables = {'params': params, 'batch_stats': batch_stats}
                output = self.model.apply(
                    variables, 
                    edge_indices, 
                    edge_features,
                    training=False  # Use running averages for evaluation
                )
                return output
            
            self._batch_eval = jit(_eval_symmetric)
    
    def evaluate_batch(self, edge_indices: jnp.ndarray, edge_features: jnp.ndarray,
                      valid_moves_mask: Optional[jnp.ndarray] = None,
                      player_roles: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Evaluate a batch of positions.
        
        Args:
            edge_indices: (batch_size, 2, num_edges) - undirected edges
            edge_features: (batch_size, num_edges, 3)
            valid_moves_mask: Optional (batch_size, num_edges) mask
            player_roles: Optional (batch_size,) - 0 for attacker, 1 for defender
            
        Returns:
            policies: (batch_size, num_edges) - masked if mask provided
            values: (batch_size, 1)
        """
        # Extract params and batch_stats
        params = self.params.get('params', self.params)
        batch_stats = self.params.get('batch_stats', {})
        
        # Call the appropriate evaluation function
        if self.asymmetric_mode:
            if player_roles is None:
                player_roles = jnp.zeros((edge_indices.shape[0],), dtype=jnp.int32)
            policies, values = self._batch_eval(params, batch_stats, edge_indices, edge_features, player_roles)
        else:
            policies, values = self._batch_eval(params, batch_stats, edge_indices, edge_features)
        
        # Apply valid moves mask if provided
        if valid_moves_mask is not None:
            # Mask invalid moves
            policies = policies * valid_moves_mask
            # Renormalize
            policies = policies / (jnp.sum(policies, axis=1, keepdims=True) + 1e-8)
        
        return policies, values
    
    def evaluate_single(self, edge_index: jnp.ndarray, edge_features: jnp.ndarray,
                       player_role: Optional[int] = None) -> Tuple[jnp.ndarray, float]:
        """
        Evaluate a single position.
        """
        # Add batch dimension
        edge_indices = edge_index[None, :, :]
        edge_features_batch = edge_features[None, :, :]
        player_roles = jnp.array([player_role if player_role is not None else 0]) if self.asymmetric_mode else None
        
        policies, values = self.evaluate_batch(
            edge_indices, edge_features_batch, player_roles=player_roles
        )
        
        return policies[0], float(values[0, 0])


# Create an alias for backward compatibility
ImprovedBatchedNeuralNetwork = FixedImprovedBatchedNeuralNetwork


if __name__ == "__main__":
    print("Testing Fixed Vectorized Neural Network...")
    print("="*60)
    
    # Test symmetric mode
    print("1. Testing Symmetric Mode:")
    net_sym = FixedImprovedBatchedNeuralNetwork(asymmetric_mode=False)
    
    # Create undirected edges
    edge_list = []
    for i in range(6):
        for j in range(i+1, 6):
            edge_list.append([i, j])
    
    edge_index = jnp.array(edge_list, dtype=jnp.int32).T
    edge_features = jnp.ones((15, 3), dtype=jnp.float32) / 3.0
    
    # Test single evaluation
    import time
    start = time.time()
    policy, value = net_sym.evaluate_single(edge_index, edge_features)
    print(f"✓ Single evaluation (first call with JIT): {time.time()-start:.3f}s")
    
    start = time.time()
    policy, value = net_sym.evaluate_single(edge_index, edge_features)
    print(f"✓ Single evaluation (cached): {time.time()-start:.3f}s")
    
    print(f"  Policy shape: {policy.shape}, sum: {jnp.sum(policy):.4f}")
    print(f"  Value: {value:.4f}")
    
    # Test batch evaluation
    batch_size = 4
    edge_indices = jnp.tile(edge_index[None, :, :], (batch_size, 1, 1))
    edge_features_batch = jnp.tile(edge_features[None, :, :], (batch_size, 1, 1))
    
    start = time.time()
    policies, values = net_sym.evaluate_batch(edge_indices, edge_features_batch)
    print(f"\n✓ Batch evaluation: {time.time()-start:.3f}s")
    print(f"  Policies shape: {policies.shape}")
    print(f"  Values shape: {values.shape}")
    
    print("\nFixed neural network working correctly with JIT!")
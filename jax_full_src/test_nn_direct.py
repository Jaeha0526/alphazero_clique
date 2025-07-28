#!/usr/bin/env python
"""Test neural network directly to ensure it's working."""

import jax
import jax.numpy as jnp
import time

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork

def test_nn_evaluation():
    """Test NN evaluation directly."""
    print("Testing Neural Network Evaluation...")
    print(f"JAX devices: {jax.devices()}")
    
    # Create boards
    batch_size = 5
    boards = VectorizedCliqueBoard(
        batch_size=batch_size,
        num_vertices=9,
        k=4,
        game_mode="symmetric"
    )
    
    # Create neural network
    nn = ImprovedBatchedNeuralNetwork(
        num_vertices=9,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=False
    )
    
    print("\nGetting board features...")
    edge_indices, edge_features = boards.get_features_for_nn_undirected()
    valid_mask = boards.get_valid_moves_mask()
    
    print(f"Edge indices shape: {edge_indices.shape}")
    print(f"Edge features shape: {edge_features.shape}")
    print(f"Valid mask shape: {valid_mask.shape}")
    
    print("\nEvaluating with NN (first call includes JIT compilation)...")
    
    try:
        # First call - includes JIT compilation
        start = time.time()
        policies, values = nn.evaluate_batch(edge_indices, edge_features, valid_mask)
        elapsed1 = time.time() - start
        
        print(f"\nFirst evaluation (with JIT): {elapsed1:.3f}s")
        
        # Second call - already JIT compiled
        start = time.time()
        policies, values = nn.evaluate_batch(edge_indices, edge_features, valid_mask)
        elapsed2 = time.time() - start
        
        print(f"Second evaluation (JIT compiled): {elapsed2:.3f}s")
        
        # Multiple calls to get average
        times = []
        for i in range(10):
            start = time.time()
            policies, values = nn.evaluate_batch(edge_indices, edge_features, valid_mask)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        print(f"Average of 10 calls: {avg_time*1000:.1f}ms")
        
        print(f"\nPolicies shape: {policies.shape}")
        print(f"Values shape: {values.shape}")
        print(f"Policy sums: {policies.sum(axis=1)}")
        
    except Exception as e:
        print(f"\nError during NN evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nn_evaluation()
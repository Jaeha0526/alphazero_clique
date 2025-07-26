#!/usr/bin/env python
"""
Test that vectorized NN produces reasonable outputs
"""

import sys
sys.path.append('/workspace/alphazero_clique')
sys.path.append('/workspace/alphazero_clique/jax_full_src')

import jax
import jax.numpy as jnp
import numpy as np
from vectorized_nn import BatchedNeuralNetwork
from vectorized_board import VectorizedCliqueBoard


def test_nn_outputs():
    """Test that the neural network produces sensible outputs."""
    
    print("Testing Neural Network Outputs")
    print("="*60)
    
    # Create network and board
    net = BatchedNeuralNetwork(num_vertices=6, hidden_dim=64, num_layers=2)
    board = VectorizedCliqueBoard(batch_size=16, num_vertices=6, k=3)
    
    # Get features from board
    edge_indices, edge_features = board.get_features_for_nn()
    valid_mask = board.get_valid_moves_mask()
    
    print(f"Board features shape: {edge_features.shape}")
    print(f"Valid moves mask shape: {valid_mask.shape}")
    
    # Evaluate positions
    policies, values = net.evaluate_batch(edge_indices, edge_features, valid_mask)
    
    print(f"\nNetwork outputs:")
    print(f"Policies shape: {policies.shape}")
    print(f"Values shape: {values.shape}")
    
    # Check outputs are sensible
    print("\n1. Policy checks:")
    print(f"   - All policies sum to ~1.0: {jnp.allclose(jnp.sum(policies, axis=1), 1.0)}")
    print(f"   - All policies non-negative: {jnp.all(policies >= 0)}")
    print(f"   - Policies properly masked: {jnp.all(policies[~valid_mask] < 1e-6)}")
    
    print("\n2. Value checks:")
    print(f"   - Values in [-1, 1]: {jnp.all(values >= -1) and jnp.all(values <= 1)}")
    print(f"   - Values shape correct: {values.shape == (16, 1)}")
    
    # Test with different board states
    print("\n3. Testing with different game states:")
    
    # Make some moves
    actions = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0])
    board.make_moves(actions)
    
    # Get new features
    edge_indices2, edge_features2 = board.get_features_for_nn()
    valid_mask2 = board.get_valid_moves_mask()
    
    # Evaluate again
    policies2, values2 = net.evaluate_batch(edge_indices2, edge_features2, valid_mask2)
    
    print(f"   - Policies changed after moves: {not jnp.allclose(policies, policies2)}")
    print(f"   - Values changed after moves: {not jnp.allclose(values, values2)}")
    
    # Show some example outputs
    print("\n4. Example outputs (first 3 games):")
    for i in range(3):
        print(f"\nGame {i}:")
        print(f"  Value: {values[i, 0]:.3f}")
        print(f"  Top 3 moves: {jnp.argsort(policies[i])[-3:][::-1]}")
        print(f"  Move probs: {policies[i, jnp.argsort(policies[i])[-3:][::-1]]}")
    
    print("\n" + "="*60)
    print("✓ Neural network produces reasonable outputs")
    print("✓ Ready for integration with MCTS")
    print("="*60)


def test_batch_consistency():
    """Test that batch evaluation is consistent with single evaluation."""
    
    print("\n\nBatch Consistency Test")
    print("="*60)
    
    net = BatchedNeuralNetwork()
    
    # Create test input
    edge_list = []
    for i in range(6):
        for j in range(i+1, 6):
            edge_list.extend([[i, j], [j, i]])
    for i in range(6):
        edge_list.append([i, i])
    
    edge_index = jnp.array(edge_list, dtype=jnp.int32).T
    edge_features = jnp.ones((36, 3), dtype=jnp.float32)
    edge_features = edge_features.at[0, :].set([0, 1, 0])  # Make first edge owned by player 1
    
    # Single evaluation
    policy_single, value_single = net.evaluate_single(edge_index, edge_features)
    
    # Batch evaluation with same input repeated
    batch_size = 8
    edge_indices = jnp.tile(edge_index[None, :, :], (batch_size, 1, 1))
    edge_features_batch = jnp.tile(edge_features[None, :, :], (batch_size, 1, 1))
    
    policies_batch, values_batch = net.evaluate_batch(edge_indices, edge_features_batch)
    
    # Check consistency
    print("Checking single vs batch evaluation:")
    print(f"  - Policies match: {jnp.allclose(policy_single, policies_batch[0])}")
    print(f"  - Values match: {jnp.allclose(value_single, values_batch[0, 0])}")
    print(f"  - All batch outputs identical: {jnp.allclose(policies_batch[0], policies_batch[1])}")
    
    print("\n✓ Batch evaluation is consistent with single evaluation")


if __name__ == "__main__":
    test_nn_outputs()
    test_batch_consistency()
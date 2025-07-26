#!/usr/bin/env python
"""Test shape compatibility between components"""

import sys
sys.path.append('/workspace/alphazero_clique')
sys.path.append('/workspace/alphazero_clique/jax_full_src')

from optimized_board import OptimizedVectorizedBoard
from vectorized_nn import BatchedNeuralNetwork

# Create components
board = OptimizedVectorizedBoard(batch_size=16)
nn = BatchedNeuralNetwork()

print("Shape Test")
print("="*40)

# Get features
edge_indices, edge_features = board.get_features_for_nn()
valid_mask = board.get_valid_moves_mask()

print(f"Board num_edges: {board.num_edges}")
print(f"Board num_vertices: {board.num_vertices}")
print(f"Edge indices shape: {edge_indices.shape}")
print(f"Edge features shape: {edge_features.shape}")
print(f"Valid mask shape: {valid_mask.shape}")

print(f"\nNN num_vertices: {nn.num_vertices}")
print(f"NN num_actions: {nn.num_actions}")
print(f"NN expected edges: {nn.num_vertices * nn.num_vertices}")

# Try evaluation
try:
    policies, values = nn.evaluate_batch(edge_indices, edge_features, valid_mask)
    print(f"\nPolicies shape: {policies.shape}")
    print(f"Values shape: {values.shape}")
except Exception as e:
    print(f"\nError: {e}")
    
print("\nThe issue is that NN expects 36 edges (6*6) but board has 15 edges")
print("This is because NN uses all vertex pairs, not just unique edges")
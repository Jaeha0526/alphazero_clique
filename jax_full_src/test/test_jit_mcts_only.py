#!/usr/bin/env python
"""
Test just the JIT MCTS to isolate the issue
"""

import sys
sys.path.append('/workspace/alphazero_clique/jax_full_src')

import time
import jax
import jax.numpy as jnp

print("Testing JIT MCTS in isolation")
print("="*40)

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from jit_mcts_simple import VectorizedJITMCTS

# Small test
batch_size = 4
num_sims = 10

print(f"\nTest parameters:")
print(f"  Batch size: {batch_size}")
print(f"  MCTS sims: {num_sims}")

# Create components
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=32, num_layers=1)
boards = VectorizedCliqueBoard(batch_size=batch_size)
mcts = VectorizedJITMCTS(batch_size=batch_size)

# Get features
edge_indices, edge_features = boards.get_features_for_nn_undirected()
valid_masks = boards.get_valid_moves_mask()

# Get initial policies and values
print("\nGetting initial NN evaluation...")
policies, values = model.evaluate_batch(edge_indices, edge_features, valid_masks)
print(f"✓ Policies shape: {policies.shape}")
print(f"✓ Values shape: {values.shape}")

# Test the vectorized search
print("\nTesting vectorized MCTS search...")
print("First call (JIT compilation)...")
start = time.time()
try:
    probs1 = mcts.search_vectorized(
        policies,
        values.squeeze(),
        valid_masks,
        num_sims,
        1.0  # temperature
    )
    time1 = time.time() - start
    print(f"✓ Success! Time: {time1:.3f}s")
    print(f"  Output shape: {probs1.shape}")
    print(f"  Probs sum: {jnp.sum(probs1, axis=1)}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Second call (cached)
if 'probs1' in locals():
    print("\nSecond call (JIT cached)...")
    start = time.time()
    probs2 = mcts.search_vectorized(
        policies,
        values.squeeze(),
        valid_masks,
        num_sims,
        1.0
    )
    time2 = time.time() - start
    print(f"✓ Time: {time2:.3f}s")
    print(f"  Speedup: {time1/time2:.1f}x")

print("\n" + "="*40)
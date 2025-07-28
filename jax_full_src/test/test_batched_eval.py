#!/usr/bin/env python
"""
Test batched neural network evaluation speed
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import jax
import jax.numpy as jnp
import numpy as np

print("Testing batched vs sequential neural network evaluation...")
print(f"Device: {jax.default_backend()}")

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork

# Create model
print("\nInitializing model...")
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=64, num_layers=2)

# Create boards
batch_sizes = [1, 8, 32, 64]

for batch_size in batch_sizes:
    print(f"\n\nTesting batch size {batch_size}:")
    
    # Create batch of boards
    boards = VectorizedCliqueBoard(batch_size=batch_size)
    edge_indices, edge_features = boards.get_features_for_nn_undirected()
    
    # Warm up
    _ = model.evaluate_batch(edge_indices, edge_features)
    
    # Time batched evaluation
    start = time.time()
    for _ in range(10):
        policies, values = model.evaluate_batch(edge_indices, edge_features)
    batched_time = time.time() - start
    
    print(f"  Batched evaluation (10 calls): {batched_time:.3f}s")
    print(f"  Time per batch: {batched_time/10:.3f}s")
    print(f"  Time per position: {batched_time/10/batch_size*1000:.1f}ms")
    
    # Compare with sequential for small batch
    if batch_size <= 8:
        start = time.time()
        for _ in range(10):
            for i in range(batch_size):
                # Evaluate one at a time
                single_indices = edge_indices[i:i+1]
                single_features = edge_features[i:i+1]
                _ = model.evaluate_batch(single_indices, single_features)
        sequential_time = time.time() - start
        
        print(f"  Sequential evaluation (10 calls): {sequential_time:.3f}s")
        print(f"  Speedup from batching: {sequential_time/batched_time:.1f}x")

# Test JIT compilation benefit
print("\n\nTesting JIT compilation benefit:")
board = VectorizedCliqueBoard(batch_size=32)
edge_indices, edge_features = board.get_features_for_nn_undirected()

# Non-JIT version
@jax.jit
def jit_eval(params, indices, features):
    return model.model.apply(params, indices, features, deterministic=True)

# Time with JIT
start = time.time()
for _ in range(100):
    _ = jit_eval(model.params, edge_indices, edge_features)
jit_time = time.time() - start

print(f"  100 JIT evaluations: {jit_time:.3f}s ({jit_time/100*1000:.1f}ms per call)")
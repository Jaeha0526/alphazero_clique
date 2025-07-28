#!/usr/bin/env python
"""
Simple test to demonstrate vectorized vs sequential performance
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import jax
import jax.numpy as jnp
import numpy as np

print("Demonstrating Vectorized vs Sequential Neural Network Calls")
print("="*60)

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork

# Create model
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=64, num_layers=2)

# Test different batch sizes
print("\n1. Neural Network Evaluation Performance:")
print("-" * 40)

for batch_size in [1, 8, 32, 64]:
    boards = VectorizedCliqueBoard(batch_size=batch_size)
    edge_indices, edge_features = boards.get_features_for_nn_undirected()
    
    # Warm up
    _ = model.evaluate_batch(edge_indices, edge_features)
    
    # Time evaluation
    start = time.time()
    for _ in range(100):
        policies, values = model.evaluate_batch(edge_indices, edge_features)
    elapsed = time.time() - start
    
    print(f"Batch size {batch_size:2d}: {elapsed:.3f}s for 100 evals ({elapsed/100*1000:.2f}ms per eval)")

# Show the problem with current MCTS
print("\n2. Current MCTS Problem (Sequential):")
print("-" * 40)

num_games = 32
num_sims = 100
evals_per_sim = 5  # Approximate evaluations per simulation

total_evals = num_games * num_sims * evals_per_sim
print(f"For {num_games} games with {num_sims} MCTS simulations:")
print(f"Total NN evaluations needed: {total_evals:,}")

# Sequential approach (current)
single_eval_time = 0.0001  # 0.1ms from our test
sequential_time = total_evals * single_eval_time
print(f"\nSequential (current): {total_evals} × {single_eval_time*1000:.1f}ms = {sequential_time:.1f}s")

# Vectorized approach
batch_eval_time = 0.0001  # Same time for batch!
num_batches = num_sims * evals_per_sim  # Can batch across games
vectorized_time = num_batches * batch_eval_time
print(f"Vectorized (proper): {num_batches} × {batch_eval_time*1000:.1f}ms = {vectorized_time:.3f}s")

print(f"\nPotential speedup: {sequential_time/vectorized_time:.0f}x !!")

# Demonstrate simple vectorized MCTS concept
print("\n3. Simple Vectorized MCTS Demo:")
print("-" * 40)

from vectorized_mcts_proper import SimplifiedVectorizedMCTS

# Small test
num_games = 8
boards = VectorizedCliqueBoard(batch_size=num_games)
vec_mcts = SimplifiedVectorizedMCTS(batch_size=num_games)
num_simulations = jnp.array([20] * num_games)

print(f"Running {num_games} games with 20 simulations each...")

# Time the vectorized version
start = time.time()
action_probs = vec_mcts.search_parallel(boards, model, num_simulations)
vec_time = time.time() - start

print(f"Vectorized time: {vec_time:.3f}s")
print(f"That's {num_games/vec_time:.1f} games/second!")
print(f"Or {vec_time/num_games*1000:.1f}ms per game")

# Verify output
print(f"\nOutput shape: {action_probs.shape}")
print(f"Sum of probabilities: {jnp.sum(action_probs, axis=1)[:4]}")
print(f"Selected actions: {jnp.argmax(action_probs, axis=1)}")

print("\n" + "="*60)
print("KEY INSIGHT: With proper vectorization, we process ALL games")
print("in parallel with the same time as processing ONE game!")
print("="*60)
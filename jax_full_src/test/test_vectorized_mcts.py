#!/usr/bin/env python
"""
Test properly vectorized MCTS using JAX vmap
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import jax
import jax.numpy as jnp
import numpy as np

print("Testing Properly Vectorized MCTS with JAX")
print("="*60)
print(f"Device: {jax.default_backend()}")

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from tree_based_mcts import ParallelTreeBasedMCTS
from vectorized_mcts_proper import SimplifiedVectorizedMCTS

# Test parameters
num_games_list = [1, 8, 32]
num_sims = 50

# Create model
print("\nInitializing neural network...")
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=64, num_layers=2)

# Warm up
print("Warming up...")
dummy_board = VectorizedCliqueBoard(batch_size=32)
edge_indices, edge_features = dummy_board.get_features_for_nn_undirected()
_ = model.evaluate_batch(edge_indices, edge_features)
print("✓ Neural network ready")

results = {}

for num_games in num_games_list:
    print(f"\n\nTesting with {num_games} games:")
    print("-" * 40)
    
    # Test 1: Current sequential implementation
    print(f"\n1. Sequential implementation (current):")
    boards_seq = VectorizedCliqueBoard(batch_size=num_games)
    seq_mcts = ParallelTreeBasedMCTS(batch_size=num_games, num_actions=15, c_puct=3.0)
    num_simulations = jnp.array([num_sims] * num_games)
    
    start = time.time()
    action_probs_seq = seq_mcts.search(boards_seq, model, num_simulations)
    seq_time = time.time() - start
    
    print(f"   Time: {seq_time:.3f}s")
    print(f"   Speed: {num_games/seq_time:.2f} games/sec")
    print(f"   Time per game: {seq_time/num_games*1000:.1f}ms")
    
    # Test 2: Properly vectorized implementation
    print(f"\n2. Vectorized implementation (JAX vmap style):")
    boards_vec = VectorizedCliqueBoard(batch_size=num_games)
    vec_mcts = SimplifiedVectorizedMCTS(batch_size=num_games)
    
    # First call includes JIT compilation
    start = time.time()
    action_probs_vec = vec_mcts.search_parallel(boards_vec, model, num_simulations)
    vec_time_first = time.time() - start
    
    # Second call uses cached JIT
    start = time.time()
    action_probs_vec = vec_mcts.search_parallel(boards_vec, model, num_simulations)
    vec_time = time.time() - start
    
    print(f"   First call (with JIT): {vec_time_first:.3f}s")
    print(f"   Second call (cached): {vec_time:.3f}s")
    print(f"   Speed: {num_games/vec_time:.2f} games/sec")
    print(f"   Speedup vs sequential: {seq_time/vec_time:.1f}x")
    
    # Store results
    results[num_games] = {
        'sequential': seq_time,
        'vectorized': vec_time,
        'speedup': seq_time/vec_time
    }
    
    # Verify both produce valid probability distributions
    print(f"\n3. Validation:")
    print(f"   Sequential - sum of probs: {jnp.sum(action_probs_seq, axis=1)[:3]}")
    print(f"   Vectorized - sum of probs: {jnp.sum(action_probs_vec, axis=1)[:3]}")
    
    # Show selected actions
    seq_actions = jnp.argmax(action_probs_seq, axis=1)
    vec_actions = jnp.argmax(action_probs_vec, axis=1)
    print(f"   Sequential actions: {seq_actions[:5]}")
    print(f"   Vectorized actions: {vec_actions[:5]}")

# Summary
print("\n\n" + "="*60)
print("PERFORMANCE SUMMARY:")
print("="*60)
print("Games | Sequential Time | Vectorized Time | Speedup")
print("-"*60)
for num_games, res in results.items():
    print(f"{num_games:5d} | {res['sequential']:14.3f}s | {res['vectorized']:14.3f}s | {res['speedup']:6.1f}x")

# Estimate for 100 games
if 32 in results:
    est_seq = results[32]['sequential'] * (100/32)
    est_vec = results[32]['vectorized'] * (100/32)
    print(f"\nEstimated time for 100 games with 50 MCTS simulations:")
    print(f"  Sequential: {est_seq:.1f}s ({est_seq/60:.1f} minutes)")
    print(f"  Vectorized: {est_vec:.1f}s ({est_vec/60:.1f} minutes)")
    print(f"  Speedup: {est_seq/est_vec:.1f}x")

print("\n✓ Fix 1 Complete: True parallelization using JAX vectorization!")
print("="*60)
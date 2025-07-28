#!/usr/bin/env python
"""
Demonstrate how parallel games SHOULD work
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

print("Demonstrating proper parallel game execution...")
print(f"Device: {jax.default_backend()}")

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork

# Create model and boards
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=64, num_layers=2)

print("\n1. Playing games with simple policy (no MCTS):")
for num_games in [1, 10, 32]:
    print(f"\n  Playing {num_games} games in parallel...")
    
    boards = VectorizedCliqueBoard(batch_size=num_games)
    move_count = 0
    start = time.time()
    
    # Play until all games end
    while jnp.any(boards.game_states == 0) and move_count < 15:
        # Get features for ALL active games at once
        edge_indices, edge_features = boards.get_features_for_nn_undirected()
        valid_mask = boards.get_valid_moves_mask()
        
        # Evaluate ALL positions in one batch
        policies, values = model.evaluate_batch(edge_indices, edge_features, valid_mask)
        
        # Select moves (just argmax for demo)
        actions = jnp.argmax(policies, axis=1)
        
        # Make moves for all games
        boards.make_moves(actions)
        move_count += 1
    
    elapsed = time.time() - start
    print(f"    Time: {elapsed:.3f}s")
    print(f"    Games/second: {num_games/elapsed:.1f}")
    print(f"    Moves/second: {num_games * move_count / elapsed:.1f}")

print("\n2. Simulating parallel MCTS (batched NN evals):")
print("   (This shows the potential speedup with proper batching)")

num_games = 32
num_sims = 100
positions_per_sim = 5  # Assume we evaluate 5 positions per MCTS simulation

# Simulate the neural network calls in MCTS
print(f"\n  Simulating {num_games} games, {num_sims} MCTS sims each")
print(f"  Total NN evaluations: {num_games * num_sims * positions_per_sim}")

# Create dummy data for all evaluations
boards = VectorizedCliqueBoard(batch_size=num_games)
edge_indices, edge_features = boards.get_features_for_nn_undirected()

# Method 1: Sequential (current implementation)
start = time.time()
for game in range(num_games):
    for sim in range(num_sims):
        for pos in range(positions_per_sim):
            # Evaluate one position at a time
            single_idx = edge_indices[0:1]  # Just use first board as dummy
            single_feat = edge_features[0:1]
            _ = model.evaluate_batch(single_idx, single_feat)
sequential_time = time.time() - start

# Method 2: Batched by game (better)
start = time.time()
for sim in range(num_sims):
    # Evaluate all games at once for each simulation
    _ = model.evaluate_batch(edge_indices, edge_features)
batched_time = time.time() - start

# Method 3: Fully batched (best - if we could batch across simulations)
batch_size = min(256, num_games * positions_per_sim)  # Reasonable batch size
total_evals = num_sims * num_games * positions_per_sim
num_batches = (total_evals + batch_size - 1) // batch_size

# Create larger batch
large_batch_indices = jnp.tile(edge_indices[0:1], (batch_size, 1, 1))
large_batch_features = jnp.tile(edge_features[0:1], (batch_size, 1, 1))

start = time.time()
for _ in range(num_batches):
    _ = model.evaluate_batch(large_batch_indices, large_batch_features)
fully_batched_time = time.time() - start

print(f"\n  Results:")
print(f"    Sequential (current): {sequential_time:.2f}s")
print(f"    Batched by game: {batched_time:.2f}s ({sequential_time/batched_time:.1f}x speedup)")
print(f"    Fully batched: {fully_batched_time:.2f}s ({sequential_time/fully_batched_time:.1f}x speedup)")

# Estimate time for 100 games with 100 MCTS
est_sequential = sequential_time * (100/num_games)
est_batched = batched_time * (100/num_games)
print(f"\n  Estimated time for 100 games:")
print(f"    Current implementation: {est_sequential:.0f}s ({est_sequential/60:.1f} minutes)")
print(f"    With proper batching: {est_batched:.0f}s ({est_batched/60:.1f} minutes)")
print(f"    Potential speedup: {est_sequential/est_batched:.1f}x")
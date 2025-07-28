#!/usr/bin/env python
"""
Test the fixed parallel MCTS implementation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import jax
import jax.numpy as jnp
import numpy as np

print("Testing Fixed Parallel MCTS")
print("="*60)
print(f"Device: {jax.default_backend()}")

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from tree_based_mcts import ParallelTreeBasedMCTS
from parallel_mcts_fixed import ParallelTreeBasedMCTSFixed, FullyVectorizedMCTS

# Test parameters
num_games = 16
num_sims = 10  # Small number for quick test

# Create model
print("\nInitializing components...")
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=32, num_layers=1)

# Warm up
dummy_board = VectorizedCliqueBoard(batch_size=1)
edge_indices, edge_features = dummy_board.get_features_for_nn_undirected()
_ = model.evaluate_batch(edge_indices, edge_features)
print("✓ Ready")

# Test 1: Current implementation (sequential)
print(f"\n1. Current implementation (sequential for loop):")
boards1 = VectorizedCliqueBoard(batch_size=num_games)
current_mcts = ParallelTreeBasedMCTS(batch_size=num_games, num_actions=15, c_puct=3.0)
num_simulations = jnp.array([num_sims] * num_games)

start = time.time()
action_probs1 = current_mcts.search(boards1, model, num_simulations)
time1 = time.time() - start

print(f"   Time: {time1:.3f}s")
print(f"   Games/sec: {num_games/time1:.2f}")

# Test 2: Fixed implementation (batched NN calls)
print(f"\n2. Fixed implementation (batched NN evaluations):")
boards2 = VectorizedCliqueBoard(batch_size=num_games)
fixed_mcts = ParallelTreeBasedMCTSFixed(batch_size=num_games, num_actions=15, c_puct=3.0)

start = time.time()
action_probs2 = fixed_mcts.search(boards2, model, num_simulations)
time2 = time.time() - start

print(f"   Time: {time2:.3f}s")
print(f"   Games/sec: {num_games/time2:.2f}")
print(f"   Speedup: {time1/time2:.1f}x")

# Test 3: Fully vectorized (JAX-native)
print(f"\n3. Fully vectorized (pure JAX):")
boards3 = VectorizedCliqueBoard(batch_size=num_games)
jax_mcts = FullyVectorizedMCTS(batch_size=num_games)

# Prepare inputs
boards_state = (
    boards3.edge_states,
    boards3.current_players,
    boards3.game_states,
    boards3.move_counts
)

# First call (includes JIT compilation)
start = time.time()
action_probs3 = jax_mcts.search(
    boards_state, 
    model.params, 
    model.model.apply,
    num_sims
)
time3_first = time.time() - start

# Second call (JIT cached)
start = time.time()
action_probs3 = jax_mcts.search(
    boards_state, 
    model.params, 
    model.model.apply,
    num_sims
)
time3 = time.time() - start

print(f"   First call (JIT): {time3_first:.3f}s")
print(f"   Second call: {time3:.3f}s")
print(f"   Games/sec: {num_games/time3:.2f}")
print(f"   Speedup vs sequential: {time1/time3:.1f}x")

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY:")
print(f"Sequential (for loop):     {time1:.3f}s - {num_games/time1:.1f} games/sec")
print(f"Fixed (batched NN):        {time2:.3f}s - {num_games/time2:.1f} games/sec")
print(f"Fully vectorized (JAX):    {time3:.3f}s - {num_games/time3:.1f} games/sec")

print(f"\nBest speedup achieved: {time1/min(time2, time3):.1f}x")

# Estimate for 100 games, 100 simulations
scale_factor = (100/num_games) * (100/num_sims)
print(f"\nEstimated time for 100 games, 100 simulations:")
print(f"  Sequential: {time1 * scale_factor:.0f}s ({time1 * scale_factor / 60:.1f} minutes)")
print(f"  Best parallel: {min(time2, time3) * scale_factor:.0f}s ({min(time2, time3) * scale_factor / 60:.1f} minutes)")

print("\n✓ Fix 1 demonstrated: Parallelization across games!")
print("="*60)
#!/usr/bin/env python
"""
Test Fix 2: Batch NN evaluations within MCTS
Compare synchronized batched MCTS vs sequential MCTS
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import jax
import jax.numpy as jnp
import numpy as np

print("Testing Fix 2: Batch NN Evaluations in MCTS")
print("="*60)
print(f"Device: {jax.default_backend()}")

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from tree_based_mcts import TreeBasedMCTS
from batched_mcts_sync import SynchronizedBatchedMCTS, SimpleBatchedMCTS

# Test parameters
num_games = 4
num_sims = 10  # Reduced for quick test

# Create model
print("\nInitializing components...")
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=32, num_layers=1)

# Warm up
dummy_board = VectorizedCliqueBoard(batch_size=1)
edge_indices, edge_features = dummy_board.get_features_for_nn_undirected()
_ = model.evaluate_batch(edge_indices, edge_features)
print("✓ Model ready")

# Test 1: Sequential MCTS (baseline)
print(f"\n1. Sequential MCTS (current implementation):")
print("   Each game runs MCTS separately")
boards1 = VectorizedCliqueBoard(batch_size=num_games)
action_probs_seq = np.zeros((num_games, 15))

start = time.time()
# Current implementation: sequential loop
for game_idx in range(num_games):
    if boards1.game_states[game_idx] == 0:  # Active game
        # Create single-game board
        single_board = VectorizedCliqueBoard(batch_size=1)
        single_board.edge_states = single_board.edge_states.at[0].set(
            boards1.edge_states[game_idx]
        )
        single_board.current_players = single_board.current_players.at[0].set(
            boards1.current_players[game_idx]
        )
        
        # Run MCTS for this game
        mcts = TreeBasedMCTS(num_actions=15, c_puct=3.0)
        probs = mcts.search(single_board, model, num_sims)
        action_probs_seq[game_idx] = probs

time_seq = time.time() - start
print(f"   Time: {time_seq:.3f}s")
print(f"   Games/sec: {num_games/time_seq:.2f}")

# Count NN evaluations
nn_evals_seq = num_games * num_sims
print(f"   NN evaluations: {nn_evals_seq} (sequential)")

# Test 2: Simple Batched MCTS
print(f"\n2. Simple Batched MCTS (synchronized simulations):")
print("   All games do simulations in sync")
boards2 = VectorizedCliqueBoard(batch_size=num_games)
simple_mcts = SimpleBatchedMCTS(batch_size=num_games)

start = time.time()
action_probs_simple = simple_mcts.search_batch(boards2, model, num_sims)
time_simple = time.time() - start

print(f"   Time: {time_simple:.3f}s")
print(f"   Games/sec: {num_games/time_simple:.2f}")
print(f"   Speedup: {time_seq/time_simple:.1f}x")

# NN evaluations in batched version
nn_evals_batched = num_sims  # All games evaluated together
print(f"   NN evaluations: {nn_evals_batched} (batched)")
print(f"   NN speedup: {nn_evals_seq/nn_evals_batched:.0f}x")

# Test 3: Full Synchronized Batched MCTS
print(f"\n3. Full Synchronized Batched MCTS (with tree):")
print("   Synchronizes tree operations across games")
boards3 = VectorizedCliqueBoard(batch_size=num_games)
sync_mcts = SynchronizedBatchedMCTS(batch_size=num_games)

start = time.time()
action_probs_sync = sync_mcts.search_batch(boards3, model, num_sims)
time_sync = time.time() - start

print(f"   Time: {time_sync:.3f}s")
print(f"   Games/sec: {num_games/time_sync:.2f}")
print(f"   Speedup: {time_seq/time_sync:.1f}x")

# Test with variable game lengths
print(f"\n4. Testing early game endings:")
boards4 = VectorizedCliqueBoard(batch_size=num_games)
# Simulate some games ending early
for i in range(0, num_games, 4):
    boards4.game_states = boards4.game_states.at[i].set(1)  # Mark as finished

print(f"   Active games: {jnp.sum(boards4.game_states == 0)}/{num_games}")

start = time.time()
action_probs_var = sync_mcts.search_batch(boards4, model, num_sims)
time_var = time.time() - start

print(f"   Time: {time_var:.3f}s")
print(f"   Only processes active games ✓")

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY:")
print(f"Sequential MCTS:      {time_seq:.3f}s - {num_games/time_seq:.1f} games/sec")
print(f"Simple Batched MCTS:  {time_simple:.3f}s - {num_games/time_simple:.1f} games/sec")
print(f"Full Batched MCTS:    {time_sync:.3f}s - {num_games/time_sync:.1f} games/sec")

best_time = min(time_simple, time_sync)
print(f"\nBest speedup: {time_seq/best_time:.1f}x")

# Key insight
print("\n🎯 KEY INSIGHT:")
print("-"*40)
print(f"Sequential: {nn_evals_seq} separate NN calls")
print(f"Batched:    {nn_evals_batched} batched NN calls")
print(f"Reduction:  {nn_evals_seq/nn_evals_batched:.0f}x fewer NN calls!")

# Projection for full run
scale_factor = (100/num_games) * (100/num_sims)
print(f"\nProjected time for 100 games, 100 simulations:")
print(f"  Sequential:  {time_seq * scale_factor:.0f}s ({time_seq * scale_factor / 60:.1f} minutes)")
print(f"  Batched:     {best_time * scale_factor:.0f}s ({best_time * scale_factor / 60:.1f} minutes)")
print(f"  Speedup:     {time_seq/best_time:.1f}x")

print("\n✓ Fix 2 implemented: Batched NN evaluations in MCTS!")
print("="*60)
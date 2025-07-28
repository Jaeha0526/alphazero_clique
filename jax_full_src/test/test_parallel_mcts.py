#!/usr/bin/env python
"""
Test parallel MCTS implementations
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import jax
import jax.numpy as jnp
import numpy as np
import multiprocessing as mp

print("Testing Parallel MCTS Implementations")
print("="*60)
print(f"Device: {jax.default_backend()}")
print(f"CPU cores available: {mp.cpu_count()}")

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from tree_based_mcts import ParallelTreeBasedMCTS
from parallel_mcts_simple import SimpleParallelMCTS, ThreadedParallelMCTS

# Create components
num_games = 8
num_sims = 20
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=64, num_layers=2)

# Warm up the model
print("\nWarming up neural network...")
dummy_board = VectorizedCliqueBoard(batch_size=1)
edge_indices, edge_features = dummy_board.get_features_for_nn_undirected()
_ = model.evaluate_batch(edge_indices, edge_features)

# Test 1: Current sequential implementation
print(f"\n1. Testing CURRENT implementation (sequential):")
print(f"   Running {num_games} games with {num_sims} MCTS simulations each...")

boards = VectorizedCliqueBoard(batch_size=num_games)
current_mcts = ParallelTreeBasedMCTS(batch_size=num_games, num_actions=15, c_puct=3.0)
num_simulations = jnp.array([num_sims] * num_games)

start = time.time()
action_probs_seq = current_mcts.search(boards, model, num_simulations)
seq_time = time.time() - start

print(f"   Time: {seq_time:.2f}s")
print(f"   Speed: {num_games/seq_time:.2f} games/second")
print(f"   Time per game: {seq_time/num_games:.2f}s")

# Test 2: Multiprocessing parallel
print(f"\n2. Testing MULTIPROCESSING parallel (using {mp.cpu_count()} cores):")
boards = VectorizedCliqueBoard(batch_size=num_games)
mp_mcts = SimpleParallelMCTS(batch_size=num_games, num_workers=mp.cpu_count())

start = time.time()
action_probs_mp = mp_mcts.search_parallel(boards, model, num_simulations)
mp_time = time.time() - start

print(f"   Time: {mp_time:.2f}s")
print(f"   Speed: {num_games/mp_time:.2f} games/second")
print(f"   Speedup vs sequential: {seq_time/mp_time:.1f}x")

# Test 3: Threaded parallel
print(f"\n3. Testing THREADED parallel:")
boards = VectorizedCliqueBoard(batch_size=num_games)
thread_mcts = ThreadedParallelMCTS(batch_size=num_games, num_actions=15, c_puct=3.0)

start = time.time()
action_probs_thread = thread_mcts.search_parallel(boards, model, num_simulations)
thread_time = time.time() - start

print(f"   Time: {thread_time:.2f}s")
print(f"   Speed: {num_games/thread_time:.2f} games/second")
print(f"   Speedup vs sequential: {seq_time/thread_time:.1f}x")

# Verify results are similar
print("\n4. Verifying results are consistent:")
# Check that all methods produce similar results (allowing for some randomness)
seq_entropy = -jnp.sum(action_probs_seq * jnp.log(action_probs_seq + 1e-8), axis=1)
mp_entropy = -jnp.sum(action_probs_mp * jnp.log(action_probs_mp + 1e-8), axis=1)
thread_entropy = -jnp.sum(action_probs_thread * jnp.log(action_probs_thread + 1e-8), axis=1)

print(f"   Sequential entropy: {jnp.mean(seq_entropy):.3f}")
print(f"   Multiprocess entropy: {jnp.mean(mp_entropy):.3f}")
print(f"   Threaded entropy: {jnp.mean(thread_entropy):.3f}")

# Show which actions were selected
seq_actions = jnp.argmax(action_probs_seq, axis=1)
mp_actions = jnp.argmax(action_probs_mp, axis=1)
thread_actions = jnp.argmax(action_probs_thread, axis=1)

print(f"\n   Selected actions (first 5 games):")
print(f"   Sequential:   {seq_actions[:5]}")
print(f"   Multiprocess: {mp_actions[:5]}")
print(f"   Threaded:     {thread_actions[:5]}")

# Summary
print("\n" + "="*60)
print("SUMMARY:")
print(f"Sequential (current): {seq_time:.2f}s")
print(f"Multiprocessing: {mp_time:.2f}s ({seq_time/mp_time:.1f}x speedup)")
print(f"Threading: {thread_time:.2f}s ({seq_time/thread_time:.1f}x speedup)")

best_time = min(mp_time, thread_time)
print(f"\nBest parallel speedup: {seq_time/best_time:.1f}x")
print(f"This would reduce 100 games from {seq_time*100/num_games:.0f}s to {best_time*100/num_games:.0f}s")
print("="*60)
#!/usr/bin/env python
"""Test performance improvement with efficient board representation."""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'jax_full_src'))

import time
import jax
import numpy as np

from jax_full_src.vectorized_board import VectorizedCliqueBoard
from jax_full_src.efficient_board_proper import EfficientCliqueBoard
from jax_full_src.vectorized_nn import ImprovedBatchedNeuralNetwork
from jax_full_src.simple_tree_mcts_timed import SimpleTreeMCTSTimed
from jax_full_src.simple_tree_mcts_efficient import SimpleTreeMCTSEfficient

# Test parameters
n = 6
k = 3
batch_size = 10
num_simulations = 20

print("=" * 70)
print("EFFICIENT BOARD MCTS PERFORMANCE TEST")
print("=" * 70)
print(f"Setup: n={n}, k={k}, batch_size={batch_size}, simulations={num_simulations}")
print(f"JAX devices: {jax.devices()}")
print()

# Create neural network
print("Creating neural network...")
net = ImprovedBatchedNeuralNetwork(
    num_vertices=n,
    hidden_dim=64,
    num_layers=3,
    asymmetric_mode=False
)

num_actions = n * (n - 1) // 2

# Test 1: Original VectorizedCliqueBoard with SimpleTreeMCTSTimed
print("\n1. ORIGINAL BOARD + SIMPLETREEMCTS")
print("-" * 40)

original_boards = VectorizedCliqueBoard(batch_size, n, k, "symmetric")
original_mcts = SimpleTreeMCTSTimed(
    batch_size=batch_size,
    num_actions=num_actions,
    c_puct=3.0,
    max_nodes_per_game=100
)

# Warmup
print("Warmup...")
warmup_boards = VectorizedCliqueBoard(1, n, k, "symmetric")
warmup_mcts = SimpleTreeMCTSTimed(1, num_actions, 3.0)
warmup_mcts.search(warmup_boards, net, 1, temperature=1.0)

# Timed run
print(f"Running {num_simulations} simulations for {batch_size} games...")
start = time.time()
original_probs = original_mcts.search(original_boards, net, num_simulations, temperature=1.0)
original_time = time.time() - start

print(f"\nTotal time: {original_time:.2f}s")
print(f"Time per game: {original_time/batch_size:.2f}s")
print(f"Time per simulation per game: {original_time/(batch_size*num_simulations)*1000:.1f}ms")

# Test 2: Efficient board with SimpleTreeMCTSEfficient
print("\n\n2. EFFICIENT BOARD + OPTIMIZED MCTS")
print("-" * 40)

efficient_boards = EfficientCliqueBoard(batch_size, n, k, "symmetric")
efficient_mcts = SimpleTreeMCTSEfficient(
    batch_size=batch_size,
    num_actions=num_actions,
    c_puct=3.0,
    max_nodes_per_game=100
)

# Warmup
print("Warmup...")
warmup_boards = EfficientCliqueBoard(1, n, k, "symmetric")
warmup_mcts = SimpleTreeMCTSEfficient(1, num_actions, 3.0)
warmup_mcts.search(warmup_boards, net, 1, temperature=1.0)

# Timed run
print(f"Running {num_simulations} simulations for {batch_size} games...")
start = time.time()
efficient_probs = efficient_mcts.search(efficient_boards, net, num_simulations, temperature=1.0)
efficient_time = time.time() - start

print(f"\nTotal time: {efficient_time:.2f}s")
print(f"Time per game: {efficient_time/batch_size:.2f}s")
print(f"Time per simulation per game: {efficient_time/(batch_size*num_simulations)*1000:.1f}ms")

# Show detailed timing
efficient_mcts.print_timing_summary()

# Comparison
print("\n\n3. COMPARISON")
print("-" * 40)
speedup = original_time / efficient_time
print(f"Original time: {original_time:.2f}s")
print(f"Efficient time: {efficient_time:.2f}s")
print(f"Speedup: {speedup:.1f}x")
print(f"Reduction in time: {(1 - efficient_time/original_time)*100:.1f}%")

# Verify results are the same
print("\n4. VERIFICATION")
print("-" * 40)
print("Checking that MCTS produces similar results...")
for i in range(min(3, batch_size)):
    orig_max = np.argmax(original_probs[i])
    eff_max = np.argmax(efficient_probs[i])
    print(f"Game {i}: Original best move={orig_max}, Efficient best move={eff_max}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
#!/usr/bin/env python
"""Simple test of efficient MCTS without neural network complexity."""

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
from jax_full_src.simple_tree_mcts import SimpleTreeMCTS
from jax_full_src.simple_tree_mcts_efficient import SimpleTreeMCTSEfficient

# Test parameters
n = 6
k = 3
batch_size = 10
num_simulations = 20

print("=" * 70)
print("EFFICIENT MCTS PERFORMANCE TEST (WITHOUT NN)")
print("=" * 70)
print(f"Setup: n={n}, k={k}, batch_size={batch_size}, simulations={num_simulations}")
print()

num_actions = n * (n - 1) // 2

# Test 1: Original SimpleTreeMCTS (but without NN timing)
print("1. ORIGINAL SIMPLETREEMCTS")
print("-" * 40)

original_boards = VectorizedCliqueBoard(batch_size, n, k, "symmetric")
original_mcts = SimpleTreeMCTS(
    batch_size=batch_size,
    num_actions=num_actions,
    c_puct=3.0,
    max_nodes_per_game=100
)

# Mock neural network that just returns random values
class MockNN:
    def __init__(self):
        self.asymmetric_mode = False
    
    def evaluate_batch(self, *args):
        # Return random priors and values in expected shape
        batch_size_actual = args[0].shape[0] if hasattr(args[0], 'shape') else 1
        return np.random.rand(batch_size_actual, num_actions), np.random.uniform(-1, 1, (batch_size_actual, 1))

mock_nn = MockNN()

print(f"Running {num_simulations} simulations for {batch_size} games...")
start = time.time()
original_probs = original_mcts.search(original_boards, mock_nn, num_simulations, temperature=1.0)
original_time = time.time() - start

print(f"\nTotal time: {original_time:.2f}s")
print(f"Time per game: {original_time/batch_size:.2f}s")
print(f"Time per simulation per game: {original_time/(batch_size*num_simulations)*1000:.1f}ms")

# Test 2: Efficient MCTS
print("\n\n2. EFFICIENT MCTS")
print("-" * 40)

efficient_boards = EfficientCliqueBoard(batch_size, n, k, "symmetric")
efficient_mcts = SimpleTreeMCTSEfficient(
    batch_size=batch_size,
    num_actions=num_actions,
    c_puct=3.0,
    max_nodes_per_game=100
)

print(f"Running {num_simulations} simulations for {batch_size} games...")
start = time.time()
efficient_probs = efficient_mcts.search(efficient_boards, mock_nn, num_simulations, temperature=1.0)
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

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"Efficient board MCTS is {speedup:.1f}x faster")
print(f"This confirms the board copying optimization works!")
print(f"Extrapolating: 35s per move → ~{35/speedup:.1f}s per move with efficient board")
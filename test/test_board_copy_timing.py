#!/usr/bin/env python
"""Direct test of board copy performance in MCTS context."""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'jax_full_src'))

import time
import jax
import jax.numpy as jnp
import numpy as np

from jax_full_src.vectorized_board import VectorizedCliqueBoard
from jax_full_src.efficient_board_proper import EfficientCliqueBoard

# Test parameters
n = 6
k = 3
num_simulations = 50  # Reduced for faster test
num_copies_per_sim = 5  # Approximate copies per simulation

print("=" * 70)
print("BOARD COPY TIMING IN MCTS CONTEXT")
print("=" * 70)
print(f"Setup: n={n}, k={k}")
print(f"Simulating {num_simulations} MCTS simulations")
print(f"Approximately {num_copies_per_sim} board copies per simulation")
print()

# Test 1: Original VectorizedCliqueBoard
print("1. ORIGINAL BOARD COPIES")
print("-" * 40)

# Create initial board with some moves
original = VectorizedCliqueBoard(1, n, k, "symmetric")
for i in range(5):
    valid = np.where(original.get_valid_moves_mask()[0])[0]
    if len(valid) > 0:
        original.make_moves(jnp.array([valid[0]]))

# Time board copies as they would happen in MCTS
copy_times = []
total_start = time.time()

for sim in range(num_simulations):
    for copy_idx in range(num_copies_per_sim):
        copy_start = time.time()
        
        # Simulate MCTS board copy pattern
        child = VectorizedCliqueBoard(1, n, k, "symmetric")
        child.adjacency_matrices = original.adjacency_matrices.copy()
        child.current_players = original.current_players.copy()
        child.game_states = original.game_states.copy()
        child.winners = original.winners.copy()
        child.move_counts = original.move_counts.copy()
        
        # Make a move (as would happen in MCTS)
        valid = np.where(child.get_valid_moves_mask()[0])[0]
        if len(valid) > 0:
            child.make_moves(jnp.array([valid[copy_idx % len(valid)]]))
        
        copy_times.append(time.time() - copy_start)

original_total = time.time() - total_start
original_avg = np.mean(copy_times) * 1000

print(f"Total copies: {len(copy_times)}")
print(f"Total time: {original_total:.2f}s")
print(f"Average per copy: {original_avg:.2f}ms")
print(f"Estimated for 200 copies (20 sims): {200 * original_avg / 1000:.2f}s")

# Test 2: Efficient board
print("\n2. EFFICIENT BOARD COPIES")
print("-" * 40)

# Create initial board with same moves
efficient = EfficientCliqueBoard(1, n, k, "symmetric")
for i in range(5):
    valid = np.where(efficient.get_valid_moves_mask()[0])[0]
    if len(valid) > 0:
        efficient.make_moves(jnp.array([valid[0]]))

# Time board copies
copy_times = []
total_start = time.time()

for sim in range(num_simulations):
    for copy_idx in range(num_copies_per_sim):
        copy_start = time.time()
        
        # Efficient copy
        child = efficient.copy()
        
        # Make a move
        valid = np.where(child.get_valid_moves_mask()[0])[0]
        if len(valid) > 0:
            child.make_moves(jnp.array([valid[copy_idx % len(valid)]]))
        
        copy_times.append(time.time() - copy_start)

efficient_total = time.time() - total_start
efficient_avg = np.mean(copy_times) * 1000

print(f"Total copies: {len(copy_times)}")
print(f"Total time: {efficient_total:.2f}s")
print(f"Average per copy: {efficient_avg:.2f}ms")
print(f"Estimated for 200 copies (20 sims): {200 * efficient_avg / 1000:.2f}s")

# Comparison
print("\n3. COMPARISON")
print("-" * 40)
speedup = original_avg / efficient_avg
time_saved_per_sim = (original_avg - efficient_avg) * 20 / 1000  # 20 copies per sim
total_saved = original_total - efficient_total

print(f"Original: {original_avg:.2f}ms per copy")
print(f"Efficient: {efficient_avg:.2f}ms per copy")
print(f"Speedup: {speedup:.1f}x")
print(f"Time saved per simulation: {time_saved_per_sim:.2f}s")
print(f"Total time saved: {total_saved:.2f}s ({total_saved/original_total*100:.1f}%)")

# Estimate MCTS improvement
print("\n4. ESTIMATED MCTS IMPROVEMENT")
print("-" * 40)
board_copy_fraction = 0.73  # 73% of MCTS time is board copying
mcts_time_per_move = 3.5  # seconds (from logs)

new_board_fraction = board_copy_fraction / speedup
new_mcts_time = mcts_time_per_move * (1 - board_copy_fraction + new_board_fraction)

print(f"Original MCTS time per move: {mcts_time_per_move:.1f}s")
print(f"Board copying was: {board_copy_fraction*100:.0f}% of time")
print(f"With efficient board: {new_board_fraction*100:.0f}% of time")
print(f"Estimated new MCTS time: {new_mcts_time:.1f}s per move")
print(f"Overall speedup: {mcts_time_per_move/new_mcts_time:.1f}x")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"Efficient board provides {speedup:.1f}x speedup for copies")
print(f"This should reduce MCTS time from ~3.5s to ~{new_mcts_time:.1f}s per move")
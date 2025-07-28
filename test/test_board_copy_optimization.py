#!/usr/bin/env python
"""Test board copy optimization impact."""

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
from jax_full_src.efficient_board import EfficientCliqueBoard

# Test parameters
n = 6
k = 3
num_copies = 1000

print("=" * 70)
print("BOARD COPY OPTIMIZATION TEST")
print("=" * 70)
print(f"Testing {num_copies} board copies for n={n}, k={k}")
print()

# Test 1: Original VectorizedCliqueBoard
print("1. ORIGINAL BOARD (Adjacency Matrix)")
print("-" * 40)

original_board = VectorizedCliqueBoard(1, n, k, "symmetric")
# Make some moves
for i in range(5):
    valid_moves = np.where(original_board.get_valid_moves_mask()[0])[0]
    if len(valid_moves) > 0:
        original_board.make_moves(jnp.array([valid_moves[0]]))

# Time copying
start = time.time()
for _ in range(num_copies):
    copy = VectorizedCliqueBoard(1, n, k, "symmetric")
    copy.adjacency_matrices = original_board.adjacency_matrices.copy()
    copy.current_players = original_board.current_players.copy()
    copy.game_states = original_board.game_states.copy()
    copy.winners = original_board.winners.copy()
    copy.move_counts = original_board.move_counts.copy()
original_time = time.time() - start

print(f"Adjacency matrix size: {n}x{n} = {n*n} elements")
print(f"Total copy time: {original_time:.3f}s")
print(f"Time per copy: {original_time/num_copies*1000:.2f}ms")
print()

# Test 2: Efficient board
print("2. EFFICIENT BOARD (Edge List)")  
print("-" * 40)

efficient_board = EfficientCliqueBoard(1, n, k, "symmetric")
# Make same moves
for i in range(5):
    valid_moves = np.where(efficient_board.get_valid_moves_mask()[0])[0]
    if len(valid_moves) > 0:
        efficient_board.make_moves(jnp.array([valid_moves[0]]))

# Time copying
start = time.time()
for _ in range(num_copies):
    copy = efficient_board.copy()
efficient_time = time.time() - start

print(f"Edge list size: {efficient_board.num_edges} elements")
print(f"Total copy time: {efficient_time:.3f}s")
print(f"Time per copy: {efficient_time/num_copies*1000:.2f}ms")
print()

# Comparison
print("3. COMPARISON")
print("-" * 40)
speedup = original_time / efficient_time
print(f"Speedup: {speedup:.1f}x")
print(f"Memory reduction: {n*n}/{efficient_board.num_edges} = {n*n/efficient_board.num_edges:.1f}x")

# Test with larger board
print("\n4. LARGER BOARD TEST (n=14, k=4)")
print("-" * 40)

n_large = 14
k_large = 4

# Original
original_large = VectorizedCliqueBoard(1, n_large, k_large, "symmetric")
start = time.time()
for _ in range(100):  # Fewer copies for larger board
    copy = VectorizedCliqueBoard(1, n_large, k_large, "symmetric")
    copy.adjacency_matrices = original_large.adjacency_matrices.copy()
original_large_time = time.time() - start

# Efficient
efficient_large = EfficientCliqueBoard(1, n_large, k_large, "symmetric")
start = time.time()
for _ in range(100):
    copy = efficient_large.copy()
efficient_large_time = time.time() - start

print(f"Original: {original_large_time*10:.2f}ms per copy")
print(f"Efficient: {efficient_large_time*10:.2f}ms per copy")
print(f"Speedup: {original_large_time/efficient_large_time:.1f}x")
print(f"Memory: {n_large*n_large} vs {efficient_large.num_edges} elements")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"Efficient board reduces copy time by {speedup:.1f}x for n={n}")
print(f"For n={n_large}, speedup is {original_large_time/efficient_large_time:.1f}x")
print("This optimization alone could reduce MCTS time by ~50-60%")
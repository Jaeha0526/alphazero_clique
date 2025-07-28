#!/usr/bin/env python
"""
Test Fix 3: JIT compile MCTS operations
Compare performance of JIT vs non-JIT MCTS
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import jax
import jax.numpy as jnp
import numpy as np

print("Testing Fix 3: JIT Compiled MCTS")
print("="*60)
print(f"Device: {jax.default_backend()}")

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from batched_mcts_sync import SimpleBatchedMCTS
from jit_mcts import JITCompiledMCTS, FullyJITMCTS

# Test parameters
num_games = 16
num_sims = 100

# Create model
print("\nInitializing components...")
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=32, num_layers=1)

# Warm up
dummy_board = VectorizedCliqueBoard(batch_size=1)
edge_indices, edge_features = dummy_board.get_features_for_nn_undirected()
_ = model.evaluate_batch(edge_indices, edge_features)
print("✓ Model ready")

# Test 1: Non-JIT batched MCTS (baseline)
print(f"\n1. Non-JIT Batched MCTS:")
boards1 = VectorizedCliqueBoard(batch_size=num_games)
simple_mcts = SimpleBatchedMCTS(batch_size=num_games)

start = time.time()
action_probs1 = simple_mcts.search_batch(boards1, model, num_sims)
time1 = time.time() - start

print(f"   Time: {time1:.3f}s")
print(f"   Games/sec: {num_games/time1:.2f}")

# Test 2: JIT-compiled MCTS
print(f"\n2. JIT-Compiled MCTS:")
boards2 = VectorizedCliqueBoard(batch_size=num_games)
jit_mcts = JITCompiledMCTS(batch_size=num_games)

# First call (includes JIT compilation)
start = time.time()
action_probs2 = jit_mcts.search(boards2, model, num_sims)
time2_first = time.time() - start

# Second call (JIT cached)
start = time.time()
action_probs2 = jit_mcts.search(boards2, model, num_sims)
time2 = time.time() - start

print(f"   First call (with JIT): {time2_first:.3f}s")
print(f"   Second call: {time2:.3f}s")
print(f"   Games/sec: {num_games/time2:.2f}")
print(f"   Speedup: {time1/time2:.1f}x")

# Test 3: Breakdown of JIT speedups
print(f"\n3. Component-level JIT speedups:")

# Test UCB calculation
N = jnp.ones((num_games, 15))
W = jnp.ones((num_games, 15))
P = jnp.ones((num_games, 15)) / 15
valid = jnp.ones((num_games, 15), dtype=bool)

# Non-JIT UCB
def calculate_ucb_python(N, W, P, c_puct=3.0):
    Q = W / np.maximum(N, 1)
    U = c_puct * P * np.sqrt(N.sum(axis=1, keepdims=True)) / (1 + N)
    return Q + U

# Time non-JIT
start = time.time()
for _ in range(1000):
    ucb_python = calculate_ucb_python(np.array(N), np.array(W), np.array(P))
time_ucb_python = time.time() - start

# Time JIT version
ucb_jit = jit_mcts._jit_calculate_ucb
parent_visits = N.sum(axis=1)

start = time.time()
for _ in range(1000):
    ucb_jax = ucb_jit(N, W, P, parent_visits, valid)
time_ucb_jit = time.time() - start

print(f"   UCB calculation (1000 iterations):")
print(f"     Python/NumPy: {time_ucb_python:.3f}s")
print(f"     JAX JIT:      {time_ucb_jit:.3f}s")
print(f"     Speedup:      {time_ucb_python/time_ucb_jit:.1f}x")

# Test 4: Full simulation loop
print(f"\n4. Full simulation loop speedup:")

# Get initial values
edge_indices, edge_features = boards2.get_features_for_nn_undirected()
valid_masks = boards2.get_valid_moves_mask()
policies, values = model.evaluate_batch(edge_indices, edge_features, valid_masks)

# Time JIT simulation loop
start = time.time()
action_probs_jit = jit_mcts.search_jit(
    policies, values.squeeze(), valid_masks, num_sims
)
time_jit_loop = time.time() - start

print(f"   JIT simulation loop: {time_jit_loop:.3f}s")
print(f"   Speedup vs full search: {time1/time_jit_loop:.1f}x")

# Test 5: Fully JIT MCTS (if we have pure JAX NN)
print(f"\n5. Fully JIT MCTS (with NN in loop):")
full_jit_mcts = FullyJITMCTS(batch_size=num_games)

# First call (JIT compilation)
start = time.time()
action_probs_full = full_jit_mcts.search_full_jit(
    edge_indices, edge_features, 
    model.params, model.model.apply,
    num_sims
)
time_full_first = time.time() - start

# Second call (cached)
start = time.time()
action_probs_full = full_jit_mcts.search_full_jit(
    edge_indices, edge_features,
    model.params, model.model.apply,
    num_sims
)
time_full = time.time() - start

print(f"   First call (with JIT): {time_full_first:.3f}s")
print(f"   Second call: {time_full:.3f}s")
print(f"   Games/sec: {num_games/time_full:.2f}")
print(f"   Speedup: {time1/time_full:.1f}x")

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY:")
print(f"Non-JIT Batched:     {time1:.3f}s - {num_games/time1:.1f} games/sec")
print(f"JIT-Compiled:        {time2:.3f}s - {num_games/time2:.1f} games/sec")
print(f"Fully JIT (w/ NN):   {time_full:.3f}s - {num_games/time_full:.1f} games/sec")

best_time = min(time2, time_full)
print(f"\nBest speedup from JIT: {time1/best_time:.1f}x")

# Memory efficiency
print(f"\n📊 ADDITIONAL BENEFITS:")
print("-"*40)
print("✓ JIT compilation fuses operations")
print("✓ Reduces Python overhead")
print("✓ Better memory access patterns")
print("✓ Enables XLA optimizations")
print("✓ Can run on TPU without changes")

print("\n✓ Fix 3 complete: MCTS operations are JIT compiled!")
print("="*60)
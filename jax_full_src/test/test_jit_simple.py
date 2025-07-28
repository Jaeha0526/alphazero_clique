#!/usr/bin/env python
"""
Test Fix 3: Simple JIT compilation test
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import jax
import jax.numpy as jnp
import numpy as np

print("Testing Fix 3: JIT Compiled MCTS (Simplified)")
print("="*60)
print(f"Device: {jax.default_backend()}")

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from batched_mcts_sync import SimpleBatchedMCTS
from jit_mcts_simple import SimpleJITMCTS, VectorizedJITMCTS

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

# Test 1: Non-JIT baseline
print(f"\n1. Non-JIT Batched MCTS (baseline):")
boards1 = VectorizedCliqueBoard(batch_size=num_games)
simple_mcts = SimpleBatchedMCTS(batch_size=num_games)

start = time.time()
action_probs1 = simple_mcts.search_batch(boards1, model, num_sims)
time1 = time.time() - start

print(f"   Time: {time1:.3f}s")
print(f"   Games/sec: {num_games/time1:.2f}")

# Test 2: Simple JIT MCTS
print(f"\n2. Simple JIT MCTS (hot paths compiled):")
boards2 = VectorizedCliqueBoard(batch_size=num_games)
jit_mcts = SimpleJITMCTS(batch_size=num_games)

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

# Test 3: Fully vectorized JIT MCTS
print(f"\n3. Fully Vectorized JIT MCTS:")
boards3 = VectorizedCliqueBoard(batch_size=num_games)
vec_mcts = VectorizedJITMCTS(batch_size=num_games)

# First call
start = time.time()
action_probs3 = vec_mcts.search(boards3, model, num_sims)
time3_first = time.time() - start

# Second call
start = time.time()
action_probs3 = vec_mcts.search(boards3, model, num_sims)
time3 = time.time() - start

print(f"   First call (with JIT): {time3_first:.3f}s")
print(f"   Second call: {time3:.3f}s")
print(f"   Games/sec: {num_games/time3:.2f}")
print(f"   Speedup: {time1/time3:.1f}x")

# Test individual operations
print(f"\n4. Breakdown - UCB calculation speedup:")

# Prepare data
N = jnp.ones((num_games, 15))
W = jnp.ones((num_games, 15))
P = jnp.ones((num_games, 15)) / 15
valid = jnp.ones((num_games, 15), dtype=bool)

# Non-JIT version
def ucb_python(N, W, P, c_puct=3.0):
    Q = W / np.maximum(N, 1)
    U = c_puct * P * np.sqrt(N.sum(axis=1, keepdims=True)) / (1 + N)
    return Q + U

# Time comparison
iterations = 1000

start = time.time()
for _ in range(iterations):
    _ = ucb_python(np.array(N), np.array(W), np.array(P))
time_python = time.time() - start

start = time.time()
for _ in range(iterations):
    _ = jit_mcts._ucb_and_select(N, W, P, valid)
time_jit = time.time() - start

print(f"   Python/NumPy ({iterations} calls): {time_python:.3f}s")
print(f"   JAX JIT ({iterations} calls): {time_jit:.3f}s")
print(f"   Speedup: {time_python/time_jit:.1f}x")

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY:")
print(f"Non-JIT:              {time1:.3f}s - {num_games/time1:.1f} games/sec")
print(f"Simple JIT:           {time2:.3f}s - {num_games/time2:.1f} games/sec")
print(f"Vectorized JIT:       {time3:.3f}s - {num_games/time3:.1f} games/sec")

best_time = min(time2, time3)
print(f"\nBest JIT speedup: {time1/best_time:.1f}x")

# Benefits
print(f"\n📊 JIT COMPILATION BENEFITS:")
print("-"*40)
print("✓ Fuses multiple operations into single kernel")
print("✓ Eliminates Python overhead in loops")
print("✓ Enables GPU-specific optimizations")
print("✓ Better memory access patterns")
print("✓ Can target TPUs without code changes")

# Projection
scale_factor = (100/num_games) * (100/num_sims)
print(f"\nProjected for 100 games, 100 simulations:")
print(f"  Non-JIT:     {time1 * scale_factor:.0f}s")
print(f"  Best JIT:    {best_time * scale_factor:.0f}s")
print(f"  Speedup:     {time1/best_time:.1f}x")

print("\n✓ Fix 3 implemented: MCTS operations are JIT compiled!")
print("="*60)
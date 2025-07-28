#!/usr/bin/env python
"""
Minimal example showing vectorization benefit
"""

import sys
sys.path.append('/workspace/alphazero_clique/jax_full_src')

import jax
import jax.numpy as jnp
import time

print("Minimal Vectorization Example")
print("="*40)

# Simple example: process multiple inputs in parallel
def simple_function(x):
    """Simulate some computation"""
    return jnp.sum(x ** 2)

# Sequential processing
print("\n1. Sequential processing:")
inputs = [jnp.ones((100, 100)) for _ in range(32)]

start = time.time()
results_seq = []
for x in inputs:
    results_seq.append(simple_function(x))
seq_time = time.time() - start
print(f"   Time: {seq_time:.3f}s")

# Vectorized processing using vmap
print("\n2. Vectorized processing (vmap):")
stacked_inputs = jnp.stack(inputs)  # Shape: (32, 100, 100)

# Vectorize the function
vectorized_fn = jax.vmap(simple_function)

start = time.time()
results_vec = vectorized_fn(stacked_inputs)
vec_time = time.time() - start
print(f"   Time: {vec_time:.3f}s")
print(f"   Speedup: {seq_time/vec_time:.1f}x")

# The concept for MCTS
print("\n3. How this applies to MCTS:")
print("-" * 40)
print("Current implementation:")
print("  for game in games:")
print("    run_mcts(game)  # Sequential!")
print("\nProper JAX implementation:")
print("  vmap(run_mcts)(games)  # Parallel!")
print("\nThe key insight: JAX can process multiple games")
print("simultaneously on the GPU using vectorization!")

# Show the specific problem
print("\n4. The specific issue in tree_based_mcts.py:")
print("-" * 40)
print("Line 348: for game_idx in range(self.batch_size):")
print("This Python for loop prevents GPU parallelization!")
print("\nInstead, we should use JAX's functional approach")
print("to process all games at once on the GPU.")
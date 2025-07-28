#!/usr/bin/env python
"""
Test JIT compilation timing
"""

import jax
import jax.numpy as jnp
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

print("Testing JIT compilation timing...")
print(f"JAX backend: {jax.default_backend()}")

# Test 1: Simple JIT
print("\n1. Simple JIT function...")
@jax.jit
def simple_fn(x):
    return jnp.sum(x ** 2)

start = time.time()
x = jnp.ones((100, 100))
result = simple_fn(x)
elapsed = time.time() - start
print(f"✓ First call (compilation): {elapsed:.3f}s")

start = time.time()
result = simple_fn(x)
elapsed = time.time() - start
print(f"✓ Second call (cached): {elapsed:.3f}s")

# Test 2: Complex JIT
print("\n2. Complex JIT function...")
@jax.jit
def complex_fn(x, y):
    for _ in range(10):
        x = jnp.dot(x, y)
        x = jax.nn.relu(x)
    return x

start = time.time()
x = jnp.ones((50, 50))
y = jnp.ones((50, 50))
result = complex_fn(x, y)
elapsed = time.time() - start
print(f"✓ First call (compilation): {elapsed:.3f}s")

# Test 3: Neural network initialization
print("\n3. Neural network initialization...")
try:
    from vectorized_nn import ImprovedBatchedNeuralNetwork
    
    start = time.time()
    model = ImprovedBatchedNeuralNetwork(
        num_vertices=6,
        hidden_dim=32,
        num_layers=2,
        asymmetric_mode=False
    )
    elapsed = time.time() - start
    print(f"✓ Model initialization: {elapsed:.3f}s")
    
    # Test evaluation
    from vectorized_board import VectorizedCliqueBoard
    board = VectorizedCliqueBoard(batch_size=1)
    edge_indices, edge_features = board.get_features_for_nn_undirected()
    
    start = time.time()
    policy, value = model.evaluate_batch(edge_indices, edge_features)
    elapsed = time.time() - start
    print(f"✓ First evaluation (compilation): {elapsed:.3f}s")
    
    start = time.time()
    policy, value = model.evaluate_batch(edge_indices, edge_features)
    elapsed = time.time() - start
    print(f"✓ Second evaluation (cached): {elapsed:.3f}s")
    
except Exception as e:
    print(f"✗ Neural network test failed: {e}")
    import traceback
    traceback.print_exc()

print("\nJIT compilation tests completed!")
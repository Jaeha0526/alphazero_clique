#!/usr/bin/env python
"""
Test creating the model without JIT pre-compilation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import time

print("Testing model creation without pre-compilation...")

# Import just the model creation function
from vectorized_nn import create_improved_model, ImprovedVectorizedCliqueGNN

# Test 1: Create the model directly
print("\n1. Creating model directly...")
start = time.time()
model, params = create_improved_model(
    num_vertices=6,
    hidden_dim=32,
    num_layers=1,
    asymmetric_mode=False
)
print(f"✓ Model created in {time.time()-start:.3f}s")

# Test 2: Test forward pass without JIT
print("\n2. Testing forward pass without JIT...")
# Create dummy input
edge_list = []
for i in range(6):
    for j in range(i+1, 6):
        edge_list.append([i, j])

edge_index = jnp.array(edge_list, dtype=jnp.int32).T[None, :, :]
edge_features = jnp.ones((1, 15, 3), dtype=jnp.float32)

start = time.time()
output = model.apply(params, edge_index, edge_features, deterministic=True)
print(f"✓ Forward pass completed in {time.time()-start:.3f}s")
print(f"  Output: policies shape {output[0].shape}, values shape {output[1].shape}")

# Test 3: Test JIT compilation separately
print("\n3. Testing JIT compilation...")

@jax.jit
def forward(params, edge_index, edge_features):
    return model.apply(params, edge_index, edge_features, deterministic=True)

start = time.time()
output = forward(params, edge_index, edge_features)
print(f"✓ JIT compilation and execution completed in {time.time()-start:.3f}s")

print("\nModel works correctly!")
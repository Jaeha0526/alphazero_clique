#!/usr/bin/env python
"""
Debug neural network initialization with print statements
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import jit
import flax.linen as nn
from typing import Tuple, Dict

print("Starting debug...")

# Manually create the model to see where it hangs
print("1. Creating model class...")

# Import the base classes
from vectorized_nn import ImprovedVectorizedCliqueGNN

print("2. Creating model instance...")
model = ImprovedVectorizedCliqueGNN(
    num_vertices=6,
    hidden_dim=32,
    num_layers=1,
    asymmetric_mode=False
)
print("✓ Model instance created")

print("3. Creating dummy input...")
edge_list = []
for i in range(6):
    for j in range(i+1, 6):
        edge_list.append([i, j])

dummy_edge_index = jnp.array(edge_list, dtype=jnp.int32).T[None, :, :]
dummy_edge_features = jnp.zeros((1, 15, 3), dtype=jnp.float32)

print("4. Initializing model parameters...")
key = jax.random.PRNGKey(0)
params = model.init({'params': key, 'dropout': key}, dummy_edge_index, dummy_edge_features)
print("✓ Parameters initialized")

print("5. Testing forward pass...")
output = model.apply(params, dummy_edge_index, dummy_edge_features, deterministic=True)
print(f"✓ Forward pass successful: {output[0].shape}, {output[1].shape}")

print("6. Testing JIT compilation...")
# Don't use mutable in JIT for now
@jit
def forward_fn(params, edge_index, edge_features):
    return model.apply(params, edge_index, edge_features, deterministic=True)

print("   Compiling JIT function...")
output = forward_fn(params, dummy_edge_index, dummy_edge_features)
print("✓ JIT compilation successful")

print("\nDebug completed successfully!")
#!/usr/bin/env python
"""
Debug timeout issue
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

print("Starting debug script...")
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from vectorized_board import VectorizedCliqueBoard
    print("✓ VectorizedCliqueBoard imported")
except Exception as e:
    print(f"✗ Failed to import VectorizedCliqueBoard: {e}")
    sys.exit(1)

try:
    from vectorized_nn import ImprovedBatchedNeuralNetwork
    print("✓ ImprovedBatchedNeuralNetwork imported")
except Exception as e:
    print(f"✗ Failed to import ImprovedBatchedNeuralNetwork: {e}")
    sys.exit(1)

try:
    from vectorized_self_play_fixed import FixedVectorizedSelfPlay, FixedSelfPlayConfig
    print("✓ FixedVectorizedSelfPlay imported")
except Exception as e:
    print(f"✗ Failed to import FixedVectorizedSelfPlay: {e}")
    sys.exit(1)

# Test 2: Simple JAX operations
print("\n2. Testing JAX operations...")
try:
    x = jnp.ones((100, 100))
    y = jnp.dot(x, x)
    print(f"✓ JAX operations work, result shape: {y.shape}")
except Exception as e:
    print(f"✗ JAX operations failed: {e}")

# Test 3: Initialize board
print("\n3. Testing board initialization...")
try:
    board = VectorizedCliqueBoard(batch_size=1, num_vertices=6, k=3)
    print("✓ Board initialized")
    print(f"  Board shape: {board.adjacency_matrices.shape}")
    print(f"  Edge states shape: {board.edge_states.shape}")
except Exception as e:
    print(f"✗ Board initialization failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Initialize neural network
print("\n4. Testing neural network initialization...")
try:
    start = time.time()
    model = ImprovedBatchedNeuralNetwork(
        num_vertices=6,
        hidden_dim=32,
        num_layers=2,
        asymmetric_mode=False
    )
    elapsed = time.time() - start
    print(f"✓ Neural network initialized in {elapsed:.2f}s")
except Exception as e:
    print(f"✗ Neural network initialization failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDebug script completed successfully!")
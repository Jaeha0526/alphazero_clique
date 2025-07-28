#!/usr/bin/env python
"""Test jnp.where behavior."""

import jax.numpy as jnp

# Test jnp.where behavior
print("Testing jnp.where...")

# Simple test
arr = jnp.array([True, False, True, False, True])
print(f"Array: {arr}")

indices = jnp.where(arr)
print(f"jnp.where(arr): {indices}")
print(f"Type: {type(indices)}")

indices_0 = jnp.where(arr)[0]
print(f"jnp.where(arr)[0]: {indices_0}")
print(f"Type: {type(indices_0)}")

# Try iterating
print("\nIterating over indices:")
for i, idx in enumerate(indices_0):
    print(f"  {i}: {idx} (type: {type(idx)})")
    if i > 5:  # Safety limit
        print("  ... stopping after 5 iterations")
        break

# Test with all False
print("\n\nTest with all False:")
arr_false = jnp.array([False, False, False])
indices_false = jnp.where(arr_false)[0]
print(f"jnp.where(all_false)[0]: {indices_false}")
print(f"Length: {len(indices_false)}")

# Test any() behavior
print("\n\nTest any() behavior:")
print(f"arr.any(): {arr.any()}")
print(f"arr_false.any(): {arr_false.any()}")
print(f"bool(arr.any()): {bool(arr.any())}")
print(f"bool(arr_false.any()): {bool(arr_false.any())}")
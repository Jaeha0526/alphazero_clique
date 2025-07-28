#!/usr/bin/env python
"""
Test a fix for the neural network JIT compilation issue
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple
import time

print("Testing neural network JIT compilation fix...")
print(f"JAX backend: {jax.default_backend()}")

# Simple test model with batch norm
class TestModel(nn.Module):
    hidden_dim: int = 32
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# Initialize model
model = TestModel()
key = jax.random.PRNGKey(0)
dummy_input = jnp.ones((4, 10))

print("\n1. Initializing model...")
start = time.time()
variables = model.init(key, dummy_input)
print(f"✓ Model initialized in {time.time()-start:.3f}s")

# Test without JIT first
print("\n2. Testing without JIT...")
start = time.time()
output, updates = model.apply(variables, dummy_input, mutable=['batch_stats'])
print(f"✓ Forward pass without JIT: {time.time()-start:.3f}s")

# Create properly JIT-compiled function
print("\n3. Creating JIT-compiled function...")
@jax.jit
def jit_apply(variables, x):
    # For inference, use running averages
    return model.apply(variables, x, training=False)

# For training with batch stats
@jax.jit
def jit_train_step(variables, x):
    def loss_fn(params):
        output, updates = model.apply(
            {'params': params, 'batch_stats': variables.get('batch_stats', {})},
            x, 
            training=True,
            mutable=['batch_stats']
        )
        return jnp.mean(output**2), updates
    
    (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables['params'])
    return loss, grads, updates

print("\n4. Testing JIT-compiled inference...")
start = time.time()
output = jit_apply(variables, dummy_input)
print(f"✓ First JIT inference (compilation): {time.time()-start:.3f}s")

start = time.time()
output = jit_apply(variables, dummy_input)
print(f"✓ Second JIT inference (cached): {time.time()-start:.3f}s")

print("\n5. Testing JIT-compiled training...")
start = time.time()
loss, grads, updates = jit_train_step(variables, dummy_input)
print(f"✓ First JIT train step (compilation): {time.time()-start:.3f}s")

start = time.time()
loss, grads, updates = jit_train_step(variables, dummy_input)
print(f"✓ Second JIT train step (cached): {time.time()-start:.3f}s")

print("\nJIT compilation test completed successfully!")
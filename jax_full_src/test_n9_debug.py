#!/usr/bin/env python
"""
Debug test for n=9, k=4 configuration
"""

import jax
import jax.numpy as jnp
import numpy as np
import time

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from jit_mcts_simple import VectorizedJITMCTS

print(f"JAX devices: {jax.devices()}")
print(f"Using device: {jax.default_backend()}")

# Test configuration
n = 9
k = 4
batch_size = 4
mcts_sims = 5

print(f"\nTest configuration: n={n}, k={k}, batch_size={batch_size}, mcts_sims={mcts_sims}")

# Create model
print("\nCreating neural network...")
model = ImprovedBatchedNeuralNetwork(
    num_vertices=n,
    hidden_dim=64,
    num_layers=3,
    asymmetric_mode=False  # symmetric
)

# Create boards
print("\nCreating boards...")
boards = VectorizedCliqueBoard(
    batch_size=batch_size,
    num_vertices=n,
    k=k,
    game_mode="symmetric"
)

# Create MCTS
num_actions = n * (n - 1) // 2
print(f"\nCreating MCTS with {num_actions} actions...")
mcts = VectorizedJITMCTS(
    batch_size=batch_size,
    num_actions=num_actions,
    c_puct=3.0
)

# Test one MCTS search
print("\nTesting MCTS search...")
start = time.time()
try:
    action_probs = mcts.search(
        boards,
        model,
        mcts_sims,
        temperature=1.0
    )
    print(f"MCTS search completed in {time.time() - start:.2f}s")
    print(f"Action probs shape: {action_probs.shape}")
except Exception as e:
    print(f"ERROR in MCTS search: {e}")
    import traceback
    traceback.print_exc()

# Test playing one move
print("\nTesting one move...")
try:
    actions = []
    for i in range(batch_size):
        action = np.random.choice(num_actions, p=np.array(action_probs[i]))
        actions.append(action)
    
    boards.make_moves(jnp.array(actions))
    print(f"Move completed successfully")
    print(f"Game states: {boards.game_states}")
except Exception as e:
    print(f"ERROR in making moves: {e}")
    import traceback
    traceback.print_exc()
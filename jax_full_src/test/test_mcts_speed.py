#!/usr/bin/env python
"""
Test MCTS speed to find bottleneck
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import jax
import jax.numpy as jnp

print("Testing MCTS performance...")
print(f"Device: {jax.default_backend()}")

# 1. Test board operations
print("\n1. Testing board operations...")
from vectorized_board import VectorizedCliqueBoard

start = time.time()
board = VectorizedCliqueBoard(batch_size=1)
print(f"   Board creation: {time.time()-start:.3f}s")

start = time.time()
edge_indices, edge_features = board.get_features_for_nn_undirected()
print(f"   Get features: {time.time()-start:.3f}s")

start = time.time()
valid_mask = board.get_valid_moves_mask()
print(f"   Get valid moves: {time.time()-start:.3f}s")

start = time.time()
board_copy = board.copy_board(0)
print(f"   Copy single board: {time.time()-start:.3f}s")

# 2. Test neural network
print("\n2. Testing neural network...")
from vectorized_nn import ImprovedBatchedNeuralNetwork

start = time.time()
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=32, num_layers=1)
print(f"   Model creation: {time.time()-start:.3f}s")

start = time.time()
policies, values = model.evaluate_batch(edge_indices, edge_features)
print(f"   First evaluation (with JIT): {time.time()-start:.3f}s")

start = time.time()
policies, values = model.evaluate_batch(edge_indices, edge_features)
print(f"   Second evaluation: {time.time()-start:.3f}s")

# 3. Test basic MCTS operations
print("\n3. Testing MCTS tree operations...")
from tree_based_mcts import MCTSNode

start = time.time()
root = MCTSNode(board)
print(f"   Create root node: {time.time()-start:.3f}s")

# 4. Test a single MCTS simulation
print("\n4. Running single MCTS simulation...")
from tree_based_mcts import TreeBasedMCTS

mcts = TreeBasedMCTS(num_actions=15, c_puct=3.0)

start = time.time()
try:
    # Just test the expand operation
    mcts._expand_node(root, model)
    print(f"   Node expansion: {time.time()-start:.3f}s")
except Exception as e:
    print(f"   Error during expansion: {e}")

print("\nPerformance test completed!")
#!/usr/bin/env python
"""
Quick test of batched MCTS concept
"""

import sys
sys.path.append('/workspace/alphazero_clique/jax_full_src')

import time
import jax.numpy as jnp
import numpy as np

print("Quick Batched MCTS Test")
print("="*40)

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from batched_mcts_sync import SimpleBatchedMCTS

# Small test
num_games = 4
num_sims = 5

# Create model
print("\nInitializing...")
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=32, num_layers=1)

# Test SimpleBatchedMCTS
print("\nTesting SimpleBatchedMCTS...")
boards = VectorizedCliqueBoard(batch_size=num_games)
simple_mcts = SimpleBatchedMCTS(batch_size=num_games)

start = time.time()
action_probs = simple_mcts.search_batch(boards, model, num_sims)
elapsed = time.time() - start

print(f"Time: {elapsed:.3f}s")
print(f"Output shape: {action_probs.shape}")
print(f"Sum of probs per game: {jnp.sum(action_probs, axis=1)}")

# Test concept: compare sequential vs batched NN calls
print("\n\nComparing NN evaluation approaches:")
print("-"*40)

# Sequential NN calls
print("1. Sequential NN evaluations:")
start = time.time()
for i in range(num_games * num_sims):
    single_board = VectorizedCliqueBoard(batch_size=1)
    edge_indices, edge_features = single_board.get_features_for_nn_undirected()
    valid_masks = single_board.get_valid_moves_mask()
    _ = model.evaluate_batch(edge_indices, edge_features, valid_masks)
seq_time = time.time() - start
print(f"   {num_games * num_sims} sequential calls: {seq_time:.3f}s")

# Batched NN calls
print("\n2. Batched NN evaluations:")
start = time.time()
for i in range(num_sims):
    batch_board = VectorizedCliqueBoard(batch_size=num_games)
    edge_indices, edge_features = batch_board.get_features_for_nn_undirected()
    valid_masks = batch_board.get_valid_moves_mask()
    _ = model.evaluate_batch(edge_indices, edge_features, valid_masks)
batch_time = time.time() - start
print(f"   {num_sims} batched calls (batch size {num_games}): {batch_time:.3f}s")

print(f"\nSpeedup from batching: {seq_time/batch_time:.1f}x")
print("\n✓ Batching NN evaluations is key to speedup!")
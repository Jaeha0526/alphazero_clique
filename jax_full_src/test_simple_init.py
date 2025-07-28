#!/usr/bin/env python
"""Test SimpleTreeMCTS initialization."""

import numpy as np
import time
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork

print("1. Creating board...")
boards = VectorizedCliqueBoard(batch_size=1, num_vertices=9, k=4, game_mode="symmetric")
print("   Board created")

print("\n2. Creating NN...")
nn = ImprovedBatchedNeuralNetwork(num_vertices=9, hidden_dim=64, num_layers=3, asymmetric_mode=False)
print("   NN created")

print("\n3. Getting features...")
features = boards.get_features_for_nn_undirected()
valid_mask = boards.get_valid_moves_mask()
print(f"   Features: {len(features)} items")

print("\n4. Evaluating with NN...")
start = time.time()
policies, values = nn.evaluate_batch(*features, valid_mask)
elapsed = time.time() - start
print(f"   NN evaluation took {elapsed:.3f}s")
print(f"   Policies shape: {policies.shape}")
print(f"   Values shape: {values.shape}")

print("\nAll basic operations work!")
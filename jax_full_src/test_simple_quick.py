#!/usr/bin/env python
"""Quick test of SimpleTreeMCTS with minimal setup."""

import numpy as np
import time
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from simple_tree_mcts import SimpleTreeMCTS

# Test with 1 game, 2 simulations
boards = VectorizedCliqueBoard(batch_size=1, num_vertices=9, k=4, game_mode="symmetric")
nn = ImprovedBatchedNeuralNetwork(num_vertices=9, hidden_dim=64, num_layers=3, asymmetric_mode=False)
mcts = SimpleTreeMCTS(batch_size=1, num_actions=36, c_puct=3.0, max_nodes_per_game=10)

print("Running MCTS search with 1 game, 2 simulations...")
start = time.time()
probs = mcts.search(boards, nn, num_simulations=2, temperature=1.0)
elapsed = time.time() - start

print(f"Completed in {elapsed:.2f}s")
print(f"Probs shape: {probs.shape}")
print(f"Probs: {probs}")
print(f"Sum: {probs.sum()}")
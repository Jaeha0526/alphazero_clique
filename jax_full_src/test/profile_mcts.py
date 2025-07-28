#!/usr/bin/env python
"""
Profile MCTS to find bottlenecks
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import cProfile
import pstats
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from tree_based_mcts import TreeBasedMCTS

# Create components
board = VectorizedCliqueBoard(batch_size=1)
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=32, num_layers=1)
mcts = TreeBasedMCTS(num_actions=15, c_puct=3.0)

# Warm up
edge_indices, edge_features = board.get_features_for_nn_undirected()
_ = model.evaluate_batch(edge_indices, edge_features)

print("Profiling MCTS with 20 simulations...")

# Profile MCTS
profiler = cProfile.Profile()
profiler.enable()

action_probs = mcts.search(board, model, num_simulations=20, game_idx=0)

profiler.disable()

# Print results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
print("\nTop 10 time-consuming functions:")
stats.print_stats(10)
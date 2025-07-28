#!/usr/bin/env python
"""
Benchmark MCTS speed
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import jax
import numpy as np

print("Benchmarking MCTS speed...")
print(f"Device: {jax.default_backend()}")

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from tree_based_mcts import TreeBasedMCTS

# Create components
print("\nInitializing components...")
board = VectorizedCliqueBoard(batch_size=1)
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=64, num_layers=2)
mcts = TreeBasedMCTS(num_actions=15, c_puct=3.0)

# Warm up the model
print("\nWarming up neural network...")
edge_indices, edge_features = board.get_features_for_nn_undirected()
_ = model.evaluate_batch(edge_indices, edge_features)

# Benchmark different numbers of simulations
for num_sims in [10, 50, 100]:
    print(f"\nTesting {num_sims} MCTS simulations...")
    
    start = time.time()
    action_probs = mcts.search(board, model, num_simulations=num_sims, game_idx=0)
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {num_sims/elapsed:.1f} simulations/second")
    print(f"  Time per simulation: {elapsed/num_sims*1000:.1f}ms")

# Test a full game
print("\n\nPlaying a full game with 50 simulations per move...")
board = VectorizedCliqueBoard(batch_size=1)
move_times = []

start_game = time.time()
move_count = 0
while board.game_states[0] == 0 and move_count < 15:
    start_move = time.time()
    
    # Get action from MCTS
    action_probs = mcts.search(board, model, num_simulations=50, game_idx=0)
    action = np.argmax(action_probs)
    
    # Make move
    board.make_moves(np.array([action]))
    
    move_time = time.time() - start_move
    move_times.append(move_time)
    print(f"  Move {move_count + 1}: {move_time:.2f}s")
    
    move_count += 1

game_time = time.time() - start_game
print(f"\nGame completed in {game_time:.2f}s")
print(f"Average time per move: {np.mean(move_times):.2f}s")
print(f"Moves per second: {len(move_times)/game_time:.2f}")

# Estimate time for 100 games
est_time = game_time * 100
print(f"\nEstimated time for 100 games: {est_time:.0f}s ({est_time/60:.1f} minutes)")
#!/usr/bin/env python
"""
Benchmark the optimized pipeline to show vectorization benefits
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import jax
import jax.numpy as jnp
import numpy as np

print("Benchmarking Optimized AlphaZero Pipeline")
print("="*60)
print(f"Device: {jax.default_backend()}")
print("")

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from vectorized_self_play_optimized import OptimizedVectorizedSelfPlay, OptimizedSelfPlayConfig

# Parameters as requested
num_games = 30
batch_size = 16
mcts_sims = 50

print(f"Parameters:")
print(f"  Games: {num_games}")
print(f"  Batch size: {batch_size}")
print(f"  MCTS simulations: {mcts_sims}")
print("")

# Create neural network
print("Initializing neural network...")
model = ImprovedBatchedNeuralNetwork(
    num_vertices=6,
    hidden_dim=64,
    num_layers=3,
    asymmetric_mode=True
)

# Warm up with dummy data
dummy_board = VectorizedCliqueBoard(batch_size=1)
edge_indices, edge_features = dummy_board.get_features_for_nn_undirected()
_ = model.evaluate_batch(edge_indices, edge_features)
print("✓ Model ready")

# Configure self-play
config = OptimizedSelfPlayConfig(
    batch_size=batch_size,
    mcts_simulations=mcts_sims,
    game_mode="asymmetric",
    use_synchronized_mcts=False  # Use JIT MCTS
)

self_play = OptimizedVectorizedSelfPlay(config)

# Run 1: First run includes JIT compilation
print("\n" + "-"*60)
print("Run 1: First run (includes JIT compilation)")
start = time.time()
game_data1 = self_play.play_games(model, num_games, verbose=True)
time1 = time.time() - start

# Run 2: Second run with cached JIT
print("\n" + "-"*60)
print("Run 2: Second run (JIT cached)")
start = time.time()
game_data2 = self_play.play_games(model, num_games, verbose=True)
time2 = time.time() - start

# Run 3: Third run to confirm timing
print("\n" + "-"*60)
print("Run 3: Third run (confirming performance)")
start = time.time()
game_data3 = self_play.play_games(model, num_games, verbose=True)
time3 = time.time() - start

# Summary
print("\n" + "="*60)
print("BENCHMARK RESULTS:")
print("="*60)
print(f"\nRun 1 (with JIT): {time1:.1f}s ({num_games/time1:.1f} games/sec)")
print(f"Run 2 (cached):   {time2:.1f}s ({num_games/time2:.1f} games/sec)")
print(f"Run 3 (cached):   {time3:.1f}s ({num_games/time3:.1f} games/sec)")

avg_cached = (time2 + time3) / 2
print(f"\nAverage (cached): {avg_cached:.1f}s ({num_games/avg_cached:.1f} games/sec)")

# Performance metrics
print(f"\n📊 PERFORMANCE METRICS:")
print("-"*40)
print(f"Games per second: {num_games/avg_cached:.1f}")
print(f"Seconds per game: {avg_cached/num_games:.2f}")
print(f"MCTS sims/second: {num_games * mcts_sims * 15 / avg_cached:.0f}")  # Approx 15 moves per game

# Projection
print(f"\n🚀 PROJECTIONS:")
print("-"*40)
games_1000 = 1000
time_1000 = avg_cached * (games_1000 / num_games)
print(f"1000 games would take: {time_1000:.0f}s ({time_1000/60:.1f} minutes)")

games_per_hour = 3600 / avg_cached * num_games
print(f"Games per hour: {games_per_hour:.0f}")

print("\n✓ Benchmark complete!")
print("="*60)
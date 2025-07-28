#!/usr/bin/env python
"""
Test the full optimized pipeline performance
Shows the speedup after JIT compilation
"""

import sys
sys.path.append('/workspace/alphazero_clique/jax_full_src')

import time
import jax
import jax.numpy as jnp

print("Testing Optimized Pipeline Performance")
print("="*60)
print(f"Device: {jax.default_backend()}")

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from vectorized_self_play_optimized import OptimizedVectorizedSelfPlay, OptimizedSelfPlayConfig
from train_jax import train_network_jax

# Test parameters
num_games_warmup = 4
num_games_test = 30
batch_size = 16
mcts_sims = 50

print(f"\nTest parameters:")
print(f"  Warmup games: {num_games_warmup}")
print(f"  Test games: {num_games_test}")
print(f"  Batch size: {batch_size}")
print(f"  MCTS simulations: {mcts_sims}")

# Create components
print("\nInitializing components...")
model = ImprovedBatchedNeuralNetwork(
    num_vertices=6,
    hidden_dim=64,
    num_layers=3,
    asymmetric_mode=True
)

config = OptimizedSelfPlayConfig(
    batch_size=batch_size,
    mcts_simulations=mcts_sims,
    game_mode="asymmetric"
)

self_play = OptimizedVectorizedSelfPlay(config)
print("✓ Ready")

# Warmup run
print("\n" + "-"*60)
print("WARMUP: JIT compilation...")
start = time.time()
warmup_data = self_play.play_games(model, num_games_warmup, verbose=False)
warmup_time = time.time() - start
print(f"Warmup completed: {num_games_warmup} games in {warmup_time:.1f}s")

# Test run 1
print("\n" + "-"*60)
print(f"TEST 1: {num_games_test} games (JIT cached)")
start = time.time()
game_data1 = self_play.play_games(model, num_games_test, verbose=True)
time1 = time.time() - start

# Test run 2
print("\n" + "-"*60)
print(f"TEST 2: {num_games_test} games (confirming performance)")
start = time.time()
game_data2 = self_play.play_games(model, num_games_test, verbose=True)
time2 = time.time() - start

# Test training speed
print("\n" + "-"*60)
print("TRAINING: Testing training performance")
print(f"Training on {len(game_data1)} examples...")
start = time.time()
train_state, policy_loss, value_loss = train_network_jax(
    model,
    game_data1,
    epochs=5,
    batch_size=32,
    learning_rate=0.001,
    verbose=False
)
train_time = time.time() - start
print(f"Training completed in {train_time:.1f}s")
print(f"Examples/second: {len(game_data1) * 5 / train_time:.0f}")

# Summary
print("\n" + "="*60)
print("PERFORMANCE SUMMARY:")
print("="*60)
print(f"\nSelf-play performance (after JIT):")
print(f"  Run 1: {time1:.1f}s ({num_games_test/time1:.1f} games/sec)")
print(f"  Run 2: {time2:.1f}s ({num_games_test/time2:.1f} games/sec)")

avg_time = (time1 + time2) / 2
print(f"\n  Average: {avg_time:.1f}s ({num_games_test/avg_time:.1f} games/sec)")

# Detailed metrics
total_moves = sum(len(g) for g in game_data1)
mcts_calls = total_moves
nn_calls_per_mcts = mcts_sims  # Batched!

print(f"\nDetailed metrics:")
print(f"  Total moves: {total_moves}")
print(f"  MCTS calls: {mcts_calls}")
print(f"  NN evaluations: {mcts_calls * nn_calls_per_mcts}")
print(f"  NN calls/second: {mcts_calls * nn_calls_per_mcts / avg_time:.0f}")

# Comparison with sequential
est_sequential_time = num_games_test * 4.2  # ~4.2s per game for sequential
print(f"\nEstimated sequential time: {est_sequential_time:.0f}s")
print(f"Actual optimized time: {avg_time:.0f}s")
print(f"🚀 Speedup: {est_sequential_time/avg_time:.1f}x")

# Full pipeline projection
print(f"\n📊 FULL TRAINING PROJECTION:")
print("-"*40)
iterations = 20
games_per_iter = 1000
total_games = iterations * games_per_iter

self_play_time = avg_time * (games_per_iter / num_games_test) * iterations
train_time_total = train_time * (games_per_iter / num_games_test) * iterations

print(f"20 iterations × 1000 games:")
print(f"  Self-play time: {self_play_time/60:.0f} minutes")
print(f"  Training time: {train_time_total/60:.0f} minutes")
print(f"  Total time: {(self_play_time + train_time_total)/60:.0f} minutes")

print("\n✓ Performance test complete!")
print("="*60)
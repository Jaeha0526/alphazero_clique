#!/usr/bin/env python
"""
Quick performance test to show optimized pipeline speedup
"""

import sys
sys.path.append('/workspace/alphazero_clique/jax_full_src')

import time
import jax

print("Quick Performance Test - Optimized AlphaZero")
print("="*50)

from vectorized_nn import ImprovedBatchedNeuralNetwork
from vectorized_self_play_optimized import OptimizedVectorizedSelfPlay, OptimizedSelfPlayConfig

# Create model
model = ImprovedBatchedNeuralNetwork(
    num_vertices=6, hidden_dim=64, num_layers=3, asymmetric_mode=True
)

# Test with smaller parameters first
config = OptimizedSelfPlayConfig(
    batch_size=8,
    mcts_simulations=20,
    game_mode="asymmetric"
)

self_play = OptimizedVectorizedSelfPlay(config)

# Warmup
print("\nWarmup (2 games)...")
start = time.time()
_ = self_play.play_games(model, 2, verbose=False)
warmup_time = time.time() - start
print(f"Warmup time: {warmup_time:.1f}s")

# Test 1
print("\nTest 1: 8 games...")
start = time.time()
data1 = self_play.play_games(model, 8, verbose=True)
time1 = time.time() - start

# Test 2 
print("\nTest 2: 8 games (cached JIT)...")
start = time.time()
data2 = self_play.play_games(model, 8, verbose=True)
time2 = time.time() - start

print("\n" + "="*50)
print("RESULTS:")
print(f"Warmup: {warmup_time:.1f}s")
print(f"Test 1: {time1:.1f}s ({8/time1:.2f} games/sec)")
print(f"Test 2: {time2:.1f}s ({8/time2:.2f} games/sec)")
print(f"\nSpeedup after JIT: {time1/time2:.1f}x")

# Now test with requested parameters
print("\n" + "="*50)
print("Testing with requested parameters...")
config2 = OptimizedSelfPlayConfig(
    batch_size=16,
    mcts_simulations=50,
    game_mode="asymmetric"
)
self_play2 = OptimizedVectorizedSelfPlay(config2)

print("\nPlaying 30 games (batch_size=16, mcts=50)...")
start = time.time()
data3 = self_play2.play_games(model, 30, verbose=True)
time3 = time.time() - start

print(f"\nCompleted 30 games in {time3:.1f}s")
print(f"That's {30/time3:.1f} games/second!")
print(f"\nFor 1000 games: ~{1000/(30/time3):.0f} seconds")
print("="*50)
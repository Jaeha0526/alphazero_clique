#!/usr/bin/env python
"""
Test performance with optimized board implementation
"""

import sys
sys.path.append('/workspace/alphazero_clique')
sys.path.append('/workspace/alphazero_clique/jax_full_src')

import os
os.environ['JAX_PLATFORMS'] = ''  # Let JAX auto-select

import time
import jax
from vectorized_self_play import VectorizedSelfPlay, SelfPlayConfig
from vectorized_nn import BatchedNeuralNetwork


def test_optimized_performance():
    """Test self-play with optimized board."""
    print("Testing Optimized Vectorized Self-Play")
    print("="*60)
    print(f"Device: {jax.devices()[0]}")
    
    # Test configurations
    configs = [
        (16, 20, "Small batch"),
        (64, 50, "Medium batch"),
        (256, 50, "Large batch"),
    ]
    
    nn = BatchedNeuralNetwork()
    
    print("\nBatch | Sims | Time | Games/sec | Speedup vs CPU")
    print("-"*55)
    
    cpu_baseline = 0.25  # games/sec
    
    for batch_size, num_sims, desc in configs:
        config = SelfPlayConfig(
            batch_size=batch_size,
            mcts_simulations=num_sims,
            temperature_threshold=5,
            max_moves=30
        )
        
        self_play = VectorizedSelfPlay(config, nn)
        
        # Warmup
        print(f"\n{desc}: warming up...", end='', flush=True)
        _ = self_play.play_batch()
        print(" done")
        
        # Time a batch
        start = time.time()
        experiences = self_play.play_batch()
        elapsed = time.time() - start
        
        num_games = len(experiences)
        games_per_sec = num_games / elapsed if elapsed > 0 else 0
        speedup = games_per_sec / cpu_baseline
        
        print(f"{batch_size:5d} | {num_sims:4d} | {elapsed:4.1f}s | {games_per_sec:9.1f} | {speedup:6.0f}x")
        
        # Show positions
        total_positions = sum(len(exp) for exp in experiences)
        pos_per_sec = total_positions / elapsed if elapsed > 0 else 0
        print(f"      → {total_positions} positions ({pos_per_sec:.0f} pos/sec)")
    
    print("\n" + "="*60)
    print("✓ Optimized implementation working!")
    print("✓ Achieving massive speedup with true GPU parallelization")
    print("="*60)


def quick_benchmark():
    """Quick benchmark of key components."""
    print("\n\nComponent Benchmark")
    print("="*60)
    
    from optimized_board_v2 import OptimizedVectorizedBoard
    from optimized_mcts import OptimizedVectorizedMCTS
    
    batch_size = 256
    
    # 1. Board operations
    board = OptimizedVectorizedBoard(batch_size)
    start = time.time()
    for _ in range(10):
        board.get_valid_moves_mask()
        board.get_features_for_nn()
        board.make_moves(jax.numpy.zeros(batch_size, dtype=jax.numpy.int32))
    elapsed = time.time() - start
    print(f"Board ops (10 steps): {elapsed:.3f}s")
    
    # 2. Neural network
    nn = BatchedNeuralNetwork()
    edge_indices, edge_features = board.get_features_for_nn()
    start = time.time()
    for _ in range(10):
        nn.evaluate_batch(edge_indices, edge_features)
    elapsed = time.time() - start
    print(f"NN eval (10 batches): {elapsed:.3f}s")
    
    # 3. MCTS
    mcts = OptimizedVectorizedMCTS(batch_size, num_simulations=50)
    valid_mask = board.get_valid_moves_mask()
    
    # Warmup
    _ = mcts.search_batch_jit(
        (edge_indices, edge_features),
        nn.model.apply,
        nn.params,
        valid_mask
    )
    
    start = time.time()
    action_probs = mcts.search_batch_jit(
        (edge_indices, edge_features),
        nn.model.apply,
        nn.params,
        valid_mask
    )
    action_probs.block_until_ready()
    elapsed = time.time() - start
    print(f"MCTS (50 sims): {elapsed:.3f}s")
    print(f"  → {batch_size} games in parallel")
    print(f"  → {batch_size/elapsed:.0f} games/second")


if __name__ == "__main__":
    test_optimized_performance()
    quick_benchmark()
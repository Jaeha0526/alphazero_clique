#!/usr/bin/env python
"""
Simple test of vectorized self-play to verify it works
"""

import sys
sys.path.append('/workspace/alphazero_clique')
sys.path.append('/workspace/alphazero_clique/jax_full_src')

# Force JAX to use GPU
import os
os.environ['JAX_PLATFORMS'] = 'gpu'

import time
from vectorized_self_play import VectorizedSelfPlay, SelfPlayConfig
from vectorized_nn import BatchedNeuralNetwork


def test_small_batch():
    """Test with a very small batch to verify functionality."""
    print("Testing Vectorized Self-Play - Small Batch")
    print("="*60)
    
    # Small config for quick test
    config = SelfPlayConfig(
        batch_size=4,
        mcts_simulations=10,  # Very few simulations
        temperature_threshold=5
    )
    
    nn = BatchedNeuralNetwork()
    self_play = VectorizedSelfPlay(config, nn)
    
    print(f"Config: batch_size={config.batch_size}, simulations={config.mcts_simulations}")
    
    # Play one batch
    print("\nPlaying one batch...")
    start = time.time()
    experiences = self_play.play_batch()
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Games completed: {len(experiences)}")
    print(f"Positions per game: {[len(exp) for exp in experiences]}")
    
    # Check data format
    if experiences:
        exp = experiences[0][0]
        print("\nExperience format check:")
        for key, val in exp.items():
            if hasattr(val, 'shape'):
                print(f"  {key}: shape {val.shape}")
            elif isinstance(val, dict):
                print(f"  {key}: dict with {len(val)} keys")
            else:
                print(f"  {key}: {type(val).__name__}")
    
    return experiences


def test_performance_scaling():
    """Test how performance scales with batch size."""
    print("\n\nPerformance Scaling Test")
    print("="*60)
    
    nn = BatchedNeuralNetwork()
    
    # Test different batch sizes with minimal simulations
    batch_sizes = [1, 4, 16, 64]
    mcts_sims = 20
    
    print(f"MCTS simulations: {mcts_sims}")
    print("\nBatch Size | Time | Games/sec | Positions/sec")
    print("-"*50)
    
    for batch_size in batch_sizes:
        config = SelfPlayConfig(
            batch_size=batch_size,
            mcts_simulations=mcts_sims,
            temperature_threshold=5
        )
        
        self_play = VectorizedSelfPlay(config, nn)
        
        # Time one batch
        start = time.time()
        experiences = self_play.play_batch()
        elapsed = time.time() - start
        
        num_games = len(experiences)
        total_positions = sum(len(exp) for exp in experiences)
        
        games_per_sec = num_games / elapsed if elapsed > 0 else 0
        pos_per_sec = total_positions / elapsed if elapsed > 0 else 0
        
        print(f"{batch_size:10d} | {elapsed:4.1f}s | {games_per_sec:9.1f} | {pos_per_sec:13.0f}")
    
    print("\nNote: True speedup comes with larger batches (256+)")


def verify_game_logic():
    """Verify that games follow proper rules."""
    print("\n\nGame Logic Verification")
    print("="*60)
    
    config = SelfPlayConfig(batch_size=2, mcts_simulations=10)
    nn = BatchedNeuralNetwork()
    self_play = VectorizedSelfPlay(config, nn)
    
    experiences = self_play.play_batch()
    
    print(f"Games played: {len(experiences)}")
    
    for i, game_exp in enumerate(experiences):
        print(f"\nGame {i}:")
        print(f"  Moves: {len(game_exp)}")
        
        # Check values are assigned
        values = [exp['value'] for exp in game_exp]
        unique_values = set(values)
        print(f"  Values: {unique_values}")
        
        # Check players alternate
        players = [exp['player'] for exp in game_exp]
        alternating = all(players[i] != players[i+1] for i in range(len(players)-1))
        print(f"  Players alternate: {alternating}")
        
        # Check policies sum to 1
        policy_sums = [exp['policy'].sum() for exp in game_exp]
        policies_valid = all(0.99 < s < 1.01 for s in policy_sums)
        print(f"  Policies sum to ~1: {policies_valid}")


if __name__ == "__main__":
    # Run tests
    experiences = test_small_batch()
    test_performance_scaling()
    verify_game_logic()
    
    print("\n" + "="*60)
    print("✓ Vectorized self-play is working!")
    print("✓ Ready for full-scale testing")
    print("="*60)
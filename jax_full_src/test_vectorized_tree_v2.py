#!/usr/bin/env python
"""Test the VectorizedTreeMCTSv2 implementation."""

import jax
import jax.numpy as jnp
import time

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from vectorized_tree_mcts_v2 import VectorizedTreeMCTSv2

def test_vectorized_tree_mcts():
    """Test the vectorized tree MCTS implementation."""
    print("Testing VectorizedTreeMCTSv2...")
    print(f"JAX devices: {jax.devices()}")
    
    # Test parameters
    batch_size = 5  # Smaller batch for testing
    num_vertices = 9
    k = 4
    num_simulations = 5  # Much smaller for initial test
    
    print(f"\nTest configuration:")
    print(f"- Batch size: {batch_size} games")
    print(f"- Board: n={num_vertices}, k={k}")
    print(f"- MCTS simulations: {num_simulations}")
    
    # Create boards
    boards = VectorizedCliqueBoard(
        batch_size=batch_size,
        num_vertices=num_vertices,
        k=k,
        game_mode="symmetric"
    )
    
    # Create neural network
    nn = ImprovedBatchedNeuralNetwork(
        num_vertices=num_vertices,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=False
    )
    
    # Create MCTS
    mcts = VectorizedTreeMCTSv2(
        batch_size=batch_size,
        num_actions=36,  # C(9,2) = 36
        c_puct=3.0,
        max_nodes_per_game=500
    )
    
    # Run search
    print("\nRunning MCTS search...")
    start_time = time.time()
    
    try:
        probs = mcts.search(boards, nn, num_simulations, temperature=1.0)
        elapsed = time.time() - start_time
        
        print(f"\nSuccess! Search completed in {elapsed:.2f}s")
        print(f"Time per game: {elapsed/batch_size:.3f}s")
        print(f"Time per simulation per game: {elapsed/batch_size/num_simulations*1000:.1f}ms")
        
        # Verify output
        print(f"\nOutput shape: {probs.shape}")
        print(f"Probs sum per game: {probs.sum(axis=1)[:5]}")  # Show first 5
        
        # Test a full move
        print("\n\nTesting full move with action selection...")
        actions = []
        for i in range(batch_size):
            valid_moves = boards.get_valid_moves_mask()[i]
            if jnp.any(valid_moves):
                # Sample from probabilities
                action = jnp.argmax(probs[i])  # For test, just take argmax
                actions.append(int(action))
            else:
                actions.append(0)
        
        actions = jnp.array(actions)
        boards.make_moves(actions)
        
        print("Move successful!")
        print(f"Active games: {jnp.sum(boards.game_states == 0)}")
        
    except Exception as e:
        print(f"\nError during search: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def benchmark_vectorized_tree():
    """Benchmark the vectorized tree MCTS at different scales."""
    print("\n\n=== Benchmarking VectorizedTreeMCTSv2 ===")
    
    configs = [
        (5, 5),      # 5 games, 5 simulations (quick test)
        (10, 5),     # 10 games, 5 simulations
        (50, 5),     # 50 games, 5 simulations
        (100, 3),    # 100 games, 3 simulations (very quick)
    ]
    
    nn = ImprovedBatchedNeuralNetwork(
        num_vertices=9,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=False
    )
    
    for batch_size, num_sims in configs:
        print(f"\n--- {batch_size} games, {num_sims} simulations ---")
        
        boards = VectorizedCliqueBoard(
            batch_size=batch_size,
            num_vertices=9,
            k=4,
            game_mode="symmetric"
        )
        
        mcts = VectorizedTreeMCTSv2(
            batch_size=batch_size,
            num_actions=36,
            c_puct=3.0,
            max_nodes_per_game=200
        )
        
        start = time.time()
        try:
            probs = mcts.search(boards, nn, num_sims, temperature=1.0)
            elapsed = time.time() - start
            
            print(f"Total time: {elapsed:.2f}s")
            print(f"Per game: {elapsed/batch_size:.3f}s")
            print(f"Per simulation per game: {elapsed/batch_size/num_sims*1000:.1f}ms")
            
            # Estimate full game time
            moves_per_game = 30
            game_time = elapsed * moves_per_game
            print(f"Estimated full game time: {game_time:.1f}s = {game_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    success = test_vectorized_tree_mcts()
    
    if success:
        benchmark_vectorized_tree()
    else:
        print("\nSkipping benchmark due to test failure.")
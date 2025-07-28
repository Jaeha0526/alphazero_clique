#!/usr/bin/env python
"""Minimal test for VectorizedTreeMCTSv2."""

import jax
import jax.numpy as jnp
import time

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from vectorized_tree_mcts_v2 import VectorizedTreeMCTSv2

def test_minimal():
    """Minimal test with 1 game, 1 simulation."""
    print("Minimal test of VectorizedTreeMCTSv2...")
    print(f"JAX devices: {jax.devices()}")
    
    # Minimal parameters
    batch_size = 1
    num_simulations = 1
    
    print(f"\nConfiguration: {batch_size} game, {num_simulations} simulation")
    
    # Create boards
    boards = VectorizedCliqueBoard(
        batch_size=batch_size,
        num_vertices=9,
        k=4,
        game_mode="symmetric"
    )
    
    # Create neural network
    nn = ImprovedBatchedNeuralNetwork(
        num_vertices=9,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=False
    )
    
    # Warm up NN to JIT compile
    print("\nWarming up NN...")
    edge_indices, edge_features = boards.get_features_for_nn_undirected()
    valid_mask = boards.get_valid_moves_mask()
    _ = nn.evaluate_batch(edge_indices, edge_features, valid_mask)
    print("NN warmup complete")
    
    # Create MCTS
    mcts = VectorizedTreeMCTSv2(
        batch_size=batch_size,
        num_actions=36,
        c_puct=3.0,
        max_nodes_per_game=10
    )
    
    # Run search
    print("\nRunning MCTS search...")
    start = time.time()
    
    try:
        probs = mcts.search(boards, nn, num_simulations, temperature=1.0)
        elapsed = time.time() - start
        
        print(f"\nSuccess! Search took {elapsed:.3f}s")
        print(f"Probs shape: {probs.shape}")
        print(f"Probs: {probs}")
        print(f"Sum: {probs.sum()}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal()
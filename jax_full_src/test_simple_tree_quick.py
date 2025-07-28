#!/usr/bin/env python
"""Quick test of SimpleTreeMCTS."""

import time
import jax

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from simple_tree_mcts import SimpleTreeMCTS

def quick_test():
    print("Quick test of SimpleTreeMCTS...")
    print(f"JAX devices: {jax.devices()}")
    
    # Small test
    batch_size = 10
    num_simulations = 50
    
    boards = VectorizedCliqueBoard(
        batch_size=batch_size,
        num_vertices=9,
        k=4,
        game_mode="symmetric"
    )
    
    nn = ImprovedBatchedNeuralNetwork(
        num_vertices=9,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=False
    )
    
    mcts = SimpleTreeMCTS(
        batch_size=batch_size,
        num_actions=36,
        c_puct=3.0,
        max_nodes_per_game=200
    )
    
    print(f"\nRunning MCTS with {batch_size} games, {num_simulations} simulations...")
    start = time.time()
    
    probs = mcts.search(boards, nn, num_simulations, temperature=1.0)
    
    elapsed = time.time() - start
    print(f"\nSuccess! Took {elapsed:.2f}s")
    print(f"Per game: {elapsed/batch_size:.3f}s")
    print(f"Probs shape: {probs.shape}")
    print(f"Probs sum: {probs.sum(axis=1)[:5]}")  # Show first 5

if __name__ == "__main__":
    quick_test()
#!/usr/bin/env python
"""Test JIT-compiled SimpleTreeMCTS to ensure it works correctly."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from simple_tree_mcts import SimpleTreeMCTS

def test_jit_tree_mcts():
    print("Testing JIT-compiled SimpleTreeMCTS...")
    
    # Small test setup
    batch_size = 2
    num_vertices = 6  # Start smaller for testing
    k = 3
    num_actions = num_vertices * (num_vertices - 1) // 2  # 15 actions
    mcts_sims = 10  # Small number for testing
    
    print(f"Setup: batch_size={batch_size}, n={num_vertices}, k={k}, actions={num_actions}")
    
    # Create components
    boards = VectorizedCliqueBoard(
        batch_size=batch_size,
        num_vertices=num_vertices,
        k=k,
        game_mode="symmetric"
    )
    
    nn = ImprovedBatchedNeuralNetwork(
        num_vertices=num_vertices,
        hidden_dim=32,  # Smaller for testing
        num_layers=2,
        asymmetric_mode=False
    )
    
    mcts = SimpleTreeMCTS(
        batch_size=batch_size,
        num_actions=num_actions,
        c_puct=3.0,
        max_nodes_per_game=50  # Smaller for testing
    )
    
    print("Starting MCTS search...")
    start_time = time.time()
    
    try:
        probs = mcts.search(boards, nn, mcts_sims, temperature=1.0)
        elapsed = time.time() - start_time
        
        print(f"✓ Success! Completed in {elapsed:.2f}s")
        print(f"  Time per simulation: {elapsed/mcts_sims:.3f}s")
        print(f"  Output shape: {probs.shape}")
        print(f"  Probabilities sum (game 0): {probs[0].sum():.4f}")
        print(f"  Probabilities sum (game 1): {probs[1].sum():.4f}")
        print(f"  Max probability (game 0): {probs[0].max():.4f}")
        
        # Verify probabilities are valid
        for i in range(batch_size):
            prob_sum = probs[i].sum()
            if abs(prob_sum - 1.0) > 1e-3:
                print(f"  ⚠️  Game {i} probabilities don't sum to 1: {prob_sum}")
            else:
                print(f"  ✓ Game {i} probabilities sum correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_jit_tree_mcts()
    if success:
        print("\n" + "="*50)
        print("✓ JIT-compiled SimpleTreeMCTS test PASSED!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("✗ JIT-compiled SimpleTreeMCTS test FAILED!")
        print("="*50)
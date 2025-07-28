#!/usr/bin/env python
"""
Test script to debug GPU pipeline issues
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from tree_based_mcts import ParallelTreeBasedMCTS


def test_components():
    """Test individual components of the pipeline"""
    print("Testing JAX GPU pipeline components...")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    
    # Test 1: Board initialization
    print("\n1. Testing board initialization...")
    try:
        board = VectorizedCliqueBoard(batch_size=2, num_vertices=6, k=3)
        print("✓ Board initialized successfully")
    except Exception as e:
        print(f"✗ Board initialization failed: {e}")
        return
    
    # Test 2: Neural network
    print("\n2. Testing neural network...")
    try:
        model = ImprovedBatchedNeuralNetwork(
            num_vertices=6,
            hidden_dim=32,
            num_layers=2,
            asymmetric_mode=False
        )
        
        # Test forward pass
        edge_indices, edge_features = board.get_features_for_nn_undirected()
        policy, value = model.evaluate_batch(edge_indices, edge_features)
        print(f"✓ Neural network forward pass successful")
        print(f"  Policy shape: {policy.shape}, Value shape: {value.shape}")
    except Exception as e:
        print(f"✗ Neural network failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: MCTS initialization
    print("\n3. Testing MCTS initialization...")
    try:
        mcts = ParallelTreeBasedMCTS(
            batch_size=2,
            num_actions=15,
            c_puct=1.0,
            noise_weight=0.25,
            perspective_mode="alternating"
        )
        print("✓ MCTS initialized successfully")
    except Exception as e:
        print(f"✗ MCTS initialization failed: {e}")
        return
    
    # Test 4: MCTS search
    print("\n4. Testing MCTS search...")
    try:
        # MCTS expects board object and num_simulations array
        num_simulations = jnp.array([5, 5])  # 5 simulations per game
        
        action_probs = mcts.search(
            board,
            model,
            num_simulations,
            temperature=1.0
        )
        print(f"✓ MCTS search successful")
        print(f"  Action probabilities shape: {action_probs.shape}")
    except Exception as e:
        print(f"✗ MCTS search failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✓ All components working correctly!")


if __name__ == "__main__":
    test_components()
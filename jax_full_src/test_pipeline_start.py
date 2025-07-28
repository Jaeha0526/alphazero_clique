#!/usr/bin/env python
"""Test the pipeline startup to see where it might be failing."""

import sys
import os
sys.path.append('/workspace/alphazero_clique/jax_full_src')

print("1. Importing modules...")
try:
    import jax
    print(f"   JAX imported, devices: {jax.devices()}")
    
    from vectorized_board import VectorizedCliqueBoard
    print("   VectorizedCliqueBoard imported")
    
    from vectorized_nn import ImprovedBatchedNeuralNetwork
    print("   ImprovedBatchedNeuralNetwork imported")
    
    from simple_tree_mcts import SimpleTreeMCTS
    print("   SimpleTreeMCTS imported")
    
    from train_jax import train_network_jax
    print("   train_network_jax imported")
    
except Exception as e:
    print(f"   Error importing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Creating test components...")
try:
    # Create board
    board = VectorizedCliqueBoard(
        batch_size=1,
        num_vertices=9,
        k=4,
        game_mode="symmetric"
    )
    print("   Board created")
    
    # Create NN
    nn = ImprovedBatchedNeuralNetwork(
        num_vertices=9,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=False
    )
    print("   Neural network created")
    
    # Test NN evaluation
    edge_indices, edge_features = board.get_features_for_nn_undirected()
    valid_mask = board.get_valid_moves_mask()
    policies, values = nn.evaluate_batch(edge_indices, edge_features, valid_mask)
    print("   NN evaluation successful")
    
    # Create MCTS
    mcts = SimpleTreeMCTS(
        batch_size=1,
        num_actions=36,
        c_puct=3.0,
        max_nodes_per_game=100
    )
    print("   MCTS created")
    
    # Test MCTS search
    probs = mcts.search(board, nn, num_simulations=5, temperature=1.0)
    print("   MCTS search successful")
    
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing experiment directory creation...")
try:
    from pathlib import Path
    exp_dir = Path("/workspace/alphazero_clique/experiments/test_startup")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Write a test file
    test_file = exp_dir / "test.txt"
    test_file.write_text("Test successful")
    
    print(f"   Created {exp_dir}")
    print(f"   Wrote test file: {test_file}")
    
    # Check if we can read it back
    content = test_file.read_text()
    print(f"   Read back: {content}")
    
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print("\nAll basic tests passed!")
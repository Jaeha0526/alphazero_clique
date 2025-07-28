#!/usr/bin/env python
"""Debug VectorizedTreeMCTSv2 step by step."""

import jax
import jax.numpy as jnp

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from vectorized_tree_mcts_v2 import VectorizedTreeMCTSv2

def debug_mcts():
    """Debug MCTS step by step."""
    print("Debugging VectorizedTreeMCTSv2...")
    
    # Create minimal setup
    batch_size = 1
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
    
    mcts = VectorizedTreeMCTSv2(
        batch_size=batch_size,
        num_actions=36,
        c_puct=3.0,
        max_nodes_per_game=10
    )
    
    # Try to trace through the initialization
    print("\n1. Creating arrays...")
    N = jnp.zeros((batch_size, 10, 36))
    W = jnp.zeros((batch_size, 10, 36))
    P = jnp.zeros((batch_size, 10, 36))
    print("   Arrays created")
    
    print("\n2. Getting root features...")
    edge_indices, edge_features = boards.get_features_for_nn_undirected()
    valid_mask = boards.get_valid_moves_mask()
    print(f"   Edge indices shape: {edge_indices.shape}")
    print(f"   Valid mask shape: {valid_mask.shape}")
    
    print("\n3. NN evaluation...")
    # First warm up the NN
    _ = nn.evaluate_batch(edge_indices, edge_features, valid_mask)
    print("   NN warmed up")
    
    policies, values = nn.evaluate_batch(edge_indices, edge_features, valid_mask)
    print(f"   Policies shape: {policies.shape}")
    print(f"   Values shape: {values.shape}")
    
    print("\n4. Testing array operations...")
    # Test the array update that might be causing issues
    P = P.at[:, 0, :].set(policies)
    print("   Array update successful")
    
    print("\n5. Testing boolean operations...")
    expanded = jnp.zeros((batch_size, 10), dtype=bool)
    expanded = expanded.at[:, 0].set(True)
    
    terminal = jnp.zeros((batch_size, 10), dtype=bool)
    
    current_nodes = jnp.zeros(batch_size, dtype=jnp.int32)
    is_expanded = expanded[jnp.arange(batch_size), current_nodes]
    is_terminal = terminal[jnp.arange(batch_size), current_nodes]
    
    print(f"   is_expanded: {is_expanded}")
    print(f"   is_terminal: {is_terminal}")
    
    should_continue = is_expanded & ~is_terminal
    print(f"   should_continue: {should_continue}")
    print(f"   any should_continue: {jnp.any(should_continue)}")
    print(f"   bool(any): {bool(jnp.any(should_continue))}")
    
    print("\nAll basic operations work!")

if __name__ == "__main__":
    debug_mcts()
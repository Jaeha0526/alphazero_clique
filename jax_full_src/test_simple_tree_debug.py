#!/usr/bin/env python
"""Debug SimpleTreeMCTS to see why trees aren't building."""

import numpy as np
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from simple_tree_mcts import SimpleTreeMCTS

def debug_simple_tree_mcts():
    print("Debugging SimpleTreeMCTS...")
    
    # Small test
    batch_size = 2
    num_simulations = 10
    
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
        max_nodes_per_game=100
    )
    
    # Look inside the search method
    print(f"\nRunning search with {num_simulations} simulations...")
    
    # Initialize trees manually to inspect
    trees = []
    for game_idx in range(batch_size):
        tree = {
            'N': {},  # Visit counts N[node_id][action]
            'W': {},  # Total values W[node_id][action]
            'P': {},  # Prior probabilities P[node_id][action]
            'children': {},  # Children mapping children[node_id][action] = child_id
            'expanded': set(),  # Set of expanded node ids
            'boards': {},  # Board states at each node
            'node_count': 0  # Number of nodes in tree
        }
        
        # Initialize root
        tree['N'][0] = np.zeros(36)
        tree['W'][0] = np.zeros(36)
        tree['children'][0] = {}
        tree['boards'][0] = boards
        tree['node_count'] = 1
        
        trees.append(tree)
        
    print(f"Initial tree structure:")
    for i, tree in enumerate(trees):
        print(f"  Game {i}: {tree['node_count']} nodes")
    
    # Do one simulation manually
    print("\nManual simulation:")
    
    # Get root features
    root_features = boards.get_features_for_nn_undirected()
    root_valid = boards.get_valid_moves_mask()
    root_policies, root_values = nn.evaluate_batch(*root_features, root_valid)
    
    print(f"Root policies shape: {root_policies.shape}")
    print(f"Root values shape: {root_values.shape}")
    
    # Store root priors
    for game_idx in range(batch_size):
        trees[game_idx]['P'][0] = np.array(root_policies[game_idx])
        trees[game_idx]['expanded'].add(0)
    
    # Try selection from root
    for game_idx, tree in enumerate(trees):
        print(f"\nGame {game_idx}:")
        print(f"  Root N: {tree['N'][0].sum()}")
        print(f"  Root P sum: {tree['P'][0].sum()}")
        
        # Calculate UCB
        N = tree['N'][0]
        W = tree['W'][0]
        P = tree['P'][0]
        
        Q = np.where(N > 0, W / N, 0.0)
        sqrt_sum = np.sqrt(N.sum() + 1)  # Add 1 to avoid sqrt(0)
        U = 3.0 * P * sqrt_sum / (1 + N)
        
        ucb = Q + U
        
        # Get valid mask
        board = tree['boards'][0]
        valid_mask = boards.get_valid_moves_mask()[game_idx]
        ucb_masked = np.where(valid_mask, ucb, -np.inf)
        
        print(f"  Valid moves: {valid_mask.sum()}")
        print(f"  Max UCB: {ucb_masked.max()}")
        print(f"  Best action: {np.argmax(ucb_masked)}")
    
    # Now run the actual search
    print("\n\nRunning actual search...")
    probs = mcts.search(boards, nn, num_simulations, temperature=1.0)
    
    print(f"\nResult probs shape: {probs.shape}")
    print(f"Probs sum: {probs.sum(axis=1)}")

if __name__ == "__main__":
    debug_simple_tree_mcts()
#!/usr/bin/env python
"""Detailed debugging of SimpleTreeMCTS."""

import numpy as np
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork

print("Testing SimpleTreeMCTS step by step...")

# Create minimal setup
boards = VectorizedCliqueBoard(batch_size=1, num_vertices=9, k=4, game_mode="symmetric")
nn = ImprovedBatchedNeuralNetwork(num_vertices=9, hidden_dim=64, num_layers=3, asymmetric_mode=False)

# Manually create the tree structure
print("\n1. Creating tree structure...")
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

print("   Tree initialized")

# Get root policy
print("\n2. Getting root policy...")
features = boards.get_features_for_nn_undirected()
valid_mask = boards.get_valid_moves_mask()
policies, values = nn.evaluate_batch(*features, valid_mask)
tree['P'][0] = np.array(policies[0])
tree['expanded'].add(0)
print(f"   Root policy set, sum: {tree['P'][0].sum()}")

# Test tree traversal
print("\n3. Testing tree traversal...")
node_id = 0
path = [(node_id, None)]

print(f"   Starting at node {node_id}")
print(f"   Is node {node_id} expanded? {node_id in tree['expanded']}")
print(f"   Children at node {node_id}: {tree['children'][node_id]}")

# Try one iteration of the while loop
if node_id in tree['expanded']:
    print("\n   Node is expanded, calculating UCB...")
    
    N = tree['N'][node_id]
    W = tree['W'][node_id]
    P = tree['P'][node_id]
    
    print(f"   N sum: {N.sum()}")
    print(f"   W sum: {W.sum()}")
    print(f"   P sum: {P.sum()}")
    
    # Q values
    Q = np.where(N > 0, W / N, 0.0)
    
    # U values
    sqrt_sum = np.sqrt(N.sum())
    print(f"   sqrt_sum: {sqrt_sum}")
    
    if sqrt_sum == 0:
        sqrt_sum = 1.0  # Avoid division by zero
        
    U = 3.0 * P * sqrt_sum / (1 + N)
    
    # UCB scores
    ucb = Q + U
    
    # Get valid mask
    board = tree['boards'][node_id]
    valid_mask = board.get_valid_moves_mask()[0]
    ucb_masked = np.where(valid_mask, ucb, -np.inf)
    
    print(f"   Valid moves: {valid_mask.sum()}")
    print(f"   Max UCB: {ucb_masked.max()}")
    
    if ucb_masked.max() > -np.inf:
        action = np.argmax(ucb_masked)
        print(f"   Selected action: {action}")
        
        # Check if child exists
        if action not in tree['children'][node_id]:
            print(f"   Child doesn't exist, would create new node")
        else:
            print(f"   Child exists: {tree['children'][node_id][action]}")

print("\nAnalysis complete!")
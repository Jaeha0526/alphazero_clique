#!/usr/bin/env python3
"""Track when specific edges get colored."""

import sys
sys.path.append('/workspace/alphazero_clique/jax_full_src')

import jax.numpy as jnp
from vectorized_board import VectorizedCliqueBoard
import pickle

def edge_to_action(i, j, n):
    """Convert vertex pair to action index."""
    if i > j:
        i, j = j, i
    count = 0
    for vi in range(n):
        for vj in range(vi + 1, n):
            if vi == i and vj == j:
                return count
            count += 1
    return -1

def test_specific_edges():
    """Track when the 4-clique edges get colored."""
    
    # Load the data
    with open('/workspace/alphazero_clique/experiments/ramsey_n_10_k4_new2/game_data/iteration_0.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Get game 2
    game_info = data['games_info'][1]
    game_moves = data['training_data'][game_info['start_idx']:game_info['end_idx']]
    
    # The problematic 4-clique is (1, 2, 6, 7)
    clique_vertices = [1, 2, 6, 7]
    clique_edges = []
    for i in range(len(clique_vertices)):
        for j in range(i + 1, len(clique_vertices)):
            v1, v2 = clique_vertices[i], clique_vertices[j]
            action_idx = edge_to_action(v1, v2, 10)
            clique_edges.append((v1, v2, action_idx))
    
    print("Tracking 4-clique edges for vertices (1, 2, 6, 7):")
    for v1, v2, action in clique_edges:
        print(f"  Edge ({v1},{v2}) = action {action}")
    
    print("\nPlaying game and tracking when these edges get colored:")
    
    # Create a board
    board = VectorizedCliqueBoard(
        batch_size=1,
        num_vertices=10,
        k=4,
        game_mode='avoid_clique'
    )
    
    # Track which edges have been colored
    colored_edges = {}
    
    # Play the moves
    for move_idx, move_data in enumerate(game_moves):
        action = move_data.get('action')
        if action is not None:
            player = int(board.current_players[0])
            
            # Check if this action colors one of our target edges
            for v1, v2, target_action in clique_edges:
                if action == target_action:
                    print(f"Move {move_idx + 1}: Player {player} colors edge ({v1},{v2}) with action {action}")
                    colored_edges[(v1, v2)] = (move_idx + 1, player)
            
            # Make the move
            board.make_moves(jnp.array([action]))
            
            # Check if all edges in our clique are colored by same player
            if len(colored_edges) == len(clique_edges):
                # Check if all same color
                players = [p for _, p in colored_edges.values()]
                if len(set(players)) == 1:
                    print(f"\n⚠️ All edges of 4-clique colored by Player {players[0]} at move {move_idx + 1}")
                    print(f"  Game state: {int(board.game_states[0])}")
                    if board.game_states[0] == 0:
                        print("  ❌ BUT GAME IS STILL ONGOING!")
    
    print(f"\nFinal summary of 4-clique edges:")
    for v1, v2, action in clique_edges:
        if (v1, v2) in colored_edges:
            move, player = colored_edges[(v1, v2)]
            print(f"  Edge ({v1},{v2}): Colored by Player {player} at move {move}")
        else:
            print(f"  Edge ({v1},{v2}): Never colored!")

if __name__ == "__main__":
    test_specific_edges()
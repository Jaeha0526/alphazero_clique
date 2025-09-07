#!/usr/bin/env python3
"""Test the clique detection logic directly."""

import sys
sys.path.append('/workspace/alphazero_clique/jax_full_src')

import jax.numpy as jnp
from vectorized_board import VectorizedCliqueBoard
import pickle

def test_specific_game():
    """Test game 2 which was incorrectly marked as complete."""
    
    # Load the data to get the exact moves
    with open('/workspace/alphazero_clique/experiments/ramsey_n_10_k4_new2/game_data/iteration_0.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Get game 2 (index 1)
    game_info = data['games_info'][1]  # Game 2
    game_moves = data['training_data'][game_info['start_idx']:game_info['end_idx']]
    
    print(f"Testing Game 2: {game_info['num_moves']} moves")
    
    # Create a board
    board = VectorizedCliqueBoard(
        batch_size=1,
        num_vertices=10,
        k=4,
        game_mode='avoid_clique'
    )
    
    # Play the moves
    for move_idx, move_data in enumerate(game_moves):
        action = move_data.get('action')
        if action is not None:
            print(f"Move {move_idx + 1}: Player {int(board.current_players[0])} plays action {action}")
            
            # Make the move
            board.make_moves(jnp.array([action]))
            
            # Check game state
            if board.game_states[0] != 0:
                print(f"  Game ended at move {move_idx + 1}!")
                print(f"  Game state: {int(board.game_states[0])}")
                print(f"  Winner: {int(board.winners[0])}")
                break
            
            # Check if we've reached move 42 (where the 4-clique should be detected)
            if move_idx >= 41:
                print(f"  Game still ongoing after move {move_idx + 1}")
                print(f"  Checking for 4-cliques manually...")
                
                # Manually check for 4-cliques
                edge_states = board.edge_states[0]
                
                # Check the specific 4-clique found by verification: (1, 2, 6, 7)
                vertices = [1, 2, 6, 7]
                edges_to_check = []
                for i in range(len(vertices)):
                    for j in range(i + 1, len(vertices)):
                        edges_to_check.append((vertices[i], vertices[j]))
                
                print(f"  Checking clique {vertices}:")
                all_same = True
                player_color = None
                for v1, v2 in edges_to_check:
                    color = edge_states[v1, v2]
                    print(f"    Edge ({v1},{v2}): color={int(color)}")
                    if player_color is None and color != 0:
                        player_color = color
                    if color != player_color or color == 0:
                        all_same = False
                
                if all_same and player_color is not None:
                    print(f"  ❌ FOUND 4-CLIQUE for player {player_color-1} but game didn't end!")
    
    print(f"\nFinal game state: {int(board.game_states[0])}")
    print(f"Total moves made: {int(board.move_counts[0])}")

if __name__ == "__main__":
    test_specific_game()
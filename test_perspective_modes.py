#!/usr/bin/env python
"""Test both perspective modes (fixed and alternating) to ensure they work correctly."""

import sys
sys.path.append('src')
import torch
from clique_board import CliqueBoard
from alpha_net_clique import CliqueGNN
from MCTS_clique import UCT_search
import encoder_decoder_clique as ed

def test_perspective_modes():
    """Test that both perspective modes work correctly."""
    print("Testing Perspective Modes")
    print("=" * 50)
    
    # Create a simple test board
    board = CliqueBoard(6, 3)
    board.make_move((0, 1))  # Player 1
    board.make_move((1, 2))  # Player 2
    
    # Create a model
    model = CliqueGNN(num_vertices=6, hidden_dim=32, num_layers=2)
    model.eval()
    
    # Test both perspective modes
    for perspective_mode in ["fixed", "alternating"]:
        print(f"\nTesting {perspective_mode} perspective mode:")
        
        # Run MCTS with each mode
        best_move, root = UCT_search(board, num_reads=50, net=model, 
                                   perspective_mode=perspective_mode)
        
        # Decode the move
        edge = ed.decode_action(board, best_move)
        print(f"Best move selected: {edge}")
        
        # Check that the policy sums to 1
        if root.child_priors is not None:
            policy_sum = root.child_priors.sum()
            print(f"Root policy sum: {policy_sum:.4f} (should be ~1.0)")
        
        # Check Q-values
        q_values = root.child_Q()
        print(f"Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
        
        # Create a terminal position to test value handling
        terminal_board = CliqueBoard(6, 3)
        terminal_board.make_move((0, 1))  # P1
        terminal_board.make_move((1, 2))  # P2  
        terminal_board.make_move((0, 2))  # P1 forms triangle, wins
        
        print(f"\nTerminal position test (P1 wins):")
        print(f"Game state: {terminal_board.game_state} (1=P1 wins)")
        
        # Get network value for this position
        state_dict = ed.prepare_state_for_network(terminal_board)
        with torch.no_grad():
            _, value = model(state_dict['edge_index'], state_dict['edge_attr'])
            
        print(f"Network value output: {value.item():.4f}")
        
        if perspective_mode == "fixed":
            print("(Fixed mode: should be positive since P1 wins)")
        else:
            print(f"(Alternating mode: current player is P{terminal_board.player + 1})")
    
    print("\nPerspective mode test completed!")

if __name__ == "__main__":
    test_perspective_modes()
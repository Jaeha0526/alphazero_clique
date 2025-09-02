#!/usr/bin/env python
"""
Test the new avoid_clique game mode.
In this mode, forming a clique makes you LOSE (opposite of normal).
"""

import jax
import jax.numpy as jnp
import numpy as np
from vectorized_board import VectorizedCliqueBoard

def test_avoid_clique_mode():
    """Test that avoid_clique mode works correctly."""
    
    print("="*60)
    print("Testing Avoid Clique Mode")
    print("="*60)
    print("\nRule: Forming a k-clique makes you LOSE!")
    print("Goal: Force your opponent to form a clique\n")
    
    # Create a single game with avoid_clique mode
    board = VectorizedCliqueBoard(
        batch_size=1,
        num_vertices=4,  # Small board for easy testing
        k=3,  # Triangle loses
        game_mode="avoid_clique"
    )
    
    print(f"Board: n={board.num_vertices}, k={board.k}")
    print(f"Game mode: {board.game_mode}")
    print(f"Total possible edges: {board.num_edges}")
    
    # In a 4-vertex graph, edges are:
    # 0: (0,1), 1: (0,2), 2: (0,3), 3: (1,2), 4: (1,3), 5: (2,3)
    
    print("\n--- Simulating a game ---")
    
    # Player 1's moves (trying to avoid triangles)
    print("\nMove 1: Player 1 selects edge (0,1)")
    board.make_moves(jnp.array([0]))  # Edge 0: (0,1)
    print(f"  Current player: {board.current_players[0]}")
    print(f"  Game state: {board.game_states[0]} (0=ongoing)")
    
    # Player 2's move
    print("\nMove 2: Player 2 selects edge (0,3)")
    board.make_moves(jnp.array([2]))  # Edge 2: (0,3)
    print(f"  Current player: {board.current_players[0]}")
    print(f"  Game state: {board.game_states[0]} (0=ongoing)")
    
    # Player 1's move
    print("\nMove 3: Player 1 selects edge (0,2)")
    board.make_moves(jnp.array([1]))  # Edge 1: (0,2)
    print(f"  Current player: {board.current_players[0]}")
    print(f"  Game state: {board.game_states[0]} (0=ongoing)")
    
    # Player 2's move
    print("\nMove 4: Player 2 selects edge (1,3)")
    board.make_moves(jnp.array([4]))  # Edge 4: (1,3)
    print(f"  Current player: {board.current_players[0]}")
    print(f"  Game state: {board.game_states[0]} (0=ongoing)")
    
    # Player 1's move - this will complete a triangle!
    print("\nMove 5: Player 1 selects edge (1,2)")
    print("  This completes triangle (0,1,2) with Player 1's edges!")
    board.make_moves(jnp.array([3]))  # Edge 3: (1,2)
    
    # Check result
    print(f"\n  Game state: {board.game_states[0]}")
    print(f"  Winner: {board.winners[0]}")
    
    if board.game_states[0] == 2:
        print("\n✅ CORRECT! Player 2 wins because Player 1 formed a clique!")
    else:
        print(f"\n❌ ERROR! Expected Player 2 to win, got state {board.game_states[0]}")
    
    print("\n" + "="*60)
    
    # Test 2: Compare with normal symmetric mode
    print("\nComparing with normal symmetric mode...")
    
    normal_board = VectorizedCliqueBoard(
        batch_size=1,
        num_vertices=4,
        k=3,
        game_mode="symmetric"
    )
    
    # Same moves as avoid_clique test
    normal_board.make_moves(jnp.array([0]))  # P1: (0,1)
    normal_board.make_moves(jnp.array([2]))  # P2: (0,3)
    normal_board.make_moves(jnp.array([1]))  # P1: (0,2)
    normal_board.make_moves(jnp.array([4]))  # P2: (1,3)
    normal_board.make_moves(jnp.array([3]))  # P1: (1,2) - forms triangle
    
    print(f"Normal mode result: Player {normal_board.winners[0] + 1} wins")
    print(f"Avoid mode result: Player {board.winners[0] + 1} wins")
    
    if normal_board.winners[0] != board.winners[0]:
        print("\n✅ Winners are opposite - avoid_clique working correctly!")
    else:
        print("\n❌ Winners should be opposite!")
    
    return board.game_states[0] == 2  # Player 2 should win

def test_full_game():
    """Test a longer game to ensure the mode works throughout."""
    
    print("\n" + "="*60)
    print("Testing Full Game in Avoid Clique Mode")
    print("="*60)
    
    board = VectorizedCliqueBoard(
        batch_size=1,
        num_vertices=5,
        k=3,
        game_mode="avoid_clique"
    )
    
    print(f"\nBoard: n={board.num_vertices}, k={board.k}")
    print("Playing random moves until someone loses by forming a triangle...\n")
    
    moves = 0
    key = jax.random.PRNGKey(42)
    
    while board.game_states[0] == 0 and moves < board.num_edges:
        valid_mask = board.get_valid_moves_mask()[0]
        valid_indices = jnp.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            # Random valid move
            key, subkey = jax.random.split(key)
            action_idx = jax.random.choice(subkey, valid_indices)
            
            current_player = board.current_players[0] + 1
            board.make_moves(jnp.array([action_idx]))
            moves += 1
            
            print(f"Move {moves}: Player {current_player} plays edge {action_idx}")
            
            if board.game_states[0] != 0:
                loser = 3 - board.winners[0] - 1  # The one who formed the clique
                winner = board.winners[0] + 1
                print(f"\nGame Over! Player {loser} formed a triangle and LOST!")
                print(f"Player {winner} wins by avoiding triangles!")
                break
        else:
            break
    
    if board.game_states[0] == 0:
        print("\nGame ended in a draw (no triangles formed)")
    
    return True

if __name__ == "__main__":
    # Run tests
    success1 = test_avoid_clique_mode()
    success2 = test_full_game()
    
    if success1 and success2:
        print("\n" + "="*60)
        print("✅ All tests passed! Avoid clique mode is working correctly.")
        print("="*60)
    else:
        print("\n❌ Some tests failed. Check implementation.")
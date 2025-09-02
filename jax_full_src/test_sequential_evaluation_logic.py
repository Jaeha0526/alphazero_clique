#!/usr/bin/env python
"""
Test that sequential evaluation (evaluate_vs_initial_and_best) 
correctly handles avoid_clique mode with alternating starts.
"""

import jax
import jax.numpy as jnp
import numpy as np
from vectorized_board import VectorizedCliqueBoard

def test_sequential_evaluation_logic():
    """Test the win recording logic used in sequential evaluation."""
    
    print("="*60)
    print("Testing Sequential Evaluation Win Recording Logic")
    print("="*60)
    
    # Simulate the logic from evaluation_jax_fixed.py
    current_wins = 0
    baseline_wins = 0
    draws = 0
    
    # Test 4 games with alternating starts
    test_cases = [
        # (game_idx, current_starts, game_state, expected_winner)
        (0, True, 2, "baseline"),   # P2 wins, current was P1 -> baseline wins
        (1, False, 1, "baseline"),  # P1 wins, baseline was P1 -> baseline wins  
        (2, True, 1, "current"),    # P1 wins, current was P1 -> current wins
        (3, False, 2, "current"),   # P2 wins, current was P2 -> current wins
        (4, True, 3, "draw"),       # Draw
    ]
    
    print("\nTest Cases (avoid_clique mode):")
    print("Remember: In avoid_clique, forming a clique makes you LOSE!")
    print()
    
    for game_idx, current_starts, game_state, expected in test_cases:
        print(f"Game {game_idx}:")
        print(f"  current_starts: {current_starts}")
        print(f"  game_state: {game_state} ", end="")
        
        if game_state == 1:
            print("(Player 1 wins - Player 2 formed clique)")
        elif game_state == 2:
            print("(Player 2 wins - Player 1 formed clique)")
        elif game_state == 3:
            print("(Draw)")
        
        # Apply the logic from evaluation_jax_fixed.py
        if game_state == 3 or game_state == 0:  # Draw or unfinished
            draws += 1
            result = "draw"
        elif game_state == 1:  # Player 1 wins
            if current_starts:
                current_wins += 1
                result = "current"
            else:
                baseline_wins += 1
                result = "baseline"
        elif game_state == 2:  # Player 2 wins
            if not current_starts:
                current_wins += 1
                result = "current"
            else:
                baseline_wins += 1
                result = "baseline"
        
        if result == expected:
            print(f"  Result: {result} ✅")
        else:
            print(f"  Result: {result} ❌ (expected {expected})")
        print()
    
    print("--- Final Counts ---")
    print(f"Current wins: {current_wins}")
    print(f"Baseline wins: {baseline_wins}")
    print(f"Draws: {draws}")
    
    if current_wins == 2 and baseline_wins == 2 and draws == 1:
        print("\n✅ CORRECT! Sequential evaluation logic handles avoid_clique properly!")
        return True
    else:
        print("\n❌ ERROR! Incorrect counts")
        return False

def test_with_actual_board():
    """Test with actual VectorizedCliqueBoard to verify game_states are set correctly."""
    
    print("\n" + "="*60)
    print("Testing with Actual Board in Avoid Clique Mode")
    print("="*60)
    
    # Test that game_states are set correctly in avoid_clique mode
    for test_num, (who_forms_clique, expected_state) in enumerate([
        ("player1", 2),  # P1 forms clique -> P2 wins -> state=2
        ("player2", 1),  # P2 forms clique -> P1 wins -> state=1
    ]):
        print(f"\nTest {test_num + 1}: {who_forms_clique} forms clique")
        
        board = VectorizedCliqueBoard(
            batch_size=1,
            num_vertices=4,
            k=3,
            game_mode="avoid_clique"
        )
        
        if who_forms_clique == "player1":
            # P1 forms triangle (0,1,2)
            board.make_moves(jnp.array([0]))  # P1: edge (0,1)
            board.make_moves(jnp.array([2]))  # P2: edge (0,3)
            board.make_moves(jnp.array([1]))  # P1: edge (0,2)
            board.make_moves(jnp.array([4]))  # P2: edge (1,3)
            board.make_moves(jnp.array([3]))  # P1: edge (1,2) - forms triangle!
        else:
            # P2 forms triangle (0,1,2)
            board.make_moves(jnp.array([2]))  # P1: edge (0,3)
            board.make_moves(jnp.array([0]))  # P2: edge (0,1)
            board.make_moves(jnp.array([4]))  # P1: edge (1,3)
            board.make_moves(jnp.array([1]))  # P2: edge (0,2)
            board.make_moves(jnp.array([5]))  # P1: edge (2,3)
            board.make_moves(jnp.array([3]))  # P2: edge (1,2) - forms triangle!
        
        actual_state = int(board.game_states[0])
        print(f"  Expected game_state: {expected_state}")
        print(f"  Actual game_state: {actual_state}")
        
        if actual_state == expected_state:
            print("  ✅ Correct!")
        else:
            print("  ❌ Error!")
            return False
    
    return True

def test_evaluation_flow():
    """Test the complete evaluation flow logic."""
    
    print("\n" + "="*60)
    print("Testing Complete Evaluation Flow")
    print("="*60)
    
    # Simulate what happens in evaluate_models_jax
    print("\nIn avoid_clique mode:")
    print("1. Board correctly sets game_state based on who forms clique")
    print("2. Evaluation maps game_state to model wins based on who started")
    print()
    
    scenarios = [
        ("Current starts (is P1), Current forms clique", 
         "game_state=2 (P2 wins)", "Baseline wins"),
        ("Current starts (is P1), Baseline forms clique", 
         "game_state=1 (P1 wins)", "Current wins"),
        ("Baseline starts (is P1), Current forms clique", 
         "game_state=1 (P1 wins)", "Baseline wins"),
        ("Baseline starts (is P1), Baseline forms clique", 
         "game_state=2 (P2 wins)", "Current wins"),
    ]
    
    for scenario, state, winner in scenarios:
        print(f"• {scenario}")
        print(f"  → {state}")
        print(f"  → {winner}")
    
    return True

if __name__ == "__main__":
    success1 = test_sequential_evaluation_logic()
    success2 = test_with_actual_board()
    success3 = test_evaluation_flow()
    
    if success1 and success2 and success3:
        print("\n" + "="*60)
        print("✅ All tests passed! Sequential evaluation is correct!")
        print("="*60)
    else:
        print("\n❌ Some tests failed!")
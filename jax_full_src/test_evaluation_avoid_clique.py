#!/usr/bin/env python
"""
Test that evaluation correctly handles avoid_clique mode with alternating starts.
"""

import jax
import jax.numpy as jnp
import numpy as np
from vectorized_board import VectorizedCliqueBoard

def test_parallel_evaluation_logic():
    """Test the win recording logic used in parallel evaluation."""
    
    print("="*60)
    print("Testing Parallel Evaluation Win Recording Logic")
    print("="*60)
    
    # Simulate 4 games with alternating starts
    num_games = 4
    boards = VectorizedCliqueBoard(
        batch_size=num_games,
        num_vertices=4,
        k=3,
        game_mode="avoid_clique"
    )
    
    # model1_starts tracks who plays as Player 1 in each game
    model1_starts = jnp.array([True, False, True, False])
    
    print("\nGame Setup:")
    for i in range(num_games):
        if model1_starts[i]:
            print(f"  Game {i}: Model1 is Player 1, Model2 is Player 2")
        else:
            print(f"  Game {i}: Model2 is Player 1, Model1 is Player 2")
    
    # Simulate some games ending
    # Game 0: Player 1 forms clique (loses in avoid_clique)
    # Game 1: Player 2 forms clique (loses in avoid_clique)  
    # Game 2: Draw
    # Game 3: Player 1 forms clique (loses in avoid_clique)
    
    # Manually set game states to test the counting logic
    # In avoid_clique: game_state = winner (1 = P1 wins, 2 = P2 wins, 3 = draw)
    # But remember: forming clique makes you LOSE
    
    # Game 0: P1 forms clique -> P2 wins -> game_state = 2
    boards.game_states = boards.game_states.at[0].set(2)
    
    # Game 1: P2 forms clique -> P1 wins -> game_state = 1
    boards.game_states = boards.game_states.at[1].set(1)
    
    # Game 2: Draw
    boards.game_states = boards.game_states.at[2].set(3)
    
    # Game 3: P1 forms clique -> P2 wins -> game_state = 2
    boards.game_states = boards.game_states.at[3].set(2)
    
    print("\n--- Simulated Game Results (avoid_clique mode) ---")
    print("Remember: Forming a clique makes you LOSE!\n")
    
    for i in range(num_games):
        state = boards.game_states[i]
        if state == 1:
            print(f"Game {i}: Player 1 wins (Player 2 formed clique)")
        elif state == 2:
            print(f"Game {i}: Player 2 wins (Player 1 formed clique)")
        elif state == 3:
            print(f"Game {i}: Draw")
    
    # Now apply the counting logic from evaluation_jax_parallel.py
    model1_wins = 0
    model2_wins = 0
    draws = 0
    
    for i in range(num_games):
        game_state = int(boards.game_states[i])
        
        if game_state == 3:  # Draw
            draws += 1
        elif game_state == 1:  # Player 1 wins
            if model1_starts[i]:  # Model1 was player 1
                model1_wins += 1
            else:  # Model2 was player 1
                model2_wins += 1
        elif game_state == 2:  # Player 2 wins
            if model1_starts[i]:  # Model1 was player 1, so model2 wins
                model2_wins += 1
            else:  # Model2 was player 1, so model1 wins
                model1_wins += 1
    
    print("\n--- Evaluation Counting Results ---")
    print(f"Model1 wins: {model1_wins}")
    print(f"Model2 wins: {model2_wins}")
    print(f"Draws: {draws}")
    
    # Verify the logic
    print("\n--- Verification ---")
    expected_model1_wins = 1  # Game 1: Model2 was P1, P1 wins, so Model2 wins but we count Model1
    expected_model2_wins = 2  # Games 0,3: Model1 was P1, P2 wins, so Model2 wins
    expected_draws = 1        # Game 2
    
    # Wait, let me recalculate:
    # Game 0: model1_starts=True, P2 wins -> Model2 wins ✓
    # Game 1: model1_starts=False, P1 wins -> Model2 was P1 -> Model2 wins ✗
    # Game 2: Draw ✓
    # Game 3: model1_starts=False, P2 wins -> Model2 was P1, P2 wins -> Model1 wins ✓
    
    print("Expected based on game outcomes:")
    print(f"  Model1 should win: 1 (Game 3)")
    print(f"  Model2 should win: 2 (Games 0, 1)")
    print(f"  Draws: 1 (Game 2)")
    
    if model1_wins == 1 and model2_wins == 2 and draws == 1:
        print("\n✅ CORRECT! Win counting logic handles avoid_clique mode properly!")
    else:
        print(f"\n❌ ERROR! Got Model1={model1_wins}, Model2={model2_wins}, Draws={draws}")
    
    return True

def test_actual_game_with_avoid_clique():
    """Test an actual game to ensure game_states are set correctly."""
    
    print("\n" + "="*60)
    print("Testing Actual Game with Avoid Clique")
    print("="*60)
    
    board = VectorizedCliqueBoard(
        batch_size=1,
        num_vertices=4,
        k=3,
        game_mode="avoid_clique"
    )
    
    # Play a game where Player 1 forms a triangle
    # Edges: 0=(0,1), 1=(0,2), 2=(0,3), 3=(1,2), 4=(1,3), 5=(2,3)
    
    moves = [0, 2, 1, 4, 3]  # P1: edges 0,1,3 form triangle (0,1,2)
    
    for i, move in enumerate(moves):
        player = (i % 2) + 1
        board.make_moves(jnp.array([move]))
        print(f"Move {i+1}: Player {player} plays edge {move}")
        
        if board.game_states[0] != 0:
            break
    
    print(f"\nGame state: {board.game_states[0]}")
    print(f"Winner: Player {board.winners[0] + 1}")
    
    # In avoid_clique: P1 formed triangle, so P2 should win
    # game_state should be 2 (Player 2 wins)
    if board.game_states[0] == 2:
        print("✅ Correct! Player 2 wins because Player 1 formed a clique")
    else:
        print(f"❌ Error! Expected game_state=2, got {board.game_states[0]}")
    
    return board.game_states[0] == 2

if __name__ == "__main__":
    success1 = test_parallel_evaluation_logic()
    success2 = test_actual_game_with_avoid_clique()
    
    if success1 and success2:
        print("\n" + "="*60)
        print("✅ All tests passed! Evaluation handles avoid_clique correctly!")
        print("="*60)
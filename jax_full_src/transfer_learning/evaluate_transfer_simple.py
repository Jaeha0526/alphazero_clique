#!/usr/bin/env python
"""
Simple evaluation without MCTS - just using raw network outputs.
Much faster for testing transfer learning benefit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from vectorized_nn import ImprovedBatchedNeuralNetwork
from vectorized_board import VectorizedCliqueBoard


def play_games_without_mcts(model1, model2, num_games=10, n=13, k=4):
    """
    Play games using just the raw policy network (no MCTS).
    Much faster for evaluation.
    """
    model1_wins = 0
    model2_wins = 0
    draws = 0
    
    for game_idx in range(num_games):
        # Alternate who starts
        model1_starts = (game_idx % 2 == 0)
        
        # Initialize board
        board = VectorizedCliqueBoard(
            batch_size=1,
            num_vertices=n,
            k=k,
            game_mode='symmetric'
        )
        
        moves = 0
        max_moves = n * (n - 1) // 2
        
        # Play game
        while board.game_states[0] == 0 and moves < max_moves:
            current_player = int(board.current_players[0])
            
            # Determine which model plays
            use_model1 = (current_player == 0 and model1_starts) or (current_player == 1 and not model1_starts)
            
            # Get board features
            edge_indices, edge_features = board.get_features_for_nn()
            
            # Get policy from appropriate model
            if use_model1:
                policies, _ = model1.evaluate_batch(edge_indices, edge_features)
            else:
                policies, _ = model2.evaluate_batch(edge_indices, edge_features)
            
            # Apply valid moves mask
            valid_mask = board.get_valid_moves_mask()
            masked_policies = policies * valid_mask
            
            # Normalize and select action
            if jnp.sum(masked_policies) > 0:
                # Add small noise for variety
                noise = jax.random.uniform(jax.random.PRNGKey(game_idx * 1000 + moves), 
                                         shape=masked_policies.shape) * 0.01
                masked_policies = masked_policies + noise * valid_mask
                masked_policies = masked_policies / jnp.sum(masked_policies)
                action = jnp.argmax(masked_policies[0])
            else:
                # No valid moves
                break
            
            board.make_moves(jnp.array([action]))
            moves += 1
        
        # Check result
        game_state = int(board.game_states[0])
        
        if game_state == 3 or game_state == 0:  # Draw
            draws += 1
        elif game_state == 1:  # Player 1 wins
            if model1_starts:
                model1_wins += 1
            else:
                model2_wins += 1
        else:  # Player 2 wins
            if not model1_starts:
                model1_wins += 1
            else:
                model2_wins += 1
        
        if (game_idx + 1) % 5 == 0:
            print(f"  After {game_idx + 1} games: Model1={model1_wins}, Model2={model2_wins}, Draws={draws}")
    
    return model1_wins, model2_wins, draws


def main():
    print("="*60)
    print("Evaluation: Transferred vs Random (No MCTS)")
    print("="*60)
    
    # Load transferred model
    print("\n📦 Loading models...")
    print("  Loading transferred model (n=9 → n=13)...")
    
    with open("checkpoint_n13k4_transferred_v2.pkl", 'rb') as f:
        checkpoint = pickle.load(f)
    
    transferred_model = ImprovedBatchedNeuralNetwork(
        num_vertices=13,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=False
    )
    transferred_model.params = checkpoint['params']
    
    # Create random model
    print("  Creating random model...")
    random_model = ImprovedBatchedNeuralNetwork(
        num_vertices=13,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=False
    )
    
    # Play games
    print(f"\n🎮 Playing 20 games (no MCTS, raw policy only)...")
    print("  This tests if transferred knowledge helps even without search")
    
    start_time = time.time()
    transferred_wins, random_wins, draws = play_games_without_mcts(
        transferred_model, random_model, num_games=20, n=13, k=4
    )
    eval_time = time.time() - start_time
    
    # Results
    total = transferred_wins + random_wins + draws
    
    print(f"\n{'='*60}")
    print(f"📊 Results (20 games in {eval_time:.1f}s):")
    print(f"{'='*60}")
    print(f"  Transferred wins: {transferred_wins} ({transferred_wins/total*100:.1f}%)")
    print(f"  Random wins: {random_wins} ({random_wins/total*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/total*100:.1f}%)")
    
    print(f"\n💡 Transfer Learning Benefit:")
    win_rate_diff = (transferred_wins - random_wins) / total * 100
    
    if win_rate_diff > 20:
        print(f"  ✅ STRONG benefit: +{win_rate_diff:.1f}% win rate")
        print(f"  The transferred model significantly outperforms random!")
    elif win_rate_diff > 5:
        print(f"  ✓ Some benefit: +{win_rate_diff:.1f}% win rate")
        print(f"  The transferred model has an edge over random.")
    elif win_rate_diff > -5:
        print(f"  ≈ Roughly equal: {win_rate_diff:+.1f}% difference")
        print(f"  No clear advantage (might need MCTS to see benefit).")
    else:
        print(f"  ⚠️ Worse than random: {win_rate_diff:.1f}% win rate")
        print(f"  The transfer might not be effective for this size jump.")


if __name__ == "__main__":
    main()
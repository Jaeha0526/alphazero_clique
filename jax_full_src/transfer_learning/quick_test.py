#!/usr/bin/env python
"""
Quick test to show transfer learning benefit.
Plays just 2 games to demonstrate the difference.
"""

import jax
import jax.numpy as jnp
import pickle
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from vectorized_nn import ImprovedBatchedNeuralNetwork
from vectorized_board import VectorizedCliqueBoard

def play_single_game(model1, model2, model1_name, model2_name, n=13, k=4):
    """Play one game between two models (no MCTS, just raw policy)."""
    
    board = VectorizedCliqueBoard(
        batch_size=1,
        num_vertices=n,
        k=k,
        game_mode='symmetric'
    )
    
    moves = 0
    max_moves = n * (n - 1) // 2
    
    while board.game_states[0] == 0 and moves < max_moves:
        current_player = int(board.current_players[0])
        
        # Get board features
        edge_indices, edge_features = board.get_features_for_nn()
        
        # Choose model based on current player
        if current_player == 0:
            policies, values = model1.evaluate_batch(edge_indices, edge_features)
            player_name = model1_name
        else:
            policies, values = model2.evaluate_batch(edge_indices, edge_features)
            player_name = model2_name
        
        # Apply valid moves mask
        valid_mask = board.get_valid_moves_mask()
        masked_policies = policies * valid_mask
        
        # Select best move (greedy)
        if jnp.sum(masked_policies) > 0:
            masked_policies = masked_policies / jnp.sum(masked_policies)
            action = jnp.argmax(masked_policies[0])
        else:
            break
        
        board.make_moves(jnp.array([action]))
        moves += 1
        
        print(f"  Move {moves}: {player_name} plays edge {action}, value={values[0,0]:.3f}")
    
    # Check winner
    if board.game_states[0] == 1:
        winner = model1_name if 0 == 0 else model2_name
    elif board.game_states[0] == 2:
        winner = model2_name if 0 == 0 else model1_name
    else:
        winner = "Draw"
    
    print(f"  Result: {winner} wins in {moves} moves")
    return winner

print("="*60)
print("Quick Transfer Learning Test")
print("="*60)

# Load transferred model
print("\n1. Loading transferred model (trained on n=9)...")
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
print("2. Creating random model...")
random_model = ImprovedBatchedNeuralNetwork(
    num_vertices=13,
    hidden_dim=64,
    num_layers=3,
    asymmetric_mode=False
)

print("\n3. Playing 2 demonstration games (no MCTS, raw policy)...")
print("-"*40)

print("\nGame 1: Transferred starts")
winner1 = play_single_game(transferred_model, random_model, "Transferred", "Random", n=13, k=4)

print("\nGame 2: Random starts")
winner2 = play_single_game(random_model, transferred_model, "Random", "Transferred", n=13, k=4)

print("\n" + "="*60)
print("Summary:")
print(f"  Game 1: {winner1}")
print(f"  Game 2: {winner2}")

if winner1 == "Transferred" and winner2 == "Transferred":
    print("\n✅ Transfer learning shows clear benefit!")
elif winner1 == "Transferred" or winner2 == "Transferred":
    print("\n✓ Transfer learning shows some benefit")
else:
    print("\n⚠️ No clear benefit yet (may need MCTS for better play)")
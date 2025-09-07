#!/usr/bin/env python3
"""Detailed check of game endings - look at the actual moves and values."""

import pickle

def analyze_detailed(filepath):
    """Look at games in detail to understand the value assignments."""
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    games = data['saved_games']
    n = data['vertices']
    k = data['k']
    max_edges = n * (n - 1) // 2
    
    print(f"Detailed analysis of first 3 games")
    print(f"Graph: n={n}, k={k}, max_edges={max_edges}")
    print(f"Game mode: {data['game_mode']}")
    print("=" * 60)
    
    for game_idx in range(min(3, len(games))):
        game = games[game_idx]
        print(f"\n--- Game {game_idx + 1} ---")
        print(f"Total moves: {game['num_moves']}")
        
        # Look at first few and last few moves
        moves = game['moves']
        
        # First 3 moves
        print("\nFirst 3 moves:")
        for i in range(min(3, len(moves))):
            move = moves[i]
            print(f"  Move {i+1}: Player {move['player']}, Value: {move['value']:+.2f}")
            print(f"    Top action: {move['top_actions'][0]}")
        
        # Last 3 moves
        print("\nLast 3 moves:")
        for i in range(max(0, len(moves)-3), len(moves)):
            move = moves[i]
            print(f"  Move {i+1}: Player {move['player']}, Value: {move['value']:+.2f}")
            print(f"    Top action: {move['top_actions'][0]}")
        
        # Determine winner
        if game['num_moves'] < max_edges:
            last_move = moves[-1]
            last_player = last_move['player']
            # In avoid_clique, last player to move formed the clique and loses
            winner = 1 - last_player
            print(f"\nResult: Player {winner} wins (Player {last_player} formed 4-clique)")
            
            # Check if values are consistent
            # All moves by the loser should have negative values
            # All moves by the winner should have positive values
            loser_values = [m['value'] for m in moves if m['player'] == last_player]
            winner_values = [m['value'] for m in moves if m['player'] == winner]
            
            print(f"Value check:")
            print(f"  Loser (Player {last_player}) values: min={min(loser_values):.2f}, max={max(loser_values):.2f}")
            print(f"  Winner (Player {winner}) values: min={min(winner_values):.2f}, max={max(winner_values):.2f}")
            
            # The issue: values seem to be from the perspective of the move outcome, not the player
            # Let's check the pattern
            if all(v == -1.0 for v in loser_values) and all(v == 1.0 for v in winner_values):
                print("  ✓ Values are consistent (winner=+1, loser=-1)")
            else:
                print("  ⚠️ Values seem inconsistent or mixed")

if __name__ == "__main__":
    analyze_detailed("experiments/ramsey_n_10_k4/game_data/iteration_0.pkl")
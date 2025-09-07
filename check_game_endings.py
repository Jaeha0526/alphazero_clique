#!/usr/bin/env python3
"""Check that all games in saved data ended properly (either 4-clique formed or all edges colored)."""

import pickle
import numpy as np

def check_game_endings(filepath):
    """Analyze how games ended in the saved data."""
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    games = data['saved_games']
    n = data['vertices']
    k = data['k']
    max_edges = n * (n - 1) // 2  # Total edges in complete graph
    
    print(f"Analyzing {len(games)} games from {filepath}")
    print(f"Graph: n={n}, k={k}")
    print(f"Max possible edges: {max_edges}")
    print(f"Game mode: {data['game_mode']}")
    print("-" * 60)
    
    # Categorize endings
    endings = {
        'clique_formed': 0,
        'all_edges_colored': 0,
        'unclear': 0
    }
    
    game_lengths = []
    
    for i, game in enumerate(games):
        num_moves = game['num_moves']
        game_lengths.append(num_moves)
        
        # In avoid_clique mode, if game ends before all edges colored,
        # someone must have formed a k-clique (and lost)
        if num_moves == max_edges:
            endings['all_edges_colored'] += 1
            result = "DRAW (all edges colored, no 4-clique)"
        elif num_moves < max_edges:
            endings['clique_formed'] += 1
            # Last player to move formed the clique and lost
            last_move = game['moves'][-1]
            loser = last_move['player']
            winner = 1 - loser
            result = f"Player {winner} WON (Player {loser} formed 4-clique on move {num_moves})"
        else:
            endings['unclear'] += 1
            result = "UNCLEAR (more moves than edges?)"
        
        if i < 5:  # Show first 5 games
            print(f"Game {i+1}: {num_moves} moves - {result}")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Games where 4-clique was formed: {endings['clique_formed']} ({100*endings['clique_formed']/len(games):.1f}%)")
    print(f"  Games ending in draw (all edges colored): {endings['all_edges_colored']} ({100*endings['all_edges_colored']/len(games):.1f}%)")
    
    if endings['all_edges_colored'] > 0:
        print(f"\n🎉 FOUND {endings['all_edges_colored']} POTENTIAL RAMSEY COUNTEREXAMPLES!")
        print("  These are games where all edges were colored without forming a 4-clique.")
    
    if endings['unclear'] > 0:
        print(f"  Unclear endings: {endings['unclear']}")
    
    print(f"\nGame length statistics:")
    print(f"  Min: {min(game_lengths)} moves")
    print(f"  Max: {max(game_lengths)} moves")
    print(f"  Mean: {np.mean(game_lengths):.1f} moves")
    print(f"  Median: {np.median(game_lengths):.0f} moves")
    
    # Check value distribution for last moves
    print(f"\nFinal move values (who won from each player's perspective):")
    player_0_wins = 0
    player_1_wins = 0
    draws = 0
    
    for game in games:
        last_move = game['moves'][-1]
        last_player = last_move['player']
        value = last_move['value']
        
        # In avoid_clique, negative value means current player loses (forms clique)
        if game['num_moves'] == max_edges:
            draws += 1
        elif value < 0:  # Current player loses
            if last_player == 0:
                player_1_wins += 1
            else:
                player_0_wins += 1
        else:  # This shouldn't happen in avoid_clique
            print(f"  Warning: Unexpected positive value {value} for last player {last_player}")
    
    print(f"  Player 0 wins: {player_0_wins} ({100*player_0_wins/len(games):.1f}%)")
    print(f"  Player 1 wins: {player_1_wins} ({100*player_1_wins/len(games):.1f}%)")
    print(f"  Draws: {draws} ({100*draws/len(games):.1f}%)")
    
    return endings

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "experiments/ramsey_n_10_k4/game_data/iteration_0.pkl"
    
    check_game_endings(filepath)
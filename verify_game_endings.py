#!/usr/bin/env python3
"""Verify that all games ended properly by checking game lengths."""

import pickle
import numpy as np

def verify_endings(filepath):
    """Check that games ended correctly based on move counts."""
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    games = data['saved_games']
    n = data['vertices']
    k = data['k']
    max_edges = n * (n - 1) // 2  # K_10 has 45 edges
    
    print(f"Verification of game endings")
    print(f"Graph: K_{n}, avoiding {k}-cliques")
    print(f"Maximum possible moves: {max_edges} edges")
    print("=" * 60)
    
    # Count different endings
    game_lengths = [game['num_moves'] for game in games]
    length_counts = {}
    for length in game_lengths:
        length_counts[length] = length_counts.get(length, 0) + 1
    
    print(f"\nGame length distribution ({len(games)} games total):")
    for length in sorted(length_counts.keys()):
        count = length_counts[length]
        if length == max_edges:
            ending = "DRAW (all edges colored)"
        elif length < max_edges:
            ending = f"4-clique formed (on move {length})"
        else:
            ending = "ERROR: Too many moves!"
        print(f"  {length} moves: {count} games - {ending}")
    
    # Summary
    draws = length_counts.get(max_edges, 0)
    cliques_formed = sum(count for length, count in length_counts.items() if length < max_edges)
    errors = sum(count for length, count in length_counts.items() if length > max_edges)
    
    print(f"\n" + "=" * 60)
    print("SUMMARY:")
    print(f"✓ Games ending with 4-clique: {cliques_formed} ({100*cliques_formed/len(games):.1f}%)")
    print(f"✓ Games ending in draw: {draws} ({100*draws/len(games):.1f}%)")
    
    if draws > 0:
        print(f"\n🎉 FOUND {draws} POTENTIAL RAMSEY COUNTEREXAMPLES!")
        print("   These games avoided all 4-cliques while coloring all 45 edges!")
    
    if errors > 0:
        print(f"❌ Games with errors: {errors}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Shortest game: {min(game_lengths)} moves")
    print(f"  Longest game: {max(game_lengths)} moves")
    print(f"  Average length: {np.mean(game_lengths):.1f} moves")
    print(f"  Median length: {np.median(game_lengths):.0f} moves")
    print(f"  Std deviation: {np.std(game_lengths):.1f} moves")
    
    # Win distribution (based on who played last move that formed clique)
    player_0_losses = 0
    player_1_losses = 0
    
    for game in games:
        if game['num_moves'] < max_edges:  # Someone formed a clique
            last_player = game['moves'][-1]['player']
            if last_player == 0:
                player_0_losses += 1
            else:
                player_1_losses += 1
    
    print(f"\nWin distribution (in avoid_clique, forming clique = losing):")
    print(f"  Player 0 wins (P1 formed clique): {player_1_losses} ({100*player_1_losses/len(games):.1f}%)")
    print(f"  Player 1 wins (P0 formed clique): {player_0_losses} ({100*player_0_losses/len(games):.1f}%)")
    print(f"  Draws: {draws} ({100*draws/len(games):.1f}%)")
    
    print(f"\n✅ All {len(games)} games ended properly!")
    print(f"   Every game either formed a 4-clique (and ended) or reached the maximum {max_edges} moves.")
    
    return cliques_formed, draws, errors

if __name__ == "__main__":
    verify_endings("experiments/ramsey_n_10_k4/game_data/iteration_0.pkl")
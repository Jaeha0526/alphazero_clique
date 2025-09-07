#!/usr/bin/env python3
"""Check if games are unique or duplicates."""

import pickle
import numpy as np

def check_uniqueness(filepath):
    """Check if games are unique or identical."""
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    games = data['saved_games']
    print(f"Checking uniqueness of {len(games)} games")
    print("=" * 60)
    
    # Extract move sequences for comparison
    game_sequences = []
    for game in games:
        # Get the sequence of actions taken
        action_sequence = []
        for move in game['moves']:
            # Get the top action (edge chosen)
            action = move['top_actions'][0][0]
            action_sequence.append(action)
        game_sequences.append(tuple(action_sequence))  # Convert to tuple for hashing
    
    # Check for exact duplicates
    unique_sequences = set(game_sequences)
    
    print(f"Total games: {len(games)}")
    print(f"Unique game sequences: {len(unique_sequences)}")
    
    if len(unique_sequences) == 1:
        print("\n❌ ALL GAMES ARE IDENTICAL!")
        print("Every single game played the exact same sequence of moves.")
    elif len(unique_sequences) < len(games):
        print(f"\n⚠️ Found duplicate games!")
        # Count duplicates
        from collections import Counter
        counts = Counter(game_sequences)
        print(f"Duplicate distribution:")
        for seq, count in counts.most_common(10):
            if count > 1:
                print(f"  - Same sequence appears {count} times")
    else:
        print("\n✓ All games are unique!")
    
    # Check first few moves across games
    print("\nFirst 5 moves of first 5 games:")
    for i in range(min(5, len(games))):
        first_moves = []
        for j in range(min(5, len(games[i]['moves']))):
            action = games[i]['moves'][j]['top_actions'][0][0]
            first_moves.append(action)
        print(f"  Game {i+1}: {first_moves}")
    
    # Check if they all start the same
    first_moves_all = []
    for game in games:
        if len(game['moves']) > 0:
            first_action = game['moves'][0]['top_actions'][0][0]
            first_moves_all.append(first_action)
    
    unique_first_moves = set(first_moves_all)
    print(f"\nFirst move diversity:")
    print(f"  Unique first moves: {len(unique_first_moves)}")
    if len(unique_first_moves) == 1:
        print(f"  ⚠️ All games start with the same move: {first_moves_all[0]}")
    else:
        from collections import Counter
        first_move_counts = Counter(first_moves_all)
        print(f"  First move distribution (top 5):")
        for move, count in first_move_counts.most_common(5):
            print(f"    Move {move}: {count} times ({100*count/len(games):.1f}%)")
    
    # Check last moves
    last_moves_all = []
    for game in games:
        if len(game['moves']) > 0:
            last_action = game['moves'][-1]['top_actions'][0][0]
            last_moves_all.append(last_action)
    
    unique_last_moves = set(last_moves_all)
    print(f"\nLast move diversity:")
    print(f"  Unique last moves: {len(unique_last_moves)}")
    if len(unique_last_moves) == 1:
        print(f"  ⚠️ All games end with the same move: {last_moves_all[0]}")
    
    # Check move probabilities for determinism
    print("\nChecking for deterministic play:")
    deterministic_count = 0
    for game_idx, game in enumerate(games[:5]):  # Check first 5 games
        print(f"\n  Game {game_idx+1} - Move probabilities:")
        for move_idx in range(min(3, len(game['moves']))):
            move = game['moves'][move_idx]
            top_prob = move['top_actions'][0][1]
            print(f"    Move {move_idx+1}: top action prob = {top_prob:.3f}")
            if top_prob == 1.0:
                deterministic_count += 1
    
    if deterministic_count > 0:
        print(f"\n⚠️ Found {deterministic_count} deterministic moves (probability = 1.0)")
        print("  This suggests MCTS is finding a single dominant path.")

if __name__ == "__main__":
    check_uniqueness("experiments/ramsey_n_10_k4/game_data/iteration_0.pkl")
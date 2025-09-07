#!/usr/bin/env python3
"""Read individual games from the new format with game boundaries."""

import pickle
import sys
from pathlib import Path

def extract_games(filepath):
    """Extract individual games using the game boundary information."""
    
    print(f"Loading {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nData format: {'NEW with boundaries' if 'games_info' in data else 'NEW without boundaries' if 'training_data' in data else 'OLD'}")
    
    if 'games_info' not in data:
        print("❌ No game boundary information available")
        print("   This file was saved before the boundary tracking was added")
        return
    
    games_info = data['games_info']
    training_data = data['training_data']
    
    print(f"\n✅ Found {len(games_info)} games with boundaries")
    print(f"Total training examples: {len(training_data)}")
    
    # Extract first few games as examples
    for game_idx in range(min(3, len(games_info))):
        game = games_info[game_idx]
        print(f"\n--- Game {game['game_id'] + 1} ---")
        print(f"  Winner: Player {game['winner']}")
        print(f"  Length: {game['num_moves']} moves")
        print(f"  Data range: [{game['start_idx']}:{game['end_idx']}]")
        
        # Extract the moves for this game
        game_moves = training_data[game['start_idx']:game['end_idx']]
        
        # Show first and last few moves
        if game_moves:
            print(f"\n  First 3 moves:")
            for i, move in enumerate(game_moves[:3]):
                print(f"    Move {i+1}: Player {move['player']}, Value {move['value']:+.2f}")
                # Show top action from policy
                policy = move['policy']
                top_action = max(enumerate(policy), key=lambda x: x[1])
                print(f"      Top action: {top_action[0]} (prob={top_action[1]:.3f})")
            
            if len(game_moves) > 3:
                print(f"\n  Last move:")
                move = game_moves[-1]
                print(f"    Move {len(game_moves)}: Player {move['player']}, Value {move['value']:+.2f}")
                policy = move['policy']
                top_action = max(enumerate(policy), key=lambda x: x[1])
                print(f"      Top action: {top_action[0]} (prob={top_action[1]:.3f})")
    
    # Check for draws (45-move games in K_10)
    max_moves = data['vertices'] * (data['vertices'] - 1) // 2
    draws = [g for g in games_info if g['num_moves'] == max_moves]
    if draws:
        print(f"\n🎉 Found {len(draws)} potential Ramsey counterexamples (games reaching {max_moves} moves)!")
        for draw in draws[:5]:  # Show first 5
            print(f"  Game {draw['game_id'] + 1}: {draw['num_moves']} moves")
    
    # Game length distribution
    lengths = [g['num_moves'] for g in games_info]
    print(f"\nGame length statistics:")
    print(f"  Min: {min(lengths)} moves")
    print(f"  Max: {max(lengths)} moves")
    print(f"  Average: {sum(lengths)/len(lengths):.1f} moves")
    
    return games_info, training_data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Try latest file
        latest = max(Path("experiments").glob("*/game_data/*.pkl"), key=lambda x: x.stat().st_mtime)
        filepath = latest
        print(f"Using most recent: {filepath}")
    
    extract_games(filepath)
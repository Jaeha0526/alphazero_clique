#!/usr/bin/env python3
"""Analyze the saved training data (new format without flawed reconstruction)."""

import pickle
import sys
from pathlib import Path

def analyze_training_data(filepath):
    """Analyze saved training data."""
    
    print(f"Loading {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print("\n" + "="*60)
    print("Training Data Analysis")
    print("="*60)
    
    # Basic info
    print(f"Iteration: {data.get('iteration', 'N/A')}")
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"Game mode: {data.get('game_mode', 'N/A')}")
    print(f"Graph: n={data.get('vertices', 'N/A')}, k={data.get('k', 'N/A')}")
    print(f"Games played: {data.get('num_games_played', 'N/A')}")
    print(f"Total training examples: {data.get('total_training_examples', 'N/A')}")
    print(f"Examples saved in file: {data.get('num_examples_saved', 'N/A')}")
    print(f"Full data mode: {data.get('is_full_data', False)}")
    
    # Game statistics (if available)
    if 'game_stats' in data:
        stats = data['game_stats']
        print("\nGame Statistics:")
        print(f"  Average game length: {stats.get('avg_game_length', 'N/A')}")
        print(f"  Total moves: {stats.get('total_moves', 'N/A')}")
        
        if 'game_length_distribution' in stats and stats['game_length_distribution']:
            dist = stats['game_length_distribution']
            print(f"\n  Game length distribution:")
            for length in sorted(dist.keys(), key=lambda x: int(x)):
                count = dist[length]
                print(f"    {length} moves: {count} games")
            
            # Check for draws (45 moves in K_10)
            max_moves = data.get('vertices', 0) * (data.get('vertices', 0) - 1) // 2
            if str(max_moves) in dist:
                draws = dist[str(max_moves)]
                print(f"\n  🎉 POTENTIAL RAMSEY COUNTEREXAMPLES: {draws} games reached {max_moves} moves!")
        
        # Win stats for asymmetric/avoid_clique
        if data.get('game_mode') in ['asymmetric', 'avoid_clique']:
            attacker_wins = stats.get('attacker_wins', 0)
            defender_wins = stats.get('defender_wins', 0)
            total = attacker_wins + defender_wins
            if total > 0:
                print(f"\n  Win distribution:")
                print(f"    Attacker/Player0 wins: {attacker_wins} ({100*attacker_wins/total:.1f}%)")
                print(f"    Defender/Player1 wins: {defender_wins} ({100*defender_wins/total:.1f}%)")
    
    # Sample some training examples
    if 'training_data' in data and data['training_data']:
        examples = data['training_data']
        print(f"\nTraining data sample (first 5 examples):")
        for i, example in enumerate(examples[:5]):
            print(f"\n  Example {i+1}:")
            print(f"    Player: {example.get('player', 'N/A')}")
            print(f"    Value: {example.get('value', 'N/A')}")
            if 'policy' in example and len(example['policy']) > 0:
                # Show top 3 actions from policy
                policy = example['policy']
                top_actions = sorted(enumerate(policy), key=lambda x: x[1], reverse=True)[:3]
                print(f"    Top 3 actions:")
                for action, prob in top_actions:
                    print(f"      Action {action}: {prob:.3f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Try to find the most recent file
        game_data_dirs = list(Path("experiments").glob("*/game_data"))
        if game_data_dirs:
            latest_dir = max(game_data_dirs, key=lambda x: x.stat().st_mtime)
            files = list(latest_dir.glob("*.pkl"))
            if files:
                filepath = max(files, key=lambda x: x.stat().st_mtime)
                print(f"Using most recent file: {filepath}")
            else:
                print("No .pkl files found")
                sys.exit(1)
        else:
            print("No game_data directories found")
            sys.exit(1)
    
    analyze_training_data(filepath)
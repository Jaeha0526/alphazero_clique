#!/usr/bin/env python3
"""Analyze saved game data from AlphaZero training."""

import pickle
import argparse
import os
import numpy as np
from pathlib import Path

def analyze_game_file(filepath):
    """Analyze a single game data file."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(filepath)}")
    print('='*60)
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Iteration: {data['iteration']}")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Game Mode: {data['game_mode']}")
    print(f"Graph: n={data['vertices']}, k={data['k']}")
    print(f"Total training examples: {data.get('total_training_examples', data.get('total_games', 'N/A'))}")
    print(f"Games played: {data.get('num_games_played', 'N/A')}")
    
    # Handle both old and new data format
    games = data.get('saved_games', data.get('sample_games', []))
    is_full_data = data.get('is_full_data', False)
    
    if is_full_data:
        print(f"FULL DATA: All {len(games)} games saved")
    else:
        print(f"Sample games saved: {len(games)}")
    
    # Analyze games
    game_lengths = [game['num_moves'] for game in games]
    print(f"\nGame lengths: min={min(game_lengths)}, max={max(game_lengths)}, avg={np.mean(game_lengths):.1f}")
    
    # Analyze move patterns
    for game_idx, game in enumerate(games[:3]):  # Show first 3 games
        print(f"\n--- Game {game_idx + 1} ---")
        print(f"  Total moves: {game['num_moves']}")
        
        # Show first few moves and last move
        for move_idx in [0, 1, 2, -1]:
            if abs(move_idx) <= len(game['moves']):
                move = game['moves'][move_idx]
                print(f"\n  Move {move['move'] + 1}:")
                print(f"    Player: {move['player']}", end="")
                if move['player_role'] is not None:
                    role = "Attacker" if move['player_role'] == 0 else "Defender"
                    print(f" ({role})", end="")
                print()
                print(f"    Final value: {move['value']:+.2f}")
                print(f"    Top 3 action probabilities:")
                for action, prob in move['top_actions'][:3]:
                    print(f"      Action {action}: {prob:.3f}")
    
    # Analyze policy entropy over moves
    print("\n--- Policy Analysis ---")
    for game in data['sample_games'][:1]:  # Just first game
        entropies = []
        for move in game['moves']:
            # Calculate entropy from top actions (approximate)
            probs = [p for _, p in move['top_actions']]
            # Add remaining probability mass
            remaining_prob = max(0, 1.0 - sum(probs))
            if remaining_prob > 0:
                # Distribute among unseen actions
                num_actions = data['vertices'] * (data['vertices'] - 1) // 2
                unseen_actions = num_actions - len(probs)
                if unseen_actions > 0:
                    probs.extend([remaining_prob / unseen_actions] * min(unseen_actions, 10))
            
            # Calculate entropy
            entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            entropies.append(entropy)
        
        print(f"  Policy entropy progression (Game 1):")
        print(f"    Early game (moves 1-3): {np.mean(entropies[:3]):.3f}")
        if len(entropies) > 6:
            print(f"    Mid game: {np.mean(entropies[3:-3]):.3f}")
        if len(entropies) >= 3:
            print(f"    Late game (last 3 moves): {np.mean(entropies[-3:]):.3f}")

def compare_iterations(experiment_dir):
    """Compare game data across multiple iterations."""
    pattern = "iteration_*.pkl"
    files = sorted(Path(experiment_dir).glob(pattern))
    
    if not files:
        print(f"No game data files found in {experiment_dir}")
        return
    
    print(f"\n{'='*60}")
    print("ITERATION COMPARISON")
    print('='*60)
    
    iteration_stats = []
    for filepath in files:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        game_lengths = [game['num_moves'] for game in data['sample_games']]
        
        # Calculate average entropy in early moves
        early_entropies = []
        for game in data['sample_games']:
            for move in game['moves'][:3]:  # First 3 moves
                probs = [p for _, p in move['top_actions']]
                if probs:
                    max_prob = max(probs)
                    early_entropies.append(-np.log(max_prob + 1e-10))  # Simple measure
        
        iteration_stats.append({
            'iteration': data['iteration'],
            'avg_length': np.mean(game_lengths),
            'avg_early_entropy': np.mean(early_entropies) if early_entropies else 0
        })
    
    # Print comparison table
    print(f"\n{'Iteration':<10} {'Avg Length':<12} {'Early Entropy':<15} {'Trend'}")
    print("-" * 50)
    
    for i, stats in enumerate(iteration_stats):
        trend = ""
        if i > 0:
            length_change = stats['avg_length'] - iteration_stats[i-1]['avg_length']
            entropy_change = stats['avg_early_entropy'] - iteration_stats[i-1]['avg_early_entropy']
            if length_change > 0.5:
                trend += "Longer games, "
            elif length_change < -0.5:
                trend += "Shorter games, "
            
            if entropy_change < -0.1:
                trend += "More confident"
            elif entropy_change > 0.1:
                trend += "More exploratory"
        
        print(f"{stats['iteration']:<10} {stats['avg_length']:<12.1f} {stats['avg_early_entropy']:<15.3f} {trend}")
    
    # Check for learning progress
    if len(iteration_stats) >= 2:
        print("\n--- Learning Indicators ---")
        
        # Check if games are getting more decisive
        first_lengths = iteration_stats[0]['avg_length']
        last_lengths = iteration_stats[-1]['avg_length']
        
        if abs(last_lengths - first_lengths) > 1:
            if last_lengths < first_lengths:
                print("✓ Games getting shorter - potentially finding winning patterns faster")
            else:
                print("⚠ Games getting longer - possibly more defensive play")
        
        # Check if policies are becoming more confident
        first_entropy = iteration_stats[0]['avg_early_entropy']
        last_entropy = iteration_stats[-1]['avg_early_entropy']
        
        if last_entropy < first_entropy * 0.8:
            print("✓ Policies becoming more confident - model learning strong patterns")
        elif last_entropy > first_entropy * 1.2:
            print("⚠ Policies becoming less confident - might need more training")
        
        print("\nRecommendation:")
        if last_entropy < first_entropy and abs(last_lengths - first_lengths) < 3:
            print("  Model appears to be learning well!")
        else:
            print("  Consider adjusting hyperparameters or training longer")

def main():
    parser = argparse.ArgumentParser(description='Analyze AlphaZero game data')
    parser.add_argument('path', help='Path to game data file or experiment directory')
    parser.add_argument('--compare', action='store_true', help='Compare all iterations in directory')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        if args.compare:
            compare_iterations(args.path)
        else:
            # Analyze all game data files in directory (try both patterns)
            pattern = "iteration_*.pkl"
            files = sorted(Path(args.path).glob(pattern))
            if not files:
                # Try old pattern for backward compatibility
                pattern = "game_data_iter_*.pkl"
                files = sorted(Path(args.path).glob(pattern))
            for filepath in files:
                analyze_game_file(filepath)
    else:
        analyze_game_file(args.path)

if __name__ == "__main__":
    main()
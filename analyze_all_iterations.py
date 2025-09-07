#!/usr/bin/env python3
"""Analyze playing patterns across all three iterations to see learning progression."""

import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def edge_to_vertices(edge_idx, n):
    """Convert edge index to vertex pair."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if count == edge_idx:
                return i, j
            count += 1
    return None, None

def analyze_iteration(filepath, iteration_num):
    """Analyze a single iteration's patterns."""
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    n = data['vertices']
    k = data['k']
    games_info = data['games_info']
    training_data = data['training_data']
    
    # Basic statistics
    game_lengths = [g['num_moves'] for g in games_info]
    max_moves = n * (n - 1) // 2
    complete_games = sum(1 for l in game_lengths if l == max_moves)
    
    # Win statistics
    p0_wins = sum(1 for g in games_info if g['winner'] == 0)
    p1_wins = sum(1 for g in games_info if g['winner'] == 1)
    draws = len(games_info) - p0_wins - p1_wins
    
    # Opening moves
    opening_moves = defaultdict(int)
    for game in games_info:
        if game['start_idx'] < game['end_idx']:
            first_move = training_data[game['start_idx']]
            first_action = first_move.get('action')
            if first_action is not None:
                opening_moves[first_action] += 1
    
    # Edge frequency
    edge_frequency = np.zeros(max_moves)
    for game in games_info:
        game_moves = training_data[game['start_idx']:game['end_idx']]
        for move in game_moves:
            action = move.get('action')
            if action is not None:
                edge_frequency[action] += 1
    
    # Policy entropy and value confidence
    early_entropy = []
    late_entropy = []
    early_values = []
    late_values = []
    
    for game in games_info:
        game_moves = training_data[game['start_idx']:game['end_idx']]
        game_length = len(game_moves)
        
        for move_idx, move in enumerate(game_moves):
            policy = move['policy']
            value = abs(move['value'])
            
            # Calculate entropy
            valid_probs = policy[policy > 0.001]
            if len(valid_probs) > 0:
                entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-10))
                
                progress = move_idx / game_length if game_length > 0 else 0
                
                if progress < 0.25:
                    early_entropy.append(entropy)
                    early_values.append(value)
                elif progress > 0.75:
                    late_entropy.append(entropy)
                    late_values.append(value)
    
    # Game ending distribution
    ending_distribution = defaultdict(int)
    for game in games_info:
        if game['num_moves'] < max_moves:
            ending_distribution[game['num_moves']] += 1
    
    return {
        'iteration': iteration_num,
        'total_games': len(games_info),
        'game_lengths': game_lengths,
        'avg_length': np.mean(game_lengths),
        'min_length': min(game_lengths),
        'max_length': max(game_lengths),
        'complete_games': complete_games,
        'p0_wins': p0_wins,
        'p1_wins': p1_wins,
        'draws': draws,
        'opening_moves': opening_moves,
        'edge_frequency': edge_frequency,
        'early_entropy': np.mean(early_entropy) if early_entropy else 0,
        'late_entropy': np.mean(late_entropy) if late_entropy else 0,
        'early_values': np.mean(early_values) if early_values else 0,
        'late_values': np.mean(late_values) if late_values else 0,
        'ending_distribution': ending_distribution
    }

def print_iteration_analysis(stats):
    """Print detailed analysis for one iteration."""
    
    print(f"\n{'='*70}")
    print(f"ITERATION {stats['iteration']} DETAILED ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nGame Statistics:")
    print(f"  Total games: {stats['total_games']}")
    print(f"  Length: min={stats['min_length']}, max={stats['max_length']}, avg={stats['avg_length']:.1f}")
    print(f"  Complete games (45 moves): {stats['complete_games']} ({100*stats['complete_games']/stats['total_games']:.1f}%)")
    
    print(f"\nWin Distribution:")
    total = stats['total_games']
    print(f"  Player 0: {stats['p0_wins']} wins ({100*stats['p0_wins']/total:.1f}%)")
    print(f"  Player 1: {stats['p1_wins']} wins ({100*stats['p1_wins']/total:.1f}%)")
    print(f"  Draws: {stats['draws']} ({100*stats['draws']/total:.1f}%)")
    
    print(f"\nTop 5 Opening Moves:")
    sorted_openings = sorted(stats['opening_moves'].items(), key=lambda x: x[1], reverse=True)[:5]
    for action, count in sorted_openings:
        v1, v2 = edge_to_vertices(action, 10)
        pct = 100 * count / stats['total_games']
        print(f"  Edge ({v1},{v2}): {count} times ({pct:.1f}%)")
    
    print(f"\nPolicy & Value Metrics:")
    print(f"  Early game entropy: {stats['early_entropy']:.3f}")
    print(f"  Late game entropy: {stats['late_entropy']:.3f}")
    print(f"  Early game |value|: {stats['early_values']:.3f}")
    print(f"  Late game |value|: {stats['late_values']:.3f}")
    
    print(f"\nMost Common Game Endings:")
    sorted_endings = sorted(stats['ending_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
    for move_num, count in sorted_endings:
        print(f"  Move {move_num}: {count} games")

def compare_all_iterations():
    """Compare patterns across all three iterations."""
    
    print("="*70)
    print("LEARNING PROGRESSION ACROSS ALL ITERATIONS")
    print("="*70)
    
    # Analyze all iterations
    stats = []
    for i in range(3):
        filepath = f"/workspace/alphazero_clique/experiments/ramsey_n_10_k4_new2/game_data/iteration_{i}.pkl"
        stats.append(analyze_iteration(filepath, i))
    
    # Print detailed analysis for each
    for stat in stats:
        print_iteration_analysis(stat)
    
    # Comparative trends
    print(f"\n{'='*70}")
    print("LEARNING TRENDS (Iteration 0 → 1 → 2)")
    print(f"{'='*70}")
    
    print(f"\n1. GAME LENGTH PROGRESSION:")
    for s in stats:
        print(f"   Iter {s['iteration']}: avg={s['avg_length']:.1f}, "
              f"min={s['min_length']}, max={s['max_length']}")
    print(f"   Trend: {stats[0]['avg_length']:.1f} → {stats[1]['avg_length']:.1f} → {stats[2]['avg_length']:.1f}")
    
    print(f"\n2. COMPLETE GAMES (Potential Ramsey Counterexamples):")
    for s in stats:
        print(f"   Iter {s['iteration']}: {s['complete_games']} games ({100*s['complete_games']/s['total_games']:.1f}%)")
    print(f"   Trend: {stats[0]['complete_games']} → {stats[1]['complete_games']} → {stats[2]['complete_games']}")
    
    print(f"\n3. WIN BALANCE SHIFT:")
    for s in stats:
        p0_pct = 100*s['p0_wins']/s['total_games']
        p1_pct = 100*s['p1_wins']/s['total_games']
        print(f"   Iter {s['iteration']}: P0={p0_pct:.1f}%, P1={p1_pct:.1f}%, Balance={p0_pct-p1_pct:+.1f}%")
    
    print(f"\n4. POLICY ENTROPY (Exploration → Exploitation):")
    print(f"   Early game: {stats[0]['early_entropy']:.3f} → {stats[1]['early_entropy']:.3f} → {stats[2]['early_entropy']:.3f}")
    print(f"   Late game:  {stats[0]['late_entropy']:.3f} → {stats[1]['late_entropy']:.3f} → {stats[2]['late_entropy']:.3f}")
    
    print(f"\n5. VALUE CONFIDENCE EVOLUTION:")
    print(f"   Early game: {stats[0]['early_values']:.3f} → {stats[1]['early_values']:.3f} → {stats[2]['early_values']:.3f}")
    print(f"   Late game:  {stats[0]['late_values']:.3f} → {stats[1]['late_values']:.3f} → {stats[2]['late_values']:.3f}")
    
    # Opening move evolution
    print(f"\n6. OPENING STRATEGY EVOLUTION:")
    for s in stats:
        top_opening = sorted(s['opening_moves'].items(), key=lambda x: x[1], reverse=True)[0]
        action, count = top_opening
        v1, v2 = edge_to_vertices(action, 10)
        pct = 100 * count / s['total_games']
        print(f"   Iter {s['iteration']}: Edge ({v1},{v2}) - {count} times ({pct:.1f}%)")
    
    # Edge usage changes
    print(f"\n7. STRATEGIC EDGE PREFERENCE CHANGES:")
    
    # Compare iteration 0→1
    edge_diff_01 = stats[1]['edge_frequency'] - stats[0]['edge_frequency']
    most_increased_01 = np.argsort(edge_diff_01)[-3:][::-1]
    
    # Compare iteration 1→2
    edge_diff_12 = stats[2]['edge_frequency'] - stats[1]['edge_frequency']
    most_increased_12 = np.argsort(edge_diff_12)[-3:][::-1]
    
    print(f"\n   Iter 0→1 most increased edges:")
    for action in most_increased_01:
        v1, v2 = edge_to_vertices(action, 10)
        change = edge_diff_01[action]
        print(f"     Edge ({v1},{v2}): {change:+.0f}")
    
    print(f"\n   Iter 1→2 most increased edges:")
    for action in most_increased_12:
        v1, v2 = edge_to_vertices(action, 10)
        change = edge_diff_12[action]
        print(f"     Edge ({v1},{v2}): {change:+.0f}")
    
    # Identify consistently popular edges across all iterations
    print(f"\n8. CONSISTENTLY IMPORTANT EDGES:")
    total_usage = stats[0]['edge_frequency'] + stats[1]['edge_frequency'] + stats[2]['edge_frequency']
    top_edges = np.argsort(total_usage)[-5:][::-1]
    for action in top_edges:
        v1, v2 = edge_to_vertices(action, 10)
        uses = [int(stats[i]['edge_frequency'][action]) for i in range(3)]
        print(f"   Edge ({v1},{v2}): {uses[0]}→{uses[1]}→{uses[2]} uses")
    
    # Check for learning convergence
    print(f"\n9. LEARNING CONVERGENCE INDICATORS:")
    
    # Check if game lengths are stabilizing
    length_changes = [abs(stats[1]['avg_length'] - stats[0]['avg_length']),
                      abs(stats[2]['avg_length'] - stats[1]['avg_length'])]
    print(f"   Avg length change: {length_changes[0]:.1f} → {length_changes[1]:.1f} (smaller = converging)")
    
    # Check if entropy is stabilizing
    entropy_changes = [abs(stats[1]['late_entropy'] - stats[0]['late_entropy']),
                       abs(stats[2]['late_entropy'] - stats[1]['late_entropy'])]
    print(f"   Late entropy change: {entropy_changes[0]:.3f} → {entropy_changes[1]:.3f}")
    
    # Check if opening preferences are stabilizing
    top_open_0 = max(stats[0]['opening_moves'].items(), key=lambda x: x[1])[0]
    top_open_1 = max(stats[1]['opening_moves'].items(), key=lambda x: x[1])[0]
    top_open_2 = max(stats[2]['opening_moves'].items(), key=lambda x: x[1])[0]
    
    if top_open_1 == top_open_2:
        print(f"   Opening preference: CONVERGING (same top move in iter 1 and 2)")
    else:
        print(f"   Opening preference: Still exploring different strategies")

if __name__ == "__main__":
    compare_all_iterations()
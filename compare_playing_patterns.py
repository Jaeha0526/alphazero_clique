#!/usr/bin/env python3
"""Compare playing patterns between iterations to see learning progress."""

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

def analyze_patterns(filepath, iteration_name):
    """Analyze playing patterns in an iteration."""
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    n = data['vertices']
    k = data['k']
    games_info = data['games_info']
    training_data = data['training_data']
    
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration_name} ANALYSIS")
    print(f"{'='*70}")
    
    # Game length statistics
    game_lengths = [g['num_moves'] for g in games_info]
    max_moves = n * (n - 1) // 2
    complete_games = sum(1 for l in game_lengths if l == max_moves)
    
    print(f"Total games: {len(games_info)}")
    print(f"Game lengths: min={min(game_lengths)}, max={max(game_lengths)}, avg={np.mean(game_lengths):.1f}")
    print(f"Complete games (45 moves): {complete_games} ({100*complete_games/len(games_info):.1f}%)")
    
    # Win statistics
    p0_wins = sum(1 for g in games_info if g['winner'] == 0)
    p1_wins = sum(1 for g in games_info if g['winner'] == 1)
    draws = len(games_info) - p0_wins - p1_wins
    
    print(f"Results: P0 wins={p0_wins} ({100*p0_wins/len(games_info):.1f}%), "
          f"P1 wins={p1_wins} ({100*p1_wins/len(games_info):.1f}%), "
          f"Draws={draws} ({100*draws/len(games_info):.1f}%)")
    
    # Opening move analysis
    opening_moves = defaultdict(int)
    opening_patterns = defaultdict(list)  # Track first 3 moves
    
    for game in games_info:
        game_moves = training_data[game['start_idx']:min(game['start_idx']+3, game['end_idx'])]
        
        # First move
        if game_moves:
            first_action = game_moves[0].get('action')
            if first_action is not None:
                opening_moves[first_action] += 1
                
                # Track first 3 moves as a pattern
                pattern = []
                for move in game_moves[:3]:
                    action = move.get('action')
                    if action is not None:
                        pattern.append(action)
                if len(pattern) == 3:
                    opening_patterns[tuple(pattern)].append(game['num_moves'])
    
    # Most common openings
    print(f"\nTop 5 opening moves:")
    sorted_openings = sorted(opening_moves.items(), key=lambda x: x[1], reverse=True)[:5]
    for action, count in sorted_openings:
        v1, v2 = edge_to_vertices(action, n)
        pct = 100 * count / len(games_info)
        print(f"  Edge ({v1},{v2}): {count} times ({pct:.1f}%)")
    
    # Analyze opening patterns and their success
    print(f"\nTop 3-move opening sequences and average game length:")
    sorted_patterns = sorted(opening_patterns.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    for pattern, game_lengths_list in sorted_patterns:
        if len(game_lengths_list) >= 3:  # Only show patterns used at least 3 times
            moves_str = ""
            for action in pattern:
                v1, v2 = edge_to_vertices(action, n)
                moves_str += f"({v1},{v2}) "
            avg_length = np.mean(game_lengths_list)
            print(f"  {moves_str}: used {len(game_lengths_list)} times, avg length={avg_length:.1f}")
    
    # Edge selection frequency heatmap
    edge_frequency = np.zeros(max_moves)
    edge_first_player = defaultdict(lambda: [0, 0])  # [P0 count, P1 count]
    
    for game in games_info:
        game_moves = training_data[game['start_idx']:game['end_idx']]
        for move in game_moves:
            action = move.get('action')
            player = move['player']
            if action is not None:
                edge_frequency[action] += 1
                edge_first_player[action][player] += 1
    
    # Edges by player preference
    print(f"\nEdges strongly preferred by each player:")
    edge_preferences = []
    for action in range(max_moves):
        p0_count, p1_count = edge_first_player[action]
        total = p0_count + p1_count
        if total > 10:  # Only consider frequently played edges
            ratio = p0_count / total if total > 0 else 0.5
            edge_preferences.append((action, ratio, total))
    
    # P0 preferred edges
    p0_edges = sorted([e for e in edge_preferences if e[1] > 0.65], key=lambda x: x[1], reverse=True)[:5]
    print(f"  P0 prefers:")
    for action, ratio, total in p0_edges:
        v1, v2 = edge_to_vertices(action, n)
        print(f"    Edge ({v1},{v2}): {ratio:.1%} by P0 (n={total})")
    
    # P1 preferred edges
    p1_edges = sorted([e for e in edge_preferences if e[1] < 0.35], key=lambda x: x[1])[:5]
    print(f"  P1 prefers:")
    for action, ratio, total in p1_edges:
        v1, v2 = edge_to_vertices(action, n)
        print(f"    Edge ({v1},{v2}): {1-ratio:.1%} by P1 (n={total})")
    
    # Critical moment analysis - when do games typically end?
    end_moves = defaultdict(int)
    for game in games_info:
        if game['num_moves'] < max_moves:
            end_moves[game['num_moves']] += 1
    
    if end_moves:
        print(f"\nGame ending distribution (non-draws):")
        peak_endings = sorted(end_moves.items(), key=lambda x: x[1], reverse=True)[:5]
        for move_num, count in peak_endings:
            print(f"  Move {move_num}: {count} games ended")
    
    # Value prediction confidence
    print(f"\nValue prediction patterns:")
    early_values = []
    mid_values = []
    late_values = []
    
    for game in games_info:
        game_moves = training_data[game['start_idx']:game['end_idx']]
        game_length = len(game_moves)
        
        for move_idx, move in enumerate(game_moves):
            value = abs(move['value'])  # Absolute value for confidence
            progress = move_idx / game_length
            
            if progress < 0.25:
                early_values.append(value)
            elif progress < 0.75:
                mid_values.append(value)
            else:
                late_values.append(value)
    
    print(f"  Average confidence (|value|):")
    print(f"    Early game: {np.mean(early_values):.3f}")
    print(f"    Mid game:   {np.mean(mid_values):.3f}")
    print(f"    Late game:  {np.mean(late_values):.3f}")
    
    # Policy entropy (diversity of moves considered)
    print(f"\nPolicy entropy (move diversity):")
    early_entropy = []
    late_entropy = []
    
    for game in games_info:
        game_moves = training_data[game['start_idx']:game['end_idx']]
        game_length = len(game_moves)
        
        for move_idx, move in enumerate(game_moves):
            policy = move['policy']
            # Calculate entropy
            valid_probs = policy[policy > 0.001]  # Only consider meaningful probabilities
            if len(valid_probs) > 0:
                entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-10))
                
                progress = move_idx / game_length
                if progress < 0.25:
                    early_entropy.append(entropy)
                elif progress > 0.75:
                    late_entropy.append(entropy)
    
    print(f"    Early game: {np.mean(early_entropy):.3f} (higher = more exploration)")
    print(f"    Late game:  {np.mean(late_entropy):.3f}")
    
    return {
        'game_lengths': game_lengths,
        'complete_games': complete_games,
        'edge_frequency': edge_frequency,
        'opening_moves': opening_moves,
        'p0_wins': p0_wins,
        'p1_wins': p1_wins,
        'draws': draws,
        'early_values': np.mean(early_values),
        'late_values': np.mean(late_values),
        'early_entropy': np.mean(early_entropy),
        'late_entropy': np.mean(late_entropy)
    }

def compare_iterations():
    """Compare patterns between iteration 0 and iteration 1."""
    
    print("="*70)
    print("COMPARATIVE ANALYSIS: Learning Progress from Iteration 0 to 1")
    print("="*70)
    
    # Analyze both iterations
    iter0_stats = analyze_patterns(
        "/workspace/alphazero_clique/experiments/ramsey_n_10_k4_new2/game_data/iteration_0.pkl",
        "0 (Random Initial)"
    )
    
    iter1_stats = analyze_patterns(
        "/workspace/alphazero_clique/experiments/ramsey_n_10_k4_new2/game_data/iteration_1.pkl", 
        "1 (After Training)"
    )
    
    # Compare key metrics
    print(f"\n{'='*70}")
    print("KEY IMPROVEMENTS:")
    print(f"{'='*70}")
    
    # Game length improvement
    avg0 = np.mean(iter0_stats['game_lengths'])
    avg1 = np.mean(iter1_stats['game_lengths'])
    print(f"\nAverage game length:")
    print(f"  Iteration 0: {avg0:.1f} moves")
    print(f"  Iteration 1: {avg1:.1f} moves")
    print(f"  Change: {avg1-avg0:+.1f} moves ({100*(avg1-avg0)/avg0:+.1f}%)")
    
    # Complete games (potential Ramsey counterexamples)
    print(f"\nComplete games (45 moves):")
    print(f"  Iteration 0: {iter0_stats['complete_games']} games")
    print(f"  Iteration 1: {iter1_stats['complete_games']} games")
    print(f"  Change: {iter1_stats['complete_games']-iter0_stats['complete_games']:+d}")
    
    # Win balance
    print(f"\nWin distribution:")
    total0 = iter0_stats['p0_wins'] + iter0_stats['p1_wins'] + iter0_stats['draws']
    total1 = iter1_stats['p0_wins'] + iter1_stats['p1_wins'] + iter1_stats['draws']
    print(f"  Iteration 0: P0={100*iter0_stats['p0_wins']/total0:.1f}%, "
          f"P1={100*iter0_stats['p1_wins']/total0:.1f}%, "
          f"Draws={100*iter0_stats['draws']/total0:.1f}%")
    print(f"  Iteration 1: P0={100*iter1_stats['p0_wins']/total1:.1f}%, "
          f"P1={100*iter1_stats['p1_wins']/total1:.1f}%, "
          f"Draws={100*iter1_stats['draws']/total1:.1f}%")
    
    # Value prediction confidence
    print(f"\nValue prediction confidence:")
    print(f"  Early game: {iter0_stats['early_values']:.3f} → {iter1_stats['early_values']:.3f}")
    print(f"  Late game:  {iter0_stats['late_values']:.3f} → {iter1_stats['late_values']:.3f}")
    
    # Policy entropy (exploration)
    print(f"\nPolicy entropy (exploration):")
    print(f"  Early game: {iter0_stats['early_entropy']:.3f} → {iter1_stats['early_entropy']:.3f}")
    print(f"  Late game:  {iter0_stats['late_entropy']:.3f} → {iter1_stats['late_entropy']:.3f}")
    
    # Edge usage comparison
    print(f"\nEdge selection changes:")
    edge_diff = iter1_stats['edge_frequency'] - iter0_stats['edge_frequency']
    most_increased = np.argsort(edge_diff)[-3:][::-1]
    most_decreased = np.argsort(edge_diff)[:3]
    
    print(f"  Most increased usage:")
    for action in most_increased:
        v1, v2 = edge_to_vertices(action, 10)
        change = edge_diff[action]
        print(f"    Edge ({v1},{v2}): {change:+.0f} more uses")
    
    print(f"  Most decreased usage:")
    for action in most_decreased:
        v1, v2 = edge_to_vertices(action, 10)
        change = edge_diff[action]
        print(f"    Edge ({v1},{v2}): {change:+.0f} uses")

if __name__ == "__main__":
    compare_iterations()
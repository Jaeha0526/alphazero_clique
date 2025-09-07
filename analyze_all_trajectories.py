#!/usr/bin/env python3
"""Analyze all game trajectories from saved data."""

import pickle
import numpy as np
from pathlib import Path
import sys

def edge_to_vertices(edge_idx, n):
    """Convert edge index to vertex pair (i, j)."""
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if count == edge_idx:
                return i, j
            count += 1
    return None, None

def analyze_trajectories(filepath):
    """Read and analyze all game trajectories."""
    
    print(f"Loading {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    n = data['vertices']
    k = data['k']
    game_mode = data.get('game_mode', 'symmetric')
    
    print(f"\n{'='*70}")
    print(f"COMPLETE TRAJECTORY ANALYSIS")
    print(f"{'='*70}")
    print(f"Configuration: K_{n}, {'avoiding' if game_mode == 'avoid_clique' else 'forming'} {k}-cliques")
    print(f"Total games: {len(data['games_info'])}")
    print(f"Total moves: {len(data['training_data'])}")
    
    games_info = data['games_info']
    training_data = data['training_data']
    
    # Statistics
    max_moves = n * (n - 1) // 2
    game_lengths = [g['num_moves'] for g in games_info]
    complete_games = sum(1 for l in game_lengths if l == max_moves)
    
    print(f"\nGame Length Distribution:")
    print(f"  Minimum: {min(game_lengths)} moves")
    print(f"  Maximum: {max(game_lengths)} moves")
    print(f"  Average: {np.mean(game_lengths):.1f} moves")
    print(f"  Complete games (all {max_moves} edges): {complete_games}")
    
    # Analyze complete games (potential Ramsey counterexamples)
    if complete_games > 0:
        print(f"\n{'='*70}")
        print(f"POTENTIAL RAMSEY COUNTEREXAMPLES ({complete_games} games)")
        print(f"{'='*70}")
        
        for i, game in enumerate(games_info):
            if game['num_moves'] == max_moves:
                print(f"\n--- Game {game['game_id'] + 1} (Complete {max_moves}-move game) ---")
                
                # Extract and display the complete trajectory
                game_moves = training_data[game['start_idx']:game['end_idx']]
                
                # Track edge colors
                edge_colors = np.zeros(max_moves, dtype=int)
                
                # Build the complete coloring
                for move_idx, move in enumerate(game_moves):
                    player = move['player']
                    action = move.get('action')
                    
                    if action is not None:
                        edge_colors[action] = player + 1
                        
                        if move_idx < 5 or move_idx >= max_moves - 2:
                            v1, v2 = edge_to_vertices(action, n)
                            prob = move['policy'][action] if action < len(move['policy']) else 0
                            print(f"  Move {move_idx + 1}: P{player} → ({v1},{v2}) [prob={prob:.3f}]")
                        elif move_idx == 5:
                            print(f"  ... ({max_moves - 7} moves omitted) ...")
                
                # Verify the coloring
                player0_edges = np.sum(edge_colors == 1)
                player1_edges = np.sum(edge_colors == 2)
                print(f"\nFinal coloring:")
                print(f"  Player 0: {player0_edges} edges")
                print(f"  Player 1: {player1_edges} edges")
                
                # Display as adjacency matrix (for small graphs)
                if n <= 10:
                    print(f"\nEdge coloring matrix (0=uncolored, 1=P0, 2=P1):")
                    matrix = np.zeros((n, n), dtype=int)
                    for edge_idx in range(max_moves):
                        v1, v2 = edge_to_vertices(edge_idx, n)
                        if v1 is not None:
                            matrix[v1, v2] = edge_colors[edge_idx]
                            matrix[v2, v1] = edge_colors[edge_idx]
                    
                    # Print matrix
                    print("     ", end="")
                    for j in range(n):
                        print(f"{j:2}", end=" ")
                    print()
                    for i in range(n):
                        print(f"  {i:2}:", end="")
                        for j in range(n):
                            if i == j:
                                print(" -", end=" ")
                            else:
                                val = matrix[i, j]
                                if val == 0:
                                    print(" .", end=" ")
                                elif val == 1:
                                    print(" R", end=" ")  # Red for Player 0
                                else:
                                    print(" B", end=" ")  # Blue for Player 1
                        print()
    
    # Analyze shortest games (early terminations)
    print(f"\n{'='*70}")
    print(f"SHORTEST GAMES (Early {k}-clique formations)")
    print(f"{'='*70}")
    
    shortest_games = sorted(enumerate(games_info), key=lambda x: x[1]['num_moves'])[:3]
    
    for idx, game in shortest_games:
        print(f"\n--- Game {game['game_id'] + 1} ({game['num_moves']} moves) ---")
        game_moves = training_data[game['start_idx']:game['end_idx']]
        
        # Show all moves for short games
        for move_idx, move in enumerate(game_moves):
            player = move['player']
            action = move.get('action')
            
            if action is not None:
                v1, v2 = edge_to_vertices(action, n)
                prob = move['policy'][action] if action < len(move['policy']) else 0
                value = move['value']
                print(f"  Move {move_idx + 1}: P{player} → ({v1},{v2}) [prob={prob:.3f}, value={value:+.3f}]")
        
        print(f"  Result: Player {game['winner']} wins (formed {k}-clique)")
    
    # Analyze move preferences (most frequently chosen edges)
    print(f"\n{'='*70}")
    print(f"MOVE PREFERENCE ANALYSIS")
    print(f"{'='*70}")
    
    # Count edge selection frequency
    edge_counts = np.zeros(max_moves)
    first_move_counts = np.zeros(max_moves)
    
    for game in games_info:
        game_moves = training_data[game['start_idx']:game['end_idx']]
        for move_idx, move in enumerate(game_moves):
            action = move.get('action')
            if action is not None:
                edge_counts[action] += 1
                if move_idx == 0:
                    first_move_counts[action] += 1
    
    # Most popular edges overall
    top_edges = np.argsort(edge_counts)[-5:][::-1]
    print("\nMost frequently played edges:")
    for rank, edge_idx in enumerate(top_edges):
        if edge_counts[edge_idx] > 0:
            v1, v2 = edge_to_vertices(edge_idx, n)
            freq = edge_counts[edge_idx] / len(games_info)
            print(f"  {rank+1}. Edge ({v1},{v2}): {edge_counts[edge_idx]:.0f} times ({freq:.1f} per game)")
    
    # Most popular opening moves
    top_first = np.argsort(first_move_counts)[-5:][::-1]
    print("\nMost popular opening moves:")
    for rank, edge_idx in enumerate(top_first):
        if first_move_counts[edge_idx] > 0:
            v1, v2 = edge_to_vertices(edge_idx, n)
            pct = 100 * first_move_counts[edge_idx] / len(games_info)
            print(f"  {rank+1}. Edge ({v1},{v2}): {first_move_counts[edge_idx]:.0f} times ({pct:.1f}%)")
    
    # Value prediction accuracy over game progression
    print(f"\n{'='*70}")
    print(f"VALUE PREDICTION ANALYSIS")
    print(f"{'='*70}")
    
    # Analyze how values change through the game
    early_values = []  # First 25% of moves
    mid_values = []    # Middle 50% of moves
    late_values = []   # Last 25% of moves
    
    for game in games_info:
        game_moves = training_data[game['start_idx']:game['end_idx']]
        game_length = len(game_moves)
        winner = game['winner']
        
        for move_idx, move in enumerate(game_moves):
            player = move['player']
            value = move['value']
            
            # Adjust value to be from winner's perspective
            if winner != player:
                value = -value
            
            # Categorize by game phase
            progress = move_idx / game_length
            if progress < 0.25:
                early_values.append(value)
            elif progress < 0.75:
                mid_values.append(value)
            else:
                late_values.append(value)
    
    if early_values:
        print(f"\nAverage value predictions (from eventual winner's perspective):")
        print(f"  Early game (0-25%):  {np.mean(early_values):+.3f}")
        print(f"  Mid game (25-75%):   {np.mean(mid_values):+.3f}")
        print(f"  Late game (75-100%): {np.mean(late_values):+.3f}")
    
    return data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "/workspace/alphazero_clique/experiments/ramsey_n_10_k4_new2/game_data/iteration_0.pkl"
    
    analyze_trajectories(filepath)
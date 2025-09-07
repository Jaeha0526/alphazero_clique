#!/usr/bin/env python3
"""Check if the 4-cliques in 'complete' games were formed on the last move."""

import sys
sys.path.append('/workspace/alphazero_clique/jax_full_src')

import pickle
import numpy as np
from itertools import combinations

def edge_to_vertices(edge_idx, n):
    """Convert edge index to vertex pair."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if count == edge_idx:
                return i, j
            count += 1
    return None, None

def check_for_k_clique(edge_colors, n, k, color):
    """Check if there's a k-clique of the given color."""
    adj = np.zeros((n, n), dtype=bool)
    
    for edge_idx in range(n * (n - 1) // 2):
        if edge_colors[edge_idx] == color:
            v1, v2 = edge_to_vertices(edge_idx, n)
            if v1 is not None:
                adj[v1, v2] = True
                adj[v2, v1] = True
    
    for vertices in combinations(range(n), k):
        is_clique = True
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if not adj[vertices[i], vertices[j]]:
                    is_clique = False
                    break
            if not is_clique:
                break
        
        if is_clique:
            return True, vertices
    
    return False, None

def analyze_complete_games():
    """Analyze games that reached 45 moves."""
    
    with open('/workspace/alphazero_clique/experiments/ramsey_n_10_k4_new2/game_data/iteration_0.pkl', 'rb') as f:
        data = pickle.load(f)
    
    n = 10
    k = 4
    max_moves = 45
    
    games_info = data['games_info']
    training_data = data['training_data']
    
    complete_games = [g for g in games_info if g['num_moves'] == max_moves]
    
    print(f"Analyzing {len(complete_games)} games that reached 45 moves:\n")
    
    for game in complete_games:
        game_id = game['game_id'] + 1
        game_moves = training_data[game['start_idx']:game['end_idx']]
        
        # Build board state WITHOUT the last move
        edge_colors_before = np.zeros(max_moves, dtype=int)
        
        for move_idx, move in enumerate(game_moves[:-1]):  # Exclude last move
            player = move['player']
            action = move.get('action')
            if action is not None:
                edge_colors_before[action] = player + 1
        
        # Check for 4-cliques before last move
        p0_before, _ = check_for_k_clique(edge_colors_before, n, k, 1)
        p1_before, _ = check_for_k_clique(edge_colors_before, n, k, 2)
        
        # Build board state WITH the last move
        edge_colors_after = edge_colors_before.copy()
        last_move = game_moves[-1]
        last_player = last_move['player']
        last_action = last_move.get('action')
        if last_action is not None:
            edge_colors_after[last_action] = last_player + 1
        
        # Check for 4-cliques after last move
        p0_after, p0_vertices = check_for_k_clique(edge_colors_after, n, k, 1)
        p1_after, p1_vertices = check_for_k_clique(edge_colors_after, n, k, 2)
        
        print(f"Game {game_id}:")
        print(f"  Before move 45: P0 clique={p0_before}, P1 clique={p1_before}")
        print(f"  After move 45:  P0 clique={p0_after}, P1 clique={p1_after}")
        
        if not (p0_before or p1_before) and (p0_after or p1_after):
            last_v1, last_v2 = edge_to_vertices(last_action, n)
            print(f"  ✅ Last move by P{last_player} at edge ({last_v1},{last_v2}) formed the 4-clique")
            if p0_after:
                print(f"     4-clique vertices: {p0_vertices}")
            if p1_after:
                print(f"     4-clique vertices: {p1_vertices}")
        elif not (p0_after or p1_after):
            print(f"  🎉 TRUE RAMSEY COUNTEREXAMPLE - No 4-cliques even after 45 moves!")
        else:
            print(f"  ⚠️ Unexpected: 4-clique existed before last move")
        print()

if __name__ == "__main__":
    analyze_complete_games()
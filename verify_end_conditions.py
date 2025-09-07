#!/usr/bin/env python3
"""Verify that games ended with correct conditions (4-clique detection)."""

import pickle
import numpy as np
from itertools import combinations

def edge_to_vertices(edge_idx, n):
    """Convert edge index to vertex pair (i, j)."""
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if count == edge_idx:
                return i, j
            count += 1
    return None, None

def check_for_k_clique(edge_colors, n, k, color):
    """Check if there's a k-clique of the given color."""
    # Build adjacency matrix for the given color
    adj = np.zeros((n, n), dtype=bool)
    
    for edge_idx in range(n * (n - 1) // 2):
        if edge_colors[edge_idx] == color:
            v1, v2 = edge_to_vertices(edge_idx, n)
            if v1 is not None:
                adj[v1, v2] = True
                adj[v2, v1] = True
    
    # Check all possible k-vertex subsets
    for vertices in combinations(range(n), k):
        # Check if these k vertices form a clique
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

def verify_game_endings(filepath):
    """Verify that games ended correctly."""
    
    print(f"Loading {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    n = data['vertices']
    k = data['k']
    game_mode = data.get('game_mode', 'symmetric')
    
    print(f"\nVerifying end conditions for K_{n}, avoiding {k}-cliques")
    print(f"Total games: {len(data['games_info'])}")
    print("=" * 70)
    
    games_info = data['games_info']
    training_data = data['training_data']
    max_moves = n * (n - 1) // 2
    
    errors = []
    
    for game_idx, game in enumerate(games_info):
        game_moves = training_data[game['start_idx']:game['end_idx']]
        num_moves = game['num_moves']
        
        # Reconstruct final board state
        edge_colors = np.zeros(max_moves, dtype=int)
        
        for move in game_moves:
            player = move['player']
            action = move.get('action')
            if action is not None:
                edge_colors[action] = player + 1
        
        # Check for 4-cliques in each color
        p0_clique, p0_vertices = check_for_k_clique(edge_colors, n, k, 1)  # Player 0's edges
        p1_clique, p1_vertices = check_for_k_clique(edge_colors, n, k, 2)  # Player 1's edges
        
        # Determine expected outcome
        if num_moves == max_moves:
            # All edges colored - should be a draw with no k-cliques
            if p0_clique or p1_clique:
                errors.append({
                    'game_id': game['game_id'],
                    'issue': f"Game {game['game_id']+1} marked as complete (45 moves) but has {k}-clique!",
                    'p0_clique': p0_vertices if p0_clique else None,
                    'p1_clique': p1_vertices if p1_clique else None
                })
                print(f"\n❌ ERROR in Game {game['game_id']+1}:")
                print(f"   Marked as complete ({max_moves} moves) but contains {k}-clique!")
                if p0_clique:
                    print(f"   Player 0 has {k}-clique at vertices: {p0_vertices}")
                if p1_clique:
                    print(f"   Player 1 has {k}-clique at vertices: {p1_vertices}")
            else:
                print(f"✅ Game {game['game_id']+1}: Correctly ended at {max_moves} moves (no {k}-cliques found)")
        else:
            # Game ended early - should have a k-clique
            winner = game['winner']
            
            # In avoid_clique mode, the player who forms a clique LOSES
            # So if Player 0 wins, Player 1 should have formed a clique
            if game_mode == 'avoid_clique':
                if winner == 0:
                    # Player 0 wins means Player 1 formed a clique
                    if not p1_clique:
                        errors.append({
                            'game_id': game['game_id'],
                            'issue': f"Game {game['game_id']+1}: P0 wins but P1 has no {k}-clique!",
                            'winner': winner,
                            'moves': num_moves
                        })
                        print(f"\n❌ ERROR in Game {game['game_id']+1}:")
                        print(f"   Player 0 declared winner but Player 1 has no {k}-clique!")
                    else:
                        print(f"✅ Game {game['game_id']+1}: Correctly ended at move {num_moves} (P1 formed {k}-clique at {p1_vertices})")
                elif winner == 1:
                    # Player 1 wins means Player 0 formed a clique
                    if not p0_clique:
                        errors.append({
                            'game_id': game['game_id'],
                            'issue': f"Game {game['game_id']+1}: P1 wins but P0 has no {k}-clique!",
                            'winner': winner,
                            'moves': num_moves
                        })
                        print(f"\n❌ ERROR in Game {game['game_id']+1}:")
                        print(f"   Player 1 declared winner but Player 0 has no {k}-clique!")
                    else:
                        print(f"✅ Game {game['game_id']+1}: Correctly ended at move {num_moves} (P0 formed {k}-clique at {p0_vertices})")
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    if errors:
        print(f"\n❌ Found {len(errors)} games with incorrect end conditions:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error['issue']}")
    else:
        print("\n✅ All games ended with correct conditions!")
    
    # Double-check the complete games
    complete_games = [g for g in games_info if g['num_moves'] == max_moves]
    print(f"\nComplete games analysis ({len(complete_games)} games):")
    
    for game in complete_games:
        game_moves = training_data[game['start_idx']:game['end_idx']]
        edge_colors = np.zeros(max_moves, dtype=int)
        
        for move in game_moves:
            player = move['player']
            action = move.get('action')
            if action is not None:
                edge_colors[action] = player + 1
        
        p0_edges = np.sum(edge_colors == 1)
        p1_edges = np.sum(edge_colors == 2)
        
        p0_clique, _ = check_for_k_clique(edge_colors, n, k, 1)
        p1_clique, _ = check_for_k_clique(edge_colors, n, k, 2)
        
        status = "✅ Valid" if not (p0_clique or p1_clique) else "❌ INVALID"
        print(f"  Game {game['game_id']+1}: P0={p0_edges} edges, P1={p1_edges} edges - {status}")

if __name__ == "__main__":
    verify_game_endings("/workspace/alphazero_clique/experiments/ramsey_n_10_k4_new2/game_data/iteration_0.pkl")
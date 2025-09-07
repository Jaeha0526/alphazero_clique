#!/usr/bin/env python3
"""View complete game replays from saved data."""

import pickle
import sys
from pathlib import Path
import numpy as np

def edge_to_vertices(edge_idx, n):
    """Convert edge index to vertex pair (i, j)."""
    # For an undirected graph, edge k corresponds to the k-th pair (i,j) with i<j
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if count == edge_idx:
                return i, j
            count += 1
    return None, None

def view_game_replay(filepath, game_idx=0):
    """View a complete game replay with all moves."""
    
    print(f"Loading {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    n = data['vertices']
    k = data['k']
    game_mode = data.get('game_mode', 'symmetric')
    
    print(f"\nGame Configuration:")
    print(f"  Mode: {game_mode}")
    print(f"  Graph: K_{n} (complete graph with {n} vertices)")
    print(f"  Goal: {'Avoid' if game_mode == 'avoid_clique' else 'Form'} {k}-cliques")
    print(f"  Max edges: {n*(n-1)//2}")
    
    # Check if we have game boundaries
    if 'games_info' not in data:
        print("\n❌ No game boundary information available")
        print("   Cannot reconstruct individual games")
        return
    
    games_info = data['games_info']
    training_data = data['training_data']
    
    if game_idx >= len(games_info):
        print(f"\n❌ Game {game_idx} not found (only {len(games_info)} games available)")
        return
    
    game = games_info[game_idx]
    print(f"\n{'='*60}")
    print(f"GAME {game_idx + 1} REPLAY")
    print(f"{'='*60}")
    print(f"Winner: Player {game['winner']}")
    print(f"Length: {game['num_moves']} moves")
    
    # Extract moves for this game
    game_moves = training_data[game['start_idx']:game['end_idx']]
    
    # Track edge colors (0=uncolored, 1=player0, 2=player1)
    edge_colors = np.zeros(n*(n-1)//2, dtype=int)
    
    print(f"\nMove-by-move replay:")
    print("-" * 40)
    
    for move_idx, move in enumerate(game_moves):
        player = move['player']
        action = move.get('action')
        value = move['value']
        policy = move['policy']
        
        if action is not None:
            # Get the edge that was colored
            v1, v2 = edge_to_vertices(action, n)
            edge_colors[action] = player + 1
            
            # Get probability of this move
            prob = policy[action] if action < len(policy) else 0
            
            print(f"\nMove {move_idx + 1}: Player {player}")
            print(f"  Edge: ({v1}, {v2}) [index {action}]")
            print(f"  Probability: {prob:.3f}")
            print(f"  Value: {value:+.2f} (from player {player}'s perspective)")
            
            # Show top 3 alternative moves
            top_actions = sorted(enumerate(policy), key=lambda x: x[1], reverse=True)[:3]
            if len(top_actions) > 1:
                print(f"  Alternatives considered:")
                for alt_action, alt_prob in top_actions[1:4]:
                    if edge_colors[alt_action] == 0:  # Only show uncolored edges
                        alt_v1, alt_v2 = edge_to_vertices(alt_action, n)
                        print(f"    ({alt_v1}, {alt_v2}): {alt_prob:.3f}")
        else:
            print(f"\nMove {move_idx + 1}: Player {player}")
            print(f"  ⚠️ Action not recorded (old data format)")
            print(f"  Value: {value:+.2f}")
    
    # Final board state summary
    print(f"\n{'='*40}")
    print("FINAL BOARD STATE:")
    player0_edges = np.sum(edge_colors == 1)
    player1_edges = np.sum(edge_colors == 2)
    uncolored = np.sum(edge_colors == 0)
    
    print(f"  Player 0 edges: {player0_edges}")
    print(f"  Player 1 edges: {player1_edges}")
    print(f"  Uncolored edges: {uncolored}")
    
    if game['num_moves'] == n*(n-1)//2:
        print(f"\n🎉 DRAW - All {n*(n-1)//2} edges colored without forming a {k}-clique!")
        if game_mode == 'avoid_clique':
            print("   This is a potential Ramsey counterexample!")
    else:
        loser = 1 - game['winner']
        print(f"\n🏆 Player {game['winner']} wins!")
        if game_mode == 'avoid_clique':
            print(f"   Player {loser} formed a {k}-clique and lost")
        else:
            print(f"   Player {game['winner']} formed a {k}-clique")

def list_games(filepath):
    """List all available games in a file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if 'games_info' not in data:
        print("No game boundaries available")
        return
    
    games_info = data['games_info']
    max_moves = data['vertices'] * (data['vertices'] - 1) // 2
    
    print(f"Found {len(games_info)} games:")
    for i, game in enumerate(games_info):
        result = "DRAW" if game['num_moves'] == max_moves else f"P{game['winner']} wins"
        print(f"  Game {i+1}: {game['num_moves']} moves - {result}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        game_idx = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0
    else:
        latest = max(Path("experiments").glob("*/game_data/*.pkl"), key=lambda x: x.stat().st_mtime)
        filepath = latest
        game_idx = 0
        print(f"Using most recent: {filepath}")
    
    if len(sys.argv) > 2 and sys.argv[2] == "list":
        list_games(filepath)
    else:
        view_game_replay(filepath, game_idx)
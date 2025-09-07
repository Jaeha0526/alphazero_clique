#!/usr/bin/env python3
"""View evaluation game replays."""

import pickle
import sys
from pathlib import Path
import numpy as np

def edge_to_vertices(edge_idx, n):
    """Convert edge index to vertex pair (i, j)."""
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if count == edge_idx:
                return i, j
            count += 1
    return None, None

def view_eval_game(filepath, game_idx=0):
    """View an evaluation game replay."""
    
    print(f"Loading {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    n = data['vertices']
    k = data['k']
    game_mode = data.get('game_mode', 'symmetric')
    
    print(f"\nEvaluation Game Data")
    print("="*60)
    print(f"  Mode: {game_mode}")
    print(f"  Graph: K_{n} ({n} vertices, max {n*(n-1)//2} edges)")
    print(f"  Goal: {'Avoid' if game_mode == 'avoid_clique' else 'Form'} {k}-cliques")
    print(f"  MCTS simulations: {data['mcts_sims']}")
    print(f"  Total games: {data['num_games']}")
    print(f"\nOverall Results:")
    print(f"  Current model wins: {data['current_wins']}")
    print(f"  Baseline model wins: {data['baseline_wins']}")
    print(f"  Draws: {data['draws']}")
    
    if 'games_info' not in data:
        print("\n❌ No game info available")
        return
    
    games_info = data['games_info']
    games_data = data['games_data']
    
    if game_idx >= len(games_info):
        print(f"\n❌ Game {game_idx} not found (only {len(games_info)} games)")
        return
    
    game = games_info[game_idx]
    print(f"\n{'='*60}")
    print(f"GAME {game_idx + 1} DETAILS")
    print(f"{'='*60}")
    print(f"Current model starts: {game['current_starts']}")
    print(f"Winner: {game['winner']}")
    print(f"Length: {game['num_moves']} moves")
    
    # Extract moves for this game
    game_moves = games_data[game['start_idx']:game['end_idx']]
    
    # Track which model played which color
    if game['current_starts']:
        player_models = {0: 'Current', 1: 'Baseline'}
    else:
        player_models = {0: 'Baseline', 1: 'Current'}
    
    print(f"\nPlayer assignments:")
    print(f"  Player 0: {player_models[0]} model")
    print(f"  Player 1: {player_models[1]} model")
    
    print(f"\nMove-by-move replay:")
    print("-" * 40)
    
    # Track edge colors
    edge_colors = np.zeros(n*(n-1)//2, dtype=int)
    
    for move in game_moves:
        player = move['player']
        action = move['action']
        model_used = move['model_used'].capitalize()
        move_num = move['move_number'] + 1
        
        # Get the edge that was colored
        v1, v2 = edge_to_vertices(action, n)
        edge_colors[action] = player + 1
        
        # Get probability of this move
        policy = move['policy']
        prob = policy[action] if action < len(policy) else 0
        
        print(f"\nMove {move_num}: Player {player} ({model_used} model)")
        print(f"  Edge: ({v1}, {v2}) [index {action}]")
        print(f"  Probability: {prob:.3f}")
        
        # Show top 3 alternatives
        top_actions = sorted(enumerate(policy), key=lambda x: x[1], reverse=True)[:4]
        if len(top_actions) > 1:
            print(f"  Top alternatives:")
            for i, (alt_action, alt_prob) in enumerate(top_actions[:3]):
                if alt_action != action and edge_colors[alt_action] == 0:
                    alt_v1, alt_v2 = edge_to_vertices(alt_action, n)
                    print(f"    {i+1}. ({alt_v1}, {alt_v2}): {alt_prob:.3f}")
    
    # Final summary
    print(f"\n{'='*40}")
    print("GAME SUMMARY:")
    player0_edges = np.sum(edge_colors == 1)
    player1_edges = np.sum(edge_colors == 2)
    uncolored = np.sum(edge_colors == 0)
    
    print(f"  {player_models[0]} model (Player 0): {player0_edges} edges")
    print(f"  {player_models[1]} model (Player 1): {player1_edges} edges")
    print(f"  Uncolored: {uncolored} edges")
    
    if game['winner'] == 'draw':
        print(f"\n🤝 DRAW - All edges colored without forming {k}-clique")
        if game_mode == 'avoid_clique':
            print("   Potential Ramsey counterexample!")
    elif game['winner'] == 'current':
        print(f"\n🏆 Current model wins!")
    else:
        print(f"\n🏆 Baseline model wins!")

def list_eval_games(directory):
    """List all evaluation game files."""
    eval_dir = Path(directory) / "eval_games"
    if not eval_dir.exists():
        print(f"No eval_games directory at {eval_dir}")
        return
    
    files = sorted(eval_dir.glob("*.pkl"))
    print(f"Found {len(files)} evaluation files:")
    
    for f in files:
        with open(f, 'rb') as file:
            data = pickle.load(file)
            print(f"\n{f.name}:")
            print(f"  Games: {data['num_games']}")
            print(f"  Results: Current {data['current_wins']} - {data['baseline_wins']} Baseline ({data['draws']} draws)")
            win_rate = data['current_wins'] / data['num_games'] if data['num_games'] > 0 else 0
            print(f"  Win rate: {win_rate:.1%}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        if Path(path).is_dir():
            list_eval_games(path)
        else:
            game_idx = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0
            view_eval_game(path, game_idx)
    else:
        print("Usage:")
        print("  View game: python view_eval_games.py <eval_file.pkl> [game_number]")
        print("  List files: python view_eval_games.py <experiment_dir>")
        print("\nExample:")
        print("  python view_eval_games.py experiments/my_exp/eval_games/iteration_5_vs_initial.pkl 1")
        print("  python view_eval_games.py experiments/my_exp")
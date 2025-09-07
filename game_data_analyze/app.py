#!/usr/bin/env python3
"""
Flask Web Application for Interactive Game Visualization
Updated to work with new data format (training_data + games_info)
"""

from flask import Flask, render_template, jsonify, request
import pickle
import os
from pathlib import Path
import networkx as nx
import numpy as np

app = Flask(__name__)

# Global storage for loaded game data
game_data_cache = {}
current_file = None


def edge_to_vertices(edge_idx, n):
    """Convert edge index to vertex pair."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if count == edge_idx:
                return i, j
            count += 1
    return None, None


def load_game_data(filepath):
    """Load game data from pickle file (new format)."""
    global current_file
    
    if filepath in game_data_cache:
        return game_data_cache[filepath]
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Check if this is evaluation data or training data
    is_eval = 'games_data' in data  # Evaluation files have games_data instead of training_data
    
    processed_data = {
        'iteration': data.get('iteration', 'unknown'),
        'n': data['vertices'],
        'k': data['k'],
        'game_mode': data.get('game_mode', 'symmetric'),
        'is_eval': is_eval
    }
    
    if is_eval:
        # Evaluation game format
        processed_data['num_games'] = data['num_games']
        processed_data['current_wins'] = data.get('current_wins', 0)
        processed_data['baseline_wins'] = data.get('baseline_wins', 0)
        processed_data['draws'] = data.get('draws', 0)
        games_info = data['games_info']
        all_moves = data['games_data']
    else:
        # Training game format
        games_info = data.get('games_info', [])
        all_moves = data.get('training_data', [])
        processed_data['num_games'] = len(games_info)
    
    # Process games for JSON serialization
    processed_data['games'] = []
    
    for game_info in games_info:
        # Extract moves for this game
        start_idx = game_info.get('start_idx', 0)
        end_idx = game_info.get('end_idx', start_idx)
        game_moves = all_moves[start_idx:end_idx]
        
        processed_game = {
            'game_id': game_info.get('game_id', 0),
            'winner': game_info.get('winner'),
            'num_moves': game_info.get('num_moves', len(game_moves)),
            'moves': []
        }
        
        # Add evaluation-specific info
        if is_eval:
            processed_game['current_starts'] = game_info.get('current_starts', True)
        
        for move_idx, move in enumerate(game_moves):
            processed_move = {
                'player': int(move['player']),
                'value': float(move.get('value', 0)),
                'move_number': move_idx
            }
            
            # Add action if available
            action = move.get('action')
            if action is not None:
                processed_move['action'] = int(action)
                # Convert to vertex pair
                v1, v2 = edge_to_vertices(action, processed_data['n'])
                if v1 is not None:
                    processed_move['edge'] = [v1, v2]
            
            # Add policy/probabilities
            policy = move.get('policy', [])
            if len(policy) > 0:
                # Get top 10 actions by probability
                top_actions = sorted(enumerate(policy), key=lambda x: x[1], reverse=True)[:10]
                processed_move['top_actions'] = []
                for act, prob in top_actions:
                    v1, v2 = edge_to_vertices(act, processed_data['n'])
                    if v1 is not None:
                        processed_move['top_actions'].append({
                            'action': int(act),
                            'prob': float(prob),
                            'edge': [v1, v2]
                        })
            
            # Add model info for evaluation games
            if is_eval:
                processed_move['model_used'] = move.get('model_used', 'unknown')
            
            processed_game['moves'].append(processed_move)
        
        processed_data['games'].append(processed_game)
    
    # Generate graph layout
    n = processed_data['n']
    G = nx.complete_graph(n)
    pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(n))
    
    # Convert positions to list format
    processed_data['node_positions'] = {str(i): [float(pos[i][0]), float(pos[i][1])] for i in range(n)}
    
    # Generate edge list with indices
    processed_data['edges'] = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            processed_data['edges'].append({
                'id': idx,
                'source': i,
                'target': j
            })
            idx += 1
    
    game_data_cache[filepath] = processed_data
    current_file = filepath
    
    return processed_data


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/list_files')
def list_files():
    """List available game data files."""
    files = []
    
    # Look for game data files
    experiments_dir = Path('../experiments')
    if experiments_dir.exists():
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                # Check for training game data
                game_data_dir = exp_dir / 'game_data'
                if game_data_dir.exists():
                    for pkl_file in sorted(game_data_dir.glob('*.pkl')):
                        files.append({
                            'path': str(pkl_file),
                            'name': f"{exp_dir.name}/game_data/{pkl_file.name}",
                            'experiment': exp_dir.name,
                            'iteration': pkl_file.stem,
                            'type': 'training'
                        })
                
                # Check for evaluation game data
                eval_dir = exp_dir / 'eval_games'
                if eval_dir.exists():
                    for pkl_file in sorted(eval_dir.glob('*.pkl')):
                        files.append({
                            'path': str(pkl_file),
                            'name': f"{exp_dir.name}/eval_games/{pkl_file.name}",
                            'experiment': exp_dir.name,
                            'iteration': pkl_file.stem,
                            'type': 'evaluation'
                        })
    
    return jsonify(files)


@app.route('/api/load_game')
def load_game():
    """Load a specific game file."""
    filepath = request.args.get('file')
    if not filepath:
        return jsonify({'error': 'No file specified'}), 400
    
    if not Path(filepath).exists():
        return jsonify({'error': 'File not found'}), 404
    
    try:
        data = load_game_data(filepath)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game_state')
def get_game_state():
    """Get the current game state for a specific game and move."""
    game_idx = int(request.args.get('game', 0))
    move_idx = int(request.args.get('move', 0))
    
    if not current_file or current_file not in game_data_cache:
        return jsonify({'error': 'No game loaded'}), 400
    
    data = game_data_cache[current_file]
    
    if game_idx >= len(data['games']):
        return jsonify({'error': 'Invalid game index'}), 400
    
    game = data['games'][game_idx]
    
    if move_idx >= len(game['moves']):
        return jsonify({'error': 'Invalid move index'}), 400
    
    # Build edge states up to current move
    n = data['n']
    edge_states = {}
    
    for m_idx in range(move_idx + 1):
        move = game['moves'][m_idx]
        action = move.get('action')
        if action is not None:
            player = move['player']
            edge_states[action] = player + 1  # 1 for P0, 2 for P1
    
    # Get current move details
    current_move = game['moves'][move_idx]
    
    # Determine if game ended at current move
    game_ended = False
    if move_idx == len(game['moves']) - 1:
        # Last move
        if game['num_moves'] == n * (n - 1) // 2:
            # All edges filled - draw
            game_ended = True
            game_result = 'draw'
        else:
            # Someone formed a k-clique
            game_ended = True
            if data['game_mode'] == 'avoid_clique':
                # In avoid_clique, forming clique means you lose
                # So winner is opposite of who made the last move
                if game['winner'] is not None:
                    game_result = f"Player {game['winner']} wins"
                else:
                    game_result = 'unknown'
            else:
                game_result = f"Player {game['winner']} wins"
    
    response = {
        'edge_states': edge_states,
        'current_move': current_move,
        'game_info': {
            'total_moves': game['num_moves'],
            'current_move_idx': move_idx,
            'winner': game.get('winner'),
            'game_id': game.get('game_id', game_idx)
        }
    }
    
    if game_ended:
        response['game_info']['game_ended'] = True
        response['game_info']['result'] = game_result
    
    # Add evaluation-specific info
    if data.get('is_eval', False):
        response['game_info']['current_starts'] = game.get('current_starts', True)
        response['game_info']['is_eval'] = True
    
    return jsonify(response)


@app.route('/api/stats')
def get_stats():
    """Get statistics for the loaded file."""
    if not current_file or current_file not in game_data_cache:
        return jsonify({'error': 'No game loaded'}), 400
    
    data = game_data_cache[current_file]
    
    # Calculate statistics
    game_lengths = [g['num_moves'] for g in data['games']]
    max_possible = data['n'] * (data['n'] - 1) // 2
    complete_games = sum(1 for l in game_lengths if l == max_possible)
    
    stats = {
        'total_games': len(data['games']),
        'avg_length': np.mean(game_lengths) if game_lengths else 0,
        'min_length': min(game_lengths) if game_lengths else 0,
        'max_length': max(game_lengths) if game_lengths else 0,
        'complete_games': complete_games,
        'complete_percentage': 100 * complete_games / len(game_lengths) if game_lengths else 0
    }
    
    # Add win statistics
    if data.get('is_eval'):
        stats['current_wins'] = data.get('current_wins', 0)
        stats['baseline_wins'] = data.get('baseline_wins', 0)
        stats['draws'] = data.get('draws', 0)
    else:
        # Count wins for training games
        p0_wins = sum(1 for g in data['games'] if g.get('winner') == 0)
        p1_wins = sum(1 for g in data['games'] if g.get('winner') == 1)
        draws = len(data['games']) - p0_wins - p1_wins
        stats['p0_wins'] = p0_wins
        stats['p1_wins'] = p1_wins
        stats['draws'] = draws
    
    return jsonify(stats)


if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*60)
    print("AlphaZero Clique Game Visualizer")
    print("="*60)
    print("\nStarting web server...")
    print("Open your browser and navigate to: http://localhost:5001")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
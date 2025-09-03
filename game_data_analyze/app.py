#!/usr/bin/env python3
"""
Flask Web Application for Interactive Game Visualization
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import pickle
import json
import os
from pathlib import Path
import networkx as nx
import numpy as np

app = Flask(__name__)

# Global storage for loaded game data
game_data_cache = {}
current_file = None


def load_game_data(filepath):
    """Load game data from pickle file."""
    global current_file
    
    if filepath in game_data_cache:
        return game_data_cache[filepath]
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    processed_data = {
        'iteration': data.get('iteration', 'unknown'),
        'n': data['vertices'],
        'k': data['k'],
        'game_mode': data.get('game_mode', 'symmetric'),
        'num_games': len(data['sample_games']),
        'games': []
    }
    
    # Process games for JSON serialization
    for game in data['sample_games']:
        processed_game = {
            'num_moves': len(game['moves']),
            'moves': []
        }
        
        for move in game['moves']:
            processed_move = {
                'player': int(move['player']),
                'value': float(move['value']),
                'top_actions': [(int(a), float(p)) for a, p in move['top_actions'][:10]]
            }
            if 'player_role' in move and move['player_role'] is not None:
                processed_move['player_role'] = int(move['player_role'])
            processed_game['moves'].append(processed_move)
        
        processed_data['games'].append(processed_game)
    
    # Generate graph layout
    n = processed_data['n']
    G = nx.complete_graph(n)
    pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(n))
    
    # Convert positions to list format
    processed_data['node_positions'] = {str(i): [float(pos[i][0]), float(pos[i][1])] for i in range(n)}
    
    # Generate edge list
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
                game_data_dir = exp_dir / 'game_data'
                if game_data_dir.exists():
                    for pkl_file in game_data_dir.glob('*.pkl'):
                        files.append({
                            'path': str(pkl_file),
                            'name': f"{exp_dir.name}/{pkl_file.name}",
                            'experiment': exp_dir.name,
                            'iteration': pkl_file.stem
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
        if move['top_actions']:
            action = move['top_actions'][0][0]
            player = move['player']
            edge_states[action] = player + 1
    
    return jsonify({
        'edge_states': edge_states,
        'current_move': game['moves'][move_idx],
        'game_info': {
            'total_moves': game['num_moves'],
            'current_move_idx': move_idx
        }
    })


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
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
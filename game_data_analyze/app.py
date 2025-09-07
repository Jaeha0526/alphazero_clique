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
file_metadata_cache = {}  # Fast metadata only
game_data_cache = {}      # Full game data (lazy loaded)
layout_cache = {}         # Graph layouts (computed once)
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


def load_file_metadata(filepath):
    """Load only metadata from pickle file for fast file selection."""
    if filepath in file_metadata_cache:
        return file_metadata_cache[filepath]
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Check if this is evaluation data or training data
    is_eval = 'games_data' in data
    
    if is_eval:
        games_info = data['games_info']
        metadata = {
            'iteration': data.get('iteration', 'unknown'),
            'n': data['vertices'],
            'k': data['k'],
            'game_mode': data.get('game_mode', 'symmetric'),
            'is_eval': True,
            'num_games': data['num_games'],
            'current_wins': data.get('current_wins', 0),
            'baseline_wins': data.get('baseline_wins', 0),
            'draws': data.get('draws', 0)
        }
    else:
        games_info = data.get('games_info', [])
        metadata = {
            'iteration': data.get('iteration', 'unknown'),
            'n': data['vertices'],
            'k': data['k'],
            'game_mode': data.get('game_mode', 'symmetric'),
            'is_eval': False,
            'num_games': len(games_info)
        }
    
    # Add basic game info (just winners and lengths, no move data)
    metadata['game_summaries'] = []
    for i, game_info in enumerate(games_info):
        summary = {
            'game_id': game_info.get('game_id', i),
            'winner': game_info.get('winner'),
            'num_moves': game_info.get('num_moves', 0)
        }
        if is_eval:
            summary['current_starts'] = game_info.get('current_starts', True)
        metadata['game_summaries'].append(summary)
    
    file_metadata_cache[filepath] = metadata
    return metadata


def get_graph_layout(n):
    """Get or create graph layout for n vertices."""
    if n in layout_cache:
        return layout_cache[n]
    
    # Use simpler circular layout for speed, fall back to spring for small graphs
    if n <= 8:
        G = nx.complete_graph(n)
        pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(n), iterations=50)
    else:
        # Circular layout is much faster for larger graphs
        pos = nx.circular_layout(nx.complete_graph(n))
    
    # Convert to the format we need
    node_positions = {str(i): [float(pos[i][0]), float(pos[i][1])] for i in range(n)}
    
    # Generate edge list
    edges = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append({'id': idx, 'source': i, 'target': j})
            idx += 1
    
    layout_data = {
        'node_positions': node_positions,
        'edges': edges
    }
    
    layout_cache[n] = layout_data
    return layout_data


def load_single_game(filepath, game_idx):
    """Load and process a single game's data on demand."""
    cache_key = f"{filepath}:game_{game_idx}"
    if cache_key in game_data_cache:
        return game_data_cache[cache_key]
    
    # Load raw data
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    is_eval = 'games_data' in data
    n = data['vertices']
    
    if is_eval:
        games_info = data['games_info']
        all_moves = data['games_data']
    else:
        games_info = data.get('games_info', [])
        all_moves = data.get('training_data', [])
    
    if game_idx >= len(games_info):
        return None
    
    # Process only the requested game
    game_info = games_info[game_idx]
    start_idx = game_info.get('start_idx', 0)
    end_idx = game_info.get('end_idx', start_idx)
    game_moves = all_moves[start_idx:end_idx]
    
    processed_game = {
        'game_id': game_info.get('game_id', game_idx),
        'winner': game_info.get('winner'),
        'num_moves': game_info.get('num_moves', len(game_moves)),
        'moves': []
    }
    
    if is_eval:
        processed_game['current_starts'] = game_info.get('current_starts', True)
    
    # Process moves
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
            v1, v2 = edge_to_vertices(action, n)
            if v1 is not None:
                processed_move['edge'] = [v1, v2]
        
        # Add policy/probabilities
        policy = move.get('policy', [])
        if len(policy) > 0:
            top_actions = sorted(enumerate(policy), key=lambda x: x[1], reverse=True)[:10]
            processed_move['top_actions'] = []
            for act, prob in top_actions:
                v1, v2 = edge_to_vertices(act, n)
                if v1 is not None:
                    processed_move['top_actions'].append({
                        'action': int(act),
                        'prob': float(prob),
                        'edge': [v1, v2]
                    })
        
        # Add visit counts if available
        visit_counts = move.get('visit_counts', [])
        if len(visit_counts) > 0:
            top_visits = sorted(enumerate(visit_counts), key=lambda x: x[1], reverse=True)[:10]
            processed_move['top_visits'] = []
            for act, visits in top_visits:
                if visits > 0:  # Only show actions that were visited
                    v1, v2 = edge_to_vertices(act, n)
                    if v1 is not None:
                        processed_move['top_visits'].append({
                            'action': int(act),
                            'visits': int(visits),
                            'edge': [v1, v2]
                        })
        
        if is_eval:
            processed_move['model_used'] = move.get('model_used', 'unknown')
        
        processed_game['moves'].append(processed_move)
    
    game_data_cache[cache_key] = processed_game
    return processed_game


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


@app.route('/api/load_file')
def load_file():
    """Load file metadata only (fast)."""
    global current_file
    filepath = request.args.get('file')
    if not filepath:
        return jsonify({'error': 'No file specified'}), 400
    
    if not Path(filepath).exists():
        return jsonify({'error': 'File not found'}), 404
    
    try:
        metadata = load_file_metadata(filepath)
        layout = get_graph_layout(metadata['n'])
        
        current_file = filepath
        
        response = {
            'metadata': metadata,
            'layout': layout
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/load_game')
def load_specific_game():
    """Load a specific game's data on demand."""
    filepath = request.args.get('file')
    game_idx = request.args.get('game_idx', type=int)
    
    if not filepath:
        return jsonify({'error': 'No file specified'}), 400
    if game_idx is None:
        return jsonify({'error': 'No game index specified'}), 400
    
    if not Path(filepath).exists():
        return jsonify({'error': 'File not found'}), 404
    
    try:
        game_data = load_single_game(filepath, game_idx)
        if game_data is None:
            return jsonify({'error': 'Invalid game index'}), 400
        
        return jsonify(game_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game_state')
def get_game_state():
    """Get the current game state for a specific game and move."""
    filepath = request.args.get('file')
    game_idx = int(request.args.get('game', 0))
    move_idx = int(request.args.get('move', 0))
    
    if not filepath:
        return jsonify({'error': 'No file specified'}), 400
    
    # Load game data if not cached
    cache_key = f"{filepath}:game_{game_idx}"
    if cache_key not in game_data_cache:
        game = load_single_game(filepath, game_idx)
        if game is None:
            return jsonify({'error': 'Invalid game index'}), 400
    else:
        game = game_data_cache[cache_key]
    
    if move_idx >= len(game['moves']):
        return jsonify({'error': 'Invalid move index'}), 400
    
    # Get file metadata for game settings
    metadata = load_file_metadata(filepath)
    n = metadata['n']
    
    # Build edge states up to current move
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
    game_result = None
    if move_idx == len(game['moves']) - 1:
        # Last move
        if game['num_moves'] == n * (n - 1) // 2:
            # All edges filled - draw
            game_ended = True
            game_result = 'draw'
        else:
            # Someone formed a k-clique
            game_ended = True
            if metadata['game_mode'] == 'avoid_clique':
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
    if metadata.get('is_eval', False):
        response['game_info']['current_starts'] = game.get('current_starts', True)
        response['game_info']['is_eval'] = True
    
    return jsonify(response)


@app.route('/api/stats')
def get_stats():
    """Get statistics for the loaded file."""
    filepath = request.args.get('file')
    if not filepath:
        return jsonify({'error': 'No file specified'}), 400
    
    metadata = load_file_metadata(filepath)
    
    # Calculate statistics from game summaries
    game_summaries = metadata['game_summaries']
    game_lengths = [g['num_moves'] for g in game_summaries]
    max_possible = metadata['n'] * (metadata['n'] - 1) // 2
    complete_games = sum(1 for l in game_lengths if l == max_possible)
    
    stats = {
        'total_games': len(game_summaries),
        'avg_length': np.mean(game_lengths) if game_lengths else 0,
        'min_length': min(game_lengths) if game_lengths else 0,
        'max_length': max(game_lengths) if game_lengths else 0,
        'complete_games': complete_games,
        'complete_percentage': 100 * complete_games / len(game_lengths) if game_lengths else 0
    }
    
    # Add win statistics
    if metadata.get('is_eval'):
        stats['current_wins'] = metadata.get('current_wins', 0)
        stats['baseline_wins'] = metadata.get('baseline_wins', 0)
        stats['draws'] = metadata.get('draws', 0)
    else:
        # Count wins for training games
        p0_wins = sum(1 for g in game_summaries if g.get('winner') == 0)
        p1_wins = sum(1 for g in game_summaries if g.get('winner') == 1)
        draws = len(game_summaries) - p0_wins - p1_wins
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
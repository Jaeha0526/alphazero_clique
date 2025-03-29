#!/usr/bin/env python

import json
import os
import io
import base64
import math
from flask import Flask, jsonify, request, render_template, send_from_directory
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from clique_board import CliqueBoard
from visualize_clique import view_clique_board, get_edge_positions

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from src)
project_root = os.path.dirname(current_dir)
# Set template and static folder paths
template_dir = os.path.join(project_root, 'templates')
static_dir = os.path.join(project_root, 'static')

# Initialize Flask with the correct template folder
app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)

# Game instances stored by game_id
game_instances = {}

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for web display"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_data

@app.route('/')
def index():
    """Serve the main game page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory(static_dir, path)

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Create a new game instance"""
    data = request.json
    num_vertices = data.get('num_vertices', 6)
    k = data.get('k', 3)
    game_mode = data.get('game_mode', 'asymmetric')
    
    # Input validation
    if not (3 <= num_vertices <= 10):
        return jsonify({'error': 'Number of vertices must be between 3 and 10'}), 400
    if not (2 <= k <= num_vertices):
        return jsonify({'error': f'k must be between 2 and {num_vertices}'}), 400
    if game_mode not in ['asymmetric', 'symmetric']:
        return jsonify({'error': 'Game mode must be either "asymmetric" or "symmetric"'}), 400
    
    # Create a new game id
    game_id = len(game_instances) + 1
    
    # Create a new game instance
    board = CliqueBoard(num_vertices, k, game_mode)
    game_instances[game_id] = board
    
    # Generate the initial board visualization
    fig, edge_positions = view_clique_board(board, return_edge_positions=True)
    img_data = fig_to_base64(fig)
    
    # Return game info
    return jsonify({
        'game_id': game_id,
        'board_image': img_data,
        'valid_moves': board.get_valid_moves(),
        'current_player': board.player + 1,
        'game_state': board.game_state,
        'move_count': board.move_count,
        'num_vertices': board.num_vertices,
        'k': board.k,
        'game_mode': board.game_mode,
        'edge_positions': edge_positions
    })

@app.route('/api/make_move', methods=['POST'])
def make_move():
    """Make a move in the game"""
    data = request.json
    game_id = data.get('game_id')
    edge = data.get('edge')  # [v1, v2]
    
    # Validate input
    if game_id not in game_instances:
        return jsonify({'error': 'Invalid game ID'}), 400
    
    board = game_instances[game_id]
    
    # Check if game is already over
    if board.game_state != 0:
        return jsonify({'error': 'Game is already over'}), 400
    
    # Validate and make the move
    if not edge or len(edge) != 2:
        return jsonify({'error': 'Invalid edge format'}), 400
    
    # Sort edge vertices to ensure consistent representation
    edge = sorted(edge)
    edge_tuple = (edge[0], edge[1])
    if not board.make_move(edge_tuple):
        return jsonify({'error': 'Invalid move'}), 400
    
    # Generate updated board visualization
    fig, edge_positions = view_clique_board(board, return_edge_positions=True)
    img_data = fig_to_base64(fig)
    
    # Return updated game state
    return jsonify({
        'board_image': img_data,
        'valid_moves': board.get_valid_moves(),
        'current_player': board.player + 1,
        'game_state': board.game_state,
        'move_count': board.move_count,
        'game_mode': board.game_mode,
        'last_move': edge,
        'edge_positions': edge_positions
    })

@app.route('/api/click_edge', methods=['POST'])
def click_edge():
    """Map a click position to an edge on the graph"""
    data = request.json
    game_id = data.get('game_id')
    x = data.get('x')
    y = data.get('y')
    edge_positions = data.get('edge_positions')
    
    # Validate input
    if game_id not in game_instances:
        return jsonify({'error': 'Invalid game ID'}), 400
    
    if x is None or y is None:
        return jsonify({'error': 'Invalid click coordinates'}), 400
    
    if not edge_positions:
        return jsonify({'error': 'No edge position data provided'}), 400
    
    board = game_instances[game_id]
    valid_moves = board.get_valid_moves()
    
    # Find the closest edge to the click point
    closest_edge = None
    min_distance = float('inf')
    
    for i, pos in enumerate(edge_positions):
        # Extract edge coordinates and vertices
        edge_x = pos['mid_x']
        edge_y = pos['mid_y']
        vertices = (pos['v1'], pos['v2'])
        
        # Calculate distance to click point
        distance = math.sqrt((edge_x - x) ** 2 + (edge_y - y) ** 2)
        
        # Check if this is a valid move
        if (vertices[0], vertices[1]) in valid_moves or (vertices[1], vertices[0]) in valid_moves:
            if distance < min_distance:
                min_distance = distance
                closest_edge = vertices
    
    # Check if we found a valid edge
    if closest_edge is None:
        return jsonify({'error': 'No valid edge found near click point'}), 400
    
    # Return the closest edge
    return jsonify({
        'edge': closest_edge
    })

@app.route('/api/game_state/<int:game_id>', methods=['GET'])
def get_game_state(game_id):
    """Get the current state of a game"""
    if game_id not in game_instances:
        return jsonify({'error': 'Invalid game ID'}), 400
    
    board = game_instances[game_id]
    
    # Generate board visualization
    fig, edge_positions = view_clique_board(board, return_edge_positions=True)
    img_data = fig_to_base64(fig)
    
    return jsonify({
        'board_image': img_data,
        'valid_moves': board.get_valid_moves(),
        'current_player': board.player + 1,
        'game_state': board.game_state,
        'move_count': board.move_count,
        'num_vertices': board.num_vertices,
        'k': board.k,
        'game_mode': board.game_mode,
        'edge_positions': edge_positions
    })

@app.route('/api/reset_game/<int:game_id>', methods=['POST'])
def reset_game(game_id):
    """Reset a game to its initial state"""
    if game_id not in game_instances:
        return jsonify({'error': 'Invalid game ID'}), 400
    
    board = game_instances[game_id]
    num_vertices = board.num_vertices
    k = board.k
    game_mode = board.game_mode
    
    # Create a new board with the same parameters
    new_board = CliqueBoard(num_vertices, k, game_mode)
    game_instances[game_id] = new_board
    
    # Generate board visualization
    fig, edge_positions = view_clique_board(new_board, return_edge_positions=True)
    img_data = fig_to_base64(fig)
    
    return jsonify({
        'board_image': img_data,
        'valid_moves': new_board.get_valid_moves(),
        'current_player': new_board.player + 1,
        'game_state': new_board.game_state,
        'move_count': new_board.move_count,
        'game_mode': new_board.game_mode,
        'edge_positions': edge_positions
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    print(f"Template directory: {template_dir}")
    print(f"Static directory: {static_dir}")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8080) 
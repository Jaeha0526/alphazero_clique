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
import torch
import glob
from alpha_net_clique import CliqueGNN
from encoder_decoder_clique import prepare_state_for_network, encode_action, decode_action
import numpy as np
from MCTS_clique import UCT_search, get_policy, get_q_values
import time
import torch.nn as nn
import torch.nn.functional as F

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

# Define the directory for playable models relative to project root
playable_model_dir = os.path.join(project_root, 'playable_models')
# Ensure the directory exists
os.makedirs(playable_model_dir, exist_ok=True)

# Game instances stored by game_id
game_instances = {}

# --- Helper Functions --- Start

# Remove the following function definitions:
# def create_encoder_decoder_for_clique(num_vertices):
#     ...
# def prepare_state_for_network(board: CliqueBoard):
#     ...

# --- Helper Functions --- End

def check_model_compatibility(checkpoint, num_vertices, k):
    """
    Check if a model checkpoint is compatible with the current architecture.
    Returns (is_compatible, error_message, suggested_hidden_dim, suggested_num_layers)
    """
    try:
        # Check basic required fields
        if 'num_vertices' not in checkpoint or 'clique_size' not in checkpoint:
            return False, "Model missing basic configuration (num_vertices or clique_size)", None, None
        
        if checkpoint['num_vertices'] != num_vertices or checkpoint['clique_size'] != k:
            return False, f"Model configuration mismatch. Game: V={num_vertices}, k={k}. Model: V={checkpoint['num_vertices']}, k={checkpoint['clique_size']}", None, None
        
        # Check for basic required structure
        state_dict = checkpoint['state_dict']
        required_keys = ['node_embedding.weight', 'edge_embedding.weight']
        missing_keys = [key for key in required_keys if key not in state_dict]
        
        if missing_keys:
            return False, f"Model is missing required keys: {missing_keys}", None, None
        
        # Get model parameters
        hidden_dim = checkpoint.get('hidden_dim', 64)
        num_layers = checkpoint.get('num_layers', 2)
        
        # Check if it has basic GNN structure
        has_gnn_layers = any('node_layers' in key for key in state_dict.keys())
        
        if not has_gnn_layers:
            return False, "Model does not have the expected GNN layer structure", hidden_dim, num_layers
            
        print(f"Found compatible model (hidden_dim={hidden_dim}, num_layers={num_layers})")
        return True, "", hidden_dim, num_layers
        
    except Exception as e:
        return False, f"Error checking model compatibility: {e}", None, None

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
    game_instances[game_id] = {
        'board': board,
        'model': None,
        'model_path': None
    }
    
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
    
    game_data = game_instances[game_id]
    board = game_data['board']
    
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
    
    board = game_instances[game_id]['board']
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
    
    board = game_instances[game_id]['board']
    
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
    
    board = game_instances[game_id]['board']
    num_vertices = board.num_vertices
    k = board.k
    game_mode = board.game_mode
    
    # Create a new board with the same parameters
    new_board = CliqueBoard(num_vertices, k, game_mode)
    game_instances[game_id] = {
        'board': new_board,
        'model': None,
        'model_path': None
    }
    
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

@app.route('/api/list_models/<int:game_id>', methods=['GET'])
def list_models(game_id):
    """List compatible models for the given game."""
    if game_id not in game_instances:
        return jsonify({'error': 'Invalid game ID'}), 400

    game_data = game_instances[game_id]
    board = game_data['board']
    current_num_vertices = board.num_vertices
    current_k = board.k

    compatible_models = []
    incompatible_models = []
    
    try:
        # Search for .pth.tar files in the playable_models directory
        model_files = glob.glob(os.path.join(playable_model_dir, "*.pth.tar"))

        for model_path in model_files:
            try:
                # Load only the metadata first to check compatibility
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                filename = os.path.basename(model_path)
                
                # Use the new compatibility checker
                is_compatible, error_msg, suggested_hidden_dim, suggested_num_layers = check_model_compatibility(
                    checkpoint, current_num_vertices, current_k
                )
                
                model_info = {
                    'filename': filename,
                    'path': model_path,
                    'num_vertices': checkpoint.get('num_vertices', 'Missing'),
                    'k': checkpoint.get('clique_size', 'Missing'),
                    'hidden_dim': checkpoint.get('hidden_dim', 'Missing'),
                    'num_layers': checkpoint.get('num_layers', 'Missing'),
                    'compatible': is_compatible
                }
                
                if is_compatible:
                    compatible_models.append(model_info)
                else:
                    model_info['error'] = error_msg
                    incompatible_models.append(model_info)

            except Exception as e:
                print(f"Error reading metadata from model {model_path}: {e}")
                model_info = {
                    'filename': os.path.basename(model_path),
                    'path': model_path,
                    'num_vertices': 'Error',
                    'k': 'Error',
                    'hidden_dim': 'Error',
                    'num_layers': 'Error',
                    'compatible': False,
                    'error': str(e)
                }
                incompatible_models.append(model_info)

    except Exception as e:
        print(f"Error listing models in {playable_model_dir}: {e}")
        return jsonify({'error': 'Failed to list models'}), 500

    return jsonify({
        'compatible_models': compatible_models,
        'incompatible_models': incompatible_models,
        'total_compatible': len(compatible_models),
        'total_incompatible': len(incompatible_models)
    })

@app.route('/api/select_model', methods=['POST'])
def select_model():
    """Load and select a model for the game."""
    data = request.json
    game_id = data.get('game_id')
    model_filename = data.get('model_filename')

    if game_id not in game_instances:
        return jsonify({'error': 'Invalid game ID'}), 400
    if not model_filename:
        return jsonify({'error': 'Model filename not provided'}), 400

    game_data = game_instances[game_id]
    board = game_data['board']
    model_path = os.path.join(playable_model_dir, model_filename)

    if not os.path.exists(model_path):
         return jsonify({'error': f'Model file not found: {model_filename}'}), 404

    try:
        # Load the full model checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Check compatibility with detailed error messages
        is_compatible, error_msg, suggested_hidden_dim, suggested_num_layers = check_model_compatibility(
            checkpoint, board.num_vertices, board.k
        )
        
        if not is_compatible:
            return jsonify({'error': error_msg}), 400

        # Use the suggested dimensions from compatibility check
        hidden_dim = suggested_hidden_dim
        num_layers = suggested_num_layers
        
        print(f"Loading model with hidden_dim={hidden_dim}, num_layers={num_layers}")

        # Use FlexibleCliqueGNN that can adapt to the checkpoint
        model = FlexibleCliqueGNN(
            num_vertices=checkpoint['num_vertices'], 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            checkpoint_state_dict=checkpoint['state_dict']
        )
        
        # Load the state dict
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError as e:
            print(f"Warning: Partial state dict loading: {e}")
            # Try partial loading
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Successfully loaded {len(pretrained_dict)} out of {len(checkpoint['state_dict'])} parameters")
                
        model.eval() # Set to evaluation mode

        # Store the loaded model in the game instance
        game_data['model'] = model
        game_data['model_path'] = model_path

        success_msg = f'Model {model_filename} selected successfully.'
        print(f"Game {game_id}: {success_msg}")
        return jsonify({'message': success_msg})

    except Exception as e:
        print(f"Error loading model {model_filename} for game {game_id}: {e}")
        game_data['model'] = None # Clear model if loading failed
        game_data['model_path'] = None
        return jsonify({'error': f'Failed to load model: {e}'}), 500

@app.route('/api/get_prediction/<int:game_id>', methods=['GET'])
def get_prediction(game_id):
    """Get the AI model's prediction for the current game state."""
    if game_id not in game_instances:
        return jsonify({'error': 'Invalid game ID'}), 400

    game_data = game_instances[game_id]
    board = game_data['board']
    model = game_data['model']

    if model is None:
        return jsonify({'error': 'No model selected for this game.'}), 400

    if board.game_state != 0: # No predictions if game is over
        return jsonify({'policy': {}, 'value': 0.0})

    try:
        # Prepare the current board state for the network
        state_data = prepare_state_for_network(board) # Get the dictionary
        edge_index = state_data['edge_index']         # Extract tensor from dict
        edge_attr = state_data['edge_attr']          # Extract tensor from dict
        
        # Ensure tensors are on the same device as the model
        device = next(model.parameters()).device

        # Run inference
        with torch.no_grad():
            # Model expects edge_index, edge_attr, and batch assignment
            # For single inference, batch is just zeros
            num_nodes = board.num_vertices
            batch_assignment = torch.zeros(num_nodes, dtype=torch.long, device=device)
            policy_output, value_output = model(edge_index, edge_attr, batch=batch_assignment)

        # Extract policy and value
        policy_tensor = policy_output.squeeze(0) # Remove batch dim
        value = value_output.item()

        # Get valid moves and map policy probabilities
        valid_moves = board.get_valid_moves()
        policy_probs = {}
        total_policy_prob_raw = 0.0 # Sum of probabilities for valid moves from the raw policy output

        for move in valid_moves:
            move_idx = encode_action(board, move)
            if 0 <= move_idx < len(policy_tensor):
                 prob = policy_tensor[move_idx].item()
                 # Ensure prob is non-negative (should be due to softmax in model, but good check)
                 prob = max(0.0, prob)
                 policy_probs[str(tuple(sorted(move)))] = prob # Use sorted tuple as key string
                 total_policy_prob_raw += prob
            else:
                 policy_probs[str(tuple(sorted(move)))] = 0.0 # Assign 0 if index is invalid or out of bounds

        # Normalize probabilities among valid moves
        normalized_policy_probs = {}
        if total_policy_prob_raw > 1e-8: # Avoid division by zero
             for move_key, prob in policy_probs.items():
                 normalized_policy_probs[move_key] = prob / total_policy_prob_raw
        else: # If sum is tiny or zero, distribute uniformly
             num_valid = len(policy_probs)
             if num_valid > 0:
                uniform_prob = 1.0 / num_valid
                for move_key in policy_probs:
                   normalized_policy_probs[move_key] = uniform_prob
             else:
                 normalized_policy_probs = {} # No valid moves, empty policy

        # Value is from the perspective of the current player to play.
        # No flipping needed based on model definition.

        return jsonify({
            'policy': normalized_policy_probs, # Dict mapping "(v1, v2)" -> prob
            'value': value
        })

    except Exception as e:
        print(f"Error getting prediction for game {game_id}: {e}")
        # Potentially clear CUDA cache if it's a CUDA error
        if "CUDA" in str(e) and torch.cuda.is_available():
             torch.cuda.empty_cache()
        return jsonify({'error': f'Prediction failed: {e}'}), 500

@app.route('/api/get_mcts_policy/<int:game_id>', methods=['GET'])
def get_mcts_policy(game_id):
    """Get the MCTS policy prediction and Q-values after a specified number of simulations."""
    # Get number of simulations from query parameter, default to 777
    num_simulations = request.args.get('simulations', default=777, type=int)

    if game_id not in game_instances:
        return jsonify({'error': 'Invalid game ID'}), 400

    game_data = game_instances[game_id]
    board = game_data['board']
    model = game_data['model']

    if model is None:
        return jsonify({'error': 'No model selected for this game.'}), 400

    if board.game_state != 0: # No predictions if game is over
        return jsonify({'policy': {}})

    try:
        # Run MCTS search
        print(f"Starting MCTS search for game {game_id} with {num_simulations} simulations...")
        start_time = time.time()
        # Ensure the board copy is used if necessary, MCTS might modify the board state internally
        # However, UCT_search seems to handle copying internally via copy.deepcopy(self.game)
        # Disable Dirichlet noise for interactive prediction by setting noise_weight=0.0
        # Pass the requested number of simulations
        _best_move, root_node = UCT_search(board, num_simulations, model, noise_weight=0.0)
        end_time = time.time()
        print(f"MCTS search completed in {end_time - start_time:.2f} seconds.")

        # Get the policy from the root node visit counts
        mcts_policy_array = get_policy(root_node)
        # Get the Q-values from the root node
        mcts_q_array = get_q_values(root_node)

        # Convert policy array to dictionary format { "(v1, v2)": probability }
        policy_dict = {}
        q_value_dict = {} # <-- Dictionary for Q-values
        valid_moves = board.get_valid_moves() # Get currently valid moves

        # --- Populate Policy and Q-value Dictionaries ---
        # Include all possible edges to show Q-values for unvisited/invalid ones too
        # Generate all possible edges directly instead of calling a non-existent method
        num_vertices = board.num_vertices
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                move = (i, j) # Create the edge tuple
                move_str = str(tuple(sorted(move))) # Use sorted tuple for consistency
                move_idx = encode_action(board, move)

                # Get Policy Probability (only for valid moves, from get_policy result)
                # Check if the generated move is in the set of valid moves
                is_valid = move in valid_moves or (j, i) in valid_moves
                if 0 <= move_idx < len(mcts_policy_array) and is_valid:
                     prob = mcts_policy_array[move_idx]
                     policy_dict[move_str] = max(0.0, float(prob))
                elif is_valid:
                     policy_dict[move_str] = 0.0
                # else: Invalid moves won't be added to policy_dict

                # Get Q-Value
                if 0 <= move_idx < len(mcts_q_array):
                    q_val = mcts_q_array[move_idx]
                    # Convert numpy NaN/float if necessary
                    q_value_dict[move_str] = None if np.isnan(q_val) else float(q_val)
                else:
                    # Should not happen if array size is correct
                    q_value_dict[move_str] = None

        # Re-normalize policy dictionary just in case (Q-values are not normalized)
        total_prob = sum(policy_dict.values())
        if total_prob > 1e-6:
            for move_key in policy_dict:
                policy_dict[move_key] /= total_prob
        elif len(policy_dict) > 0:
             uniform_prob = 1.0 / len(policy_dict)
             for move_key in policy_dict:
                 policy_dict[move_key] = uniform_prob

        return jsonify({
            'policy': policy_dict,
            'q_values': q_value_dict # <-- Add Q-values to response
        })

    except Exception as e:
        print(f"Error getting MCTS policy for game {game_id}: {e}")
        # Potentially clear CUDA cache if it's a CUDA error
        if "CUDA" in str(e) and torch.cuda.is_available():
             torch.cuda.empty_cache()
        return jsonify({'error': f'MCTS Prediction failed: {e}'}), 500

@app.route('/api/list_all_models', methods=['GET'])
def list_all_models():
    """List ALL models found in the playable_models directory and their params."""
    all_models_info = []
    try:
        model_files = glob.glob(os.path.join(playable_model_dir, "*.pth.tar"))
        print(f"Found files in {playable_model_dir}: {model_files}") # Debug print

        for model_path in model_files:
            model_info = {'filename': os.path.basename(model_path)}
            try:
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                # Check if keys exist before accessing
                if 'num_vertices' in checkpoint:
                    model_info['num_vertices'] = checkpoint['num_vertices']
                else:
                     model_info['num_vertices'] = 'Missing'

                if 'clique_size' in checkpoint:
                    model_info['k'] = checkpoint['clique_size']
                else:
                    model_info['k'] = 'Missing'

            except Exception as e:
                print(f"Error reading metadata from model {model_path}: {e}")
                model_info['num_vertices'] = 'Error'
                model_info['k'] = 'Error'

            all_models_info.append(model_info)

    except Exception as e:
        print(f"Error listing all models in {playable_model_dir}: {e}")
        # Return error but maybe also partial results if any were collected
        return jsonify({'error': 'Failed to list all models', 'models': all_models_info}), 500

    if not all_models_info:
        print(f"No .pth.tar files found in {playable_model_dir}")

    return jsonify({'models': all_models_info})

class FlexibleCliqueGNN(CliqueGNN):
    """
    Flexible version of CliqueGNN that can adapt to different model architectures.
    """
    def __init__(self, num_vertices=6, hidden_dim=64, num_layers=2, checkpoint_state_dict=None):
        # Don't call super().__init__ yet, we need to analyze the checkpoint first
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        
        # Input embedding layers
        self.node_embedding = nn.Linear(1, hidden_dim)
        self.edge_embedding = nn.Linear(3, hidden_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.xavier_uniform_(self.edge_embedding.weight)
        
        # Dynamically create GNN layers
        self.node_layers = nn.ModuleList()
        self.edge_layers = nn.ModuleList()
        for _ in range(num_layers):
            from alpha_net_clique import EdgeAwareGNNBlock, EdgeBlock
            self.node_layers.append(EdgeAwareGNNBlock(hidden_dim, hidden_dim, hidden_dim))
            self.edge_layers.append(EdgeBlock(hidden_dim, hidden_dim, hidden_dim))
        
        # Policy head (same for all architectures)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize policy head
        for m in self.policy_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Analyze checkpoint to determine architecture
        if checkpoint_state_dict is not None:
            has_attention = any('attention' in key for key in checkpoint_state_dict.keys())
            
            # Check value head input size from checkpoint
            value_head_input_size = hidden_dim  # Default
            if 'value_head.0.weight' in checkpoint_state_dict:
                value_head_input_size = checkpoint_state_dict['value_head.0.weight'].shape[0]
            
            print(f"Detected value head input size: {value_head_input_size}, has attention: {has_attention}")
            
        else:
            # Default to new architecture
            has_attention = True
            value_head_input_size = 2 * hidden_dim
        
        # Create appropriate architecture
        if has_attention:
            # New architecture with attention
            self.node_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1)
            )
            
            self.edge_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1)
            )
            
            # Value head expects 2*hidden_dim input
            self.value_head = nn.Sequential(
                nn.BatchNorm1d(2*hidden_dim),
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            )
            
        else:
            # Older architecture without attention
            self.node_attention = None
            self.edge_attention = None
            
            # Value head expects hidden_dim input (global pooled features)
            self.value_head = nn.Sequential(
                nn.BatchNorm1d(value_head_input_size),
                nn.Linear(value_head_input_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            )
        
        # Initialize value head
        for m in self.value_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        self.has_attention = has_attention
    
    def forward(self, edge_index, edge_attr, batch=None, debug=False):
        # Same as parent class until value calculation
        if batch is None:
            batch = torch.zeros(edge_index.max().item() + 1, dtype=torch.long, device=edge_index.device)
        
        num_nodes_per_graph = torch.bincount(batch)
        num_graphs = len(num_nodes_per_graph)
        
        # Initialize node and edge features
        node_indices = torch.zeros(len(batch), device=edge_index.device).float().unsqueeze(1)
        x = self.node_embedding(node_indices)
        edge_features = self.edge_embedding(edge_attr)
        
        # Apply GNN layers
        for i, (node_layer, edge_layer) in enumerate(zip(self.node_layers, self.edge_layers)):
            x_new = node_layer(x, edge_index, edge_features) 
            edge_features = edge_layer(x, edge_index, edge_features)
            x = x_new
        
        # Generate edge scores
        edge_scores = self.policy_head(edge_features)
        
        # Process each graph separately
        policies = []
        values = []
        
        node_cumsum = torch.cat([torch.tensor([0], device=edge_index.device), torch.cumsum(num_nodes_per_graph, dim=0)])
        
        for i in range(num_graphs):
            start_idx = node_cumsum[i]
            end_idx = node_cumsum[i + 1]
            num_nodes = num_nodes_per_graph[i]
            
            graph_mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
            graph_edge_index = edge_index[:, graph_mask]
            graph_edge_scores = edge_scores[graph_mask]
            
            # Create policy (same logic as parent)
            num_edges = num_nodes * (num_nodes - 1) // 2
            edge_map = {}
            idx = 0
            for j in range(num_nodes):
                for k in range(j+1, num_nodes):
                    edge_map[(j, k)] = idx
                    edge_map[(k, j)] = idx
                    idx += 1
            
            canonical_scores = {}
            for j in range(graph_edge_index.shape[1]):
                src = int(graph_edge_index[0, j].item() - start_idx)
                dst = int(graph_edge_index[1, j].item() - start_idx)
                
                if src != dst:
                     canonical_edge = tuple(sorted((src, dst)))
                     if canonical_edge not in canonical_scores:
                          canonical_scores[canonical_edge] = graph_edge_scores[j].item()

            policy = torch.zeros(num_edges, device=edge_index.device)
            for edge_tuple, edge_idx in edge_map.items():
                 canonical_edge = tuple(sorted(edge_tuple))
                 if edge_tuple == canonical_edge:
                     policy[edge_idx] = canonical_scores.get(canonical_edge, 0.0)
            
            policy = F.softmax(policy, dim=0)
            policies.append(policy)
            
            # Value calculation - depends on architecture
            graph_nodes = x[start_idx:end_idx]
            
            if self.has_attention:
                # New architecture with attention pooling
                graph_edge_features = edge_features[graph_mask]
                
                if len(graph_nodes) > 0:
                    node_attn_scores = self.node_attention(graph_nodes)
                    node_attn_weights = F.softmax(node_attn_scores, dim=0)
                    node_features_pooled = torch.sum(graph_nodes * node_attn_weights, dim=0)
                else:
                    node_features_pooled = torch.zeros(self.hidden_dim, device=edge_index.device)
                
                if len(graph_edge_features) > 0:
                    edge_attn_scores = self.edge_attention(graph_edge_features)
                    edge_attn_weights = F.softmax(edge_attn_scores, dim=0)
                    edge_features_pooled = torch.sum(graph_edge_features * edge_attn_weights, dim=0)
                else:
                    edge_features_pooled = torch.zeros(self.hidden_dim, device=edge_index.device)
                
                combined_features = torch.cat([node_features_pooled, edge_features_pooled], dim=0)
                
                if self.training:
                    values.append(combined_features)
                else:
                    value = self.value_head(combined_features.unsqueeze(0))
                    values.append(value)
            else:
                # Older architecture with simple global pooling
                if len(graph_nodes) > 0:
                    node_features_pooled = torch.mean(graph_nodes, dim=0)
                else:
                    node_features_pooled = torch.zeros(self.hidden_dim, device=edge_index.device)
                
                value = self.value_head(node_features_pooled.unsqueeze(0))
                values.append(value)
        
        policies = torch.stack(policies)
        
        if self.has_attention and self.training:
            values_tensor = torch.stack(values)
            values = self.value_head(values_tensor)
        else:
            values = torch.stack(values)
        
        return policies, values

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    print(f"Template directory: {template_dir}")
    print(f"Static directory: {static_dir}")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8080) 
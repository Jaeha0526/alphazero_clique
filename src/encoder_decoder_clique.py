#!/usr/bin/env python
import numpy as np
import torch
from typing import Tuple, List, Dict, Any

def encode_board(board) -> np.ndarray:
    """
    Encode the Clique Game board into a format suitable for neural network input.
    
    Args:
        board: CliqueBoard instance
        
    Returns:
        board_encoding: Encoded board state as a NumPy array
    """
    # Get the number of vertices
    num_vertices = board.num_vertices
    
    # Get the number of possible edges in the complete graph
    num_edges = num_vertices * (num_vertices - 1) // 2
    
    # Create a flattened encoding of the edge states
    # For each edge, we create a one-hot encoding of the state (unselected, player1, player2)
    encoding = np.zeros((num_edges, 3), dtype=np.float32)
    
    idx = 0
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            state = board.edge_states[i, j]
            encoding[idx, state] = 1.0
            idx += 1
    
    # Add additional features
    # 1. Current player (0 for player 1, 1 for player 2)
    player_feature = np.ones(num_edges, dtype=np.float32) * board.player
    
    # 2. Move count as a normalized value
    # Normalization: divide by max possible moves (all edges)
    move_count_feature = np.ones(num_edges, dtype=np.float32) * (board.move_count / num_edges)
    
    # 3. Edge index features (normalized position)
    edge_i_feature = np.zeros(num_edges, dtype=np.float32)
    edge_j_feature = np.zeros(num_edges, dtype=np.float32)
    
    idx = 0
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            edge_i_feature[idx] = i / (num_vertices - 1)  # Normalize to [0, 1]
            edge_j_feature[idx] = j / (num_vertices - 1)  # Normalize to [0, 1]
            idx += 1
    
    # Concatenate all features
    features = np.column_stack([
        encoding,            # Edge states (3 features per edge)
        player_feature[:, np.newaxis],     # Current player
        move_count_feature[:, np.newaxis], # Move count
        edge_i_feature[:, np.newaxis],     # First vertex index
        edge_j_feature[:, np.newaxis]      # Second vertex index
    ])
    
    return features

def decode_action(board, move_idx: int) -> Tuple[int, int]:
    """
    Decode a move index into an edge (i, j).
    
    Args:
        board: CliqueBoard instance
        move_idx: Index of the move
        
    Returns:
        edge: Tuple (i, j) representing the edge
    """
    num_vertices = board.num_vertices
    
    # Reconstruct edge from index
    idx = 0
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            if idx == move_idx:
                return (i, j)
            idx += 1
    
    # If not found, return invalid edge
    return (-1, -1)

def encode_action(board, edge: Tuple[int, int]) -> int:
    """
    Encode an edge (i, j) into a move index.
    
    Args:
        board: CliqueBoard instance
        edge: Tuple (i, j) representing the edge
        
    Returns:
        move_idx: Index of the move
    """
    i, j = min(edge), max(edge)  # Ensure i < j
    num_vertices = board.num_vertices
    
    # Calculate index directly
    idx = 0
    for x in range(num_vertices):
        for y in range(x+1, num_vertices):
            if x == i and y == j:
                return idx
            idx += 1
    
    # If not found, return invalid index
    return -1

def prepare_state_for_network(board) -> Dict[str, Any]:
    """
    Prepare the board state for input to the GNN network.
    
    Args:
        board: CliqueBoard instance
        
    Returns:
        state_dict: Dictionary with edge_index and edge_attr tensors
    """
    board_state = board.get_board_state()
    num_vertices = board.num_vertices
    
    # Get edge indices
    edge_indices = []
    edge_features = []
    
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            # Add both directions for undirected graph
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            # One-hot encode the edge state
            state = board_state['edge_states'][i, j]
            if state == 0:  # unselected
                feat = [1, 0, 0]
            elif state == 1:  # player 1
                feat = [0, 1, 0]
            else:  # player 2
                feat = [0, 0, 1]
            
            edge_features.append(feat)
            edge_features.append(feat)  # Same feature for both directions
    
    # Add self-loops for each vertex with special feature
    for i in range(num_vertices):
        edge_indices.append([i, i])
        edge_features.append([0, 0, 0])  # Special feature for self-loops
    
    # Convert to PyTorch tensors (leave on CPU - device will be set by caller)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t()  # Shape [2, E]
    edge_attr = torch.tensor(edge_features, dtype=torch.float)     # Shape [E, 3]
    
    return {
        'edge_index': edge_index,
        'edge_attr': edge_attr
    }

def get_valid_moves_mask(board) -> np.ndarray:
    """
    Get a binary mask of valid moves.
    
    Args:
        board: CliqueBoard instance
        
    Returns:
        mask: Binary mask where 1 indicates a valid move
    """
    num_vertices = board.num_vertices
    num_edges = num_vertices * (num_vertices - 1) // 2
    
    # Initialize mask with zeros
    mask = np.zeros(num_edges, dtype=np.float32)
    
    # Get valid moves from the board
    valid_moves = board.get_valid_moves()
    
    # Mark valid moves in the mask
    for edge in valid_moves:
        idx = encode_action(board, edge)
        if idx >= 0:
            mask[idx] = 1.0
    
    return mask

def apply_valid_moves_mask(policy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a valid moves mask to a policy vector.
    
    Args:
        policy: Policy vector from the network
        mask: Binary mask where 1 indicates a valid move
        
    Returns:
        masked_policy: Policy with invalid moves zeroed out and renormalized
    """
    masked_policy = policy * mask
    
    # Renormalize if any valid moves exist
    if np.sum(masked_policy) > 0:
        masked_policy /= np.sum(masked_policy)
    
    return masked_policy

# Example usage
if __name__ == "__main__":
    from clique_board import CliqueBoard
    
    # Create a sample board
    board = CliqueBoard(6, 3)  # 6 vertices, need 3-clique to win
    
    # Make some moves
    board.make_move((0, 1))  # Player 1
    board.make_move((1, 2))  # Player 2
    
    # Encode the board
    encoded_board = encode_board(board)
    print(f"Encoded board shape: {encoded_board.shape}")
    
    # Get valid moves mask
    mask = get_valid_moves_mask(board)
    print(f"Valid moves mask shape: {mask.shape}")
    print(f"Number of valid moves: {int(np.sum(mask))}")
    
    # Prepare state for GNN
    state_dict = prepare_state_for_network(board)
    print(f"Edge index shape: {state_dict['edge_index'].shape}")
    print(f"Edge attr shape: {state_dict['edge_attr'].shape}")
    
    # Test encoding and decoding actions
    edge = (2, 3)
    move_idx = encode_action(board, edge)
    decoded_edge = decode_action(board, move_idx)
    print(f"Edge {edge} encoded as {move_idx}, decoded back to {decoded_edge}") 
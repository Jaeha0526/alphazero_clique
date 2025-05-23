#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import collections
import math
import copy
import torch.multiprocessing as mp
from alpha_net_clique import CliqueGNN
import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random
import time
from clique_board import CliqueBoard
import encoder_decoder_clique as ed
from visualize_clique import view_clique_board
import matplotlib.pyplot as plt
import argparse

@dataclass
class UCTNode:
    game: CliqueBoard
    move: Optional[int] # Move taken TO reach this node
    parent: Optional['UCTNode']
    number_visits: float = 0.0 # Visits TO this node (N(s))
    is_expanded: bool = False
    children: Dict[int, 'UCTNode'] = None
    child_priors: np.ndarray = None # P(s, a) from NN for children
    child_number_visits: np.ndarray = None # N(s, a) for children
    child_total_value: np.ndarray = None # W(s, a) for children
    action_idxes: List[int] = None # Valid action indices from this node

    def __post_init__(self):
        if self.children is None:
            self.children = {}
        
        num_vertices = self.game.num_vertices
        num_edges = num_vertices * (num_vertices - 1) // 2
        
        if self.child_priors is None:
            self.child_priors = np.zeros([num_edges], dtype=np.float32)
        # Initialize child stats arrays
        if self.child_number_visits is None:
            self.child_number_visits = np.zeros([num_edges], dtype=np.float32)
        if self.child_total_value is None:
            self.child_total_value = np.zeros([num_edges], dtype=np.float32)
        if self.action_idxes is None:
            self.action_idxes = []
    
    def child_Q(self) -> np.ndarray:
        """Calculate Q-values for children using this node's internal stats."""
        # Read from self's arrays, using (1 + N) denominator
        return self.child_total_value / (1.0 + self.child_number_visits)
    
    def child_U(self) -> np.ndarray:
        """Calculate Exploration bonus U for children."""
        # Use self.number_visits (visits TO this node) for sqrt(N(s))
        # Use self.child_number_visits for N(s,a) in denominator
        sqrt_N_s = math.sqrt(max(1.0, self.number_visits))
        return sqrt_N_s * (abs(self.child_priors) / (1.0 + self.child_number_visits))

    def best_child(self) -> int:
        # This should now work correctly with the internal stats
        if self.action_idxes:
            bestmove_scores = self.child_Q() + self.child_U()
            # Select best score among valid action indices
            best_valid_idx = np.argmax(bestmove_scores[self.action_idxes])
            bestmove = self.action_idxes[best_valid_idx]
        else:
            # Fallback if action_idxes not populated (shouldn't happen in normal search)
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove
    
    def select_leaf(self) -> 'UCTNode':
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current
    
    def add_dirichlet_noise(self, action_idxs: List[int], child_priors: np.ndarray, noise_weight: float = 0.25) -> np.ndarray:
        """Add Dirichlet noise to the child priors at the root node for exploration."""
        # Skip noise if weight is zero or not enough actions
        if noise_weight <= 0 or len(action_idxs) <= 1:
            return child_priors

        try:
            # Extract the valid child priors
            valid_child_priors = child_priors[action_idxs]

            # Generate Dirichlet noise (alpha=0.3 is common)
            noise_alpha = 0.3
            noise = np.random.dirichlet([noise_alpha] * len(action_idxs))

            # Mix the priors with noise using the provided weight
            valid_child_priors = (1 - noise_weight) * valid_child_priors + noise_weight * noise

            # Put the modified priors back
            child_priors[action_idxs] = valid_child_priors
        except Exception as e:
            print(f"Warning: Error adding Dirichlet noise: {e}") # Add warning
            # Return original priors if error occurs
            pass

        return child_priors

    def expand(self, child_priors: np.ndarray, noise_weight: float = 0.25) -> None:
        """Expand the node using network priors, optionally adding noise if root."""
        self.is_expanded = True
        action_idxs = []
        c_p = child_priors.copy() # Use the passed priors
        
        valid_moves = self.game.get_valid_moves()
        for edge in valid_moves:
            move_idx = ed.encode_action(self.game, edge)
            if move_idx >= 0:
                action_idxs.append(move_idx)
                
        if not action_idxs:
            self.is_expanded = False # Cannot expand if no valid moves
            return
            
        self.action_idxes = action_idxs
        
        # --- Normalize passed priors and store in self.child_priors ---
        mask = np.zeros_like(c_p, dtype=bool)
        mask[action_idxs] = True
        c_p[~mask] = 0.0 # Zero out invalid moves
        
        valid_sum = c_p[mask].sum()
        if valid_sum > 1e-8:
            c_p[mask] = c_p[mask] / valid_sum # Normalize
        else:
            # If sum is zero (e.g., network predicts 0 for all valid), use uniform
            uniform_prob = 1.0 / len(action_idxs)
            c_p[mask] = uniform_prob
            
        # Add noise if this is the root node (parent is None)
        if self.parent is None and noise_weight > 0:
            c_p = self.add_dirichlet_noise(action_idxs, c_p, noise_weight)
            
        # Set the child priors for this node
        self.child_priors = c_p
        # Note: child_number_visits and child_total_value start at 0
            
    def maybe_add_child(self, move: int) -> 'UCTNode':
        if move not in self.children:
            copy_board = copy.deepcopy(self.game)
            copy_board = make_move_on_board(copy_board, move)
            self.children[move] = UCTNode(
                copy_board, move, parent=self)
        return self.children[move]
    
    def backup(self, value_estimate: float) -> None:
        """Propagate the estimated value up the tree, updating stats."""
        node = self
        # value_estimate is always from P1's perspective
        
        while node is not None: 
            # Increment visits TO this node
            node.number_visits += 1
            
            # Update stats in the PARENT node for the move leading to THIS node
            parent_node = node.parent
            if parent_node is not None:
                move_idx = node.move # The move taken from parent to reach this node
                if move_idx is not None:
                    # Add value from the perspective of the player whose turn it was AT THE PARENT node
                    value_to_add = value_estimate if parent_node.game.player == 0 else -value_estimate
                    parent_node.child_total_value[move_idx] += value_to_add
                    parent_node.child_number_visits[move_idx] += 1 # Increment parent's child visit count
            
            # Move up the tree for the next iteration
            node = parent_node

def UCT_search(game_state: CliqueBoard, num_reads: int, net: nn.Module,
               noise_weight: float = 0.25) -> Tuple[int, UCTNode]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # Create root node with parent=None
    root = UCTNode(game_state, move=None, parent=None)
    
    for i in range(num_reads):
        try:
            leaf = root.select_leaf()
            
            state_dict = ed.prepare_state_for_network(leaf.game)
            edge_index = state_dict['edge_index'].to(device)
            edge_attr = state_dict['edge_attr'].to(device)
            
            with torch.no_grad():
                child_priors, value_estimate = net(edge_index, edge_attr)
                child_priors = child_priors.cpu().numpy().squeeze()
                value_estimate = value_estimate.item()
                
            if leaf.game.game_state != 0 or not leaf.game.get_valid_moves():
                # If game is over, use the game result for backup
                if leaf.game.game_state == 1:  # Player 1 wins
                    value = 1.0
                elif leaf.game.game_state == 2:  # Player 2 wins
                    value = -1.0
                else:  # Draw or ongoing but no valid moves
                    value = 0.0
                    
                leaf.backup(value)
                continue
                
            # Expand the leaf node, passing noise weight
            # expand now correctly sets self.child_priors
            leaf.expand(child_priors, noise_weight)
            # Backup the value estimate from the network
            leaf.backup(value_estimate)
            
        except Exception as e:
            print(f"Error during MCTS simulation {i}: {e}") # Added print
            continue
        
    # Find the move with the most visits, using the root's internal stats
    if root.action_idxes: # Check if root was expanded and has valid actions
        # Use root.child_number_visits array
        valid_visits = root.child_number_visits[root.action_idxes]
        best_action_local_idx = np.argmax(valid_visits)
        best_move = root.action_idxes[best_action_local_idx]
        return best_move, root
    else:
        # Root expansion failed or no valid moves, return a random valid move if any
        valid_moves = game_state.get_valid_moves()
        if valid_moves:
            move_idx = ed.encode_action(game_state, random.choice(valid_moves))
            return move_idx, root
        else:
            # No valid moves from the start
            num_edges = game_state.num_vertices * (game_state.num_vertices - 1) // 2
            return 0 % num_edges, root # Return a default valid index (e.g., 0) if possible

# --- Restore standalone helper function --- 
def make_move_on_board(board: CliqueBoard, move: int) -> CliqueBoard:
    """Make a move on the board given by move index"""
    edge = ed.decode_action(board, move)
    if edge != (-1, -1):
        board.make_move(edge)
    return board
# --- End restored helper function ---

def get_policy(root: UCTNode) -> np.ndarray:
    """Calculates the policy based on visit counts stored in the root node."""
    num_edges = len(root.child_priors) # Use length of priors array for size
    policy = np.zeros([num_edges], dtype=np.float32)
    
    # Use root.child_number_visits array directly
    visits = root.child_number_visits
    total_visits = visits.sum()
    
    if total_visits > 0:
        policy = visits / total_visits
    
    # Apply valid moves mask
    mask = ed.get_valid_moves_mask(root.game)
    policy = ed.apply_valid_moves_mask(policy, mask)
    return policy

def get_q_values(root: UCTNode) -> np.ndarray:
    """Calculates the Q-values for all potential actions from the root."""
    # Use root's internal stats arrays
    q_values = root.child_total_value / (1.0 + root.child_number_visits)
    # Set Q for unvisited moves explicitly? The division yields 0/1=0, which is fine.
    # Q for invalid moves will also be 0.
    return q_values

def save_as_pickle(filename: str, data: List) -> None:
    """Save data to a pickle file.
    
    Args:
        filename: Full path to the file (including directory)
        data: Data to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the data
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

def load_pickle(filename: str) -> List:
    """Load data from a pickle file.
    
    Args:
        filename: Full path to the file (including directory)
    """
    with open(filename, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def MCTS_self_play(clique_net: nn.Module, num_games: int, 
                   num_vertices: int = 6, clique_size: int = 3, cpu: int = 0,
                   mcts_sims: int = 500, game_mode: str = "symmetric",
                   iteration: int = 0, data_dir: str = "./datasets/clique",
                   noise_weight: float = 0.25) -> None:
    """
    Run self-play games using MCTS and save the games.
    
    Args:
        clique_net: Neural network model
        num_games: Number of games to play
        num_vertices: Number of vertices in the graph
        clique_size: Size of clique needed to win
        cpu: CPU index for multiprocessing
        mcts_sims: Number of MCTS simulations per move
        game_mode: "symmetric" or "asymmetric" game mode
        iteration: Current iteration number
        data_dir: Directory to save game data
        noise_weight: Weight for Dirichlet noise during self-play (0 to disable)
    """
    print(f"Starting self-play on CPU {cpu} for iteration {iteration}")
    
    # Create directory for saving games
    os.makedirs(data_dir, exist_ok=True)
    
    for game_idx in range(num_games):
        print(f"\nCPU {cpu}: Starting game {game_idx+1}/{num_games}")
        
        # Initialize a new game
        board = CliqueBoard(num_vertices, clique_size, game_mode)
        game_over = False
        
        # Store game states and policies
        game_states = []
        policies = []
        
        # Maximum possible moves
        max_moves = num_vertices * (num_vertices - 1) // 2
        
        # Play until game over or max moves reached
        while not game_over and board.move_count < max_moves:
            # Print current board state
            print(f"\nMove {board.move_count + 1}:")
            print(f"Current player: {'Player 1' if board.player == 0 else 'Player 2'}")
            print(board)
            
            # Get best move using MCTS, passing noise weight
            best_move, root = UCT_search(board, mcts_sims, clique_net, noise_weight=noise_weight)
            
            # Get policy from root node (MCTS)
            mcts_policy = get_policy(root)
            print(f"MCTS Policy: {mcts_policy}")
            
            # Get model's direct policy prediction
            state_dict = ed.prepare_state_for_network(board)
            with torch.no_grad():
                model_policy, _ = clique_net(state_dict['edge_index'], state_dict['edge_attr'])
                model_policy = model_policy.squeeze().cpu().numpy()
            print(f"Model Policy: {model_policy}")
            
            # Use MCTS policy for actual moves
            policy = mcts_policy
            
            # Store current state and policy
            game_states.append(board.copy())
            policies.append(policy)
            
            # Make the move
            edge = ed.decode_action(board, best_move)
            board = make_move_on_board(board, best_move)
            
            # Print the move made
            print(f"Selected edge: {edge}")
            
            # Check if game is over
            if board.game_state != 0:
                game_over = True
                if board.game_state == 1:
                    print("Player 1 wins!")
                elif board.game_state == 2:
                    print("Player 2 wins!")
                elif board.game_state == 3:
                    print("Game drawn!")
            # Check for draw in symmetric mode
            elif not board.get_valid_moves() and game_mode == "symmetric":
                game_over = True
                board.game_state = 3  # Set draw state
                print("Game drawn - no more valid moves!")
        
        # Store final state
        game_states.append(board.copy())
        
        # Save all moves from this game in a single file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        game_examples = []
        for i in range(len(policies)):
            # Use prepare_state_for_network to properly format the board state
            state_dict = ed.prepare_state_for_network(game_states[i])
            board_state = {
                'edge_index': state_dict['edge_index'].numpy(),
                'edge_attr': state_dict['edge_attr'].numpy()
            }
            example = {
                'board_state': board_state,
                'policy': policies[i],
                'value': 1.0 if board.game_state == 1 else -1.0 if board.game_state == 2 else 0.0
            }
            game_examples.append(example)
        
        filename = f"{data_dir}/game_{timestamp}_cpu{cpu}_game{game_idx}_iter{iteration}.pkl"
        save_as_pickle(filename, game_examples)
        
        print(f"\nCPU {cpu}: Game {game_idx+1} completed. Result: {board.game_state}")
        print(f"Game examples saved to: {filename}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MCTS self-play for Clique Game")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--vertices", type=int, default=6, help="Number of vertices in the graph")
    parser.add_argument("--clique-size", type=int, default=3, help="Size of clique needed to win")
    parser.add_argument("--mcts-sims", type=int, default=500, help="Number of MCTS simulations per move")
    parser.add_argument("--cpu", type=int, default=0, help="CPU index for multiprocessing")
    parser.add_argument("--game-mode", type=str, default="symmetric", help="Game mode: symmetric or asymmetric")
    parser.add_argument("--noise-weight", type=float, default=0.25, help="Weight for Dirichlet noise during self-play (0 to disable)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize network
    net = CliqueGNN(num_vertices=args.vertices)
    
    # Load pretrained model if exists
    model_path = "./model_data/clique_net.pth.tar"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print("No pretrained model found, using random initialization")
    
    # Move model to CPU first for sharing across processes
    net = net.cpu()
    net.share_memory()
    
    # Start self-play process, passing noise weight from args
    MCTS_self_play(net, args.num_games, args.vertices, args.clique_size, args.cpu, args.mcts_sims, args.game_mode, noise_weight=args.noise_weight) 
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
    move: Optional[int]
    parent: Optional['UCTNode']
    is_expanded: bool = False
    children: Dict[int, 'UCTNode'] = None
    child_priors: np.ndarray = None
    child_total_value: np.ndarray = None
    child_number_visits: np.ndarray = None
    action_idxes: List[int] = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}
        
        # Calculate number of edges in the complete graph
        num_vertices = self.game.num_vertices
        num_edges = num_vertices * (num_vertices - 1) // 2
        
        if self.child_priors is None:
            self.child_priors = np.zeros([num_edges], dtype=np.float32)
        if self.child_total_value is None:
            self.child_total_value = np.zeros([num_edges], dtype=np.float32)
        if self.child_number_visits is None:
            self.child_number_visits = np.zeros([num_edges], dtype=np.float32)
        if self.action_idxes is None:
            self.action_idxes = []
    
    @property
    def number_visits(self) -> float:
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value: float) -> None:
        self.parent.child_number_visits[self.move] = value
    
    @property
    def total_value(self) -> float:
        return self.parent.child_total_value[self.move]
    
    @total_value.setter
    def total_value(self, value: float) -> None:
        self.parent.child_total_value[self.move] = value
    
    def child_Q(self) -> np.ndarray:
        # Add small epsilon to avoid division by zero
        return self.child_total_value / (1 + self.child_number_visits)
    
    def child_U(self) -> np.ndarray:
        # Exploration term with safety for division
        return math.sqrt(max(1, self.number_visits)) * (
            abs(self.child_priors) / (1 + self.child_number_visits))
    
    def best_child(self) -> int:
        if self.action_idxes:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove
    
    def select_leaf(self) -> 'UCTNode':
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current
    
    def add_dirichlet_noise(self, action_idxs: List[int], child_priors: np.ndarray) -> np.ndarray:
        """Add Dirichlet noise to the child priors at the root node for exploration"""
        if len(action_idxs) <= 1:
            return child_priors
            
        try:
            # Extract the valid child priors
            valid_child_priors = child_priors[action_idxs]
            
            # Generate Dirichlet noise
            noise = np.random.dirichlet([0.3] * len(action_idxs))
            
            # Mix the priors with noise
            valid_child_priors = 0.75 * valid_child_priors + 0.25 * noise
            
            # Put the modified priors back
            child_priors[action_idxs] = valid_child_priors
        except Exception as e:
            pass
        
        return child_priors
        
    def expand(self, child_priors: np.ndarray) -> None:
        self.is_expanded = True
        action_idxs = []
        c_p = child_priors.copy()
        
        # Get all valid moves
        valid_moves = self.game.get_valid_moves()
        
        for edge in valid_moves:
            move_idx = ed.encode_action(self.game, edge)
            if move_idx >= 0:  # Only add valid encoded moves
                action_idxs.append(move_idx)
                
        if not action_idxs:
            self.is_expanded = False
            return
            
        self.action_idxes = action_idxs
        
        # Zero out invalid moves
        c_p[~np.isin(np.arange(len(c_p)), action_idxs)] = 0.0
        
        # Normalize valid move probabilities
        valid_probs = c_p[action_idxs]
        
        if valid_probs.sum() > 0:
            valid_probs = valid_probs / valid_probs.sum()
            c_p[action_idxs] = valid_probs
        else:
            # If no valid probabilities, use uniform distribution
            c_p[action_idxs] = 1.0 / len(action_idxs)
        
        # Check if this is the root node (has DummyNode as parent)
        is_root = isinstance(self.parent, DummyNode)
        
        # Only add Dirichlet noise at the root of the search tree
        if is_root:
            c_p = self.add_dirichlet_noise(action_idxs, c_p)
            
        # Set child priors
        self.child_priors = c_p
    
    def make_move_on_board(self, board: CliqueBoard, move: int) -> CliqueBoard:
        """Make a move on the board given by move index"""
        edge = ed.decode_action(board, move)
        if edge != (-1, -1):
            board.make_move(edge)
        return board
            
    def maybe_add_child(self, move: int) -> 'UCTNode':
        if move not in self.children:
            copy_board = copy.deepcopy(self.game)
            copy_board = self.make_move_on_board(copy_board, move)
            self.children[move] = UCTNode(
                copy_board, move, parent=self)
        return self.children[move]
    
    def backup(self, value_estimate: float) -> None:
        current = self
        while current.parent is not None:
            current.number_visits += 1
            # Flip value for player 1 vs player 2 perspective
            if current.game.player == 0:  # Player 1's perspective (next move is Player 2)
                current.total_value += value_estimate
            else:  # Player 2's perspective (next move is Player 1)
                current.total_value += -value_estimate
            current = current.parent

class DummyNode:
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)

def UCT_search(game_state: CliqueBoard, num_reads: int, net: nn.Module) -> Tuple[int, UCTNode]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)  # Ensure model is on the correct device
    
    # Create root node with DummyNode as parent
    root = UCTNode(game_state, move=None, parent=DummyNode())
    
    for i in range(num_reads):
        try:
            leaf = root.select_leaf()
            
            # Convert board to network input
            state_dict = ed.prepare_state_for_network(leaf.game)
            edge_index = state_dict['edge_index'].to(device)
            edge_attr = state_dict['edge_attr'].to(device)
            
            with torch.no_grad():
                child_priors, value_estimate = net(edge_index, edge_attr)
                # Reshape child_priors to be 1D array
                child_priors = child_priors.cpu().numpy().squeeze()
                value_estimate = value_estimate.item()
                
            # Check if game is over or if there are no valid moves
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
                
            # Expand the leaf node
            leaf.expand(child_priors)
            leaf.backup(value_estimate)
            
        except Exception as e:
            # Continue with next iteration to keep the search robust
            continue
        
    # Find the move with the most visits
    if len(root.child_number_visits) > 0:
        best_move = np.argmax(root.child_number_visits)
        return best_move, root
    else:
        # No valid moves, return a random valid move if any
        valid_moves = game_state.get_valid_moves()
        if valid_moves:
            move_idx = ed.encode_action(game_state, random.choice(valid_moves))
            return move_idx, root
        else:
            return 0, root

def make_move_on_board(board: CliqueBoard, move: int) -> CliqueBoard:
    """Make a move on the board given by move index"""
    edge = ed.decode_action(board, move)
    if edge != (-1, -1):
        board.make_move(edge)
    return board

def get_policy(root: UCTNode) -> np.ndarray:
    num_vertices = root.game.num_vertices
    num_edges = num_vertices * (num_vertices - 1) // 2
    
    policy = np.zeros([num_edges], dtype=np.float32)
    valid_indices = np.where(root.child_number_visits != 0)[0]
    if len(valid_indices) > 0:
        policy[valid_indices] = root.child_number_visits[valid_indices] / root.child_number_visits.sum()
    return policy

def save_as_pickle(filename: str, data: List) -> None:
    os.makedirs("./datasets/clique/", exist_ok=True)
    complete_name = os.path.join("./datasets/clique/", filename + ".pkl")
    with open(complete_name, 'wb') as output:
        pickle.dump(data, output)

def load_pickle(filename: str) -> List:
    complete_name = os.path.join("./datasets/clique/", filename)
    with open(complete_name, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def MCTS_self_play(clique_net: nn.Module, num_games: int, 
                   num_vertices: int = 6, clique_size: int = 3, cpu: int = 0,
                   mcts_sims: int = 500) -> None:
    """
    Play multiple games of Clique Game against itself using MCTS and save the data.
    
    Args:
        clique_net: Neural network model to use for MCTS
        num_games: Number of games to play
        num_vertices: Number of vertices in the graph
        clique_size: Size of clique needed for Player 1 to win
        cpu: CPU index for multiprocessing
        mcts_sims: Number of MCTS simulations per move
    """
    os.makedirs("./model_data", exist_ok=True)
    os.makedirs("./datasets/clique", exist_ok=True)
    
    # Set device - since we use spawn, it's safe to check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # The model should already be on CPU from the parent process
    # We'll move it to the current device (CPU or GPU) within this process
    clique_net = clique_net.to(device)
    clique_net.eval()  # Set to evaluation mode
    
    print(f"Process {cpu}: Using device {device}")
    
    for game_idx in range(num_games):
        # Initialize a new game
        current_board = CliqueBoard(num_vertices, clique_size)
        game_over = False
        dataset = []
        states = []  # Track states to detect repetitions
        value = 0
        
        # Store game states for visualization
        game_states = [current_board.copy()]
        
        print(f"Game {game_idx+1}/{num_games}")
        
        # Play until game is over or move limit reached
        max_moves = num_vertices * (num_vertices - 1) // 2  # Max possible moves (all edges)
        while not game_over and current_board.move_count < max_moves:
            # Store current state to check for repetitions
            current_state_str = str(current_board.edge_states)
            states.append(current_state_str)
            
            # Get current board state
            board_state = current_board.get_board_state()
            
            # Use MCTS to find the best move
            best_move, root = UCT_search(current_board, mcts_sims, clique_net)
            
            # Get the policy from MCTS visits
            policy = get_policy(root)
            
            # Store the state and policy
            dataset.append([board_state, policy])
            
            # Make the best move
            current_board = make_move_on_board(current_board, best_move)
            game_states.append(current_board.copy())
            
            # Print current game state
            print(f"Move {current_board.move_count}:")
            print(current_board)
            edge = ed.decode_action(current_board, best_move)
            player = "Player 1" if current_board.player == 1 else "Player 2" # Player has already switched
            print(f"{player} selected edge {edge}")
            
            # Check if game is over
            if current_board.game_state != 0:
                game_over = True
                if current_board.game_state == 1:  # Player 1 wins
                    value = 1
                    print("Player 1 wins!")
                else:  # Player 2 wins
                    value = -1
                    print("Player 2 wins!")
            
            # Check for draw by board filled
            if not current_board.get_valid_moves():
                game_over = True
                print("Game drawn - board filled!")
                
        # Prepare dataset with values
        dataset_with_values = []
        for idx, (s, p) in enumerate(dataset):
            # Calculate final reward from perspective of the player who made the move
            player_perspective = 1 if idx % 2 == 1 else -1
            final_value = value * player_perspective
            
            dataset_with_values.append([s, p, final_value])
            
        # Save dataset
        save_as_pickle(
            f"clique_game_cpu{cpu}_{game_idx}_{datetime.datetime.today().strftime('%Y-%m-%d')}",
            dataset_with_values
        )
        
        # Visualize and save the game
        if cpu == 0 and game_idx == 0:  # Only visualize first game on first CPU
            save_dir = "./model_data/game_visualizations"
            os.makedirs(save_dir, exist_ok=True)
            
            for i, state in enumerate(game_states):
                fig = view_clique_board(state)
                plt.savefig(os.path.join(save_dir, f"clique_game_{game_idx}_move_{i}.png"))
                plt.close(fig)
            
            print(f"Game visualization saved to {save_dir}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MCTS self-play for Clique Game")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--vertices", type=int, default=6, help="Number of vertices in the graph")
    parser.add_argument("--clique-size", type=int, default=3, help="Size of clique needed to win")
    parser.add_argument("--mcts-sims", type=int, default=500, help="Number of MCTS simulations per move")
    parser.add_argument("--cpu", type=int, default=0, help="CPU index for multiprocessing")
    
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
    
    # Start self-play process
    MCTS_self_play(net, args.num_games, args.vertices, args.clique_size, args.cpu, args.mcts_sims) 
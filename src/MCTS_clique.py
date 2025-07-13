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
        # The AlphaZero paper uses c_puct = 1.0, but we're using higher values
        c_puct = 3.0  # Significantly higher exploration for better search breadth
        sqrt_N_s = math.sqrt(max(1.0, self.number_visits))
        return c_puct * sqrt_N_s * (abs(self.child_priors) / (1.0 + self.child_number_visits))

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

            # The AlphaZero paper uses alpha = 0.03 for Chess and 0.3 for Go
            # For the Clique Game, we're using a balanced value tuned for 6-vertex graphs
            noise_alpha = 0.3  # More exploration with slightly more uniform noise
            
            # The value of alpha depends on the number of valid actions
            # For games with fewer legal moves, we want more focused exploration
            if len(action_idxs) < 5:
                noise_alpha = 0.15  # Slightly more exploration for small action spaces
            elif len(action_idxs) > 15:
                noise_alpha = 0.4  # More uniform for large action spaces
            
            # Generate Dirichlet noise
            noise = np.random.dirichlet([noise_alpha] * len(action_idxs))

            # Stabilize extremes in the noise distribution
            noise = np.clip(noise, 0.01, 0.99)
            
            # Mix the priors with noise using the provided weight
            valid_child_priors = (1 - noise_weight) * valid_child_priors + noise_weight * noise

            # Put the modified priors back
            child_priors[action_idxs] = valid_child_priors
            
            # Ensure the priors sum to 1.0 after noise addition
            if valid_child_priors.sum() > 0:
                child_priors[action_idxs] = valid_child_priors / valid_child_priors.sum()
                
            # Print noise information periodically
            if np.random.random() < 0.01:  # 1% chance to print
                print(f"Dirichlet noise added: alpha={noise_alpha}, "
                      f"num_actions={len(action_idxs)}, weight={noise_weight}")
                
        except Exception as e:
            print(f"Warning: Error adding Dirichlet noise: {e}")
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
    
    def backup(self, value_estimate: float, perspective_mode: str = "alternating") -> None:
        """Propagate the estimated value up the tree, updating stats.
        
        Args:
            value_estimate: Value from either fixed (P1) or alternating (current player) perspective
            perspective_mode: "fixed" or "alternating"
        """
        node = self
        current_value = value_estimate
        
        # For terminal nodes, we apply a higher weight to reinforce learning of winning/losing positions
        # This can help the network more quickly learn the value of terminal states
        terminal_state_weight = 1.0
        if self.game.game_state != 0:  # Terminal state (game over)
            terminal_state_weight = 2.0  # Increase the weight for terminal states
        
        while node is not None: 
            # Increment visits TO this node
            node.number_visits += 1
            
            # Update stats in the PARENT node for the move leading to THIS node
            parent_node = node.parent
            if parent_node is not None:
                move_idx = node.move # The move taken from parent to reach this node
                if move_idx is not None:
                    if perspective_mode == "fixed":
                        # Fixed perspective: value is always from P1's perspective
                        # Need to flip if parent is P2
                        value_to_add = terminal_state_weight * (current_value if parent_node.game.player == 0 else -current_value)
                    else:  # alternating
                        # Alternating perspective: value flips at each level
                        value_from_parent_perspective = -current_value
                        value_to_add = terminal_state_weight * value_from_parent_perspective
                    
                    parent_node.child_total_value[move_idx] += value_to_add
                    parent_node.child_number_visits[move_idx] += 1 # Increment parent's child visit count
            
            # Update current_value for next level
            if perspective_mode == "alternating":
                # Flip value for next level up (alternating players)
                current_value = -current_value
            # For fixed perspective, current_value stays the same
            
            # Move up the tree for the next iteration
            node = parent_node

def UCT_search(game_state: CliqueBoard, num_reads: int, net: nn.Module,
               perspective_mode: str = "alternating", 
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
                # Pass player role for asymmetric models
                if hasattr(net, 'asymmetric_mode') and net.asymmetric_mode:
                    player_role = leaf.game.player  # 0 for attacker (Player 1), 1 for defender (Player 2)
                    child_priors, value_estimate = net(edge_index, edge_attr, player_role=player_role)
                else:
                    child_priors, value_estimate = net(edge_index, edge_attr)
                child_priors = child_priors.cpu().numpy().squeeze()
                value_estimate = value_estimate.item()
                
            if leaf.game.game_state != 0 or not leaf.game.get_valid_moves():
                # If game is over, use the game result based on perspective mode
                if perspective_mode == "fixed":
                    # Fixed perspective: always from Player 1's perspective
                    if leaf.game.game_state == 1:  # Player 1 wins
                        value = 1.0
                    elif leaf.game.game_state == 2:  # Player 2 wins
                        value = -1.0
                    else:  # Draw or ongoing but no valid moves
                        value = 0.0
                else:  # alternating
                    # Alternating perspective: from current player's perspective
                    current_player = leaf.game.player
                    if leaf.game.game_state == 1:  # Player 1 wins
                        value = 1.0 if current_player == 0 else -1.0
                    elif leaf.game.game_state == 2:  # Player 2 wins
                        value = -1.0 if current_player == 0 else 1.0
                    else:  # Draw or ongoing but no valid moves
                        value = 0.0
                    
                leaf.backup(value, perspective_mode)
                continue
                
            # Expand the leaf node, passing noise weight
            # expand now correctly sets self.child_priors
            leaf.expand(child_priors, noise_weight)
            # Backup the value estimate from the network
            leaf.backup(value_estimate, perspective_mode)
            
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

def get_policy(root: UCTNode, temperature: float = 1.0) -> np.ndarray:
    """
    Calculates the policy based on visit counts stored in the root node.
    
    Args:
        root: The root MCTS node
        temperature: Temperature parameter for controlling exploration vs exploitation
                     - temperature=1.0: standard MCTS policy (proportional to visit counts)
                     - temperature → 0: deterministic policy (highest visit count only)
                     - temperature > 1: more uniform policy (more exploration)
    
    Returns:
        policy: Distribution over all possible actions
    """
    num_edges = len(root.child_priors)  # Use length of priors array for size
    policy = np.zeros([num_edges], dtype=np.float32)
    
    # Use root.child_number_visits array directly
    visits = root.child_number_visits
    
    # Apply temperature by raising visits to power of 1/temperature
    if temperature != 0:
        # Avoid division by zero
        # For high temperatures (>1), this will make the policy more uniform
        # For low temperatures (<1), this will make the policy more deterministic
        visits_temp = np.power(visits + 1e-8, 1.0 / max(temperature, 1e-8))
    else:
        # For temperature=0, use a deterministic policy
        best_action = np.argmax(visits)
        visits_temp = np.zeros_like(visits)
        visits_temp[best_action] = 1.0
    
    # Normalize to create a probability distribution
    total_visits_temp = visits_temp.sum()
    if total_visits_temp > 0:
        policy = visits_temp / total_visits_temp
    
    # Apply valid moves mask to ensure only valid moves have non-zero probability
    mask = ed.get_valid_moves_mask(root.game)
    policy = ed.apply_valid_moves_mask(policy, mask)
    
    # Verify the policy is valid
    if policy.sum() < 1e-6:
        # Fallback to uniform policy over valid moves if something went wrong
        print("Warning: Invalid policy distribution from MCTS. Using uniform fallback.")
        valid_moves = root.game.get_valid_moves()
        if valid_moves:
            uniform_policy = np.zeros([num_edges], dtype=np.float32)
            for move in valid_moves:
                move_idx = ed.encode_action(root.game, move)
                if move_idx >= 0:
                    uniform_policy[move_idx] = 1.0 / len(valid_moves)
            policy = uniform_policy
    
    return policy

def get_q_values(root: UCTNode) -> np.ndarray:
    """Calculates the Q-values for all potential actions from the root."""
    # Use root's internal stats arrays
    q_values = root.child_total_value / (1.0 + root.child_number_visits)
    # Set Q for unvisited moves explicitly? The division yields 0/1=0, which is fine.
    # Q for invalid moves will also be 0.
    return q_values

def get_varied_mcts_sims(base_sims: int, skill_variation: float) -> Tuple[int, int]:
    """
    Get varied MCTS simulation counts for both players based on skill variation.
    
    Args:
        base_sims: Base number of MCTS simulations
        skill_variation: Percentage variation (0.0 to 1.0). E.g., 0.3 = ±30% variation
                        When > 0, players get different simulation counts randomly
    
    Returns:
        Tuple of (player1_sims, player2_sims)
    """
    if skill_variation <= 0:
        # No variation - both players use same simulation count
        return base_sims, base_sims
    
    # Clamp skill_variation to reasonable bounds
    skill_variation = min(skill_variation, 0.8)  # Max 80% variation
    
    # Generate percentage-based variations for each player
    # Use uniform distribution within the specified percentage bounds
    min_factor = 1.0 - skill_variation
    max_factor = 1.0 + skill_variation
    
    player1_factor = np.random.uniform(min_factor, max_factor)
    player2_factor = np.random.uniform(min_factor, max_factor)
    
    player1_sims = max(1, int(base_sims * player1_factor))
    player2_sims = max(1, int(base_sims * player2_factor))
    
    return player1_sims, player2_sims

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
                   perspective_mode: str = "alternating", noise_weight: float = 0.25,
                   skill_variation: float = 0.0) -> None:
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
        perspective_mode: "fixed" (Player 1) or "alternating" (current player)
        noise_weight: Weight for Dirichlet noise during self-play (0 to disable)
        skill_variation: Variation in MCTS simulation counts (0 = no variation)
    """
    print(f"Starting self-play on CPU {cpu} for iteration {iteration}")
    
    # Create directory for saving games
    os.makedirs(data_dir, exist_ok=True)
    
    for game_idx in range(num_games):
        print(f"\nCPU {cpu}: Starting game {game_idx+1}/{num_games}")
        
        # Get varied MCTS simulation counts for this game
        player1_sims, player2_sims = get_varied_mcts_sims(mcts_sims, skill_variation)
        if skill_variation > 0:
            print(f"Skill variation enabled: P1={player1_sims} sims, P2={player2_sims} sims")
        
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
            
            # Calculate annealing temperature based on move number
            # Start with high temperature (1.0) for exploration
            # Gradually decrease as the game progresses for more exploitation
            max_moves = num_vertices * (num_vertices - 1) // 2
            move_progress = board.move_count / max_moves
            
            # Temperature annealing - more aggressive schedule
            if move_progress < 0.2:  # First 20% of the game
                temperature = 1.0  # High exploration
            elif move_progress < 0.4:  # Next 20% of the game
                temperature = 0.8  # Still good exploration
            elif move_progress < 0.6:  # Middle 20% of the game
                temperature = 0.5  # Balanced exploration/exploitation
            elif move_progress < 0.8:  # Next 20% of the game
                temperature = 0.2  # More exploitation
            else:  # Last 20% of the game
                temperature = 0.1  # Strong exploitation
            
            # Also reduce noise weight as the game progresses
            current_noise_weight = noise_weight * (1.0 - move_progress)
            
            # Use player-specific simulation counts based on skill variation
            current_mcts_sims = player1_sims if board.player == 0 else player2_sims
            
            # Get best move using MCTS with adjusted noise weight and player-specific sims
            best_move, root = UCT_search(board, current_mcts_sims, clique_net, 
                                        perspective_mode=perspective_mode, 
                                        noise_weight=current_noise_weight)
            
            # Get policy from root node using temperature
            mcts_policy = get_policy(root, temperature=temperature)
            print(f"MCTS Policy (temp={temperature:.2f}): {mcts_policy}")
            
            # Get model's direct policy prediction for comparison
            state_dict = ed.prepare_state_for_network(board)
            with torch.no_grad():
                model_policy, value_estimate = clique_net(state_dict['edge_index'], state_dict['edge_attr'])
                model_policy = model_policy.squeeze().cpu().numpy()
                value_estimate = value_estimate.item()
            print(f"Model Policy: {model_policy}")
            print(f"Model Value Estimate: {value_estimate:.4f}")
            
            # Use MCTS policy for actual moves and training
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
            # Get value based on perspective mode
            if perspective_mode == "fixed":
                # Fixed perspective: always from Player 1's perspective
                if board.game_state == 1:  # Player 1 wins
                    value = 1.0
                elif board.game_state == 2:  # Player 2 wins
                    value = -1.0
                else:  # Draw
                    value = 0.0
            else:  # alternating
                # Alternating perspective: from current player's perspective
                current_player = game_states[i].player
                if board.game_state == 1:  # Player 1 wins
                    value = 1.0 if current_player == 0 else -1.0
                elif board.game_state == 2:  # Player 2 wins
                    value = -1.0 if current_player == 0 else 1.0
                else:  # Draw
                    value = 0.0
            
            # Add player role for asymmetric training
            current_player = game_states[i].player
            example = {
                'board_state': board_state,
                'policy': policies[i],
                'value': value,
                'player_role': current_player  # 0 = attacker/Player1, 1 = defender/Player2
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
    MCTS_self_play(net, args.num_games, args.vertices, args.clique_size, args.cpu, args.mcts_sims, args.game_mode, 
                   perspective_mode="alternating", noise_weight=args.noise_weight, skill_variation=0.0) 
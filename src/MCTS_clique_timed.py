#!/usr/bin/env python
"""
MCTS with detailed timing instrumentation for performance analysis.
"""
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
    child_priors: np.ndarray = None
    child_total_value: np.ndarray = None  # W(s,a)
    child_number_visits: np.ndarray = None  # N(s,a)
    value_sum: float = 0.0 # Accumulated value FROM this node's perspective
    player: int = 0  # Player whose turn it is at this node
    depth: int = 0  # Depth in tree
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
        self.player = self.game.player
        if self.parent:
            self.depth = self.parent.depth + 1
    
    def child_Q(self) -> np.ndarray:
        """Calculate Q values for all children using (1 + N) formula matching PyTorch version."""
        return self.child_total_value / (1.0 + self.child_number_visits)
    
    def child_U(self, c_puct: float) -> np.ndarray:
        """Calculate U values for all children. Uses visits TO this node."""
        return c_puct * np.sqrt(self.number_visits) * self.child_priors / (1.0 + self.child_number_visits)
    
    def select_leaf(self, c_puct: float = 3.0, timing_stats: dict = None) -> 'UCTNode':
        """Select a leaf node by recursively applying UCB."""
        current = self
        
        while current.is_expanded and current.game.game_state == 0:
            # UCB calculation timing
            ucb_start = time.time()
            
            child_action_score = current.child_Q() + current.child_U(c_puct)
            valid_moves = current.game.get_valid_moves()
            
            if len(valid_moves) == 0:
                break
                
            masked_scores = np.ones(len(child_action_score)) * -np.inf
            masked_scores[valid_moves] = child_action_score[valid_moves]
            
            if np.all(masked_scores == -np.inf):
                break
                
            best_move = np.argmax(masked_scores)
            
            if timing_stats is not None:
                timing_stats['ucb_calc'].append(time.time() - ucb_start)
            
            # Move to child
            current = current.maybe_add_child(best_move, timing_stats)
        
        return current
    
    def maybe_add_child(self, move: int, timing_stats: dict = None) -> 'UCTNode':
        """Add child if it doesn't exist, then return it."""
        if move not in self.children:
            # Board copy timing
            copy_start = time.time()
            
            # IMPORTANT: Make a deep copy of the board, don't share state
            new_board = copy.deepcopy(self.game)
            new_board.make_move(move)
            
            if timing_stats is not None:
                timing_stats['board_copy'].append(time.time() - copy_start)
            
            self.children[move] = UCTNode(
                game=new_board, 
                move=move,
                parent=self
            )
        
        return self.children[move]
    
    def backup(self, value_estimate: float, perspective_mode: str = "alternating") -> None:
        """Propagate the estimated value up the tree, updating stats."""
        node = self
        current_value = value_estimate
        
        terminal_state_weight = 1.0
        if self.game.game_state != 0:
            terminal_state_weight = 2.0
        
        while node is not None: 
            node.number_visits += 1
            node.value_sum += current_value * terminal_state_weight
            
            if node.parent:
                parent_node = node.parent
                parent_node.child_number_visits[node.move] += 1
                parent_node.child_total_value[node.move] += current_value * terminal_state_weight
            
            parent_node = node.parent
            if perspective_mode == "alternating":
                current_value = -current_value
            
            node = parent_node

def UCT_search_timed(game_state: CliqueBoard, num_reads: int, net: nn.Module,
                     perspective_mode: str = "alternating", 
                     noise_weight: float = 0.25) -> Tuple[int, UCTNode, dict]:
    """UCT search with detailed timing instrumentation."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # Timing statistics
    timing_stats = collections.defaultdict(list)
    total_start = time.time()
    
    # Create root node
    init_start = time.time()
    root = UCTNode(game_state, move=None, parent=None)
    timing_stats['initialization'].append(time.time() - init_start)
    
    # Detailed timing for simulations
    sim_times = {
        'selection': [],
        'nn_prep': [],
        'nn_eval': [],
        'expansion': [],
        'backup': [],
        'total': []
    }
    
    for i in range(num_reads):
        sim_start = time.time()
        
        try:
            # Selection phase
            select_start = time.time()
            leaf = root.select_leaf(timing_stats=timing_stats)
            sim_times['selection'].append(time.time() - select_start)
            
            # Neural network preparation
            nn_prep_start = time.time()
            state_dict = ed.prepare_state_for_network(leaf.game)
            edge_index = state_dict['edge_index'].to(device)
            edge_attr = state_dict['edge_attr'].to(device)
            sim_times['nn_prep'].append(time.time() - nn_prep_start)
            
            # Neural network evaluation
            nn_eval_start = time.time()
            with torch.no_grad():
                if hasattr(net, 'asymmetric_mode') and net.asymmetric_mode:
                    player_role = leaf.game.player
                    child_priors, value_estimate = net(edge_index, edge_attr, player_role=player_role)
                else:
                    child_priors, value_estimate = net(edge_index, edge_attr)
            
            child_priors = child_priors.squeeze().cpu().numpy()
            value_estimate = value_estimate.item()
            sim_times['nn_eval'].append(time.time() - nn_eval_start)
            
            # Expansion phase
            expand_start = time.time()
            if leaf.game.game_state == 0:
                leaf.is_expanded = True
                num_moves = child_priors.shape[0]
                leaf.child_priors = child_priors
                leaf.child_total_value = np.zeros(num_moves, dtype=np.float32)
                leaf.child_number_visits = np.zeros(num_moves, dtype=np.float32)
                
                # Add Dirichlet noise to root
                if leaf == root and noise_weight > 0:
                    valid_moves = leaf.game.get_valid_moves()
                    if len(valid_moves) > 0:
                        noise = np.random.dirichlet([0.3] * len(valid_moves))
                        leaf.child_priors[valid_moves] = (1 - noise_weight) * leaf.child_priors[valid_moves] + noise_weight * noise
            sim_times['expansion'].append(time.time() - expand_start)
            
            # Backup phase
            backup_start = time.time()
            leaf.backup(value_estimate, perspective_mode)
            sim_times['backup'].append(time.time() - backup_start)
            
        except Exception as e:
            print(f"Error in MCTS simulation {i}: {e}")
            import traceback
            traceback.print_exc()
            break
        
        sim_times['total'].append(time.time() - sim_start)
    
    total_time = time.time() - total_start
    
    # Prepare timing analysis
    timing_analysis = {
        'total_time': total_time,
        'num_simulations': num_reads,
        'sim_times': sim_times,
        'operation_stats': {}
    }
    
    # Calculate operation statistics
    for op, times in timing_stats.items():
        if times:
            timing_analysis['operation_stats'][op] = {
                'count': len(times),
                'avg_ms': np.mean(times) * 1000,
                'total_s': np.sum(times)
            }
    
    # Print timing analysis
    print(f"\n      === ORIGINAL MCTS TIMING ANALYSIS ===")
    print(f"      Simulations: {num_reads}")
    print(f"      Total MCTS time: {total_time:.3f}s")
    
    print(f"\n      Per-simulation timing (avg over {num_reads} sims):")
    print(f"        Selection: {np.mean(sim_times['selection'])*1000:.1f}ms")
    print(f"        NN Prep: {np.mean(sim_times['nn_prep'])*1000:.1f}ms")
    print(f"        NN Eval: {np.mean(sim_times['nn_eval'])*1000:.1f}ms")
    print(f"        Expansion: {np.mean(sim_times['expansion'])*1000:.1f}ms")
    print(f"        Backup: {np.mean(sim_times['backup'])*1000:.1f}ms")
    print(f"        Total: {np.mean(sim_times['total'])*1000:.1f}ms")
    
    print(f"\n      Detailed operation counts and timing:")
    for op, stats in timing_analysis['operation_stats'].items():
        print(f"        {op}: {stats['count']} calls, {stats['avg_ms']:.2f}ms avg, {stats['total_s']:.3f}s total")
    
    # Count nodes in tree
    def count_nodes(node):
        return 1 + sum(count_nodes(child) for child in node.children.values())
    
    total_nodes = count_nodes(root)
    print(f"\n      Total nodes in tree: {total_nodes}")
    
    # Get best move
    if root.children:
        best_moves = root.game.get_valid_moves()
        masked_visits = np.zeros(root.child_number_visits.shape)
        masked_visits[best_moves] = root.child_number_visits[best_moves]
        best_move = int(np.argmax(masked_visits))
    else:
        best_move = 0
    
    return best_move, root, timing_analysis


def MCTS_self_play_timed(network: nn.Module, num_games: int, vertices: int, k: int, cpu: int, 
                         num_simulations: int, game_mode: str = "symmetric", iteration: int = 0, 
                         save_dir: str = './datasets/', perspective_mode: str = "alternating",
                         noise_weight: float = 0.25, skill_variation: float = 0.0):
    """
    Self-play with timing analysis for specified number of games.
    """
    print(f"\nOriginal MCTS Self-Play - Process {cpu}")
    print(f"Games: {num_games}, n={vertices}, k={k}, simulations={num_simulations}")
    
    # Aggregate timing across all games
    all_timing = {
        'games': [],
        'total_game_times': [],
        'avg_move_times': [],
        'moves_per_game': []
    }
    
    for idx in range(num_games):
        game_start = time.time()
        print(f"\n  Game {idx+1}/{num_games}")
        
        examples = []
        board = CliqueBoard(vertices, k, game_mode)
        
        move_count = 0
        move_times = []
        
        while board.game_state == 0:
            move_start = time.time()
            
            # Run MCTS with timing
            best_move, root, timing_analysis = UCT_search_timed(
                board, num_simulations, network, 
                perspective_mode=perspective_mode,
                noise_weight=noise_weight
            )
            
            move_time = time.time() - move_start
            move_times.append(move_time)
            
            print(f"    Move {move_count}: {move_time:.2f}s")
            
            # Get action probs from root
            action_probs = np.zeros(vertices * (vertices - 1) // 2)
            valid_moves = board.get_valid_moves()
            
            if len(valid_moves) > 0 and hasattr(root, 'child_number_visits'):
                visits = root.child_number_visits[valid_moves]
                if visits.sum() > 0:
                    action_probs[valid_moves] = visits / visits.sum()
                else:
                    action_probs[valid_moves] = 1.0 / len(valid_moves)
            
            # Store example
            examples.append({
                'board_state': copy.deepcopy(board),
                'policy': action_probs,
                'value': None,
                'player_role': board.player if game_mode == "asymmetric" else None
            })
            
            # Make move
            board.make_move(best_move)
            move_count += 1
        
        # Assign values
        winner = board.winner
        for ex in examples:
            if perspective_mode == "alternating":
                ex['value'] = 1.0 if ex['board_state'].player == winner else -1.0
            else:
                ex['value'] = 1.0 if winner == 0 else -1.0
        
        game_time = time.time() - game_start
        avg_move_time = np.mean(move_times)
        
        print(f"  Game completed in {game_time:.1f}s ({move_count} moves, {avg_move_time:.2f}s/move)")
        
        all_timing['games'].append(idx)
        all_timing['total_game_times'].append(game_time)
        all_timing['avg_move_times'].append(avg_move_time)
        all_timing['moves_per_game'].append(move_count)
        
        # Save game data
        filename = f"game_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_cpu{cpu}_game{idx}_iter{iteration}.pkl"
        with open(os.path.join(save_dir, filename), 'wb') as f:
            pickle.dump(examples, f)
    
    # Summary
    print(f"\n=== ORIGINAL MCTS SELF-PLAY SUMMARY ===")
    print(f"Total games: {num_games}")
    print(f"Total time: {sum(all_timing['total_game_times']):.1f}s")
    print(f"Avg game time: {np.mean(all_timing['total_game_times']):.1f}s")
    print(f"Avg moves per game: {np.mean(all_timing['moves_per_game']):.1f}")
    print(f"Avg time per move: {np.mean(all_timing['avg_move_times']):.2f}s")
    
    return all_timing


if __name__ == "__main__":
    # Test timing analysis
    import sys
    sys.path.append('.')
    
    # Create test network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CliqueGNN(num_vertices=6, hidden_dim=64, num_layers=3).to(device)
    net.eval()
    
    # Run self-play with timing
    os.makedirs('./test_timing', exist_ok=True)
    MCTS_self_play_timed(net, num_games=2, vertices=6, k=3, cpu=0, 
                         num_simulations=20, save_dir='./test_timing')
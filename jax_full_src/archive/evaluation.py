#!/usr/bin/env python
"""
Evaluation functions for AlphaZero - JAX compatible version
"""

import torch
from typing import Tuple
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from MCTS_clique import MCTS_self_play, UCT_search
from clique_board import CliqueBoard
import encoder_decoder_clique as ed


def evaluate_against_mcts(model: torch.nn.Module, num_games: int = 20,
                         mcts_simulations: int = 100, num_vertices: int = 6,
                         k: int = 3, game_mode: str = "asymmetric") -> float:
    """
    Evaluate model against pure MCTS.
    
    Returns:
        Win rate of the model
    """
    wins = 0
    draws = 0
    # Model will be moved to device in UCT_search
    
    for game in range(num_games):
        # Alternate who goes first
        model_player = game % 2
        
        # Play one game
        board = CliqueBoard(num_vertices, k, game_mode)
        
        while board.game_state == 0 and board.move_count < (num_vertices * (num_vertices - 1) // 2):
            if board.player == model_player:
                # Model with MCTS
                best_edge, _ = UCT_search(board, mcts_simulations//2, model)
            else:
                # Pure MCTS (no model)
                best_edge, _ = UCT_search(board, mcts_simulations, None)
            
            if best_edge is None:
                break
                
            board.make_move(best_edge)
        
        # Check result
        if board.game_state == 1:  # Player 1 won
            if model_player == 0:
                wins += 1
        elif board.game_state == 2:  # Player 2 won
            if model_player == 1:
                wins += 1
        else:
            draws += 0.5
    
    win_rate = (wins + draws) / num_games
    return win_rate


def evaluate_head_to_head(model1: torch.nn.Module, model2: torch.nn.Module,
                         num_games: int = 20, num_vertices: int = 6,
                         k: int = 3, game_mode: str = "asymmetric") -> float:
    """
    Evaluate model1 against model2.
    
    Returns:
        Win rate of model1
    """
    wins = 0
    draws = 0
    # Models will be moved to device in UCT_search
    
    for game in range(num_games):
        # Alternate who goes first
        model1_player = game % 2
        
        # Play one game
        board = CliqueBoard(num_vertices, k, game_mode)
        
        while board.game_state == 0 and board.move_count < (num_vertices * (num_vertices - 1) // 2):
            if board.player == model1_player:
                # Model 1
                best_edge, _ = UCT_search(board, 50, model1)
            else:
                # Model 2
                best_edge, _ = UCT_search(board, 50, model2)
            
            if best_edge is None:
                break
                
            board.make_move(best_edge)
        
        # Check result
        if board.game_state == 1:  # Player 1 won
            if model1_player == 0:
                wins += 1
        elif board.game_state == 2:  # Player 2 won
            if model1_player == 1:
                wins += 1
        else:
            draws += 0.5
    
    win_rate = (wins + draws) / num_games
    return win_rate
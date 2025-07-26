#!/usr/bin/env python
"""
Evaluation functions for AlphaZero
"""

import torch
from typing import Tuple
from mcts_clique import MCTS
from clique_board_numpy import CliqueBoard


def evaluate_against_mcts(model: torch.nn.Module, num_games: int = 20,
                         mcts_simulations: int = 100) -> float:
    """
    Evaluate model against pure MCTS.
    
    Returns:
        Win rate of the model
    """
    wins = 0
    draws = 0
    
    for game in range(num_games):
        # Alternate who goes first
        model_player = game % 2
        
        # Play one game
        board = CliqueBoard()
        
        while not board.is_game_over():
            if board.current_player == model_player:
                # Model with MCTS
                mcts = MCTS(model, c_puct=1.0)
                action = mcts.search(board, num_simulations=mcts_simulations//2)
            else:
                # Pure MCTS
                mcts = MCTS(None, c_puct=1.0)
                action = mcts.search(board, num_simulations=mcts_simulations)
            
            board.make_move(action)
        
        # Check result
        winner = board.get_winner()
        if winner == model_player:
            wins += 1
        elif winner == -1:
            draws += 0.5
    
    win_rate = (wins + draws) / num_games
    return win_rate


def evaluate_head_to_head(model1: torch.nn.Module, model2: torch.nn.Module,
                         num_games: int = 20) -> float:
    """
    Evaluate model1 against model2.
    
    Returns:
        Win rate of model1
    """
    wins = 0
    draws = 0
    
    for game in range(num_games):
        # Alternate who goes first
        model1_player = game % 2
        
        # Play one game
        board = CliqueBoard()
        
        while not board.is_game_over():
            if board.current_player == model1_player:
                # Model 1
                mcts = MCTS(model1, c_puct=1.0)
                action = mcts.search(board, num_simulations=50)
            else:
                # Model 2
                mcts = MCTS(model2, c_puct=1.0)
                action = mcts.search(board, num_simulations=50)
            
            board.make_move(action)
        
        # Check result
        winner = board.get_winner()
        if winner == model1_player:
            wins += 1
        elif winner == -1:
            draws += 0.5
    
    win_rate = (wins + draws) / num_games
    return win_rate
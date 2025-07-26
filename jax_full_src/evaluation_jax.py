#!/usr/bin/env python
"""
JAX-based evaluation functions
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from vectorized_board import VectorizedCliqueBoard
from vectorized_mcts import SimplifiedVectorizedMCTS
from vectorized_nn import BatchedNeuralNetwork


def evaluate_against_mcts_jax(
    model_state,
    model,
    num_games: int = 20,
    mcts_simulations: int = 100
) -> float:
    """
    Evaluate model against pure MCTS using JAX.
    
    Returns:
        Win rate of the model
    """
    wins = 0
    draws = 0
    
    # Create MCTS instances
    model_mcts = SimplifiedVectorizedMCTS(
        batch_size=1,
        num_actions=15,
        c_puct=1.0
    )
    
    # For pure MCTS, we need a dummy uniform policy network
    pure_mcts = SimplifiedVectorizedMCTS(
        batch_size=1,
        num_actions=15,
        c_puct=1.0
    )
    
    for game_idx in range(num_games):
        # Alternate who goes first
        model_player = game_idx % 2
        
        # Play one game
        board = VectorizedCliqueBoard(batch_size=1)
        
        move_count = 0
        max_moves = 15  # Maximum possible moves in 6-vertex clique game
        
        while move_count < max_moves:
            # Check if game is over
            if board.game_states[0] != 0:
                break
            
            # Get current player
            current_player = int(board.current_players[0])
            
            # Select MCTS to use
            if current_player == model_player:
                # Model with MCTS
                edge_indices, edge_features = board.get_features_for_nn()
                action_probs = model_mcts.search_batch_jit(
                    edge_indices, edge_features,
                    board.get_valid_moves_mask(),
                    jax.random.PRNGKey(game_idx * 100 + move_count)
                )
                # Sample action from probabilities
                action = np.random.choice(15, p=action_probs[0])
            else:
                # Pure MCTS
                edge_indices, edge_features = board.get_features_for_nn()
                action_probs = pure_mcts.search_batch_jit(
                    edge_indices, edge_features,
                    board.get_valid_moves_mask(),
                    jax.random.PRNGKey(game_idx * 100 + move_count + 1000)
                )
                action = np.random.choice(15, p=action_probs[0])
            
            # Make move
            board.make_moves(jnp.array([action]))
            move_count += 1
        
        # Check result
        game_state = int(board.game_states[0])
        if game_state == 1:  # Player 1 won
            if model_player == 0:
                wins += 1
        elif game_state == 2:  # Player 2 won
            if model_player == 1:
                wins += 1
        else:
            draws += 0.5
    
    win_rate = (wins + draws) / num_games
    return win_rate


def evaluate_head_to_head_jax(
    model1_state,
    model2_state,
    model,
    num_games: int = 20
) -> float:
    """
    Evaluate model1 against model2 using JAX.
    
    Returns:
        Win rate of model1
    """
    wins = 0
    draws = 0
    
    # Create MCTS instances
    mcts1 = OptimizedVectorizedMCTS(
        nn=model,
        c_puct=1.0,
        num_simulations=50,
        num_actions=15
    )
    
    mcts2 = OptimizedVectorizedMCTS(
        nn=model,
        c_puct=1.0,
        num_simulations=50,
        num_actions=15
    )
    
    for game_idx in range(num_games):
        # Alternate who goes first
        model1_player = game_idx % 2
        
        # Play one game
        board = VectorizedCliqueBoard(batch_size=1)
        
        move_count = 0
        max_moves = 15
        
        while move_count < max_moves:
            # Check if game is over
            if board.game_states[0] != 0:
                break
            
            # Get current player
            current_player = int(board.current_players[0])
            
            # Select MCTS to use
            if current_player == model1_player:
                # Model 1
                edge_indices, edge_features = board.get_features_for_nn()
                action_probs = mcts1.search_batch_jit(
                    edge_indices, edge_features,
                    board.get_valid_moves_mask(),
                    jax.random.PRNGKey(game_idx * 100 + move_count)
                )
                action = np.random.choice(15, p=action_probs[0])
            else:
                # Model 2
                edge_indices, edge_features = board.get_features_for_nn()
                action_probs = mcts2.search_batch_jit(
                    edge_indices, edge_features,
                    board.get_valid_moves_mask(),
                    jax.random.PRNGKey(game_idx * 100 + move_count + 1000)
                )
                action = np.random.choice(15, p=action_probs[0])
            
            # Make move
            board.make_moves(jnp.array([action]))
            move_count += 1
        
        # Check result
        game_state = int(board.game_states[0])
        if game_state == 1:  # Player 1 won
            if model1_player == 0:
                wins += 1
        elif game_state == 2:  # Player 2 won
            if model1_player == 1:
                wins += 1
        else:
            draws += 0.5
    
    win_rate = (wins + draws) / num_games
    return win_rate
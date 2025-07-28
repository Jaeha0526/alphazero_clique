#!/usr/bin/env python
"""
JAX-based evaluation functions for tree-based MCTS implementation
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict
from vectorized_board import VectorizedCliqueBoard
from tree_based_mcts import ParallelTreeBasedMCTS
from vectorized_nn import ImprovedBatchedNeuralNetwork


def evaluate_head_to_head_jax(
    model1: ImprovedBatchedNeuralNetwork,
    model2: ImprovedBatchedNeuralNetwork,
    num_games: int = 20,
    num_vertices: int = 6,
    k: int = 3,
    game_mode: str = 'symmetric',
    mcts_sims: int = 50,
    batch_size: int = 1,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Evaluate model1 against model2 using JAX with tree-based MCTS.
    
    Returns:
        Dictionary with evaluation results
    """
    
    wins = 0
    draws = 0
    losses = 0
    
    # Create MCTS instances for both models
    mcts1 = ParallelTreeBasedMCTS(
        batch_size=1,
        num_vertices=num_vertices,
        k=k,
        c_puct=1.0,
        num_simulations=mcts_sims
    )
    
    mcts2 = ParallelTreeBasedMCTS(
        batch_size=1,
        num_vertices=num_vertices,
        k=k,
        c_puct=1.0,
        num_simulations=mcts_sims
    )
    
    # Play games
    for game_idx in range(num_games):
        # Alternate who goes first
        model1_starts = (game_idx % 2 == 0)
        
        # Initialize board
        board = VectorizedCliqueBoard(
            batch_size=1,
            num_vertices=num_vertices,
            k=k,
            game_mode=game_mode
        )
        
        # Play game
        while board.game_states[0] == 0:
            current_player = int(board.current_players[0])
            
            # Determine which model/MCTS to use
            if (current_player == 1 and model1_starts) or (current_player == 2 and not model1_starts):
                # Model 1's turn
                edge_indices, edge_features = board.get_features_for_nn_undirected()
                
                # Get action from MCTS
                action_probs = mcts1.search(
                    board.boards,
                    board.current_players,
                    board.game_states,
                    model1,
                    jax.random.PRNGKey(game_idx * 1000 + board.move_counts[0])
                )
                
                # Select action (deterministic for evaluation)
                valid_mask = board.get_valid_moves_mask()
                masked_probs = action_probs[0] * valid_mask[0]
                action = jnp.argmax(masked_probs)
            else:
                # Model 2's turn
                edge_indices, edge_features = board.get_features_for_nn_undirected()
                
                # Get action from MCTS
                action_probs = mcts2.search(
                    board.boards,
                    board.current_players,
                    board.game_states,
                    model2,
                    jax.random.PRNGKey(game_idx * 1000 + board.move_counts[0] + 500)
                )
                
                # Select action
                valid_mask = board.get_valid_moves_mask()
                masked_probs = action_probs[0] * valid_mask[0]
                action = jnp.argmax(masked_probs)
            
            # Make move
            board.make_moves(jnp.array([action]))
        
        # Check result from model1's perspective
        game_state = int(board.game_states[0])
        winner = board.get_winner(0)
        
        if winner == 0:
            draws += 1
        elif (winner == 1 and model1_starts) or (winner == 2 and not model1_starts):
            wins += 1
        else:
            losses += 1
    
    # Calculate statistics
    win_rate = wins / num_games
    draw_rate = draws / num_games
    loss_rate = losses / num_games
    
    return {
        'player1_wins': wins,
        'draws': draws,
        'player2_wins': losses
    }
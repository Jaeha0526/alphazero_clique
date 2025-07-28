#!/usr/bin/env python
"""
Optimized JAX evaluation using vectorized MCTS.
Much faster than the original sequential evaluation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import Tuple

from vectorized_board import VectorizedCliqueBoard
from jit_mcts_simple import VectorizedJITMCTS
from vectorized_nn import ImprovedBatchedNeuralNetwork


def evaluate_head_to_head_optimized(
    model: ImprovedBatchedNeuralNetwork,
    params1: dict,  # First model parameters
    params2: dict,  # Second model parameters  
    num_games: int = 20,
    num_vertices: int = 6,
    k: int = 3,
    num_mcts_sims: int = 50,
    game_mode: str = "asymmetric",
    verbose: bool = False
) -> float:
    """
    Vectorized evaluation of two models.
    
    Args:
        model: Neural network model (same architecture for both)
        params1: Parameters for first model
        params2: Parameters for second model
        num_games: Number of games to play
        num_vertices: Graph size
        k: Clique size
        num_mcts_sims: MCTS simulations per move
        game_mode: "symmetric" or "asymmetric"
        verbose: Print progress
        
    Returns:
        Win rate of model1 against model2
    """
    if verbose:
        print(f"Evaluating models: {num_games} games, {num_mcts_sims} MCTS sims")
    
    # Use smaller batches for evaluation
    batch_size = min(8, num_games)
    mcts = VectorizedJITMCTS(batch_size=batch_size, num_actions=15, c_puct=1.0)
    
    model1_wins = 0
    total_decided_games = 0
    games_played = 0
    
    start_time = time.time()
    
    while games_played < num_games:
        current_batch_size = min(batch_size, num_games - games_played)
        
        # Initialize batch of games
        boards = VectorizedCliqueBoard(
            batch_size=current_batch_size,
            num_vertices=num_vertices,
            k=k,
            game_mode=game_mode
        )
        
        move_count = 0
        max_moves = num_vertices * (num_vertices - 1) // 2
        
        # Play games until all finish
        while jnp.any(boards.game_states == 0) and move_count < max_moves:
            # Determine which model to use for each game
            # Model1 plays as Player 1 in even-indexed games, Player 2 in odd-indexed games
            active_mask = boards.game_states == 0
            
            # Get moves for all active games
            active_indices = jnp.where(active_mask)[0]
            
            if len(active_indices) == 0:
                break
                
            # Create boards for active games
            active_boards = VectorizedCliqueBoard(batch_size=len(active_indices))
            for i, game_idx in enumerate(active_indices):
                active_boards.edge_states = active_boards.edge_states.at[i].set(
                    boards.edge_states[game_idx]
                )
                active_boards.current_players = active_boards.current_players.at[i].set(
                    boards.current_players[game_idx]
                )
                active_boards.game_states = active_boards.game_states.at[i].set(
                    boards.game_states[game_idx]
                )
                active_boards.move_counts = active_boards.move_counts.at[i].set(
                    boards.move_counts[game_idx]
                )
            
            # Determine which model each active game should use
            actions = []
            for i, orig_game_idx in enumerate(active_indices):
                # Model1 plays as Player 1 in even games, Player 2 in odd games
                current_player = active_boards.current_players[i]
                use_model1 = (orig_game_idx % 2 == 0 and current_player == 0) or \
                           (orig_game_idx % 2 == 1 and current_player == 1)
                
                # Create single-game board for this decision
                single_board = VectorizedCliqueBoard(batch_size=1)
                single_board.edge_states = single_board.edge_states.at[0].set(
                    active_boards.edge_states[i]
                )
                single_board.current_players = single_board.current_players.at[0].set(
                    active_boards.current_players[i]
                )
                single_board.game_states = single_board.game_states.at[0].set(0)
                
                # Use appropriate model parameters
                model_params = params1 if use_model1 else params2
                model.params = model_params
                
                # Get MCTS probabilities
                if 1 not in mcts.mcts_instances:
                    mcts.mcts_instances[1] = VectorizedJITMCTS(
                        batch_size=1, num_actions=15, c_puct=1.0
                    )
                single_mcts = mcts.mcts_instances[1]
                
                mcts_probs = single_mcts.search(
                    single_board, model, num_mcts_sims, temperature=0.0
                )
                
                # Sample action
                action = jnp.argmax(mcts_probs[0])
                actions.append(int(action))
            
            # Apply actions to original boards
            board_actions = jnp.zeros(current_batch_size, dtype=jnp.int32)
            for i, action in enumerate(actions):
                board_actions = board_actions.at[active_indices[i]].set(action)
            
            boards.make_moves(board_actions)
            move_count += 1
        
        # Count results
        for game_idx in range(current_batch_size):
            if boards.game_states[game_idx] != 0:  # Game finished
                winner = int(boards.winners[game_idx])
                if winner != 0:  # Not a draw
                    total_decided_games += 1
                    # Model1 wins if it was Player 1 and winner is 1, or Player 2 and winner is 2
                    model1_was_player1 = (games_played + game_idx) % 2 == 0
                    model1_won = (model1_was_player1 and winner == 1) or \
                               (not model1_was_player1 and winner == 2)
                    if model1_won:
                        model1_wins += 1
        
        games_played += current_batch_size
        
        if verbose and games_played % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Played {games_played}/{num_games} games ({elapsed:.1f}s)")
    
    # Calculate win rate
    win_rate = model1_wins / max(total_decided_games, 1)
    
    if verbose:
        total_time = time.time() - start_time
        print(f"Evaluation complete: {model1_wins}/{total_decided_games} wins ({win_rate:.1%}) in {total_time:.1f}s")
    
    return win_rate


def evaluate_head_to_head_jax(
    model: ImprovedBatchedNeuralNetwork,
    params1: dict,
    params2: dict,
    num_games: int = 20,
    num_vertices: int = 6,
    clique_size: int = 3,
    num_mcts_sims: int = 50,
    game_mode: str = "asymmetric"
) -> float:
    """
    Compatibility wrapper for the original evaluation interface.
    """
    return evaluate_head_to_head_optimized(
        model=model,
        params1=params1,
        params2=params2,
        num_games=num_games,
        num_vertices=num_vertices,
        k=clique_size,
        num_mcts_sims=num_mcts_sims,
        game_mode=game_mode,
        verbose=True
    )
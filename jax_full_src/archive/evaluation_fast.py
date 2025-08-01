#!/usr/bin/env python
"""
Ultra-fast evaluation for testing - plays fewer games with simpler MCTS
"""

import jax
import jax.numpy as jnp
import numpy as np
import time

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


def evaluate_head_to_head_fast(
    model: ImprovedBatchedNeuralNetwork,
    params1: dict,
    params2: dict,
    num_games: int = 10,  # Fewer games
    num_vertices: int = 6,
    clique_size: int = 3,
    game_mode: str = "asymmetric"
) -> float:
    """
    Fast evaluation using direct policy network (no MCTS for speed).
    """
    print(f"Fast evaluation: {num_games} games (policy-only)")
    
    model1_wins = 0
    total_games = 0
    
    start_time = time.time()
    
    # Play games in batch
    batch_size = min(8, num_games)
    games_played = 0
    
    while games_played < num_games:
        current_batch_size = min(batch_size, num_games - games_played)
        
        # Initialize batch of games
        boards = VectorizedCliqueBoard(
            batch_size=current_batch_size,
            num_vertices=num_vertices,
            k=clique_size,
            game_mode=game_mode
        )
        
        move_count = 0
        max_moves = 20  # Limit moves for speed
        
        # Play games until all finish or max moves
        while jnp.any(boards.game_states == 0) and move_count < max_moves:
            active_mask = boards.game_states == 0
            
            if not jnp.any(active_mask):
                break
            
            # Use simple policy selection (no MCTS)
            edge_indices, edge_features = boards.get_features_for_nn_undirected()
            valid_masks = boards.get_valid_moves_mask()
            
            actions = jnp.zeros(current_batch_size, dtype=jnp.int32)
            
            for game_idx in range(current_batch_size):
                if active_mask[game_idx]:
                    # Determine which model to use
                    current_player = boards.current_players[game_idx]
                    use_model1 = (games_played + game_idx) % 2 == 0 and current_player == 0 or \
                               (games_played + game_idx) % 2 == 1 and current_player == 1
                    
                    # Set model parameters
                    model.params = params1 if use_model1 else params2
                    
                    # Get policy (single game)
                    single_edge_indices = edge_indices[game_idx:game_idx+1]
                    single_edge_features = edge_features[game_idx:game_idx+1] 
                    single_valid_mask = valid_masks[game_idx:game_idx+1]
                    
                    policies, _ = model.evaluate_batch(
                        single_edge_indices, single_edge_features, single_valid_mask
                    )
                    
                    # Apply valid moves mask and select action
                    masked_policy = jnp.where(single_valid_mask[0], policies[0], -jnp.inf)
                    action = jnp.argmax(masked_policy)
                    actions = actions.at[game_idx].set(action)
            
            boards.make_moves(actions)
            move_count += 1
        
        # Count results
        for game_idx in range(current_batch_size):
            if boards.game_states[game_idx] != 0:  # Game finished
                winner = int(boards.winners[game_idx])
                if winner != 0:  # Not a draw
                    total_games += 1
                    # Check if model1 won
                    model1_was_player1 = (games_played + game_idx) % 2 == 0
                    model1_won = (model1_was_player1 and winner == 1) or \
                               (not model1_was_player1 and winner == 2)
                    if model1_won:
                        model1_wins += 1
        
        games_played += current_batch_size
    
    # Calculate win rate
    win_rate = model1_wins / max(total_games, 1) if total_games > 0 else 0.5
    
    eval_time = time.time() - start_time
    print(f"Fast evaluation: {model1_wins}/{total_games} wins ({win_rate:.1%}) in {eval_time:.1f}s")
    
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
    Fast evaluation wrapper for compatibility.
    """
    return evaluate_head_to_head_fast(
        model=model,
        params1=params1,
        params2=params2,
        num_games=min(10, num_games),  # Cap at 10 games for speed
        num_vertices=num_vertices,
        clique_size=clique_size,
        game_mode=game_mode
    )
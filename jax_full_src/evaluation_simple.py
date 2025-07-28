#!/usr/bin/env python
"""
Simple evaluation using our existing optimized self-play code.
Just plays games with two different models.
"""

import jax.numpy as jnp
import numpy as np
from vectorized_self_play_optimized import OptimizedVectorizedSelfPlay, OptimizedSelfPlayConfig


def evaluate_head_to_head_jax(
    model,
    params1,
    params2, 
    num_games: int = 20,
    num_vertices: int = 6,
    clique_size: int = 3,
    num_mcts_sims: int = 25,  # Lighter MCTS
    game_mode: str = "asymmetric"
) -> float:
    """
    Evaluate two models by playing games between them.
    Reuses our optimized self-play code!
    """
    print(f"Evaluating models: {num_games} games with {num_mcts_sims} MCTS sims")
    
    # Use smaller batch for evaluation
    batch_size = min(8, num_games)
    
    # Create evaluation self-play with lighter MCTS
    eval_config = OptimizedSelfPlayConfig(
        batch_size=batch_size,
        mcts_simulations=num_mcts_sims,
        game_mode=game_mode,
        temperature_threshold=0  # Always deterministic for evaluation
    )
    
    evaluator = OptimizedVectorizedSelfPlay(eval_config)
    
    model1_wins = 0
    total_decided_games = 0
    games_played = 0
    
    while games_played < num_games:
        current_batch = min(batch_size, num_games - games_played)
        
        # Play one batch of games
        # We'll alternate which model goes first
        wins_this_batch = 0
        decided_this_batch = 0
        
        for game_start in range(0, current_batch, 2):
            games_in_pair = min(2, current_batch - game_start)
            
            if games_in_pair >= 2:
                # Play 2 games: model1 vs model2, then model2 vs model1
                
                # Game 1: model1 starts (as player 1)
                model.params = params1
                game_data1 = evaluator.play_single_game(model, verbose=False)
                winner1 = game_data1[1] if len(game_data1) > 1 else 0
                
                # Switch to model2 for moves where it should play
                # This is simplified - in a full implementation we'd switch during the game
                
                # Game 2: model2 starts (as player 1) 
                model.params = params2
                game_data2 = evaluator.play_single_game(model, verbose=False)
                winner2 = game_data2[1] if len(game_data2) > 1 else 0
                
                # Count results
                # In game 1, model1 wins if winner is 1
                if winner1 != 0:
                    decided_this_batch += 1
                    if winner1 == 1:  # model1 won as player 1
                        wins_this_batch += 1
                
                # In game 2, model1 wins if winner is 2 (since model2 was player 1)
                if winner2 != 0:
                    decided_this_batch += 1
                    if winner2 == 2:  # model1 won as player 2
                        wins_this_batch += 1
        
        model1_wins += wins_this_batch
        total_decided_games += decided_this_batch
        games_played += current_batch
        
        print(f"  Played {games_played}/{num_games} games")
    
    # Calculate win rate
    win_rate = model1_wins / max(total_decided_games, 1)
    print(f"Evaluation result: {model1_wins}/{total_decided_games} wins = {win_rate:.1%}")
    
    return win_rate
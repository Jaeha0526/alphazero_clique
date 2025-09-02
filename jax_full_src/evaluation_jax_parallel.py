"""
Parallel evaluation for symmetric games - process all games in a single batch
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import Optional, Dict
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from mctx_final_optimized import MCTXFinalOptimized
from mctx_true_jax import MCTXTrueJAX
from ramsey_counterexample_saver import RamseyCounterexampleSaver


def evaluate_vs_initial_and_best_parallel(
    current_model: ImprovedBatchedNeuralNetwork,
    initial_model: ImprovedBatchedNeuralNetwork,
    best_model: Optional[ImprovedBatchedNeuralNetwork] = None,
    config: dict = None
) -> Dict[str, float]:
    """
    Parallel evaluation for symmetric games.
    Process all games in a single batch for massive speedup.
    """
    if config is None:
        config = {
            'num_games': 20,
            'num_vertices': 6,
            'k': 3,
            'mcts_sims': 30,
            'game_mode': 'symmetric',
            'c_puct': 3.0
        }
    
    num_games = config['num_games']
    num_vertices = config['num_vertices']
    k = config['k']
    mcts_sims = config.get('mcts_sims', 30)
    c_puct = config.get('c_puct', 3.0)
    temperature = 0.0  # Deterministic for evaluation
    
    print(f"\nParallel Evaluation: {num_games} games in single batch")
    start_time = time.time()
    
    # Evaluate against initial model
    initial_results = evaluate_models_parallel(
        model1=current_model,
        model2=initial_model,
        num_games=num_games,
        num_vertices=num_vertices,
        k=k,
        mcts_sims=mcts_sims,
        c_puct=c_puct,
        temperature=temperature,
        game_mode=config.get('game_mode', 'symmetric')
    )
    
    results = {
        'win_rate_vs_initial': initial_results['model1_win_rate'],
        'draw_rate_vs_initial': initial_results['draw_rate'],
        'eval_time_vs_initial': initial_results['eval_time'],
        'vs_initial_details': initial_results
    }
    
    # Evaluate against best model if provided
    if best_model is not None:
        best_results = evaluate_models_parallel(
            model1=current_model,
            model2=best_model,
            num_games=num_games,
            num_vertices=num_vertices,
            k=k,
            mcts_sims=mcts_sims,
            c_puct=c_puct,
            temperature=temperature,
            game_mode=config.get('game_mode', 'symmetric')
        )
        
        results.update({
            'win_rate_vs_best': best_results['model1_win_rate'],
            'draw_rate_vs_best': best_results['draw_rate'],
            'eval_time_vs_best': best_results['eval_time'],
            'vs_best_details': best_results
        })
    else:
        results.update({
            'win_rate_vs_best': -1,
            'draw_rate_vs_best': -1,
            'eval_time_vs_best': 0,
            'vs_best_details': None
        })
    
    total_time = time.time() - start_time
    print(f"Total evaluation time: {total_time:.1f}s ({num_games/(total_time+0.001):.1f} games/sec)")
    
    return results


def evaluate_models_parallel(
    model1: ImprovedBatchedNeuralNetwork,
    model2: ImprovedBatchedNeuralNetwork,
    num_games: int = 20,
    num_vertices: int = 6,
    k: int = 3,
    mcts_sims: int = 30,
    c_puct: float = 3.0,
    temperature: float = 0.0,
    game_mode: str = 'symmetric'
) -> Dict[str, float]:
    """
    Evaluate two models against each other in parallel.
    All games run simultaneously in a single batch.
    """
    start_time = time.time()
    
    # Initialize Ramsey saver if in avoid_clique mode
    ramsey_saver = None
    if game_mode == "avoid_clique":
        ramsey_saver = RamseyCounterexampleSaver()
    
    # Create batch of games
    boards = VectorizedCliqueBoard(
        batch_size=num_games,
        num_vertices=num_vertices,
        k=k,
        game_mode=game_mode
    )
    
    # Alternate who starts - model1 starts in even games, model2 in odd games
    model1_starts = jnp.array([i % 2 == 0 for i in range(num_games)])
    
    # Create MCTS instances for both models
    num_actions = num_vertices * (num_vertices - 1) // 2
    
    # Use True MCTX for maximum speed (same as self-play)
    use_true_mctx = True  # Always use the fastest version
    
    if use_true_mctx:
        mcts1 = MCTXTrueJAX(
            batch_size=num_games,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,
            c_puct=c_puct,
            num_vertices=num_vertices
        )
        
        mcts2 = MCTXTrueJAX(
            batch_size=num_games,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,
            c_puct=c_puct,
            num_vertices=num_vertices
        )
    else:
        mcts1 = MCTXFinalOptimized(
            batch_size=num_games,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,
            num_vertices=num_vertices,
            c_puct=c_puct
        )
        
        mcts2 = MCTXFinalOptimized(
            batch_size=num_games,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,
            num_vertices=num_vertices,
            c_puct=c_puct
        )
    
    move_count = 0
    max_moves = num_vertices * (num_vertices - 1) // 2
    
    # Play all games simultaneously
    while jnp.any(boards.game_states == 0) and move_count < max_moves:
        active_games = jnp.sum(boards.game_states == 0)
        
        # Determine which model should play for each game
        # If model1_starts[i] is True:  model1 is player 0, model2 is player 1
        # If model1_starts[i] is False: model2 is player 0, model1 is player 1
        use_model1 = (boards.current_players == 0) == model1_starts
        
        # Get action probabilities from both models
        probs1 = mcts1.search(boards, model1, mcts_sims, temperature)
        probs2 = mcts2.search(boards, model2, mcts_sims, temperature)
        
        # Select actions based on which model should play
        actions = []
        for i in range(num_games):
            if boards.game_states[i] != 0:  # Game already finished
                actions.append(0)  # Dummy action
            else:
                if use_model1[i]:
                    probs = probs1[i]
                else:
                    probs = probs2[i]
                
                # Get valid moves and select action based on MCTS
                valid_mask = boards.get_valid_moves_mask()[i]
                masked_probs = probs * valid_mask
                
                # For evaluation, just pick the most visited action (like PyTorch)
                if jnp.sum(masked_probs) > 0:
                    # Simply take the action with highest probability (most visits)
                    action = jnp.argmax(masked_probs)
                else:
                    valid_actions = jnp.where(valid_mask)[0]
                    action = valid_actions[0] if len(valid_actions) > 0 else 0
                
                actions.append(int(action))
        
        # Make moves
        boards.make_moves(jnp.array(actions))
        move_count += 1
        
        if move_count % 5 == 0:  # Progress update every 5 moves
            print(f"  Move {move_count}: {active_games} active games")
    
    # Count results
    model1_wins = 0
    model2_wins = 0
    draws = 0
    
    for i in range(num_games):
        game_state = int(boards.game_states[i])
        
        if game_state == 3:  # Draw
            draws += 1
        elif game_state == 1:  # Player 1 wins
            if model1_starts[i]:  # Model1 was player 1
                model1_wins += 1
            else:  # Model2 was player 1
                model2_wins += 1
        elif game_state == 2:  # Player 2 wins
            if model1_starts[i]:  # Model1 was player 1, so model2 wins
                model2_wins += 1
            else:  # Model2 was player 1, so model1 wins
                model1_wins += 1
    
    # Save Ramsey counterexamples if in avoid_clique mode
    if ramsey_saver is not None:
        saved_files = ramsey_saver.save_batch_counterexamples(
            boards=boards,
            source="parallel_evaluation"
        )
        if saved_files:
            print(f"  💎 Saved {len(saved_files)} Ramsey counterexamples!")
    
    eval_time = time.time() - start_time
    
    # Print summary
    print(f"\nParallel Evaluation Results ({num_games} games in {eval_time:.1f}s):")
    print(f"  Model1 wins: {model1_wins} ({model1_wins/num_games:.1%})")
    print(f"  Model2 wins: {model2_wins} ({model2_wins/num_games:.1%})")
    print(f"  Draws: {draws} ({draws/num_games:.1%})")
    print(f"  Games/sec: {num_games/eval_time:.1f}")
    
    return {
        'model1_wins': model1_wins,
        'model2_wins': model2_wins,
        'draws': draws,
        'model1_win_rate': model1_wins / num_games,
        'model2_win_rate': model2_wins / num_games,
        'draw_rate': draws / num_games,
        'eval_time': eval_time,
        'games_breakdown': {
            'wins': model1_wins,
            'losses': model2_wins,
            'draws': draws
        }
    }
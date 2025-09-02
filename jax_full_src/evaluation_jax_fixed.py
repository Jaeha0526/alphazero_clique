#!/usr/bin/env python
"""
Fixed JAX-based evaluation functions
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional
from vectorized_board import VectorizedCliqueBoard
from mctx_final_optimized import MCTXFinalOptimized
from mctx_true_jax import MCTXTrueJAX
from vectorized_nn import ImprovedBatchedNeuralNetwork
import time
from ramsey_counterexample_saver import RamseyCounterexampleSaver


def evaluate_models_jax(
    current_model: ImprovedBatchedNeuralNetwork,
    baseline_model: ImprovedBatchedNeuralNetwork,
    num_games: int = 20,
    num_vertices: int = 6,
    k: int = 3,
    game_mode: str = 'symmetric',
    mcts_sims: int = 50,
    c_puct: float = 3.0,
    temperature: float = 0.0,  # 0 for deterministic play during evaluation
    verbose: bool = False,
    decided_games_only: bool = False
) -> Dict[str, float]:
    """
    Evaluate current model against baseline model by playing games.
    
    Args:
        current_model: Current neural network model
        baseline_model: Baseline model to compare against (e.g., initial or best model)
        num_games: Number of games to play
        num_vertices: Number of vertices in the graph
        k: Size of clique needed to win
        game_mode: "symmetric" or "asymmetric" game mode
        mcts_sims: Number of MCTS simulations per move
        c_puct: Exploration constant for MCTS
        temperature: Temperature for action selection (0 = deterministic)
        verbose: Whether to print game progress
        decided_games_only: If True, calculate win rate from decided games only (exclude draws)
        
    Returns:
        Dictionary with evaluation results including win rate
    """
    
    # Track results
    current_wins = 0
    baseline_wins = 0
    draws = 0
    
    # Initialize Ramsey saver if in avoid_clique mode
    ramsey_saver = None
    if game_mode == "avoid_clique":
        ramsey_saver = RamseyCounterexampleSaver()
    
    # Create MCTS instances
    num_actions = num_vertices * (num_vertices - 1) // 2
    # Use True MCTX if specified
    use_true_mctx = game_mode.get('use_true_mctx', False) if isinstance(game_mode, dict) else False
    
    if use_true_mctx:
        mcts_current = MCTXTrueJAX(
            batch_size=1,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,  # Only need sims + 1 nodes
            c_puct=c_puct
        )
        
        mcts_baseline = MCTXTrueJAX(
            batch_size=1,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,  # Only need sims + 1 nodes
            c_puct=c_puct
        )
    else:
        mcts_current = MCTXFinalOptimized(
            batch_size=1,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,  # Only need sims + 1 nodes
            num_vertices=num_vertices,
            c_puct=c_puct
        )
        
        mcts_baseline = MCTXFinalOptimized(
            batch_size=1,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,  # Only need sims + 1 nodes
            num_vertices=num_vertices,
            c_puct=c_puct
        )
    
    # Play games
    for game_idx in range(num_games):
        if verbose:
            print(f"Evaluation Game {game_idx+1}/{num_games}")
        
        # Alternate who goes first
        current_starts = (game_idx % 2 == 0)
        
        # Initialize board
        board = VectorizedCliqueBoard(
            batch_size=1,
            num_vertices=num_vertices,
            k=k,
            game_mode=game_mode
        )
        
        move_count = 0
        max_moves = num_vertices * (num_vertices - 1) // 2
        
        # Play game
        while board.game_states[0] == 0 and move_count < max_moves:
            current_player = int(board.current_players[0])
            
            # Determine which model/MCTS to use
            # If current_starts=True: current model is player 0, baseline is player 1
            # If current_starts=False: baseline is player 0, current model is player 1
            use_current = (current_player == 0 and current_starts) or (current_player == 1 and not current_starts)
            
            if use_current:
                # Current model's turn
                action_probs = mcts_current.search(
                    board, 
                    current_model,
                    mcts_sims,
                    temperature
                )
            else:
                # Baseline model's turn
                action_probs = mcts_baseline.search(
                    board,
                    baseline_model,
                    mcts_sims,
                    temperature
                )
            
            # Select action based on MCTS visit counts
            # For evaluation, just pick the most visited action (like PyTorch)
            valid_mask = board.get_valid_moves_mask()
            masked_probs = action_probs[0] * valid_mask[0]
            
            # Ensure we have valid probabilities
            if jnp.sum(masked_probs) > 0:
                # Simply take the action with highest probability (most visits)
                action = jnp.argmax(masked_probs)
            else:
                # No valid moves (shouldn't happen)
                break
            
            # Make move
            board.make_moves(jnp.array([action]))
            move_count += 1
        
        # Check result
        # game_states: 0=ongoing, 1=player1 wins, 2=player2 wins, 3=draw
        # winners: -1=none, 0=player1, 1=player2
        game_state = int(board.game_states[0])
        winner = int(board.winners[0])
        
        if game_state == 3 or game_state == 0:  # Draw or unfinished
            draws += 1
            if verbose:
                print(f"  Game {game_idx+1}: Draw")
            
            # Save Ramsey counterexample if in avoid_clique mode
            if ramsey_saver is not None and game_state == 3:
                ramsey_saver.save_counterexample(
                    edge_states=board.edge_states[0],
                    num_vertices=num_vertices,
                    k=k,
                    source="evaluation",
                    game_idx=game_idx
                )
        elif game_state == 1:  # Player 1 wins
            if current_starts:
                current_wins += 1
                if verbose:
                    print(f"  Game {game_idx+1}: Current model wins")
            else:
                baseline_wins += 1
                if verbose:
                    print(f"  Game {game_idx+1}: Baseline model wins")
        elif game_state == 2:  # Player 2 wins
            if not current_starts:
                current_wins += 1
                if verbose:
                    print(f"  Game {game_idx+1}: Current model wins")
            else:
                baseline_wins += 1
                if verbose:
                    print(f"  Game {game_idx+1}: Baseline model wins")
    
    # Calculate statistics
    if decided_games_only and (current_wins + baseline_wins) > 0:
        # Calculate win rate excluding draws
        win_rate = current_wins / (current_wins + baseline_wins)
    else:
        # Calculate win rate including draws
        win_rate = current_wins / num_games
    
    return {
        'current_wins': current_wins,
        'baseline_wins': baseline_wins,
        'draws': draws,
        'win_rate': win_rate,
        'total_games': num_games
    }


def evaluate_vs_initial_and_best(
    current_model: ImprovedBatchedNeuralNetwork,
    initial_model: ImprovedBatchedNeuralNetwork,
    best_model: Optional[ImprovedBatchedNeuralNetwork] = None,
    config: dict = None
) -> Dict[str, float]:
    """
    Evaluate current model against both initial and best models.
    
    Returns:
        Dictionary with win rates vs initial and best
    """
    if config is None:
        config = {
            'num_games': 21,
            'num_vertices': 6,
            'k': 3,
            'game_mode': 'symmetric',
            'mcts_sims': 30,
            'c_puct': 3.0
        }
    
    print("\nEvaluating against initial model...")
    start_time = time.time()
    
    # Evaluate vs initial
    initial_results = evaluate_models_jax(
        current_model=current_model,
        baseline_model=initial_model,
        num_games=config['num_games'],
        num_vertices=config['num_vertices'],
        k=config['k'],
        game_mode=config['game_mode'],
        mcts_sims=config['mcts_sims'],
        c_puct=config.get('c_puct', 3.0),
        temperature=0.0,  # Deterministic for evaluation
        verbose=False
    )
    
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.1f}s")
    print(f"Results: {initial_results['current_wins']} wins, "
          f"{initial_results['baseline_wins']} losses, {initial_results['draws']} draws")
    print(f"Win rate vs initial: {initial_results['win_rate']:.1%}")
    
    results = {
        'win_rate_vs_initial': initial_results['win_rate'],
        'eval_time_vs_initial': eval_time
    }
    
    # Evaluate vs best if provided
    if best_model is not None:
        print("\nEvaluating against best model...")
        start_time = time.time()
        
        best_results = evaluate_models_jax(
            current_model=current_model,
            baseline_model=best_model,
            num_games=config['num_games'],
            num_vertices=config['num_vertices'],
            k=config['k'],
            game_mode=config['game_mode'],
            mcts_sims=config['mcts_sims'],
            c_puct=config.get('c_puct', 3.0),
            temperature=0.0,
            verbose=False
        )
        
        eval_time = time.time() - start_time
        print(f"Evaluation completed in {eval_time:.1f}s")
        print(f"Results: {best_results['current_wins']} wins, "
              f"{best_results['baseline_wins']} losses, {best_results['draws']} draws")
        print(f"Win rate vs best: {best_results['win_rate']:.1%}")
        
        results['win_rate_vs_best'] = best_results['win_rate']
        results['eval_time_vs_best'] = eval_time
    
    return results
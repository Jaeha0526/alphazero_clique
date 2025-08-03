#!/usr/bin/env python
"""
Enhanced evaluation for asymmetric games that tracks attacker/defender performance separately.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Tuple
from vectorized_board import VectorizedCliqueBoard
from mctx_final_optimized import MCTXFinalOptimized
from mctx_true_jax import MCTXTrueJAX
from vectorized_nn import ImprovedBatchedNeuralNetwork
import time


def evaluate_models_asymmetric_detailed(
    current_model: ImprovedBatchedNeuralNetwork,
    baseline_model: ImprovedBatchedNeuralNetwork,
    num_games: int = 40,  # More games for better statistics
    num_vertices: int = 6,
    k: int = 3,
    mcts_sims: int = 50,
    c_puct: float = 3.0,
    temperature: float = 0.0,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate models in asymmetric game with separate tracking for attacker/defender roles.
    
    Args:
        current_model: Current neural network model
        baseline_model: Baseline model to compare against
        num_games: Number of games to play (should be even for balance)
        num_vertices: Number of vertices in the graph
        k: Size of clique needed to win
        mcts_sims: Number of MCTS simulations per move
        c_puct: Exploration constant for MCTS
        temperature: Temperature for action selection (0 = deterministic)
        verbose: Whether to print detailed game progress
        
    Returns:
        Dictionary with detailed evaluation results
    """
    
    # Ensure even number of games for perfect balance
    if num_games % 2 != 0:
        num_games += 1
        if verbose:
            print(f"Adjusted num_games to {num_games} for balanced evaluation")
    
    # Track results by role
    current_as_attacker = {'games': 0, 'wins': 0}
    current_as_defender = {'games': 0, 'wins': 0}
    baseline_as_attacker = {'games': 0, 'wins': 0}
    baseline_as_defender = {'games': 0, 'wins': 0}
    draws = 0
    
    # Create MCTS instances
    num_actions = num_vertices * (num_vertices - 1) // 2
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
    start_time = time.time()
    
    for game_idx in range(num_games):
        if verbose and game_idx % 10 == 0:
            print(f"Playing games {game_idx+1}-{min(game_idx+10, num_games)}/{num_games}")
        
        # Alternate who goes first
        current_starts = (game_idx % 2 == 0)
        
        # Track roles for this game
        if current_starts:
            current_role = "attacker"
            baseline_role = "defender"
            current_as_attacker['games'] += 1
            baseline_as_defender['games'] += 1
        else:
            current_role = "defender"
            baseline_role = "attacker"
            current_as_defender['games'] += 1
            baseline_as_attacker['games'] += 1
        
        if verbose and game_idx < 4:  # Show first few games
            print(f"  Game {game_idx+1}: Current={current_role}, Baseline={baseline_role}")
        
        # Initialize board
        board = VectorizedCliqueBoard(
            batch_size=1,
            num_vertices=num_vertices,
            k=k,
            game_mode="asymmetric"
        )
        
        move_count = 0
        max_moves = num_vertices * (num_vertices - 1) // 2
        
        # Play game
        while board.game_states[0] == 0 and move_count < max_moves:
            current_player = int(board.current_players[0])
            
            # Determine which model to use
            use_current = (current_player == 0 and current_starts) or \
                         (current_player == 1 and not current_starts)
            
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
            
            # Select action
            valid_mask = board.get_valid_moves_mask()
            masked_probs = action_probs[0] * valid_mask[0]
            
            # Ensure we have valid probabilities
            if jnp.sum(masked_probs) > 0:
                if temperature == 0:
                    # Deterministic: choose highest probability
                    action = jnp.argmax(masked_probs)
                else:
                    # Sample from distribution
                    action = jax.random.categorical(
                        jax.random.PRNGKey(game_idx * 1000 + move_count),
                        jnp.log(masked_probs + 1e-8)
                    )
            else:
                # No valid moves (shouldn't happen)
                break
            
            # Make move
            board.make_moves(jnp.array([action]))
            move_count += 1
        
        # Check result and update role-specific stats
        game_state = int(board.game_states[0])
        
        if game_state == 3 or game_state == 0:  # Draw or unfinished
            draws += 1
            if verbose and game_idx < 4:
                print(f"    Result: Draw")
        elif game_state == 1:  # Player 0 (Attacker) wins
            if current_starts:
                # Current model was attacker and won
                current_as_attacker['wins'] += 1
                if verbose and game_idx < 4:
                    print(f"    Result: Current wins as attacker")
            else:
                # Baseline model was attacker and won
                baseline_as_attacker['wins'] += 1
                if verbose and game_idx < 4:
                    print(f"    Result: Baseline wins as attacker")
        elif game_state == 2:  # Player 1 (Defender) wins
            if current_starts:
                # Baseline model was defender and won
                baseline_as_defender['wins'] += 1
                if verbose and game_idx < 4:
                    print(f"    Result: Baseline wins as defender")
            else:
                # Current model was defender and won
                current_as_defender['wins'] += 1
                if verbose and game_idx < 4:
                    print(f"    Result: Current wins as defender")
    
    eval_time = time.time() - start_time
    
    # Calculate win rates by role
    current_attacker_rate = current_as_attacker['wins'] / max(1, current_as_attacker['games'])
    current_defender_rate = current_as_defender['wins'] / max(1, current_as_defender['games'])
    baseline_attacker_rate = baseline_as_attacker['wins'] / max(1, baseline_as_attacker['games'])
    baseline_defender_rate = baseline_as_defender['wins'] / max(1, baseline_as_defender['games'])
    
    # Overall statistics
    current_total_wins = current_as_attacker['wins'] + current_as_defender['wins']
    baseline_total_wins = baseline_as_attacker['wins'] + baseline_as_defender['wins']
    overall_win_rate = current_total_wins / num_games
    
    # Create detailed results
    results = {
        # Overall stats
        'total_games': num_games,
        'current_wins': current_total_wins,
        'baseline_wins': baseline_total_wins,
        'draws': draws,
        'win_rate': overall_win_rate,
        
        # Current model stats by role
        'current_attacker_games': current_as_attacker['games'],
        'current_attacker_wins': current_as_attacker['wins'],
        'current_attacker_rate': current_attacker_rate,
        'current_defender_games': current_as_defender['games'],
        'current_defender_wins': current_as_defender['wins'],
        'current_defender_rate': current_defender_rate,
        
        # Baseline model stats by role
        'baseline_attacker_games': baseline_as_attacker['games'],
        'baseline_attacker_wins': baseline_as_attacker['wins'],
        'baseline_attacker_rate': baseline_attacker_rate,
        'baseline_defender_games': baseline_as_defender['games'],
        'baseline_defender_wins': baseline_as_defender['wins'],
        'baseline_defender_rate': baseline_defender_rate,
        
        # Time
        'eval_time': eval_time
    }
    
    if verbose:
        print(f"\nEvaluation Summary ({num_games} games in {eval_time:.1f}s):")
        print(f"Current Model:")
        print(f"  As Attacker: {current_as_attacker['wins']}/{current_as_attacker['games']} ({current_attacker_rate:.1%})")
        print(f"  As Defender: {current_as_defender['wins']}/{current_as_defender['games']} ({current_defender_rate:.1%})")
        print(f"  Overall: {current_total_wins}/{num_games} ({overall_win_rate:.1%})")
        print(f"Baseline Model:")
        print(f"  As Attacker: {baseline_as_attacker['wins']}/{baseline_as_attacker['games']} ({baseline_attacker_rate:.1%})")
        print(f"  As Defender: {baseline_as_defender['wins']}/{baseline_as_defender['games']} ({baseline_defender_rate:.1%})")
        print(f"Draws: {draws}")
    
    return results


def evaluate_vs_initial_and_best_asymmetric(
    current_model: ImprovedBatchedNeuralNetwork,
    initial_model: ImprovedBatchedNeuralNetwork,
    best_model: Optional[ImprovedBatchedNeuralNetwork] = None,
    config: dict = None
) -> Dict[str, float]:
    """
    Evaluate current model against both initial and best models with detailed role tracking.
    
    Returns:
        Dictionary with detailed win rates vs initial and best
    """
    if config is None:
        config = {
            'num_games': 40,  # More games for better statistics
            'num_vertices': 6,
            'k': 3,
            'mcts_sims': 30,
            'c_puct': 3.0
        }
    
    print("\nEvaluating against initial model (detailed asymmetric)...")
    
    # Evaluate vs initial
    initial_results = evaluate_models_asymmetric_detailed(
        current_model=current_model,
        baseline_model=initial_model,
        num_games=config['num_games'],
        num_vertices=config['num_vertices'],
        k=config['k'],
        mcts_sims=config['mcts_sims'],
        c_puct=config.get('c_puct', 3.0),
        temperature=0.0,
        verbose=True
    )
    
    results = {
        'win_rate_vs_initial': initial_results['win_rate'],
        'vs_initial_attacker_rate': initial_results['current_attacker_rate'],
        'vs_initial_defender_rate': initial_results['current_defender_rate'],
        'vs_initial_details': initial_results
    }
    
    # Evaluate vs best if provided
    if best_model is not None:
        print("\nEvaluating against best model (detailed asymmetric)...")
        
        best_results = evaluate_models_asymmetric_detailed(
            current_model=current_model,
            baseline_model=best_model,
            num_games=config['num_games'],
            num_vertices=config['num_vertices'],
            k=config['k'],
            mcts_sims=config['mcts_sims'],
            c_puct=config.get('c_puct', 3.0),
            temperature=0.0,
            verbose=True
        )
        
        results.update({
            'win_rate_vs_best': best_results['win_rate'],
            'vs_best_attacker_rate': best_results['current_attacker_rate'],
            'vs_best_defender_rate': best_results['current_defender_rate'],
            'vs_best_details': best_results
        })
    
    return results
"""
Parallel evaluation for asymmetric games - process all games in a single batch
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


def evaluate_vs_initial_and_best_asymmetric_parallel(
    current_model: ImprovedBatchedNeuralNetwork,
    initial_model: ImprovedBatchedNeuralNetwork,
    best_model: Optional[ImprovedBatchedNeuralNetwork] = None,
    config: dict = None
) -> Dict[str, float]:
    """
    Parallel evaluation for asymmetric games with role tracking.
    Process all games in a single batch for massive speedup.
    """
    if config is None:
        config = {
            'num_games': 40,
            'num_vertices': 6,
            'k': 3,
            'game_mode': 'asymmetric',
            'mcts_sims': 30,
            'c_puct': 3.0
        }
    
    num_games = config['num_games']
    
    print(f"\nParallel Asymmetric Evaluation: {num_games} games in single batch")
    start_time = time.time()
    
    # Evaluate against initial model
    initial_results = evaluate_models_asymmetric_parallel(
        current_model=current_model,
        baseline_model=initial_model,
        num_games=num_games,
        num_vertices=config['num_vertices'],
        k=config['k'],
        mcts_sims=config.get('mcts_sims', 30),
        c_puct=config.get('c_puct', 3.0),
        temperature=0.0
    )
    
    results = {
        'win_rate_vs_initial': initial_results['current_win_rate'],
        'vs_initial_attacker_rate': initial_results['current_attacker_rate'],
        'vs_initial_defender_rate': initial_results['current_defender_rate'],
        'vs_initial_details': initial_results
    }
    
    # Evaluate against best model if provided
    if best_model is not None:
        best_results = evaluate_models_asymmetric_parallel(
            current_model=current_model,
            baseline_model=best_model,
            num_games=num_games,
            num_vertices=config['num_vertices'],
            k=config['k'],
            mcts_sims=config.get('mcts_sims', 30),
            c_puct=config.get('c_puct', 3.0),
            temperature=0.0
        )
        
        results.update({
            'win_rate_vs_best': best_results['current_win_rate'],
            'vs_best_attacker_rate': best_results['current_attacker_rate'],
            'vs_best_defender_rate': best_results['current_defender_rate'],
            'vs_best_details': best_results
        })
    else:
        results.update({
            'win_rate_vs_best': -1,
            'vs_best_attacker_rate': -1,
            'vs_best_defender_rate': -1,
            'vs_best_details': None
        })
    
    total_time = time.time() - start_time
    print(f"Total evaluation time: {total_time:.1f}s ({num_games/(total_time+0.001):.1f} games/sec)")
    
    return results


def evaluate_models_asymmetric_parallel(
    current_model: ImprovedBatchedNeuralNetwork,
    baseline_model: ImprovedBatchedNeuralNetwork,
    num_games: int = 40,
    num_vertices: int = 6,
    k: int = 3,
    mcts_sims: int = 30,
    c_puct: float = 3.0,
    temperature: float = 0.0
) -> Dict[str, float]:
    """
    Evaluate two models in asymmetric games in parallel.
    Ensures balanced role assignment (equal games as attacker/defender).
    """
    start_time = time.time()
    
    # Ensure even number of games for balanced roles
    if num_games % 4 != 0:
        print(f"Warning: Adjusting num_games from {num_games} to {(num_games//4)*4} for balanced evaluation")
        num_games = (num_games // 4) * 4
    
    # Create batch of games
    boards = VectorizedCliqueBoard(
        batch_size=num_games,
        num_vertices=num_vertices,
        k=k,
        game_mode='asymmetric'
    )
    
    # Role assignment for balanced evaluation:
    # Quarter 1: Current=attacker, Baseline=defender, Current starts
    # Quarter 2: Current=defender, Baseline=attacker, Baseline starts  
    # Quarter 3: Current=attacker, Baseline=defender, Baseline starts
    # Quarter 4: Current=defender, Baseline=attacker, Current starts
    quarter_size = num_games // 4
    
    current_is_attacker = jnp.array(
        [True] * quarter_size +   # Q1: current attacks first
        [False] * quarter_size +  # Q2: baseline attacks first
        [True] * quarter_size +   # Q3: baseline starts but current attacks
        [False] * quarter_size    # Q4: current starts but baseline attacks
    )
    
    current_starts = jnp.array(
        [True] * quarter_size +   # Q1: current starts
        [False] * quarter_size +  # Q2: baseline starts
        [False] * quarter_size +  # Q3: baseline starts
        [True] * quarter_size     # Q4: current starts
    )
    
    # Create MCTS instances for both models
    num_actions = num_vertices * (num_vertices - 1) // 2
    
    # Use True MCTX for maximum speed (same as self-play)
    use_true_mctx = True  # Always use the fastest version
    
    if use_true_mctx:
        mcts_current = MCTXTrueJAX(
            batch_size=num_games,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,
            c_puct=c_puct,
            num_vertices=num_vertices
        )
        
        mcts_baseline = MCTXTrueJAX(
            batch_size=num_games,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,
            c_puct=c_puct,
            num_vertices=num_vertices
        )
    else:
        mcts_current = MCTXFinalOptimized(
            batch_size=num_games,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,
            num_vertices=num_vertices,
            c_puct=c_puct
        )
        
        mcts_baseline = MCTXFinalOptimized(
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
        # In asymmetric: Player 0 is attacker, Player 1 is defender
        is_attacker_turn = boards.current_players == 0
        
        # Current model plays when:
        # - It's attacker's turn and current is attacker, OR
        # - It's defender's turn and current is defender
        use_current = (is_attacker_turn == current_is_attacker)
        
        # Get action probabilities from both models
        probs_current = mcts_current.search(boards, current_model, mcts_sims, temperature)
        probs_baseline = mcts_baseline.search(boards, baseline_model, mcts_sims, temperature)
        
        # Select actions based on which model should play
        actions = []
        for i in range(num_games):
            if boards.game_states[i] != 0:  # Game already finished
                actions.append(0)  # Dummy action
            else:
                if use_current[i]:
                    probs = probs_current[i]
                else:
                    probs = probs_baseline[i]
                
                # Get valid moves and select best action
                valid_mask = boards.get_valid_moves_mask()[i]
                masked_probs = probs * valid_mask
                
                # For temperature=0, argmax
                action = jnp.argmax(masked_probs)
                actions.append(int(action))
        
        # Make moves
        boards.make_moves(jnp.array(actions))
        move_count += 1
        
        if move_count % 5 == 0:  # Progress update
            print(f"  Move {move_count}: {active_games} active games")
    
    # Count results by role
    current_as_attacker_wins = 0
    current_as_defender_wins = 0
    baseline_as_attacker_wins = 0
    baseline_as_defender_wins = 0
    draws = 0
    
    for i in range(num_games):
        game_state = int(boards.game_states[i])
        
        if game_state == 3:  # Draw (shouldn't happen in asymmetric)
            draws += 1
        elif game_state == 1:  # Attacker wins
            if current_is_attacker[i]:
                current_as_attacker_wins += 1
            else:
                baseline_as_attacker_wins += 1
        elif game_state == 2:  # Defender wins
            if current_is_attacker[i]:
                baseline_as_defender_wins += 1
            else:
                current_as_defender_wins += 1
    
    # Calculate statistics
    current_attacker_games = int(jnp.sum(current_is_attacker))
    current_defender_games = num_games - current_attacker_games
    
    current_total_wins = current_as_attacker_wins + current_as_defender_wins
    baseline_total_wins = baseline_as_attacker_wins + baseline_as_defender_wins
    
    current_attacker_rate = current_as_attacker_wins / max(current_attacker_games, 1)
    current_defender_rate = current_as_defender_wins / max(current_defender_games, 1)
    baseline_attacker_rate = baseline_as_attacker_wins / max(current_defender_games, 1)  # Baseline attacks when current defends
    baseline_defender_rate = baseline_as_defender_wins / max(current_attacker_games, 1)  # Baseline defends when current attacks
    
    eval_time = time.time() - start_time
    
    # Print summary
    print(f"\nParallel Asymmetric Results ({num_games} games in {eval_time:.1f}s):")
    print(f"Current Model:")
    print(f"  As Attacker: {current_as_attacker_wins}/{current_attacker_games} ({current_attacker_rate:.1%})")
    print(f"  As Defender: {current_as_defender_wins}/{current_defender_games} ({current_defender_rate:.1%})")
    print(f"  Overall: {current_total_wins}/{num_games} ({current_total_wins/num_games:.1%})")
    print(f"Baseline Model:")
    print(f"  As Attacker: {baseline_as_attacker_wins}/{current_defender_games} ({baseline_attacker_rate:.1%})")
    print(f"  As Defender: {baseline_as_defender_wins}/{current_attacker_games} ({baseline_defender_rate:.1%})")
    print(f"  Overall: {baseline_total_wins}/{num_games} ({baseline_total_wins/num_games:.1%})")
    print(f"Draws: {draws}")
    print(f"Games/sec: {num_games/eval_time:.1f}")
    
    return {
        # Overall statistics
        'total_games': num_games,
        'current_wins': current_total_wins,
        'baseline_wins': baseline_total_wins,
        'draws': draws,
        'current_win_rate': current_total_wins / num_games,
        'baseline_win_rate': baseline_total_wins / num_games,
        
        # Current model by role
        'current_attacker_games': current_attacker_games,
        'current_attacker_wins': current_as_attacker_wins,
        'current_attacker_rate': current_attacker_rate,
        'current_defender_games': current_defender_games,
        'current_defender_wins': current_as_defender_wins,
        'current_defender_rate': current_defender_rate,
        
        # Baseline model by role
        'baseline_attacker_games': current_defender_games,
        'baseline_attacker_wins': baseline_as_attacker_wins,
        'baseline_attacker_rate': baseline_attacker_rate,
        'baseline_defender_games': current_attacker_games,
        'baseline_defender_wins': baseline_as_defender_wins,
        'baseline_defender_rate': baseline_defender_rate,
        
        # Timing
        'eval_time': eval_time
    }
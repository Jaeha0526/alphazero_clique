"""
Truly parallel evaluation - run ALL games (vs initial AND vs best) in a single batch
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


def evaluate_vs_initial_and_best_truly_parallel(
    current_model: ImprovedBatchedNeuralNetwork,
    initial_model: ImprovedBatchedNeuralNetwork,
    best_model: Optional[ImprovedBatchedNeuralNetwork] = None,
    config: dict = None
) -> Dict[str, float]:
    """
    TRULY parallel evaluation - run all games in a single batch.
    If best_model provided: runs 42 games (21 vs initial + 21 vs best) in one batch.
    If no best_model: runs 21 games vs initial only.
    """
    if config is None:
        config = {
            'num_games': 21,
            'num_vertices': 6,
            'k': 3,
            'mcts_sims': 30,
            'game_mode': 'symmetric',
            'c_puct': 3.0
        }
    
    num_games_per_opponent = config['num_games']
    num_vertices = config['num_vertices']
    k = config['k']
    mcts_sims = config.get('mcts_sims', 30)
    c_puct = config.get('c_puct', 3.0)
    temperature = 0.0  # Deterministic for evaluation
    game_mode = config.get('game_mode', 'symmetric')
    
    # Determine total batch size
    # Check if best model is different from initial model
    if best_model is not None:
        # Use tree_map to check if params are the same
        import jax.tree_util as tree
        params_equal = tree.tree_all(
            tree.tree_map(lambda a, b: jnp.allclose(a, b), 
                         best_model.params, 
                         initial_model.params)
        )
        if not params_equal:
            # Run games against both opponents
            total_games = num_games_per_opponent * 2
            eval_both = True
            print(f"\nTruly Parallel Evaluation: {total_games} games ({num_games_per_opponent} vs initial + {num_games_per_opponent} vs best) in ONE batch")
        else:
            # Best model is same as initial, only evaluate against initial
            total_games = num_games_per_opponent
            eval_both = False
            print(f"\nParallel Evaluation: {total_games} games vs initial (best model same as initial)")
    else:
        # Only evaluate against initial
        total_games = num_games_per_opponent
        eval_both = False
        print(f"\nParallel Evaluation: {total_games} games vs initial")
    
    start_time = time.time()
    
    # Initialize Ramsey saver if in avoid_clique mode
    ramsey_saver = None
    if game_mode == "avoid_clique":
        ramsey_saver = RamseyCounterexampleSaver()
    
    # Create batch of games
    boards = VectorizedCliqueBoard(
        batch_size=total_games,
        num_vertices=num_vertices,
        k=k,
        game_mode=game_mode
    )
    
    # Setup game assignments
    # First half: vs initial, Second half: vs best (if applicable)
    if eval_both:
        # Games 0 to num_games_per_opponent-1: vs initial
        # Games num_games_per_opponent to 2*num_games_per_opponent-1: vs best
        opponent_is_initial = jnp.array(
            [True] * num_games_per_opponent + [False] * num_games_per_opponent
        )
    else:
        opponent_is_initial = jnp.ones(total_games, dtype=bool)
    
    # Alternate who starts within each group
    current_starts = jnp.array([i % 2 == 0 for i in range(total_games)])
    
    # Create MCTS instances
    num_actions = num_vertices * (num_vertices - 1) // 2
    # Check if we should force Python MCTS for evaluation
    python_eval = config.get('python_eval', False)
    use_true_mctx = False if python_eval else config.get('use_true_mctx', True)
    
    if python_eval:
        print("  Using Python MCTS for evaluation (no compilation overhead)")
    
    if use_true_mctx:
        mcts_current = MCTXTrueJAX(
            batch_size=total_games,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,
            c_puct=c_puct,
            num_vertices=num_vertices
        )
        
        mcts_initial = MCTXTrueJAX(
            batch_size=total_games,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,
            c_puct=c_puct,
            num_vertices=num_vertices
        )
        
        if eval_both:
            mcts_best = MCTXTrueJAX(
                batch_size=total_games,
                num_actions=num_actions,
                max_nodes=mcts_sims + 1,
                c_puct=c_puct,
                num_vertices=num_vertices
            )
    else:
        mcts_current = MCTXFinalOptimized(
            batch_size=total_games,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,
            num_vertices=num_vertices,
            c_puct=c_puct
        )
        
        mcts_initial = MCTXFinalOptimized(
            batch_size=total_games,
            num_actions=num_actions,
            max_nodes=mcts_sims + 1,
            num_vertices=num_vertices,
            c_puct=c_puct
        )
        
        if eval_both:
            mcts_best = MCTXFinalOptimized(
                batch_size=total_games,
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
        
        # Get action probabilities from all models
        probs_current = mcts_current.search(boards, current_model, mcts_sims, temperature)
        probs_initial = mcts_initial.search(boards, initial_model, mcts_sims, temperature)
        
        if eval_both:
            probs_best = mcts_best.search(boards, best_model, mcts_sims, temperature)
        
        # Select actions based on game assignment and current player
        actions = []
        for i in range(total_games):
            if boards.game_states[i] != 0:  # Game already finished
                actions.append(0)  # Dummy action
                continue
            
            # Determine which model plays this turn
            current_plays = (boards.current_players[i] == 0) == current_starts[i]
            
            if current_plays:
                # Current model's turn
                probs = probs_current[i]
            else:
                # Opponent's turn - depends on which opponent
                if opponent_is_initial[i]:
                    probs = probs_initial[i]
                else:
                    probs = probs_best[i]
            
            # Get valid moves and select best action
            valid_mask = boards.get_valid_moves_mask()[i]
            masked_probs = probs * valid_mask
            
            # Select action (deterministic - argmax)
            if jnp.sum(masked_probs) > 0:
                action = jnp.argmax(masked_probs)
            else:
                valid_actions = jnp.where(valid_mask)[0]
                action = valid_actions[0] if len(valid_actions) > 0 else 0
            
            actions.append(int(action))
        
        # Make moves
        boards.make_moves(jnp.array(actions))
        move_count += 1
        
        if move_count % 10 == 0:  # Progress update
            print(f"  Move {move_count}: {active_games} active games")
    
    # Count results for games vs initial
    initial_wins = 0
    current_wins_vs_initial = 0
    draws_vs_initial = 0
    
    # Count results for games vs best
    best_wins = 0
    current_wins_vs_best = 0
    draws_vs_best = 0
    
    for i in range(total_games):
        game_state = int(boards.game_states[i])
        
        if opponent_is_initial[i]:
            # Game was vs initial model
            if game_state == 3:  # Draw
                draws_vs_initial += 1
            elif game_state == 1:  # Player 1 wins
                if current_starts[i]:
                    current_wins_vs_initial += 1
                else:
                    initial_wins += 1
            elif game_state == 2:  # Player 2 wins
                if not current_starts[i]:
                    current_wins_vs_initial += 1
                else:
                    initial_wins += 1
        else:
            # Game was vs best model
            if game_state == 3:  # Draw
                draws_vs_best += 1
            elif game_state == 1:  # Player 1 wins
                if current_starts[i]:
                    current_wins_vs_best += 1
                else:
                    best_wins += 1
            elif game_state == 2:  # Player 2 wins
                if not current_starts[i]:
                    current_wins_vs_best += 1
                else:
                    best_wins += 1
    
    # Save Ramsey counterexamples if in avoid_clique mode
    if ramsey_saver is not None:
        saved_files = ramsey_saver.save_batch_counterexamples(
            boards=boards,
            source="truly_parallel_evaluation"
        )
        if saved_files:
            print(f"  💎 Saved {len(saved_files)} Ramsey counterexamples!")
    
    eval_time = time.time() - start_time
    
    # Print summary
    print(f"\nTruly Parallel Evaluation Results ({total_games} games in {eval_time:.1f}s):")
    print(f"  vs Initial: {current_wins_vs_initial} wins, {initial_wins} losses, {draws_vs_initial} draws")
    print(f"  Win rate vs initial: {current_wins_vs_initial/num_games_per_opponent:.1%}")
    
    if eval_both:
        print(f"  vs Best: {current_wins_vs_best} wins, {best_wins} losses, {draws_vs_best} draws")
        print(f"  Win rate vs best: {current_wins_vs_best/num_games_per_opponent:.1%}")
    
    print(f"  Games/sec: {total_games/eval_time:.1f}")
    
    # Prepare results
    results = {
        'win_rate_vs_initial': current_wins_vs_initial / num_games_per_opponent,
        'draw_rate_vs_initial': draws_vs_initial / num_games_per_opponent,
        'eval_time_vs_initial': eval_time if not eval_both else eval_time/2,  # Approximate
        'vs_initial_details': {
            'wins': current_wins_vs_initial,
            'losses': initial_wins,
            'draws': draws_vs_initial,
            'eval_time': eval_time if not eval_both else eval_time/2
        }
    }
    
    if eval_both:
        results.update({
            'win_rate_vs_best': current_wins_vs_best / num_games_per_opponent,
            'draw_rate_vs_best': draws_vs_best / num_games_per_opponent,
            'eval_time_vs_best': eval_time/2,  # Approximate
            'vs_best_details': {
                'wins': current_wins_vs_best,
                'losses': best_wins,
                'draws': draws_vs_best,
                'eval_time': eval_time/2
            }
        })
    else:
        results.update({
            'win_rate_vs_best': -1,
            'draw_rate_vs_best': -1,
            'eval_time_vs_best': 0,
            'vs_best_details': None
        })
    
    return results
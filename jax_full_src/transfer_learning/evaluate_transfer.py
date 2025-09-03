#!/usr/bin/env python
"""
Evaluate transferred model vs randomly initialized model.
This helps measure the benefit of transfer learning.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vectorized_nn import ImprovedBatchedNeuralNetwork
from vectorized_board import VectorizedCliqueBoard
from mctx_final_optimized import MCTXFinalOptimized
from mctx_true_jax import MCTXTrueJAX


def create_random_model(num_vertices: int, hidden_dim: int = 64, 
                       num_layers: int = 3, asymmetric: bool = False) -> ImprovedBatchedNeuralNetwork:
    """Create a randomly initialized model."""
    print(f"  Creating random model: n={num_vertices}, hidden_dim={hidden_dim}")
    model = ImprovedBatchedNeuralNetwork(
        num_vertices=num_vertices,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        asymmetric_mode=asymmetric
    )
    return model


def load_transferred_model(checkpoint_path: str, num_vertices: int, 
                          hidden_dim: int = 64, num_layers: int = 3,
                          asymmetric: bool = False) -> ImprovedBatchedNeuralNetwork:
    """Load a model with transferred weights."""
    print(f"  Loading transferred model from: {checkpoint_path}")
    
    # Create model structure
    model = ImprovedBatchedNeuralNetwork(
        num_vertices=num_vertices,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        asymmetric_mode=asymmetric
    )
    
    # Load transferred parameters
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    model.params = checkpoint['params']
    
    # Print transfer info if available
    if 'transfer_info' in checkpoint:
        info = checkpoint['transfer_info']
        print(f"    Transfer info: n={info['source_vertices']} → n={info['target_vertices']}")
    
    return model


def evaluate_models(
    model1: ImprovedBatchedNeuralNetwork,
    model2: ImprovedBatchedNeuralNetwork,
    model1_name: str,
    model2_name: str,
    num_games: int = 20,
    num_vertices: int = 13,
    k: int = 4,
    mcts_sims: int = 30,
    use_true_mctx: bool = False,
    parallel_games: bool = False
) -> Dict[str, Any]:
    """
    Evaluate two models against each other.
    
    Args:
        model1, model2: Models to compare
        model1_name, model2_name: Names for display
        num_games: Number of games to play
        num_vertices: Graph size
        k: Clique size
        mcts_sims: MCTS simulations per move
        use_true_mctx: Use pure JAX MCTS
        parallel_games: Run games in parallel
    
    Returns:
        Dictionary with results
    """
    print(f"\n🎮 Evaluating {model1_name} vs {model2_name}")
    print(f"  Settings: n={num_vertices}, k={k}, {num_games} games, {mcts_sims} MCTS sims")
    
    num_actions = num_vertices * (num_vertices - 1) // 2
    
    if parallel_games and num_games > 1:
        # Parallel evaluation
        batch_size = min(num_games, 32)  # Limit batch size for memory
        print(f"  Running {batch_size} games in parallel")
        
        results = evaluate_parallel(
            model1, model2, model1_name, model2_name,
            batch_size, num_vertices, k, mcts_sims, use_true_mctx
        )
        
        # Run remaining games if needed
        remaining = num_games - batch_size
        if remaining > 0:
            print(f"  Running {remaining} more games...")
            extra_results = evaluate_sequential(
                model1, model2, model1_name, model2_name,
                remaining, num_vertices, k, mcts_sims, use_true_mctx
            )
            # Combine results
            results['model1_wins'] += extra_results['model1_wins']
            results['model2_wins'] += extra_results['model2_wins']
            results['draws'] += extra_results['draws']
    else:
        # Sequential evaluation
        results = evaluate_sequential(
            model1, model2, model1_name, model2_name,
            num_games, num_vertices, k, mcts_sims, use_true_mctx
        )
    
    # Calculate win rates
    total_games = results['model1_wins'] + results['model2_wins'] + results['draws']
    results['model1_win_rate'] = results['model1_wins'] / total_games
    results['model2_win_rate'] = results['model2_wins'] / total_games
    results['draw_rate'] = results['draws'] / total_games
    
    return results


def evaluate_sequential(
    model1: ImprovedBatchedNeuralNetwork,
    model2: ImprovedBatchedNeuralNetwork,
    model1_name: str,
    model2_name: str,
    num_games: int,
    num_vertices: int,
    k: int,
    mcts_sims: int,
    use_true_mctx: bool
) -> Dict[str, Any]:
    """Run sequential evaluation (one game at a time)."""
    
    num_actions = num_vertices * (num_vertices - 1) // 2
    
    # Create MCTS instances
    MCTSClass = MCTXTrueJAX if use_true_mctx else MCTXFinalOptimized
    
    mcts1 = MCTSClass(
        batch_size=1,
        num_actions=num_actions,
        max_nodes=mcts_sims + 1,
        c_puct=3.0,
        num_vertices=num_vertices
    )
    
    mcts2 = MCTSClass(
        batch_size=1,
        num_actions=num_actions,
        max_nodes=mcts_sims + 1,
        c_puct=3.0,
        num_vertices=num_vertices
    )
    
    model1_wins = 0
    model2_wins = 0
    draws = 0
    game_lengths = []
    
    for game_idx in range(num_games):
        # Alternate who starts
        model1_starts = (game_idx % 2 == 0)
        
        # Initialize board
        board = VectorizedCliqueBoard(
            batch_size=1,
            num_vertices=num_vertices,
            k=k,
            game_mode='symmetric'
        )
        
        move_count = 0
        max_moves = num_actions
        
        # Play game
        while board.game_states[0] == 0 and move_count < max_moves:
            current_player = int(board.current_players[0])
            
            # Determine which model to use
            use_model1 = (current_player == 0 and model1_starts) or (current_player == 1 and not model1_starts)
            
            if use_model1:
                action_probs = mcts1.search(board, model1, mcts_sims, temperature=0.0)
            else:
                action_probs = mcts2.search(board, model2, mcts_sims, temperature=0.0)
            
            # Select action (deterministic for evaluation)
            valid_moves = board.get_valid_moves_mask()[0]
            masked_probs = action_probs[0] * valid_moves
            
            if jnp.sum(masked_probs) > 0:
                action = jnp.argmax(masked_probs)
            else:
                # Random valid move if something goes wrong
                valid_indices = jnp.where(valid_moves)[0]
                if len(valid_indices) > 0:
                    action = valid_indices[0]
                else:
                    break
            
            board.make_moves(jnp.array([action]))
            move_count += 1
        
        game_lengths.append(move_count)
        
        # Check result
        game_state = int(board.game_states[0])
        
        if game_state == 3 or game_state == 0:  # Draw
            draws += 1
            result = "Draw"
        elif game_state == 1:  # Player 1 wins
            if model1_starts:
                model1_wins += 1
                result = model1_name
            else:
                model2_wins += 1
                result = model2_name
        else:  # Player 2 wins
            if not model1_starts:
                model1_wins += 1
                result = model1_name
            else:
                model2_wins += 1
                result = model2_name
        
        if (game_idx + 1) % 5 == 0:
            print(f"    Game {game_idx + 1}/{num_games}: {result} wins (length: {move_count})")
    
    return {
        'model1_wins': model1_wins,
        'model2_wins': model2_wins,
        'draws': draws,
        'game_lengths': game_lengths,
        'avg_game_length': np.mean(game_lengths) if game_lengths else 0
    }


def evaluate_parallel(
    model1: ImprovedBatchedNeuralNetwork,
    model2: ImprovedBatchedNeuralNetwork,
    model1_name: str,
    model2_name: str,
    batch_size: int,
    num_vertices: int,
    k: int,
    mcts_sims: int,
    use_true_mctx: bool
) -> Dict[str, Any]:
    """Run parallel evaluation (multiple games at once)."""
    
    num_actions = num_vertices * (num_vertices - 1) // 2
    
    # Create MCTS instances for batch
    MCTSClass = MCTXTrueJAX if use_true_mctx else MCTXFinalOptimized
    
    mcts1 = MCTSClass(
        batch_size=batch_size,
        num_actions=num_actions,
        max_nodes=mcts_sims + 1,
        c_puct=3.0,
        num_vertices=num_vertices
    )
    
    mcts2 = MCTSClass(
        batch_size=batch_size,
        num_actions=num_actions,
        max_nodes=mcts_sims + 1,
        c_puct=3.0,
        num_vertices=num_vertices
    )
    
    # Initialize boards
    boards = VectorizedCliqueBoard(
        batch_size=batch_size,
        num_vertices=num_vertices,
        k=k,
        game_mode='symmetric'
    )
    
    # Track who starts each game
    model1_starts = jnp.array([i % 2 == 0 for i in range(batch_size)])
    
    move_count = 0
    max_moves = num_actions
    
    # Play all games
    while jnp.any(boards.game_states == 0) and move_count < max_moves:
        current_players = boards.current_players
        
        # Determine which model to use for each game
        use_model1_mask = ((current_players == 0) & model1_starts) | ((current_players == 1) & ~model1_starts)
        
        # Get moves from both models
        if jnp.any(use_model1_mask):
            probs1 = mcts1.search(boards, model1, mcts_sims, temperature=0.0)
        else:
            probs1 = jnp.zeros((batch_size, num_actions))
            
        if jnp.any(~use_model1_mask):
            probs2 = mcts2.search(boards, model2, mcts_sims, temperature=0.0)
        else:
            probs2 = jnp.zeros((batch_size, num_actions))
        
        # Combine probabilities based on which model should play
        action_probs = jnp.where(use_model1_mask[:, None], probs1, probs2)
        
        # Select actions
        valid_moves = boards.get_valid_moves_mask()
        masked_probs = action_probs * valid_moves
        actions = jnp.argmax(masked_probs, axis=1)
        
        boards.make_moves(actions)
        move_count += 1
        
        if move_count % 10 == 0:
            active_games = jnp.sum(boards.game_states == 0)
            print(f"    Move {move_count}: {active_games}/{batch_size} games still active")
    
    # Count results
    model1_wins = 0
    model2_wins = 0
    draws = 0
    
    for i in range(batch_size):
        game_state = int(boards.game_states[i])
        
        if game_state == 3 or game_state == 0:  # Draw
            draws += 1
        elif game_state == 1:  # Player 1 wins
            if model1_starts[i]:
                model1_wins += 1
            else:
                model2_wins += 1
        else:  # Player 2 wins
            if not model1_starts[i]:
                model1_wins += 1
            else:
                model2_wins += 1
    
    return {
        'model1_wins': model1_wins,
        'model2_wins': model2_wins,
        'draws': draws,
        'game_lengths': [move_count] * batch_size,  # Approximate
        'avg_game_length': move_count
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate transferred model vs random initialization')
    
    parser.add_argument('transferred_checkpoint', type=str,
                        help='Path to transferred model checkpoint')
    parser.add_argument('--num_vertices', type=int, required=True,
                        help='Number of vertices (must match transferred model)')
    parser.add_argument('--k', type=int, required=True,
                        help='Clique size')
    parser.add_argument('--num_games', type=int, default=20,
                        help='Number of games to play')
    parser.add_argument('--mcts_sims', type=int, default=30,
                        help='MCTS simulations per move')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--use_true_mctx', action='store_true',
                        help='Use pure JAX MCTS')
    parser.add_argument('--parallel', action='store_true',
                        help='Run games in parallel')
    parser.add_argument('--asymmetric', action='store_true',
                        help='Use asymmetric game mode')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Evaluating Transfer Learning Benefit")
    print(f"{'='*60}")
    
    # Create models
    print("\n📦 Loading models...")
    
    transferred_model = load_transferred_model(
        args.transferred_checkpoint,
        args.num_vertices,
        args.hidden_dim,
        args.num_layers,
        args.asymmetric
    )
    
    random_model = create_random_model(
        args.num_vertices,
        args.hidden_dim,
        args.num_layers,
        args.asymmetric
    )
    
    # Evaluate
    start_time = time.time()
    
    results = evaluate_models(
        transferred_model,
        random_model,
        "Transferred",
        "Random",
        args.num_games,
        args.num_vertices,
        args.k,
        args.mcts_sims,
        args.use_true_mctx,
        args.parallel
    )
    
    eval_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*60}")
    print(f"📊 Results after {args.num_games} games:")
    print(f"{'='*60}")
    print(f"  Transferred model wins: {results['model1_wins']} ({results['model1_win_rate']:.1%})")
    print(f"  Random model wins: {results['model2_wins']} ({results['model2_win_rate']:.1%})")
    print(f"  Draws: {results['draws']} ({results['draw_rate']:.1%})")
    print(f"  Average game length: {results['avg_game_length']:.1f} moves")
    print(f"  Evaluation time: {eval_time:.1f}s ({args.num_games/eval_time:.2f} games/sec)")
    
    print(f"\n💡 Transfer Learning Benefit:")
    benefit = results['model1_win_rate'] - results['model2_win_rate']
    if benefit > 0.2:
        print(f"  ✅ Significant benefit: +{benefit:.1%} win rate improvement")
    elif benefit > 0:
        print(f"  ✓ Some benefit: +{benefit:.1%} win rate improvement")
    else:
        print(f"  ⚠️ No clear benefit (may need more training)")
    
    return results


if __name__ == "__main__":
    main()
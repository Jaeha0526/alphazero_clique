#!/usr/bin/env python
"""
Main entry point for JAX AlphaZero pipeline with tree-based MCTS.
Provides command-line interface identical to PyTorch version.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import pickle
import os
import time
from datetime import datetime
import json
import argparse
from pathlib import Path
from jax import random
from typing import Any

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from vectorized_self_play_fixed import FixedVectorizedSelfPlay, FixedSelfPlayConfig
from train_jax import train_network_jax
from evaluation_jax import evaluate_head_to_head_jax


def save_checkpoint(model: ImprovedBatchedNeuralNetwork, 
                    params: Any,
                    optimizer_state: Any,
                    iteration: int, 
                    checkpoint_dir: str):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'params': params,
        'optimizer_state': optimizer_state,
        'iteration': iteration,
        'asymmetric_mode': model.asymmetric_mode,
        'model_config': {
            'num_vertices': model.num_vertices,
            'hidden_dim': model.hidden_dim,
            'num_gnn_layers': model.num_layers,
            'asymmetric_mode': model.asymmetric_mode,
        }
    }
    
    filepath = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    # Also save as best model
    best_path = os.path.join(checkpoint_dir, 'best_model.pkl')
    with open(best_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Saved checkpoint to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="JAX AlphaZero with Tree-Based MCTS")
    
    # Training parameters
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations')
    parser.add_argument('--self-play-games', type=int, default=100,
                        help='Self-play games per iteration')
    parser.add_argument('--mcts-sims', type=int, default=50,
                        help='MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--self-play-batch', type=int, default=32,
                        help='Self-play batch size')
    
    # Game parameters
    parser.add_argument('--vertices', type=int, default=6,
                        help='Number of vertices')
    parser.add_argument('--k', type=int, default=3,
                        help='Clique size to win')
    parser.add_argument('--game-mode', type=str, default='symmetric',
                        choices=['symmetric', 'asymmetric'],
                        help='Game mode')
    parser.add_argument('--perspective-mode', type=str, default='alternating',
                        choices=['fixed', 'alternating'],
                        help='Value perspective mode')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    
    # Other parameters
    parser.add_argument('--skill-variation', type=float, default=0.0,
                        help='Skill variation for self-play')
    parser.add_argument('--noise-weight', type=float, default=0.25,
                        help='Dirichlet noise weight')
    parser.add_argument('--c-puct', type=float, default=3.0,
                        help='PUCT exploration constant')
    parser.add_argument('--experiment-name', type=str, default='jax_experiment',
                        help='Experiment name')
    parser.add_argument('--eval-games', type=int, default=20,
                        help='Number of evaluation games')
    
    args = parser.parse_args()
    
    # Set up directories
    exp_dir = Path(f'experiments/{args.experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = exp_dir / 'models'
    game_data_dir = exp_dir / 'datasets'
    checkpoint_dir.mkdir(exist_ok=True)
    game_data_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("JAX AlphaZero Pipeline with Tree-Based MCTS")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {jax.default_backend()}")
    print(f"Game mode: {args.game_mode}")
    print(f"Perspective: {args.perspective_mode}")
    print(f"MCTS simulations: {args.mcts_sims}")
    print(f"C_PUCT: {args.c_puct}")
    print(f"Self-play batch size: {args.self_play_batch}")
    print("=" * 60)
    
    # Initialize model
    asymmetric = args.game_mode == 'asymmetric'
    model = ImprovedBatchedNeuralNetwork(
        num_vertices=args.vertices,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        asymmetric_mode=asymmetric
    )
    
    # Get initial parameters from model
    params = model.params
    
    # Training log
    training_log = {
        'config': config,
        'iterations': [],
        'total_iterations': args.iterations,
        'start_time': datetime.now().isoformat()
    }
    
    # Main training loop
    for iteration in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{args.iterations}")
        print(f"{'='*60}")
        
        iter_start = time.time()
        iter_log = {'iteration': iteration}
        
        # 1. Self-Play
        print("\n1. Running self-play with tree-based MCTS...")
        self_play_config = FixedSelfPlayConfig(
            batch_size=args.self_play_batch,
            num_vertices=args.vertices,
            k=args.k,
            game_mode=args.game_mode,
            mcts_simulations=args.mcts_sims,
            c_puct=args.c_puct,
            noise_weight=args.noise_weight,
            perspective_mode=args.perspective_mode,
            skill_variation=args.skill_variation
        )
        
        self_play = FixedVectorizedSelfPlay(self_play_config, model)
        model.params = params  # Update model params
        
        sp_start = time.time()
        games = self_play.play_games(args.self_play_games)
        sp_time = time.time() - sp_start
        
        print(f"Generated {len(games)} games in {sp_time:.1f}s ({len(games)/sp_time:.1f} games/sec)")
        
        # Calculate average game length
        game_lengths = [len(game) for game in games]
        avg_length = np.mean(game_lengths) if game_lengths else 0
        print(f"Average game length: {avg_length:.1f} moves")
        
        iter_log['self_play_time'] = sp_time
        iter_log['games_generated'] = len(games)
        iter_log['avg_game_length'] = avg_length
        
        # Save game data
        game_file = game_data_dir / f'games_iter_{iteration}.pkl'
        with open(game_file, 'wb') as f:
            pickle.dump(games, f)
        
        # 2. Training
        print("\n2. Training network...")
        
        # Prepare training data
        all_examples = []
        for game in games:
            all_examples.extend(game)
        
        print(f"Training on {len(all_examples)} positions")
        
        # Train
        train_start = time.time()
        train_state, train_metrics = train_network_jax(
            model,
            all_examples,
            params,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            asymmetric_mode=asymmetric,
            perspective_mode=args.perspective_mode,
            l2_weight=1e-5
        )
        train_time = time.time() - train_start
        
        # Update params from train state
        params = train_state.params
        optimizer_state = train_state
        
        print(f"Training completed in {train_time:.1f}s")
        
        if train_metrics:
            avg_policy_loss = np.mean([m['loss'] for m in train_metrics])
            avg_value_loss = np.mean([m.get('value_loss', 0) for m in train_metrics])
            print(f"Avg policy loss: {avg_policy_loss:.4f}")
            print(f"Avg value loss: {avg_value_loss:.4f}")
            
            iter_log['train_time'] = train_time
            iter_log['avg_policy_loss'] = float(avg_policy_loss)
            iter_log['avg_value_loss'] = float(avg_value_loss)
        
        # 3. Evaluation
        print("\n3. Evaluating model...")
        
        # Create models for evaluation
        new_model = ImprovedBatchedNeuralNetwork(
            num_vertices=args.vertices,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            asymmetric_mode=asymmetric
        )
        new_model.params = params
        
        # Always evaluate against initial model
        initial_model = ImprovedBatchedNeuralNetwork(
            num_vertices=args.vertices,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            asymmetric_mode=asymmetric
        )
        # Initial model uses random params
        
        # Evaluate against initial
        eval_start = time.time()
        results_vs_initial = evaluate_head_to_head_jax(
            new_model, initial_model,
            num_games=args.eval_games,
            num_vertices=args.vertices,
            k=args.k,
            game_mode=args.game_mode,
            mcts_sims=30,
            batch_size=4,
            verbose=False
        )
        
        wins = results_vs_initial['player1_wins']
        draws = results_vs_initial['draws']
        losses = results_vs_initial['player2_wins']
        win_rate_vs_initial = wins / (wins + draws + losses) if (wins + draws + losses) > 0 else 0
        
        print(f"Win rate vs initial: {win_rate_vs_initial:.1%} ({wins}W-{draws}D-{losses}L)")
        
        iter_log['eval_time'] = time.time() - eval_start
        iter_log['win_rate_vs_initial'] = float(win_rate_vs_initial)
        iter_log['wins_vs_initial'] = wins
        iter_log['draws_vs_initial'] = draws
        iter_log['losses_vs_initial'] = losses
        
        # Save checkpoint
        if iteration % 5 == 0 or iteration == args.iterations - 1:
            save_checkpoint(model, params, optimizer_state, iteration, str(checkpoint_dir))
        
        # Log iteration
        iter_time = time.time() - iter_start
        iter_log['total_time'] = iter_time
        training_log['iterations'].append(iter_log)
        
        print(f"\nIteration {iteration} completed in {iter_time:.1f}s")
        
        # Save training log after each iteration
        with open(exp_dir / 'training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)
    
    # Final summary
    training_log['end_time'] = datetime.now().isoformat()
    training_log['final_win_rate_vs_initial'] = float(win_rate_vs_initial)
    
    # Calculate average speed
    total_games = sum(it['games_generated'] for it in training_log['iterations'])
    total_sp_time = sum(it['self_play_time'] for it in training_log['iterations'])
    training_log['avg_games_per_second'] = total_games / total_sp_time if total_sp_time > 0 else 0
    
    with open(exp_dir / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # Generate plots
    try:
        import matplotlib.pyplot as plt
        
        iterations = range(len(training_log['iterations']))
        policy_losses = [it.get('avg_policy_loss', 0) for it in training_log['iterations']]
        value_losses = [it.get('avg_value_loss', 0) for it in training_log['iterations']]
        win_rates = [it.get('win_rate_vs_initial', 0) for it in training_log['iterations']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1.plot(iterations, policy_losses, 'b-', label='Policy Loss')
        ax1.plot(iterations, value_losses, 'r-', label='Value Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.legend()
        ax1.grid(True)
        
        # Win rate plot
        ax2.plot(iterations, win_rates, 'g-', marker='o')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Win Rate')
        ax2.set_title('Win Rate vs Initial Model')
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(exp_dir / 'training_losses.png')
        plt.close()
        
        print(f"\nPlots saved to {exp_dir}/training_losses.png")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Results saved to: {exp_dir}")
    print(f"Final win rate vs initial: {win_rate_vs_initial:.1%}")
    print(f"Average self-play speed: {training_log['avg_games_per_second']:.1f} games/sec")
    print("="*60)


if __name__ == "__main__":
    main()
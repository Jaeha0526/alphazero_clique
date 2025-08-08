#!/usr/bin/env python
"""
Optimized JAX AlphaZero pipeline with all performance fixes.
Uses JIT-compiled MCTS and batched neural network evaluations.
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
import matplotlib.pyplot as plt

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from mctx_final_optimized import MCTXFinalOptimized
from mctx_true_jax import MCTXTrueJAX
from train_jax import train_network_jax
from train_jax_fully_optimized import train_network_jax_optimized
from train_jax_with_validation import train_network_jax_with_validation
from evaluation_jax_fixed import evaluate_vs_initial_and_best
from evaluation_jax_asymmetric import evaluate_vs_initial_and_best_asymmetric
from evaluation_jax_parallel import evaluate_vs_initial_and_best_parallel
from evaluation_jax_asymmetric_parallel import evaluate_vs_initial_and_best_asymmetric_parallel


class OptimizedSelfPlay:
    """Self-play using JIT-compiled MCTS."""
    
    def __init__(self, config):
        self.config = config
        # For n=9 vertices, we have C(9,2) = 36 possible edges
        self.num_actions = config.num_vertices * (config.num_vertices - 1) // 2
        
        # Statistics tracking
        self.game_statistics = {
            'total_games': 0,
            'attacker_wins': 0,
            'defender_wins': 0,
            'game_lengths': [],
            'move_counts_per_game': [],
            'avg_game_length': 0.0,
            'win_ratio_attacker': 0.0,
            'win_ratio_defender': 0.0,
            'length_distribution': {}
        }
    
    def play_games(self, neural_network, num_games):
        """Play games using optimized MCTS."""
        all_game_data = []
        games_played = 0
        
        # Reset statistics for this batch
        batch_stats = {
            'games_played': 0,
            'attacker_wins': 0,
            'defender_wins': 0,
            'game_lengths': [],
            'total_moves': 0
        }
        
        while games_played < num_games:
            batch_size = min(self.config.batch_size, num_games - games_played)
            print(f"\nStarting batch of {batch_size} games (Total: {games_played}/{num_games})")
            
            # Create MCTS with correct batch size for this iteration
            num_actions = self.config.num_vertices * (self.config.num_vertices - 1) // 2
            print(f"  Creating MCTXFinalOptimized: {num_actions} actions, batch_size={batch_size}")
            # Use the final optimized MCTX implementation
            # Choose MCTS implementation based on config
            if getattr(self.config, 'use_true_mctx', False):
                print(f"  Creating True MCTX (JAX primitives): {num_actions} actions, batch_size={batch_size}")
                mcts = MCTXTrueJAX(
                    batch_size=batch_size,
                    num_actions=num_actions,
                    max_nodes=self.config.mcts_simulations + 1,  # Only need sims + 1 nodes
                    c_puct=self.config.c_puct,
                    num_vertices=self.config.num_vertices
                )
            else:
                print(f"  Creating MCTXFinalOptimized (Python loops): {num_actions} actions, batch_size={batch_size}")
                mcts = MCTXFinalOptimized(
                    batch_size=batch_size,
                    num_actions=num_actions,
                    max_nodes=self.config.mcts_simulations + 1,  # Only need sims + 1 nodes
                    num_vertices=self.config.num_vertices,
                    c_puct=self.config.c_puct
                )
            
            # Initialize boards
            print(f"  Initializing boards: n={self.config.num_vertices}, k={self.config.k}, mode={self.config.game_mode}")
            boards = VectorizedCliqueBoard(
                batch_size=batch_size,
                num_vertices=self.config.num_vertices,
                k=self.config.k,
                game_mode=self.config.game_mode
            )
            
            game_data = [[] for _ in range(batch_size)]
            
            # Play until all games finish
            move_count = 0
            max_moves = self.config.num_vertices * (self.config.num_vertices - 1) // 2
            while jnp.any(boards.game_states == 0) and move_count < max_moves:
                active_games = jnp.sum(boards.game_states == 0)
                print(f"  Batch {batch_size} games - Move {move_count}, Active games: {active_games}")
                
                # Get MCTS action probabilities using JIT-compiled search
                mcts_start = time.time()
                print(f"    Starting MCTS search...")
                try:
                    mcts_probs = mcts.search(
                        boards,
                        neural_network,
                        self.config.mcts_simulations,
                        temperature=1.0 if len(game_data[0]) < self.config.temperature_threshold else 0.0
                    )
                    mcts_time = time.time() - mcts_start
                    print(f"    MCTS search completed in {mcts_time:.2f}s")
                except Exception as e:
                    print(f"    ERROR in MCTS: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                move_count += 1
                
                # Store data for training
                edge_indices, edge_features = boards.get_features_for_nn_undirected()
                for i in range(batch_size):
                    if boards.game_states[i] == 0:
                        game_data[i].append({
                            'edge_indices': edge_indices[i],
                            'edge_features': edge_features[i],
                            'policy': mcts_probs[i],
                            'player': boards.current_players[i],
                            'player_role': int(boards.current_players[i]) if self.config.game_mode == "asymmetric" else None
                        })
                
                # Sample actions
                active_mask = boards.game_states == 0
                actions = []
                for i in range(batch_size):
                    if active_mask[i]:
                        # Normalize probabilities to ensure they sum to 1
                        probs = np.array(mcts_probs[i])
                        # Ensure we have the right number of probabilities
                        if len(probs) != num_actions:
                            print(f"Warning: Expected {num_actions} action probs, got {len(probs)}")
                            # Resize if needed
                            if len(probs) < num_actions:
                                # Pad with zeros
                                probs = np.pad(probs, (0, num_actions - len(probs)), 'constant')
                            else:
                                # Truncate
                                probs = probs[:num_actions]
                        probs = probs / probs.sum()
                        action = np.random.choice(num_actions, p=probs)
                        actions.append(action)
                    else:
                        actions.append(0)
                
                # Make moves
                boards.make_moves(jnp.array(actions))
            
            # Process finished games
            for i in range(batch_size):
                winner = int(boards.winners[i])
                game_length = len(game_data[i])
                
                # Update batch statistics
                batch_stats['games_played'] += 1
                batch_stats['game_lengths'].append(game_length)
                batch_stats['total_moves'] += game_length
                
                # Track wins by role (for asymmetric games)
                if self.config.game_mode == "asymmetric":
                    if winner == 0:  # Attacker wins
                        batch_stats['attacker_wins'] += 1
                    elif winner == 1:  # Defender wins
                        batch_stats['defender_wins'] += 1
                
                for move_data in game_data[i]:
                    # Perspective-based value
                    if self.config.perspective_mode == "alternating":
                        value = 1.0 if move_data['player'] == winner else -1.0
                    else:
                        value = 1.0 if winner == 1 else -1.0
                    
                    move_data['value'] = value
                    all_game_data.append(move_data)
            
            games_played += batch_size
        
        # Update overall statistics
        self._update_statistics(batch_stats)
        
        # Print statistics summary
        self._print_statistics_summary(batch_stats)
            
        return all_game_data
    
    def _update_statistics(self, batch_stats):
        """Update overall game statistics."""
        self.game_statistics['total_games'] += batch_stats['games_played']
        self.game_statistics['attacker_wins'] += batch_stats['attacker_wins']
        self.game_statistics['defender_wins'] += batch_stats['defender_wins']
        self.game_statistics['game_lengths'].extend(batch_stats['game_lengths'])
        self.game_statistics['move_counts_per_game'].extend(batch_stats['game_lengths'])
        
        # Calculate averages and ratios
        total_games = self.game_statistics['total_games']
        if total_games > 0:
            self.game_statistics['avg_game_length'] = np.mean(self.game_statistics['game_lengths'])
            
            if self.config.game_mode == "asymmetric":
                total_asym_games = self.game_statistics['attacker_wins'] + self.game_statistics['defender_wins']
                if total_asym_games > 0:
                    self.game_statistics['win_ratio_attacker'] = self.game_statistics['attacker_wins'] / total_asym_games
                    self.game_statistics['win_ratio_defender'] = self.game_statistics['defender_wins'] / total_asym_games
        
        # Update length distribution
        self._update_length_distribution()
    
    def _update_length_distribution(self):
        """Calculate game length distribution."""
        if not self.game_statistics['game_lengths']:
            return
            
        lengths = self.game_statistics['game_lengths']
        unique_lengths, counts = np.unique(lengths, return_counts=True)
        
        self.game_statistics['length_distribution'] = {
            int(length): int(count) for length, count in zip(unique_lengths, counts)
        }
    
    def _print_statistics_summary(self, batch_stats):
        """Print summary of game statistics."""
        print(f"\n📊 Self-Play Statistics Summary:")
        print(f"  Games in this batch: {batch_stats['games_played']}")
        
        if batch_stats['game_lengths']:
            avg_length = np.mean(batch_stats['game_lengths'])
            min_length = min(batch_stats['game_lengths'])
            max_length = max(batch_stats['game_lengths'])
            print(f"  Average game length: {avg_length:.1f} moves")
            print(f"  Game length range: {min_length} - {max_length} moves")
        
        if self.config.game_mode == "asymmetric":
            total_decisive = batch_stats['attacker_wins'] + batch_stats['defender_wins']
            if total_decisive > 0:
                attacker_rate = batch_stats['attacker_wins'] / total_decisive
                defender_rate = batch_stats['defender_wins'] / total_decisive
                print(f"  Win rates - Attacker: {attacker_rate:.1%}, Defender: {defender_rate:.1%}")
        
        # Overall statistics
        print(f"\n📈 Overall Statistics (Total: {self.game_statistics['total_games']} games):")
        if self.game_statistics['total_games'] > 0:
            print(f"  Average game length: {self.game_statistics['avg_game_length']:.1f} moves")
            
            if self.config.game_mode == "asymmetric":
                print(f"  Overall win rates - Attacker: {self.game_statistics['win_ratio_attacker']:.1%}, Defender: {self.game_statistics['win_ratio_defender']:.1%}")
            
            # Show length distribution for recent games
            if len(self.game_statistics['length_distribution']) <= 10:
                print(f"  Game length distribution: {dict(sorted(self.game_statistics['length_distribution'].items()))}")
    
    def get_statistics(self):
        """Get current game statistics."""
        return self.game_statistics.copy()


def save_checkpoint(model: ImprovedBatchedNeuralNetwork, 
                    params: Any,
                    optimizer_state: Any,
                    iteration: int, 
                    checkpoint_dir: str):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'params': params,
        'iteration': iteration,
        'asymmetric_mode': model.asymmetric_mode,
        'model_config': {
            'num_vertices': model.num_vertices,
            'hidden_dim': model.hidden_dim,
            'num_gnn_layers': model.num_layers,
            'asymmetric_mode': model.asymmetric_mode,
        }
    }
    # Don't save optimizer_state as it's not easily pickleable
    
    filepath = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Saved checkpoint to {filepath}")
    return filepath


def load_checkpoint(checkpoint_path: str, model: ImprovedBatchedNeuralNetwork):
    """Load model checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    return checkpoint['params'], checkpoint.get('optimizer_state', None), checkpoint['iteration']


def plot_learning_curve(log_file_path: str):
    """
    Plot learning curves exactly like the original pipeline.
    Shows Policy Loss, Value Loss, and Win Rate vs Initial on 3 axes.
    """
    if not os.path.exists(log_file_path):
        print(f"Log file not found at {log_file_path}. Cannot plot learning curve.")
        return
    
    try:
        with open(log_file_path, 'r') as f:
            log_data = json.load(f)
    except Exception as e:
        print(f"Error loading log file {log_file_path}: {e}")
        return
    
    # Extract data from log (same as original pipeline)
    plot_data = [
        (entry["iteration"], 
         entry.get("validation_policy_loss"), 
         entry.get("validation_value_loss"), 
         entry.get("evaluation_win_rate_vs_initial"))
        for entry in log_data
        if entry.get("validation_policy_loss") is not None and entry.get("validation_value_loss") is not None
    ]
    
    if len(plot_data) < 1:
        print("Not enough valid data points (with losses) in log file to plot learning curve.")
        return
    
    # Extract individual series
    iterations = [p[0] for p in plot_data]
    policy_losses = [p[1] for p in plot_data]
    value_losses = [p[2] for p in plot_data]
    win_rates_initial = [p[3] if p[3] is not None and p[3] >= 0 else 0.5 for p in plot_data]
    
    # Create the exact same plot as original pipeline (3-axis plot)
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot Policy Loss (Axis 1) - Red, left axis
    color = 'tab:red'
    ax1.set_xlabel('Iteration', fontsize=14)
    ax1.set_ylabel('Validation Policy Loss', color=color, fontsize=14)
    ax1.plot(iterations, policy_losses, color=color, marker='o', linestyle='-', linewidth=2, markersize=5, label='Policy Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')

    # Create a second y-axis for Value Loss (Axis 2) - Blue, right axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Value Loss', color=color, fontsize=14)
    ax2.plot(iterations, value_losses, color=color, marker='s', linestyle='--', linewidth=2, markersize=5, label='Value Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    # Create a third y-axis for Win Rate vs Initial (Axis 3) - Green, far right axis
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color = 'tab:green'
    ax3.set_ylabel('Win Rate vs Initial', color=color, fontsize=14)
    ax3.plot(iterations, win_rates_initial, color=color, marker='^', linestyle=':', linewidth=2, markersize=6, label='Win Rate vs Initial')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc='lower left')

    # Add title with hyperparameters (like original)
    title = f"Training Losses & Win Rate vs Initial\n"
    title += f"Optimized JAX AlphaZero - GPU Accelerated"
    plt.title(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot with same filename as original
    plot_dir = os.path.dirname(log_file_path)
    plot_filename = os.path.join(plot_dir, "training_losses.png")
    
    try:
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Learning curve saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close(fig)  # Free memory


def main():
    parser = argparse.ArgumentParser(description='Optimized JAX AlphaZero Training')
    parser.add_argument('--num_iterations', type=int, default=20,
                        help='Number of training iterations')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of self-play games per iteration')
    parser.add_argument('--game_batch_size', type=int, default=32,
                        help='Number of games to play in parallel during self-play')
    parser.add_argument('--training_batch_size', type=int, default=32,
                        help='Batch size for neural network training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs per iteration')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='checkpoints_jax_optimized',
                        help='Directory to save checkpoints')
    parser.add_argument('--asymmetric', action='store_true',
                        help='Use asymmetric game mode')
    parser.add_argument('--vertices', type=int, default=6,
                        help='Number of vertices in the graph')
    parser.add_argument('--k', type=int, default=3,
                        help='Clique size to win')
    parser.add_argument('--mcts_sims', type=int, default=50,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--experiment_name', type=str, default='optimized_jax_run',
                        help='Name for this experiment')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--use_true_mctx', action='store_true',
                        help='Use true MCTX with JAX primitives (slower but pure JAX)')
    parser.add_argument('--parallel_evaluation', action='store_true',
                        help='Use parallel evaluation for massive speedup')
    parser.add_argument('--use_validation', action='store_true',
                        help='Use validation split and early stopping (like PyTorch)')
    
    args = parser.parse_args()
    
    print(f"Starting AlphaZero with args: {args}")
    
    # Set up experiment directory (relative to current working directory)
    experiments_dir = Path("./experiments")
    experiment_dir = experiments_dir / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize
    print(f"JAX devices: {jax.devices()}")
    print(f"Using device: {jax.default_backend()}")
    
    # Configuration
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        batch_size: int = args.game_batch_size  # For self-play
        num_vertices: int = args.vertices
        k: int = args.k
        game_mode: str = "asymmetric" if args.asymmetric else "symmetric"
        mcts_simulations: int = args.mcts_sims
        temperature_threshold: int = 10
        c_puct: float = 3.0
        perspective_mode: str = "alternating"
        use_true_mctx: bool = args.use_true_mctx
    
    config = Config()
    print(f"Config created: {config}")
    
    # Create model
    print("Creating neural network (this may take 1-2 minutes for JAX compilation)...")
    model = ImprovedBatchedNeuralNetwork(
        num_vertices=config.num_vertices,  # Use config values
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=args.asymmetric
    )
    
    # Keep a copy of the initial model for evaluation
    initial_model = ImprovedBatchedNeuralNetwork(
        num_vertices=config.num_vertices,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=args.asymmetric
    )
    # Copy initial parameters
    initial_model.params = model.params
    print("Neural network created successfully!")
    
    # Create self-play instance
    self_play = OptimizedSelfPlay(config)
    
    # Initialize training log
    log_file_path = experiment_dir / "training_log.json"
    training_log = []
    
    # Store initial model params for evaluation
    initial_params = model.params
    print("Setup complete, starting training...")
    
    # Load existing log if resuming
    start_iteration = 0
    optimizer_state = None
    if args.resume_from:
        print(f"Loading checkpoint from {args.resume_from}")
        params, optimizer_state, start_iteration = load_checkpoint(args.resume_from, model)
        model.params = params
        print(f"Resumed from iteration {start_iteration}")
        
        # Load existing log
        if log_file_path.exists():
            try:
                with open(log_file_path, 'r') as f:
                    training_log = json.load(f)
                print(f"Loaded existing log with {len(training_log)} entries")
            except Exception as e:
                print(f"Could not load existing log: {e}")
    
    # Training loop
    for iteration in range(start_iteration, args.num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{args.num_iterations}")
        print(f"{'='*60}")
        
        # Self-play
        print(f"\nGenerating {args.num_episodes} self-play games...")
        start_time = time.time()
        
        game_data = self_play.play_games(model, args.num_episodes)
        
        self_play_time = time.time() - start_time
        print(f"Self-play completed in {self_play_time:.1f}s")
        print(f"Games per second: {args.num_episodes / self_play_time:.1f}")
        print(f"Total training examples: {len(game_data)}")
        
        # Train network
        print(f"\nTraining network for {args.num_epochs} epochs...")
        start_time = time.time()
        
        # Initialize variables for asymmetric losses
        attacker_policy_loss = None
        defender_policy_loss = None
        
        # Choose training function based on validation flag
        if args.use_validation:
            print("Using training with VALIDATION SPLIT and EARLY STOPPING (like PyTorch)")
            # Train with validation returns (state, policy_loss, value_loss, history)
            train_state, policy_loss, value_loss, train_history = train_network_jax_with_validation(
                model,
                game_data,
                epochs=args.num_epochs,
                batch_size=args.training_batch_size,
                learning_rate=0.001,
                initial_state=optimizer_state,
                asymmetric_mode=args.asymmetric,
                validation_split=0.1,  # 90/10 split like PyTorch
                early_stopping_patience=5,  # Same as PyTorch
                early_stopping_min_delta=0.001  # Same as PyTorch
            )
            # Store training history for analysis (if needed later)
        else:
            # Use optimized training without validation
            use_optimized_training = getattr(args, 'optimized_training', True)
            
            if use_optimized_training:
                print("Using OPTIMIZED training (JIT + vectorized batches)")
                train_fn = train_network_jax_optimized
            else:
                print("Using standard training")
                train_fn = train_network_jax
            
            # Train network - handle both symmetric and asymmetric return values
            training_result = train_fn(
                model,
                game_data,
                epochs=args.num_epochs,
                batch_size=args.training_batch_size,
                learning_rate=0.001,
                initial_state=optimizer_state,
                asymmetric_mode=args.asymmetric
            )
            
            # Unpack results based on asymmetric mode
            if args.asymmetric and len(training_result) == 5:
                train_state, policy_loss, value_loss, attacker_policy_loss, defender_policy_loss = training_result
            else:
                train_state, policy_loss, value_loss = training_result[:3]  # Handle both cases safely
        
        # Update model params and optimizer state
        model.params = train_state.params
        optimizer_state = train_state
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f}s")
        # Print final losses with asymmetric breakdown if available
        if args.asymmetric and attacker_policy_loss is not None and defender_policy_loss is not None:
            print(f"Final losses - Policy: {policy_loss:.4f} (Attacker: {attacker_policy_loss:.4f}, Defender: {defender_policy_loss:.4f}), Value: {value_loss:.4f}")
        else:
            print(f"Final losses - Policy: {policy_loss:.4f}, Value: {value_loss:.4f}")
        
        # Evaluate against initial model
        eval_config = {
            'num_games': 40 if args.asymmetric else 21,  # More games for asymmetric
            'num_vertices': args.vertices,
            'k': args.k,
            'game_mode': "asymmetric" if args.asymmetric else "symmetric",
            'mcts_sims': 30,  # Fewer simulations for faster evaluation
            'c_puct': 3.0
        }
        
        # Use enhanced evaluation for asymmetric games
        if args.asymmetric:
            if args.parallel_evaluation:
                print("Using PARALLEL asymmetric evaluation")
                eval_results = evaluate_vs_initial_and_best_asymmetric_parallel(
                    current_model=model,
                    initial_model=initial_model,
                    best_model=None,  # Could add best model tracking later
                    config=eval_config
                )
            else:
                eval_results = evaluate_vs_initial_and_best_asymmetric(
                    current_model=model,
                    initial_model=initial_model,
                    best_model=None,  # Could add best model tracking later
                    config=eval_config
                )
            win_rate_vs_initial = eval_results['win_rate_vs_initial']
            attacker_rate = eval_results['vs_initial_attacker_rate']
            defender_rate = eval_results['vs_initial_defender_rate']
            eval_time = eval_results['vs_initial_details']['eval_time']
            
            print(f"\nDetailed Asymmetric Results:")
            print(f"  As Attacker: {attacker_rate:.1%}")
            print(f"  As Defender: {defender_rate:.1%}")
        else:
            if args.parallel_evaluation:
                print("Using PARALLEL symmetric evaluation") 
                eval_results = evaluate_vs_initial_and_best_parallel(
                    current_model=model,
                    initial_model=initial_model,
                    best_model=None,
                    config=eval_config
                )
            else:
                eval_results = evaluate_vs_initial_and_best(
                    current_model=model,
                    initial_model=initial_model,
                    best_model=None,
                    config=eval_config
                )
            win_rate_vs_initial = eval_results['win_rate_vs_initial']
            eval_time = eval_results['eval_time_vs_initial']
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model, model.params, optimizer_state, 
            iteration + 1, 
            str(experiment_dir / "checkpoints")
        )
        
        # Get self-play statistics
        selfplay_stats = self_play.get_statistics()
        
        # Log metrics to training log
        iteration_metrics = {
            'iteration': iteration + 1,
            'self_play_time': self_play_time,
            'training_time': training_time,
            'eval_time': eval_time,
            'games_per_second': args.num_episodes / self_play_time,
            'total_examples': len(game_data),
            'total_time': self_play_time + training_time + eval_time,
            'validation_policy_loss': float(policy_loss),
            'validation_value_loss': float(value_loss),
            'evaluation_win_rate_vs_initial': float(win_rate_vs_initial),
            'timestamp': datetime.now().isoformat(),
            # Self-play statistics
            'selfplay_stats': {
                'avg_game_length': selfplay_stats['avg_game_length'],
                'total_games_played': selfplay_stats['total_games'],
                'game_length_distribution': selfplay_stats['length_distribution']
            }
        }
        
        # Add asymmetric training losses if available
        if args.asymmetric:
            if attacker_policy_loss is not None:
                iteration_metrics['validation_attacker_policy_loss'] = float(attacker_policy_loss)
            if defender_policy_loss is not None:
                iteration_metrics['validation_defender_policy_loss'] = float(defender_policy_loss)
        
        # Add validation history if using validation split
        if args.use_validation and 'train_history' in locals():
            iteration_metrics['training_history'] = {
                'train_policy_losses': train_history['train_policy_loss'],
                'train_value_losses': train_history['train_value_loss'],
                'val_policy_losses': train_history['val_policy_loss'],
                'val_value_losses': train_history['val_value_loss'],
                'epochs_trained': len(train_history['train_policy_loss']),
                'early_stopped': len(train_history['train_policy_loss']) < args.num_epochs
            }
            if train_history.get('val_attacker_loss'):
                iteration_metrics['training_history']['val_attacker_losses'] = train_history['val_attacker_loss']
                iteration_metrics['training_history']['val_defender_losses'] = train_history['val_defender_loss']
            if train_history.get('train_attacker_loss'):
                iteration_metrics['training_history']['train_attacker_losses'] = train_history['train_attacker_loss']
                iteration_metrics['training_history']['train_defender_losses'] = train_history['train_defender_loss']
        
        # Add asymmetric-specific metrics if applicable
        if args.asymmetric:
            iteration_metrics.update({
                'attacker_win_rate_vs_initial': float(attacker_rate),
                'defender_win_rate_vs_initial': float(defender_rate),
                'attacker_games': eval_results['vs_initial_details']['current_attacker_games'],
                'defender_games': eval_results['vs_initial_details']['current_defender_games']
            })
            # Add self-play asymmetric statistics
            iteration_metrics['selfplay_stats'].update({
                'selfplay_attacker_win_rate': selfplay_stats['win_ratio_attacker'],
                'selfplay_defender_win_rate': selfplay_stats['win_ratio_defender'],
                'selfplay_attacker_wins': selfplay_stats['attacker_wins'],
                'selfplay_defender_wins': selfplay_stats['defender_wins']
            })
        
        # Add to training log
        training_log.append(iteration_metrics)
        
        # Save training log after each iteration
        try:
            with open(log_file_path, 'w') as f:
                json.dump(training_log, f, indent=2)
            print(f"Updated training log: {log_file_path}")
        except Exception as e:
            print(f"Error saving training log: {e}")
        
        # Generate plots after each iteration
        try:
            if len(training_log) >= 1:  # Plot even from iteration 1
                plot_learning_curve(str(log_file_path))
                
                # Generate enhanced plots for asymmetric mode
                if args.asymmetric:
                    from plot_asymmetric_metrics import plot_asymmetric_learning_curves, plot_role_balance
                    plot_asymmetric_learning_curves(str(log_file_path))
                    plot_role_balance(str(log_file_path))
        except Exception as e:
            print(f"Error generating plots: {e}")
        
        # Also save individual metrics file for compatibility
        metrics_path = experiment_dir / f"metrics_iter_{iteration + 1}.json"
        with open(metrics_path, 'w') as f:
            json.dump(iteration_metrics, f, indent=2)
        
        print(f"\nIteration {iteration + 1} completed!")
        print(f"Self-play: {self_play_time:.1f}s ({args.num_episodes / self_play_time:.1f} games/sec)")
        print(f"Training: {training_time:.1f}s")
        print(f"Total time: {self_play_time + training_time:.1f}s")
    
    # Final summary and plots
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    
    if training_log:
        # Calculate summary statistics
        total_self_play_time = sum(entry['self_play_time'] for entry in training_log)
        total_training_time = sum(entry['training_time'] for entry in training_log)
        total_examples = sum(entry['total_examples'] for entry in training_log)
        avg_games_per_sec = sum(entry['games_per_second'] for entry in training_log) / len(training_log)
        
        print(f"\n📊 TRAINING SUMMARY:")
        print(f"Total iterations: {len(training_log)}")
        print(f"Total self-play time: {total_self_play_time:.1f}s")
        print(f"Total training time: {total_training_time:.1f}s")
        print(f"Total training examples: {total_examples}")
        print(f"Average speed: {avg_games_per_sec:.1f} games/sec")
        
        # Show speedup comparison
        if len(training_log) >= 2:
            first_speed = training_log[0]['games_per_second']
            last_speed = training_log[-1]['games_per_second']
            speedup = last_speed / first_speed if first_speed > 0 else 1.0
            print(f"Speedup (last vs first): {speedup:.1f}x")
        
        # Generate final comprehensive plots
        try:
            plot_learning_curve(str(log_file_path))
            print(f"\n📈 Final plots saved to: {experiment_dir / 'training_losses.png'}")
        except Exception as e:
            print(f"Error generating final plots: {e}")
    
    print(f"\nExperiment results saved to: {experiment_dir}")
    print(f"Training log: {log_file_path}")
    print("✓ Optimized AlphaZero training complete!")


if __name__ == "__main__":
    main()
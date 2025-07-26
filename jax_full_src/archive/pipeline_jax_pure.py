#!/usr/bin/env python
"""
Pure JAX AlphaZero Pipeline - No PyTorch Dependencies
"""

import os
import sys
import time
import json
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# Local imports
from vectorized_self_play import VectorizedSelfPlay, SelfPlayConfig
from vectorized_nn import BatchedNeuralNetwork
from train_jax import train_network_jax, save_model_jax, load_model_jax, TrainState
from evaluation_jax import evaluate_against_mcts_jax, evaluate_head_to_head_jax


class PureJAXAlphaZeroPipeline:
    """Complete AlphaZero pipeline using only JAX."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = jax.devices()[0]
        
        # Setup directories
        self.setup_directories()
        
        # Save configuration
        with open(os.path.join(self.dirs['root'], 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize components
        self._initialize_components()
        
        # Training history
        self.training_history = {
            'iteration': [],
            'self_play_time': [],
            'training_time': [],
            'evaluation_time': [],
            'games_per_second': [],
            'win_rate_vs_mcts': [],
            'win_rate_vs_initial': [],
            'policy_loss': [],
            'value_loss': []
        }
        
        # Log data
        self.log_data = {
            'hyperparameters': config,
            'log': []
        }
        
        print(f"Initialized Pure JAX Pipeline on {self.device}")
    
    def setup_directories(self):
        """Create experiment directories."""
        self.dirs = {
            'root': f"experiments/{self.config['experiment_name']}",
            'models': f"experiments/{self.config['experiment_name']}/models",
            'self_play': f"experiments/{self.config['experiment_name']}/self_play_data",
            'plots': f"experiments/{self.config['experiment_name']}/plots"
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize neural networks and self-play system."""
        # JAX neural network
        self.nn_model = BatchedNeuralNetwork(
            num_vertices=self.config['num_vertices'],
            hidden_dim=self.config['hidden_dim']
        )
        
        # Training state (will be initialized in first training)
        self.train_state = None
        
        # Self-play configuration
        self.self_play_config = SelfPlayConfig(
            batch_size=self.config['batch_size'],
            num_vertices=self.config['num_vertices'],
            k=self.config['k'],
            game_mode=self.config['game_mode'],
            mcts_simulations=self.config['mcts_simulations'],
            temperature_threshold=self.config['temperature_threshold'],
            c_puct=self.config['c_puct']
        )
        
        # Initialize self-play system
        self.self_play = VectorizedSelfPlay(self.self_play_config, self.nn_model)
        
        # Save initial model
        self._save_initial_model()
    
    def _save_initial_model(self):
        """Save the initial random model."""
        initial_path = os.path.join(self.dirs['models'], 'model_iter0.pkl')
        if not os.path.exists(initial_path):
            # Create dummy state for initial model
            rng = jax.random.PRNGKey(42)
            from train_jax import create_train_state
            initial_state = create_train_state(self.nn_model, 0.001, rng)
            save_model_jax(initial_state, initial_path)
            print(f"Saved initial model to: {initial_path}")
    
    def run_self_play(self, num_games: int, iteration: int) -> Tuple[List, float]:
        """Run vectorized self-play to generate training data."""
        print(f"\n--- Self-Play (Iteration {iteration}) ---")
        print(f"Generating {num_games} games using TRUE parallel self-play")
        print(f"Batch size: {self.config['batch_size']} games in parallel")
        
        start_time = time.time()
        
        # Generate games in batches
        all_experiences = self.self_play.play_games(num_games, verbose=True)
        
        elapsed = time.time() - start_time
        games_per_second = num_games / elapsed
        
        # Flatten experiences
        flattened_experiences = []
        for game_experiences in all_experiences:
            flattened_experiences.extend(game_experiences)
        
        print(f"\nSelf-play complete:")
        print(f"  Total games: {len(all_experiences)}")
        print(f"  Total positions: {len(flattened_experiences)}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Performance: {games_per_second:.1f} games/second")
        print(f"  Speedup: {games_per_second/0.25:.0f}x vs CPU baseline")
        
        # Save experiences
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"selfplay_iter{iteration}_{timestamp}.pkl"
        filepath = os.path.join(self.dirs['self_play'], filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(flattened_experiences, f)
        
        print(f"  Saved to: {filename}")
        
        return flattened_experiences, games_per_second
    
    def train_network(self, experiences: List[Dict], iteration: int) -> Tuple[float, float, float]:
        """Train the neural network using JAX."""
        print(f"\n--- Training (Iteration {iteration}) ---")
        
        start_time = time.time()
        
        # Train the network
        self.train_state, avg_policy_loss, avg_value_loss = train_network_jax(
            self.nn_model,
            experiences,
            epochs=self.config['epochs_per_iteration'],
            batch_size=self.config['training_batch_size'],
            learning_rate=self.config['learning_rate'],
            verbose=True
        )
        
        elapsed = time.time() - start_time
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.dirs['models'], f'model_iter{iteration}.pkl')
        save_model_jax(self.train_state, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Update neural network with new parameters
        self.self_play.nn.params = self.train_state.params
        
        return elapsed, avg_policy_loss, avg_value_loss
    
    def evaluate_model(self, iteration: int) -> Tuple[float, float, float]:
        """Evaluate current model performance."""
        print(f"\n--- Evaluation (Iteration {iteration}) ---")
        
        start_time = time.time()
        
        # Evaluate against pure MCTS
        print("1. Evaluating against pure MCTS...")
        win_rate_mcts = evaluate_against_mcts_jax(
            self.train_state,
            self.nn_model,
            num_games=self.config['eval_games'],
            mcts_simulations=self.config['eval_mcts_simulations']
        )
        print(f"   Win rate vs MCTS: {win_rate_mcts:.1%}")
        
        # Evaluate against initial model
        win_rate_initial = 0.5  # Default
        print("2. Evaluating against initial model...")
        initial_path = os.path.join(self.dirs['models'], 'model_iter0.pkl')
        if os.path.exists(initial_path):
            initial_state = load_model_jax(self.nn_model, initial_path)
            win_rate_initial = evaluate_head_to_head_jax(
                self.train_state,
                initial_state,
                self.nn_model,
                num_games=self.config['eval_games']
            )
            print(f"   Win rate vs initial: {win_rate_initial:.1%}")
        
        elapsed = time.time() - start_time
        print(f"Evaluation time: {elapsed:.1f}s")
        
        return win_rate_mcts, win_rate_initial, elapsed
    
    def plot_learning_curves(self):
        """Plot learning curves."""
        if len(self.training_history['iteration']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Training Progress - {self.config["experiment_name"]}', fontsize=16)
        
        # Plot 1: Win rates
        ax = axes[0, 0]
        ax.plot(self.training_history['iteration'], self.training_history['win_rate_vs_mcts'], 
                'b-', label='vs MCTS', marker='o')
        ax.plot(self.training_history['iteration'], self.training_history['win_rate_vs_initial'], 
                'r-', label='vs Initial', marker='^')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Win Rate')
        ax.set_title('Evaluation Win Rates')
        ax.legend()
        ax.grid(True)
        
        # Plot 2: Losses
        ax = axes[0, 1]
        ax.plot(self.training_history['iteration'], self.training_history['policy_loss'], 
                'b-', label='Policy Loss', marker='o')
        ax.plot(self.training_history['iteration'], self.training_history['value_loss'], 
                'r-', label='Value Loss', marker='s')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True)
        
        # Plot 3: Performance
        ax = axes[1, 0]
        ax.plot(self.training_history['iteration'], self.training_history['games_per_second'], 
                'g-', marker='o')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Games/Second')
        ax.set_title('Self-Play Performance')
        ax.grid(True)
        
        # Plot 4: Time breakdown
        ax = axes[1, 1]
        width = 0.35
        x = np.array(self.training_history['iteration'])
        ax.bar(x - width/2, self.training_history['self_play_time'], width, label='Self-Play')
        ax.bar(x + width/2, self.training_history['training_time'], width, label='Training')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Time per Component')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.dirs['root'], 'learning_curves.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Learning curves saved to {plot_path}")
    
    def run_iteration(self, iteration: int):
        """Run one complete iteration."""
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")
        
        iteration_start = time.time()
        
        # Self-play
        experiences, games_per_sec = self.run_self_play(
            self.config['games_per_iteration'],
            iteration
        )
        
        # Training
        train_time, policy_loss, value_loss = self.train_network(experiences, iteration)
        
        # Evaluation
        win_rate_mcts, win_rate_initial, eval_time = self.evaluate_model(iteration)
        
        # Update history
        self.training_history['iteration'].append(iteration)
        self.training_history['self_play_time'].append(len(experiences) / games_per_sec)
        self.training_history['training_time'].append(train_time)
        self.training_history['evaluation_time'].append(eval_time)
        self.training_history['games_per_second'].append(games_per_sec)
        self.training_history['win_rate_vs_mcts'].append(win_rate_mcts)
        self.training_history['win_rate_vs_initial'].append(win_rate_initial)
        self.training_history['policy_loss'].append(policy_loss)
        self.training_history['value_loss'].append(value_loss)
        
        # Update log data
        iteration_metrics = {
            'iteration': iteration,
            'self_play_time': self.training_history['self_play_time'][-1],
            'training_time': train_time,
            'evaluation_time': eval_time,
            'games_per_second': games_per_sec,
            'total_games': len(experiences),
            'win_rate_vs_mcts': win_rate_mcts,
            'win_rate_vs_initial': win_rate_initial,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'timestamp': datetime.now().isoformat()
        }
        self.log_data['log'].append(iteration_metrics)
        
        # Save JSON log
        log_path = os.path.join(self.dirs['root'], 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.log_data, f, indent=2)
        
        # Save plots
        self.plot_learning_curves()
        
        iteration_time = time.time() - iteration_start
        print(f"\nIteration {iteration} completed in {iteration_time:.1f}s")
    
    def run(self):
        """Run the complete training pipeline."""
        print("\n" + "="*70)
        print("PURE JAX ALPHAZERO PIPELINE - CLIQUE GAME")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config['batch_size']} parallel games")
        print(f"MCTS simulations: {self.config['mcts_simulations']}")
        print(f"Training with JAX throughout - no PyTorch!")
        print("="*70)
        
        for iteration in range(self.config['num_iterations']):
            self.run_iteration(iteration)
            
            # Check early stopping
            if (self.training_history['win_rate_vs_mcts'][-1] > 
                self.config['target_win_rate']):
                print(f"\nTarget win rate achieved! ({self.config['target_win_rate']:.1%})")
                break
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print(f"Final performance: {self.training_history['games_per_second'][-1]:.1f} games/sec")
        print(f"Final win rate vs MCTS: {self.training_history['win_rate_vs_mcts'][-1]:.1%}")
        print("="*70)


if __name__ == "__main__":
    # Configuration
    config = {
        'experiment_name': 'pure_jax_3iter',
        'num_iterations': 3,
        'games_per_iteration': 100,
        'batch_size': 32,
        'mcts_simulations': 50,
        'temperature_threshold': 10,
        'c_puct': 1.0,
        
        # Game settings
        'num_vertices': 6,
        'k': 3,
        'game_mode': 'asymmetric',
        
        # Network settings
        'hidden_dim': 64,
        'epochs_per_iteration': 10,
        'training_batch_size': 32,
        'learning_rate': 0.001,
        
        # Evaluation settings
        'eval_games': 10,
        'eval_mcts_simulations': 50,
        'target_win_rate': 0.8
    }
    
    # Run pipeline
    pipeline = PureJAXAlphaZeroPipeline(config)
    pipeline.run()
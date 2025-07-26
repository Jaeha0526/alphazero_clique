#!/usr/bin/env python
"""
Simplified training script that demonstrates the working pipeline
"""

import warnings
warnings.filterwarnings('ignore', message='.*DataLoader.*deprecated.*')

import os
os.environ['DISABLE_WANDB'] = 'true'  # Disable wandb to avoid hanging
os.environ['JAX_PLATFORM_NAME'] = 'gpu'  # Force GPU

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from pipeline_vectorized import VectorizedAlphaZeroPipeline

# Standard training configuration (reduced for demonstration)
config = {
    'experiment_name': 'standard_training_3iter',
    'num_iterations': 3,
    'games_per_iteration': 50,  # Reduced from 100
    'batch_size': 16,  # Reduced from 32
    'mcts_simulations': 25,  # Reduced from 50
    'temperature_threshold': 10,
    'c_puct': 1.0,
    
    # Game settings
    'num_vertices': 6,
    'k': 3,
    'game_mode': 'asymmetric',
    
    # Network settings
    'hidden_dim': 64,
    'num_layers': 3,
    'epochs_per_iteration': 5,  # Reduced from 10
    'training_batch_size': 32,
    'learning_rate': 0.001,
    
    # Evaluation settings
    'eval_games': 5,  # Reduced from 10
    'eval_mcts_simulations': 25,  # Reduced from 50
    'target_win_rate': 0.8
}

print("="*70)
print("STANDARD TRAINING RUN - 3 ITERATIONS")
print("="*70)
print(f"Configuration:")
print(f"  - Games per iteration: {config['games_per_iteration']}")
print(f"  - Batch size: {config['batch_size']} (parallel games)")
print(f"  - MCTS simulations: {config['mcts_simulations']}")
print(f"  - Training epochs: {config['epochs_per_iteration']}")
print(f"  - Evaluation games: {config['eval_games']}")
print("="*70)

# Create and run pipeline
pipeline = VectorizedAlphaZeroPipeline(config)
pipeline.run()

print("\nTraining completed successfully!")
print(f"Results saved in: experiments/{config['experiment_name']}/")
print("  - Models: experiments/{config['experiment_name']}/models/")
print("  - Logs: experiments/{config['experiment_name']}/training_log.json")
print("  - Plots: experiments/{config['experiment_name']}/learning_curves.png")
#!/usr/bin/env python
"""
Working pipeline execution with all features
"""

import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['DISABLE_WANDB'] = 'true'

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# Import after paths are set
from pipeline_vectorized import VectorizedAlphaZeroPipeline

def main():
    # Configuration for 3 iterations
    config = {
        'experiment_name': 'working_3iter',
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
        'num_layers': 3,
        'epochs_per_iteration': 10,
        'training_batch_size': 32,
        'learning_rate': 0.001,
        
        # Evaluation settings
        'eval_games': 10,
        'eval_mcts_simulations': 50,
        'target_win_rate': 0.8
    }
    
    print("="*70)
    print("JAX VECTORIZED ALPHAZERO - 3 ITERATION TRAINING")
    print("="*70)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Games per iteration: {config['games_per_iteration']}")
    print(f"Parallel batch size: {config['batch_size']}")
    print(f"MCTS simulations: {config['mcts_simulations']}")
    print(f"Expected time: ~5-10 minutes")
    print("="*70)
    
    # Create and run pipeline
    try:
        pipeline = VectorizedAlphaZeroPipeline(config)
        pipeline.run()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Results saved in: experiments/{config['experiment_name']}/")
        print(f"- Models: experiments/{config['experiment_name']}/models/")
        print(f"- Training log: experiments/{config['experiment_name']}/training_log.json")
        print(f"- Learning curves: experiments/{config['experiment_name']}/learning_curves.png")
        print(f"- Self-play data: experiments/{config['experiment_name']}/self_play_data/")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
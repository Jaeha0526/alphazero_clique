#!/usr/bin/env python
"""
Simple test of the pipeline with minimal settings
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from pipeline_vectorized import VectorizedAlphaZeroPipeline

# Minimal config
config = {
    'experiment_name': 'test_simple',
    'num_iterations': 1,
    'games_per_iteration': 4,
    'batch_size': 2,
    'mcts_simulations': 5,
    'temperature_threshold': 5,
    'c_puct': 1.0,
    'num_vertices': 6,
    'k': 3,
    'game_mode': 'asymmetric',
    'hidden_dim': 32,
    'num_layers': 2,
    'epochs_per_iteration': 2,
    'training_batch_size': 16,
    'learning_rate': 0.001,
    'eval_games': 1,
    'eval_mcts_simulations': 5,
    'target_win_rate': 0.95
}

print("Creating pipeline...")
pipeline = VectorizedAlphaZeroPipeline(config)

print("\nRunning self-play only...")
experiences, games_per_sec = pipeline.run_self_play(4, 0)
print(f"Generated {len(experiences)} experiences")

print("\nTraining network...")
try:
    train_time, policy_loss, value_loss = pipeline.train_network(experiences, 0)
    print(f"Training completed in {train_time:.1f}s")
    print(f"Losses: policy={policy_loss:.4f}, value={value_loss:.4f}")
except Exception as e:
    print(f"Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\nRunning evaluation...")
try:
    win_rate_mcts, win_rate_prev, win_rate_initial, eval_time = pipeline.evaluate_model(0)
    print(f"Evaluation completed in {eval_time:.1f}s")
    print(f"Win rates: vs MCTS={win_rate_mcts:.1%}, vs initial={win_rate_initial:.1%}")
except Exception as e:
    print(f"Evaluation failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed!")
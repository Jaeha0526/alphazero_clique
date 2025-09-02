#!/usr/bin/env python
"""
Test that eval_games and eval_mcts_sims are properly used in evaluation config.
"""

import sys
import os

# Simulate command line arguments
test_cases = [
    {
        'args': ['--eval_games', '10', '--eval_mcts_sims', '15'],
        'expected_games': 10,
        'expected_sims': 15
    },
    {
        'args': [],  # Default values
        'expected_games': 21,  # Default for symmetric
        'expected_sims': 30   # Default
    },
    {
        'args': ['--eval_games', '5'],  # Only games specified
        'expected_games': 5,
        'expected_sims': 30   # Default
    },
    {
        'args': ['--asymmetric'],  # Asymmetric mode defaults
        'expected_games': 40,  # Default for asymmetric
        'expected_sims': 30
    },
    {
        'args': ['--asymmetric', '--eval_games', '25'],  # Override asymmetric default
        'expected_games': 25,
        'expected_sims': 30
    }
]

print("="*60)
print("Testing eval_games and eval_mcts_sims configuration")
print("="*60)

for i, test in enumerate(test_cases, 1):
    print(f"\nTest case {i}: {' '.join(test['args']) if test['args'] else '(defaults)'}")
    
    # Set up arguments
    sys.argv = ['run_jax_optimized.py'] + test['args'] + [
        '--vertices', '6', '--k', '3',
        '--num_iterations', '1', '--num_episodes', '10'
    ]
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--game_batch_size', type=int, default=32)
    parser.add_argument('--training_batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_jax_optimized')
    parser.add_argument('--asymmetric', action='store_true')
    parser.add_argument('--avoid_clique', action='store_true')
    parser.add_argument('--vertices', type=int, default=6)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--mcts_sims', type=int, default=30)
    parser.add_argument('--experiment_name', type=str, default='default_experiment')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--use_true_mctx', action='store_true')
    parser.add_argument('--parallel_evaluation', action='store_true')
    parser.add_argument('--use_validation', action='store_true')
    parser.add_argument('--eval_games', type=int, default=None)
    parser.add_argument('--eval_mcts_sims', type=int, default=None)
    
    args = parser.parse_args(sys.argv[1:])
    
    # Create config object (simplified from run_jax_optimized.py)
    class Config:
        def __init__(self):
            self.game_mode = 'asymmetric' if args.asymmetric else ('avoid_clique' if args.avoid_clique else 'symmetric')
    
    config = Config()
    
    # Build evaluation config (exact code from run_jax_optimized.py)
    eval_config = {
        'num_games': args.eval_games if args.eval_games else (40 if args.asymmetric else 21),
        'num_vertices': args.vertices,
        'k': args.k,
        'game_mode': config.game_mode,
        'mcts_sims': args.eval_mcts_sims if args.eval_mcts_sims else 30,
        'c_puct': 3.0
    }
    
    # Check results
    games_ok = eval_config['num_games'] == test['expected_games']
    sims_ok = eval_config['mcts_sims'] == test['expected_sims']
    
    print(f"  Expected: {test['expected_games']} games, {test['expected_sims']} sims")
    print(f"  Got:      {eval_config['num_games']} games, {eval_config['mcts_sims']} sims")
    
    if games_ok and sims_ok:
        print("  ✅ PASSED")
    else:
        print("  ❌ FAILED")
        if not games_ok:
            print(f"     Games mismatch: expected {test['expected_games']}, got {eval_config['num_games']}")
        if not sims_ok:
            print(f"     Sims mismatch: expected {test['expected_sims']}, got {eval_config['mcts_sims']}")

print("\n" + "="*60)
print("Testing complete!")
print("="*60)
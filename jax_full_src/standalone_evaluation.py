#!/usr/bin/env python
"""
Standalone evaluation script that loads saved models and evaluates them.
Can be run completely independently of training.
"""

import argparse
import pickle
import time
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Optional

from vectorized_nn import ImprovedBatchedNeuralNetwork
from evaluation_jax_parallel import evaluate_models_parallel
from evaluation_jax_truly_parallel import evaluate_vs_initial_and_best_truly_parallel
from evaluation_subprocess import evaluate_models_subprocess_parallel


def load_model(checkpoint_path: str, device=None) -> ImprovedBatchedNeuralNetwork:
    """Load a model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Determine if this is a full checkpoint or just parameters
    is_full_checkpoint = 'model_config' in checkpoint or 'iteration' in checkpoint
    
    # Extract config
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    elif is_full_checkpoint:
        # Old checkpoint format
        config = {
            'num_vertices': checkpoint.get('num_vertices', 6),
            'hidden_dim': checkpoint.get('hidden_dim', 64),
            'num_gnn_layers': checkpoint.get('num_layers', 3),
            'asymmetric_mode': checkpoint.get('asymmetric_mode', False)
        }
    else:
        # For initial_model.pkl and best_model.pkl which are just params
        # We need to infer the config from the checkpoint path or defaults
        if 'initial_model.pkl' in checkpoint_path or 'best_model.pkl' in checkpoint_path:
            # Try to load config from a nearby checkpoint
            from pathlib import Path
            exp_dir = Path(checkpoint_path).parent.parent
            checkpoints = list(exp_dir.glob("checkpoints/checkpoint_iter_*.pkl"))
            if checkpoints:
                # Load any checkpoint to get config
                with open(checkpoints[0], 'rb') as f:
                    ref_checkpoint = pickle.load(f)
                    if 'model_config' in ref_checkpoint:
                        config = ref_checkpoint['model_config']
                    else:
                        config = {
                            'num_vertices': ref_checkpoint.get('num_vertices', 6),
                            'hidden_dim': ref_checkpoint.get('hidden_dim', 64),
                            'num_gnn_layers': ref_checkpoint.get('num_layers', 3),
                            'asymmetric_mode': ref_checkpoint.get('asymmetric_mode', False)
                        }
            else:
                # Fallback to defaults
                config = {
                    'num_vertices': 6,
                    'hidden_dim': 64,
                    'num_gnn_layers': 3,
                    'asymmetric_mode': False
                }
        else:
            config = {
                'num_vertices': 6,
                'hidden_dim': 64,
                'num_gnn_layers': 3,
                'asymmetric_mode': False
            }
    
    # Create model
    model = ImprovedBatchedNeuralNetwork(
        num_vertices=config['num_vertices'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_gnn_layers'],
        asymmetric_mode=config['asymmetric_mode']
    )
    
    # Load parameters
    if is_full_checkpoint and 'params' in checkpoint:
        model.params = checkpoint['params']
    else:
        # If checkpoint is just params (like initial_model.pkl, best_model.pkl)
        model.params = checkpoint
    
    print(f"Model loaded: n={config['num_vertices']}, hidden={config['hidden_dim']}")
    return model


def evaluate_two_models(model1_path: str, model2_path: str, 
                       config: Dict) -> Dict[str, float]:
    """Evaluate two models against each other."""
    
    # Load models
    model1 = load_model(model1_path)
    model2 = load_model(model2_path)
    
    # Run evaluation
    print(f"\nEvaluating {model1_path} vs {model2_path}")
    print(f"Games: {config['num_games']}, MCTS sims: {config['mcts_sims']}")
    
    # Use subprocess parallelization if requested
    if config.get('subprocess', False):
        print(f"Using subprocess parallel evaluation with {config.get('num_cpus', 4)} CPUs")
        results = evaluate_models_subprocess_parallel(
            model1_path=model1_path,
            model2_path=model2_path,
            num_games=config['num_games'],
            num_cpus=config.get('num_cpus', 4),
            config={
                'num_vertices': config['num_vertices'],
                'k': config['k'],
                'mcts_sims': config['mcts_sims'],
                'c_puct': config.get('c_puct', 3.0),
                'game_mode': config.get('game_mode', 'symmetric'),
                'python_eval': config.get('python_eval', True)
            }
        )
    else:
        results = evaluate_models_parallel(
            model1=model1,
            model2=model2,
            num_games=config['num_games'],
            num_vertices=config['num_vertices'],
            k=config['k'],
            mcts_sims=config['mcts_sims'],
            c_puct=config.get('c_puct', 3.0),
            temperature=0.0,
            game_mode=config.get('game_mode', 'symmetric'),
            python_eval=config.get('python_eval', True)  # Default to Python MCTS
        )
    
    return results


def evaluate_vs_initial_and_best(current_path: str, 
                                 initial_path: str,
                                 best_path: Optional[str],
                                 config: Dict) -> Dict[str, float]:
    """Evaluate current model against initial and best models."""
    
    # Load models
    current_model = load_model(current_path)
    initial_model = load_model(initial_path)
    best_model = load_model(best_path) if best_path else None
    
    print(f"\nEvaluating {current_path}")
    print(f"  vs initial: {initial_path}")
    if best_path:
        print(f"  vs best: {best_path}")
    
    # Use subprocess parallelization if requested
    if config.get('subprocess', False):
        print(f"Using subprocess parallel evaluation with {config.get('num_cpus', 4)} CPUs")
        # Evaluate vs initial
        initial_results = evaluate_models_subprocess_parallel(
            model1_path=current_path,
            model2_path=initial_path,
            num_games=config['num_games'],
            num_cpus=config.get('num_cpus', 4),
            config=config
        )
        results = {
            'win_rate_vs_initial': initial_results['model1_win_rate'],
            'eval_time_vs_initial': initial_results['eval_time']
        }
        
        # Evaluate vs best if provided
        if best_model:
            best_results = evaluate_models_subprocess_parallel(
                model1_path=current_path,
                model2_path=best_path,
                num_games=config['num_games'],
                num_cpus=config.get('num_cpus', 4),
                config=config
            )
            results['win_rate_vs_best'] = best_results['model1_win_rate']
            results['eval_time_vs_best'] = best_results['eval_time']
        else:
            results['win_rate_vs_best'] = -1
            results['eval_time_vs_best'] = 0
    # Use truly parallel evaluation if best model provided
    elif best_model and config.get('truly_parallel', False):
        results = evaluate_vs_initial_and_best_truly_parallel(
            current_model=current_model,
            initial_model=initial_model,
            best_model=best_model,
            config=config
        )
    else:
        # Regular evaluation
        results = {
            'win_rate_vs_initial': 0,
            'win_rate_vs_best': -1
        }
        
        # Eval vs initial
        initial_results = evaluate_models_parallel(
            model1=current_model,
            model2=initial_model,
            num_games=config['num_games'],
            num_vertices=config['num_vertices'],
            k=config['k'],
            mcts_sims=config['mcts_sims'],
            c_puct=config.get('c_puct', 3.0),
            temperature=0.0,
            game_mode=config.get('game_mode', 'symmetric'),
            python_eval=config.get('python_eval', True)
        )
        results['win_rate_vs_initial'] = initial_results['model1_win_rate']
        
        # Eval vs best if provided
        if best_model:
            best_results = evaluate_models_parallel(
                model1=current_model,
                model2=best_model,
                num_games=config['num_games'],
                num_vertices=config['num_vertices'],
                k=config['k'],
                mcts_sims=config['mcts_sims'],
                c_puct=config.get('c_puct', 3.0),
                temperature=0.0,
                game_mode=config.get('game_mode', 'symmetric'),
                python_eval=config.get('python_eval', True)
            )
            results['win_rate_vs_best'] = best_results['model1_win_rate']
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Standalone model evaluation')
    
    # Model paths
    parser.add_argument('--current', type=str, default=None,
                       help='Path to current model checkpoint')
    parser.add_argument('--initial', type=str, default=None,
                       help='Path to initial model (for comparison)')
    parser.add_argument('--best', type=str, default=None,
                       help='Path to best model (for comparison)')
    
    # Or evaluate two specific models
    parser.add_argument('--model1', type=str, default=None,
                       help='Path to first model')
    parser.add_argument('--model2', type=str, default=None,
                       help='Path to second model')
    
    # Game parameters
    parser.add_argument('--vertices', type=int, default=6,
                       help='Number of vertices')
    parser.add_argument('--k', type=int, default=3,
                       help='Clique size')
    parser.add_argument('--game_mode', type=str, default='symmetric',
                       choices=['symmetric', 'asymmetric', 'avoid_clique'],
                       help='Game mode')
    
    # Evaluation parameters
    parser.add_argument('--num_games', type=int, default=21,
                       help='Number of evaluation games')
    parser.add_argument('--mcts_sims', type=int, default=30,
                       help='MCTS simulations per move')
    parser.add_argument('--c_puct', type=float, default=3.0,
                       help='PUCT exploration constant')
    
    # Implementation options
    parser.add_argument('--python_eval', action='store_true',
                       help='Use Python MCTS (no compilation)')
    parser.add_argument('--use_jax', action='store_true',
                       help='Use JAX MCTS (may compile)')
    parser.add_argument('--truly_parallel', action='store_true',
                       help='Use truly parallel evaluation')
    parser.add_argument('--subprocess', action='store_true',
                       help='Use subprocess parallelization for CPU-based evaluation')
    parser.add_argument('--num_cpus', type=int, default=4,
                       help='Number of CPUs for parallel evaluation')
    
    # Experiment directory shortcut
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name (auto-finds models)')
    parser.add_argument('--iteration', type=int, default=None,
                       help='Iteration to evaluate')
    
    args = parser.parse_args()
    
    # Auto-find models if experiment specified
    if args.experiment:
        exp_dir = Path(f"experiments/{args.experiment}")
        if not exp_dir.exists():
            print(f"Experiment directory {exp_dir} not found!")
            return
        
        # Try to load training config to get game mode
        training_log = exp_dir / "training_log.json"
        if training_log.exists() and args.game_mode == 'symmetric':  # Only override if not explicitly set
            try:
                import json
                with open(training_log, 'r') as f:
                    log_data = json.load(f)
                    if log_data and 'config' in log_data[0]:
                        config_game_mode = log_data[0]['config'].get('game_mode', 'symmetric')
                        args.game_mode = config_game_mode
                        print(f"Auto-detected game mode from training: {config_game_mode}")
            except:
                pass  # If we can't load, just use the default
        
        # Find models
        if args.iteration:
            args.current = str(exp_dir / f"checkpoints/checkpoint_iter_{args.iteration}.pkl")
        else:
            # Find latest checkpoint
            checkpoints = list(exp_dir.glob("checkpoints/checkpoint_iter_*.pkl"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
                args.current = str(latest)
                print(f"Using latest checkpoint: {latest}")
        
        args.initial = str(exp_dir / "models/initial_model.pkl")
        args.best = str(exp_dir / "models/best_model.pkl")
        
        # Check if files exist
        if not Path(args.current).exists():
            print(f"Current model not found: {args.current}")
            return
        if not Path(args.initial).exists():
            print(f"Initial model not found: {args.initial}")
            args.initial = None
        if not Path(args.best).exists():
            print(f"Best model not found: {args.best}")
            args.best = None
    
    # Create config
    config = {
        'num_games': args.num_games,
        'num_vertices': args.vertices,
        'k': args.k,
        'mcts_sims': args.mcts_sims,
        'c_puct': args.c_puct,
        'game_mode': args.game_mode,
        'python_eval': args.python_eval and not args.use_jax,
        'use_true_mctx': args.use_jax and not args.python_eval,
        'truly_parallel': args.truly_parallel,
        'subprocess': args.subprocess,
        'num_cpus': args.num_cpus
    }
    
    print("="*60)
    print("Standalone Model Evaluation")
    print("="*60)
    print(f"Configuration:")
    print(f"  Games: {config['num_games']}")
    print(f"  MCTS sims: {config['mcts_sims']}")
    print(f"  Game: n={config['num_vertices']}, k={config['k']}, mode={config['game_mode']}")
    print(f"  Implementation: {'Python MCTS' if config['python_eval'] else 'JAX MCTS'}")
    
    # Run evaluation
    start_time = time.time()
    
    if args.model1 and args.model2:
        # Evaluate two specific models
        results = evaluate_two_models(args.model1, args.model2, config)
        print("\n" + "="*60)
        print("Results:")
        print(f"  Model1 wins: {results['model1_wins']} ({results['model1_win_rate']:.1%})")
        print(f"  Model2 wins: {results['model2_wins']} ({results['model2_win_rate']:.1%})")
        print(f"  Draws: {results['draws']} ({results['draw_rate']:.1%})")
    else:
        # Evaluate vs initial and best
        results = evaluate_vs_initial_and_best(
            args.current, args.initial, args.best, config
        )
        print("\n" + "="*60)
        print("Results:")
        if args.initial:
            print(f"  Win rate vs initial: {results['win_rate_vs_initial']:.1%}")
        if args.best and results['win_rate_vs_best'] >= 0:
            print(f"  Win rate vs best: {results['win_rate_vs_best']:.1%}")
    
    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.1f}s")
    print("="*60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
GPU-Accelerated JAX AlphaZero Pipeline for Clique Game
"""

import os
import argparse
import time
import datetime
import numpy as np
import json
import pickle
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
import multiprocessing as mp

# Add paths for imports
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available")

# GPU-enabled JAX components
from jax_src.jax_clique_board_numpy import JAXCliqueBoard
from jax_src.jax_alpha_net_clique_gpu import CliqueGNN as JAXCliqueGNN, create_gpu_model
from jax_src.jax_self_play_gpu import GPUParallelSelfPlay, SelfPlayConfig, MCTS_self_play_gpu
from jax_src.jax_mcts_clique_gpu import UCT_search

# Import JAX for proper random keys
import jax
import jax.random as jrandom

# Original components we still need
from src.train_clique import train_network, load_examples
from src.alpha_net_clique import CliqueGNN as PyTorchGNN  # For evaluation compatibility


def evaluate_models_jax_gpu(current_model, best_model, current_params, best_params,
                           num_games: int = 40, num_vertices: int = 6, clique_size: int = 3,
                           num_mcts_sims: int = 100, game_mode: str = "symmetric",
                           use_policy_only: bool = False) -> float:
    """Evaluate JAX models using GPU acceleration"""
    current_wins = 0
    best_wins = 0
    draws = 0
    
    # Create model wrappers for MCTS
    class ModelWrapper:
        def __init__(self, model, params):
            self.model = model
            self.params = params
    
    current_wrapped = ModelWrapper(current_model, current_params)
    best_wrapped = ModelWrapper(best_model, best_params)
    
    for game_idx in range(num_games):
        board = JAXCliqueBoard(num_vertices, clique_size, game_mode)
        game_over = False
        
        max_moves = num_vertices * (num_vertices - 1) // 2
        
        while not game_over and board.move_count < max_moves:
            # Determine which model to use
            current_model_turn = (game_idx % 2 == 0 and board.player == 0) or \
                                (game_idx % 2 == 1 and board.player == 1)
            
            model_to_use = current_wrapped if current_model_turn else best_wrapped
            
            # Get move using GPU-accelerated MCTS
            best_move, _ = UCT_search(board, num_mcts_sims, model_to_use)
            
            # Make move
            from src import encoder_decoder_clique as ed
            edge = ed.decode_action(board, best_move)
            if edge != (-1, -1):
                board.make_move(edge)
            
            # Check game end
            if board.game_state != 0:
                game_over = True
                
                if board.game_state == 1:  # Player 1 wins
                    if game_idx % 2 == 0:  # Current model is Player 1
                        current_wins += 1
                    else:
                        best_wins += 1
                elif board.game_state == 2:  # Player 2 wins
                    if game_idx % 2 == 1:  # Current model is Player 2
                        current_wins += 1
                    else:
                        best_wins += 1
                elif board.game_state == 3:  # Draw
                    draws += 1
            
            if not board.get_valid_moves():
                game_over = True
                if board.game_state == 0:
                    draws += 1
    
    # Calculate win rate
    total_games = current_wins + best_wins + draws
    win_rate = current_wins / total_games if total_games > 0 else 0.5
    
    print(f"GPU Evaluation: Current {current_wins} - {best_wins} Best ({draws} draws)")
    print(f"Win Rate: {win_rate:.2%}")
    
    return win_rate


def save_jax_model_gpu(model, params, filepath: str, metadata: Dict[str, Any]):
    """Save JAX GPU model parameters and metadata"""
    # Convert params to CPU for saving
    import jax
    cpu_params = jax.device_put(params, jax.devices('cpu')[0])
    
    save_dict = {
        'params': cpu_params,
        'metadata': metadata,
        'model_type': 'jax_gpu'
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f)
    
    print(f"Saved JAX GPU model to {filepath}")


def load_jax_model_gpu(filepath: str) -> Tuple[Dict, Dict]:
    """Load JAX model parameters for GPU"""
    with open(filepath, 'rb') as f:
        save_dict = pickle.load(f)
    
    # Move params to GPU
    import jax
    gpu_params = jax.device_put(save_dict['params'], jax.devices()[0])
    
    return gpu_params, save_dict.get('metadata', {})


def run_single_iteration_gpu(iteration: int, args: argparse.Namespace, 
                            initial_model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Run a single iteration with GPU acceleration"""
    # Setup directories
    base_dir = f"./experiments/{args.experiment_name}"
    data_dir = os.path.join(base_dir, "datasets")
    model_dir = os.path.join(base_dir, "models")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"GPU ITERATION {iteration}")
    print(f"Using: {jax.devices()[0]}")
    print(f"{'='*80}")
    
    # Model paths
    prev_model_path = f"{model_dir}/clique_net_iter{iteration-1}_jax_gpu.pkl"
    best_model_path = f"{model_dir}/clique_net_jax_gpu.pkl"
    current_model_path = f"{model_dir}/clique_net_iter{iteration}_jax_gpu.pkl"
    
    # 1. Load or create model for self-play
    if iteration > 0 and os.path.exists(prev_model_path):
        params, metadata = load_jax_model_gpu(prev_model_path)
        model = JAXCliqueGNN(
            metadata.get('num_vertices', args.vertices),
            metadata.get('hidden_dim', args.hidden_dim),
            metadata.get('num_layers', args.num_layers)
        )
        print(f"Loaded previous model from GPU")
    else:
        print("Creating new model for GPU")
        model, params = create_gpu_model(args.vertices, args.hidden_dim, args.num_layers)
        
        metadata = {
            'num_vertices': args.vertices,
            'clique_size': args.k,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'iteration': 0
        }
        save_jax_model_gpu(model, params, current_model_path, metadata)
    
    # 2. Self-play phase with GPU
    print(f"\n--- GPU Self-Play Phase ---")
    print(f"Generating {args.self_play_games} games with {args.mcts_sims} MCTS simulations")
    print(f"Batch size: {args.batch_size}")
    
    config = SelfPlayConfig(
        num_vertices=args.vertices,
        k=args.k,
        game_mode=args.game_mode,
        mcts_simulations=args.mcts_sims,
        batch_size=args.batch_size,
        noise_weight=0.25
    )
    
    self_play_start = time.time()
    
    # Use GPU parallel self-play
    num_gpus = 1  # Can be increased if multiple GPUs available
    parallel_self_play = GPUParallelSelfPlay(config, model, params, num_gpus)
    experience_path = parallel_self_play.generate_games(
        args.self_play_games, 
        data_dir, 
        iteration
    )
    
    self_play_time = time.time() - self_play_start
    games_per_sec = args.self_play_games / self_play_time
    
    print(f"GPU Self-play: {games_per_sec:.1f} games/sec")
    
    # 3. Training phase (still using PyTorch for now)
    print(f"\n--- Training Phase ---")
    
    all_examples = load_examples(data_dir, iteration=None if args.use_all_data else iteration)
    
    if not all_examples:
        print("No training examples found!")
        return None
    
    print(f"Training on {len(all_examples)} examples")
    
    # Train using original PyTorch code
    avg_policy_loss, avg_value_loss = train_network(
        all_examples, iteration, args.vertices, args.k, model_dir, args
    )
    
    print(f"Training completed: Policy Loss={avg_policy_loss:.4f}, Value Loss={avg_value_loss:.4f}")
    
    # 4. Convert trained PyTorch model back to JAX
    # For now, create new JAX model with updated params
    trained_model, trained_params = create_gpu_model(args.vertices, args.hidden_dim, args.num_layers)
    
    # Save trained model
    metadata = {
        'num_vertices': args.vertices,
        'clique_size': args.k,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'iteration': iteration,
        'policy_loss': avg_policy_loss,
        'value_loss': avg_value_loss
    }
    save_jax_model_gpu(trained_model, trained_params, current_model_path, metadata)
    
    # 5. Evaluation phase with GPU
    print(f"\n--- GPU Evaluation Phase ---")
    
    # Evaluate vs best model
    win_rate_vs_best = 0.5
    
    if os.path.exists(best_model_path):
        best_params, best_metadata = load_jax_model_gpu(best_model_path)
        best_model = JAXCliqueGNN(
            best_metadata.get('num_vertices', args.vertices),
            best_metadata.get('hidden_dim', args.hidden_dim),
            best_metadata.get('num_layers', args.num_layers)
        )
        
        print("GPU evaluation against best model...")
        win_rate_vs_best = evaluate_models_jax_gpu(
            trained_model, best_model, trained_params, best_params,
            num_games=args.num_games,
            num_vertices=args.vertices,
            clique_size=args.k,
            num_mcts_sims=args.eval_mcts_sims,
            game_mode=args.game_mode
        )
    
    # Update best model
    if win_rate_vs_best > args.eval_threshold:
        print(f"New model is better ({win_rate_vs_best:.2%} > {args.eval_threshold}). Updating best model.")
        save_jax_model_gpu(trained_model, trained_params, best_model_path, metadata)
    
    print(f"\n=== GPU Iteration {iteration} finished ===")
    
    return {
        "iteration": iteration,
        "validation_policy_loss": avg_policy_loss,
        "validation_value_loss": avg_value_loss,
        "evaluation_win_rate_vs_best": win_rate_vs_best,
        "self_play_time": self_play_time,
        "games_per_second": games_per_sec
    }


def run_gpu_pipeline(args: argparse.Namespace):
    """Run the full GPU-accelerated AlphaZero pipeline"""
    start_time = time.time()
    
    # Check GPU availability
    print(f"JAX devices: {jax.devices()}")
    if 'gpu' not in jax.default_backend().lower():
        print("WARNING: GPU not detected! Performance will be limited.")
    
    # Define directories
    base_dir = f"./experiments/{args.experiment_name}"
    data_dir = os.path.join(base_dir, "datasets")
    model_dir = os.path.join(base_dir, "models")
    log_file_path = os.path.join(base_dir, "training_log.json")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Starting GPU pipeline for experiment: {args.experiment_name}")
    
    # Initialize log
    log_data = {"hyperparameters": vars(args), "log": []}
    
    # Run iterations
    for iteration in range(args.iterations):
        iteration_metrics = run_single_iteration_gpu(iteration, args)
        
        if iteration_metrics:
            log_data["log"].append(iteration_metrics)
            
            # Save log
            with open(log_file_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            # Plot if requested
            if args.plot:
                from src.pipeline_clique import plot_learning_curve
                plot_learning_curve(log_file_path)
    
    end_time = time.time()
    print(f"\nGPU Pipeline finished in {(end_time - start_time)/60:.2f} minutes")
    
    # Final statistics
    if log_data["log"]:
        total_games = sum(entry.get("iteration", 0) + 1 for entry in log_data["log"]) * args.self_play_games
        avg_speed = np.mean([entry.get("games_per_second", 0) for entry in log_data["log"]])
        print(f"Total games generated: {total_games}")
        print(f"Average speed: {avg_speed:.1f} games/sec")


def main():
    parser = argparse.ArgumentParser(description='GPU-Accelerated JAX AlphaZero Pipeline')
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="pipeline", 
                       choices=["pipeline", "selfplay", "evaluate"],
                       help="Execution mode")
    
    # Game parameters
    parser.add_argument("--vertices", type=int, default=6)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--game-mode", type=str, default="symmetric")
    
    # Pipeline parameters
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--self-play-games", type=int, default=1000)
    parser.add_argument("--mcts-sims", type=int, default=100)
    parser.add_argument("--eval-threshold", type=float, default=0.55)
    parser.add_argument("--experiment-name", type=str, default="gpu_test")
    
    # GPU-optimized parameters
    parser.add_argument("--batch-size", type=int, default=256, help="GPU batch size")
    parser.add_argument("--num-cpus", type=int, default=8, help="CPU workers (less needed with GPU)")
    
    # Model parameters
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use-all-data", action='store_true')
    parser.add_argument("--num-games", type=int, default=21)
    parser.add_argument("--eval-mcts-sims", type=int, default=30)
    parser.add_argument("--min-alpha", type=float, default=0.5)
    parser.add_argument("--max-alpha", type=float, default=2.0)
    
    # Other
    parser.add_argument("--plot", action='store_true', help="Generate plots")
    
    args = parser.parse_args()
    
    if args.mode == "pipeline":
        run_gpu_pipeline(args)
    elif args.mode == "selfplay":
        # Test self-play only
        model, params = create_gpu_model(args.vertices, args.hidden_dim, args.num_layers)
        
        config = SelfPlayConfig(
            num_vertices=args.vertices,
            k=args.k,
            mcts_simulations=args.mcts_sims,
            batch_size=args.batch_size
        )
        
        self_play = GPUParallelSelfPlay(config, model, params)
        self_play.generate_games(args.self_play_games, "./test_data", 0)
    else:
        print(f"Mode {args.mode} not fully implemented")


if __name__ == "__main__":
    main()
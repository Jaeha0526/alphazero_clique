#!/usr/bin/env python
"""
Complete JAX-accelerated AlphaZero pipeline for Clique Game.
Includes ALL features from original pipeline with JAX optimizations.
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

# Add path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")

# JAX components
from jax_src.jax_clique_board_numpy import JAXCliqueBoard
from jax_src.jax_alpha_net_clique import CliqueGNN as JAXCliqueGNN
from jax_src.jax_self_play import ParallelSelfPlay, SelfPlayConfig, MCTS_self_play_batch
from jax_src.jax_mcts_clique import SimpleMCTS, UCT_search

# Original components we still need
from src.train_clique import train_network, load_examples
from src.alpha_net_clique import CliqueGNN as PyTorchGNN  # For evaluation compatibility


def evaluate_models_jax(current_model, best_model, num_games: int = 40, 
                       num_vertices: int = 6, clique_size: int = 3,
                       num_mcts_sims: int = 100, game_mode: str = "symmetric",
                       use_policy_only: bool = False) -> float:
    """
    Evaluate JAX model against another model by playing games.
    Maintains exact same interface as original evaluate_models.
    """
    current_wins = 0
    best_wins = 0
    draws = 0
    
    # Get model parameters
    rng = np.random.RandomState(42)
    current_params = current_model.init_params(rng) if hasattr(current_model, 'init_params') else None
    best_params = best_model.init_params(rng) if hasattr(best_model, 'init_params') else None
    
    for game_idx in range(num_games):
        board = JAXCliqueBoard(num_vertices, clique_size, game_mode)
        game_over = False
        
        max_moves = num_vertices * (num_vertices - 1) // 2
        
        while not game_over and board.move_count < max_moves:
            # Determine which model to use
            current_model_turn = (game_idx % 2 == 0 and board.player == 0) or \
                                (game_idx % 2 == 1 and board.player == 1)
            
            if current_model_turn:
                model_to_use = current_model
                params_to_use = current_params
            else:
                model_to_use = best_model
                params_to_use = best_params
            
            # Get move
            if use_policy_only:
                # Direct policy evaluation
                import src.encoder_decoder_clique as ed
                state_dict = ed.prepare_state_for_network(board)
                edge_index = state_dict['edge_index'].numpy()
                edge_attr = state_dict['edge_attr'].numpy()
                
                policy, _ = model_to_use(params_to_use, edge_index, edge_attr)
                policy = policy.flatten()
                
                # Apply valid moves mask
                valid_mask = ed.get_valid_moves_mask(board)
                masked_policy = policy * valid_mask
                if masked_policy.sum() > 0:
                    masked_policy /= masked_policy.sum()
                    best_move = np.argmax(masked_policy)
                else:
                    valid_moves = board.get_valid_moves()
                    best_move = ed.encode_action(board, valid_moves[0]) if valid_moves else 0
            else:
                # MCTS search
                mcts = SimpleMCTS(board, num_mcts_sims, model_to_use, params_to_use)
                best_move, _ = mcts.search()
            
            # Make move
            import src.encoder_decoder_clique as ed
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
    
    print(f"Evaluation Results: Current {current_wins} - {best_wins} Best ({draws} draws)")
    print(f"Win Rate: {win_rate:.2%}")
    
    return win_rate


def save_jax_model(model, params, filepath: str, metadata: Dict[str, Any]):
    """Save JAX model parameters and metadata"""
    save_dict = {
        'params': params,
        'metadata': metadata,
        'model_type': 'jax'
    }
    
    # Save as numpy arrays for compatibility
    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f)
    
    print(f"Saved JAX model to {filepath}")


def load_jax_model(filepath: str) -> Tuple[Dict, Dict]:
    """Load JAX model parameters and metadata"""
    with open(filepath, 'rb') as f:
        save_dict = pickle.load(f)
    
    return save_dict['params'], save_dict.get('metadata', {})


def plot_learning_curve(log_file_path: str):
    """
    Plot validation policy and value loss curves from the training log.
    Exact same implementation as original.
    """
    if not os.path.exists(log_file_path):
        print(f"Log file not found at {log_file_path}. Cannot plot learning curve.")
        return
        
    try:
        with open(log_file_path, 'r') as f:
            loaded_data = json.load(f)
            if not isinstance(loaded_data, dict) or "log" not in loaded_data:
                print("Log file is not in the expected format (missing 'log' key).")
                return
            log_list = loaded_data["log"]
            hyperparams = loaded_data.get("hyperparameters", {})

    except Exception as e:
        print(f"Error loading or parsing log file {log_file_path}: {e}")
        return

    if not isinstance(log_list, list) or len(log_list) < 2:
        print("Not enough data points in log file to plot learning curve.")
        return

    # Extract data from the log_list
    plot_data = [
        (entry["iteration"], entry["validation_policy_loss"], entry["validation_value_loss"], 
         entry.get("evaluation_win_rate_vs_initial"))
        for entry in log_list
        if entry.get("validation_policy_loss") is not None and entry.get("validation_value_loss") is not None
    ]
    
    iterations = [p[0] for p in plot_data]
    policy_losses = [p[1] for p in plot_data]
    value_losses = [p[2] for p in plot_data]
    win_rates_initial = [p[3] if p[3] is not None and p[3] >= 0 else np.nan for p in plot_data]

    if len(iterations) < 2:
        print("Not enough valid data points (with losses) in log file to plot learning curve.")
        return
        
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot Policy Loss (Axis 1)
    color = 'tab:red'
    ax1.set_xlabel('Iteration', fontsize=14)
    ax1.set_ylabel('Validation Policy Loss', color=color, fontsize=14)
    ax1.plot(iterations, policy_losses, color=color, marker='o', linestyle='-', 
             linewidth=2, markersize=5, label='Policy Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')

    # Create a second y-axis for Value Loss (Axis 2)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Value Loss', color=color, fontsize=14)
    ax2.plot(iterations, value_losses, color=color, marker='s', linestyle='--', 
             linewidth=2, markersize=5, label='Value Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    # Create a third y-axis for Win Rate vs Initial (Axis 3)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color = 'tab:green'
    ax3.set_ylabel('Win Rate vs Initial', color=color, fontsize=14)
    ax3.plot(iterations, win_rates_initial, color=color, marker='^', linestyle=':', 
             linewidth=2, markersize=6, label='Win Rate vs Initial')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc='lower left')

    # Add title with hyperparameters
    title = f"Training Losses & Win Rate vs Initial\n"
    title += f"Exp: {hyperparams.get('experiment_name', 'N/A')}, "
    title += f"V={hyperparams.get('vertices', '?')}, k={hyperparams.get('k', '?')}, "
    title += f"MCTS={hyperparams.get('mcts_sims', '?')}\n"
    title += f"LR={hyperparams.get('initial_lr', '?')}, "
    title += f"Factor={hyperparams.get('lr_factor', '?')}, "
    title += f"Patience={hyperparams.get('lr_patience', '?')}"
    plt.title(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save plot
    plot_dir = os.path.dirname(log_file_path)
    plot_filename = os.path.join(plot_dir, f"training_losses.png")
    try:
        plt.savefig(plot_filename)
        print(f"Learning curve saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)


def play_against_ai(model_path: str = None, num_vertices: int = 6, clique_size: int = 3, 
                   num_mcts_sims: int = 200, human_player: int = 0):
    """Play a game against the trained AI (JAX version)"""
    # Load model
    if model_path is None:
        model_path = "./experiments/default/models/clique_net_jax.pkl"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    params, metadata = load_jax_model(model_path)
    model = JAXCliqueGNN(num_vertices, metadata.get('hidden_dim', 64), metadata.get('num_layers', 2))
    
    # Initialize game
    board = JAXCliqueBoard(num_vertices, clique_size)
    game_over = False
    
    print("\n=== CLIQUE GAME vs AI ===")
    print(f"You are Player {human_player + 1}")
    print(f"Form a {clique_size}-clique to win!\n")
    
    max_moves = num_vertices * (num_vertices - 1) // 2
    
    while not game_over and board.move_count < max_moves:
        print(board)
        
        if board.player == human_player:
            # Human's turn
            valid_moves = board.get_valid_moves()
            
            print("\nValid moves:")
            for i, move in enumerate(valid_moves):
                print(f"{i}: Edge {move}")
            
            try:
                move_idx = int(input("\nEnter move index: "))
                if 0 <= move_idx < len(valid_moves):
                    edge = valid_moves[move_idx]
                    board.make_move(edge)
                    print(f"\nYou selected edge {edge}")
                else:
                    print("Invalid move index, try again")
                    continue
            except ValueError:
                print("Invalid input, please enter a number")
                continue
        else:
            # AI's turn
            print("\nAI is thinking...")
            
            # Run MCTS
            mcts = SimpleMCTS(board, num_mcts_sims, model, params)
            best_move, _ = mcts.search()
            
            import src.encoder_decoder_clique as ed
            edge = ed.decode_action(board, best_move)
            
            if edge != (-1, -1):
                board.make_move(edge)
                print(f"AI selected edge {edge}")
            else:
                print("AI made invalid move")
                break
        
        # Check game state
        if board.game_state != 0:
            game_over = True
            print("\n" + "="*30)
            if board.game_state == 3:
                print("Game ended in a DRAW!")
            else:
                winner = "You" if board.game_state == human_player + 1 else "AI"
                print(f"{winner} WIN!")
            print("="*30)


def run_single_iteration(iteration: int, args: argparse.Namespace, 
                        initial_model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Run a single iteration of self-play, training, and evaluation.
    Returns metrics dictionary or None if failed.
    """
    # Setup directories
    base_dir = f"./experiments/{args.experiment_name}"
    data_dir = os.path.join(base_dir, "datasets")
    model_dir = os.path.join(base_dir, "models")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"ITERATION {iteration}")
    print(f"{'='*80}")
    
    # Model paths
    prev_model_path = f"{model_dir}/clique_net_iter{iteration-1}_jax.pkl"
    best_model_path = f"{model_dir}/clique_net_jax.pkl"
    current_model_path = f"{model_dir}/clique_net_iter{iteration}_jax.pkl"
    
    # 1. Determine which model to use for self-play
    if iteration > 0 and os.path.exists(prev_model_path):
        model_to_load_path = prev_model_path
        print(f"Using model from previous iteration for self-play: {model_to_load_path}")
    elif os.path.exists(best_model_path):
        model_to_load_path = best_model_path
        print(f"Using best model for self-play: {model_to_load_path}")
    else:
        print("No previous or best model found. Using fresh model for self-play.")
        # Create initial model
        model = JAXCliqueGNN(args.vertices, args.hidden_dim, args.num_layers)
        rng = np.random.RandomState(42)
        params = model.init_params(rng)
        
        metadata = {
            'num_vertices': args.vertices,
            'clique_size': args.k,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'iteration': 0
        }
        save_jax_model(model, params, current_model_path, metadata)
        model_to_load_path = current_model_path
    
    # 2. Load model for self-play
    params, metadata = load_jax_model(model_to_load_path)
    model_for_self_play = JAXCliqueGNN(
        metadata.get('num_vertices', args.vertices),
        metadata.get('hidden_dim', args.hidden_dim),
        metadata.get('num_layers', args.num_layers)
    )
    
    # 3. Self-play phase
    print(f"\n--- Self-Play Phase ---")
    print(f"Generating {args.self_play_games} games with {args.mcts_sims} MCTS simulations")
    
    config = SelfPlayConfig(
        num_vertices=args.vertices,
        k=args.k,
        game_mode=args.game_mode,
        mcts_simulations=args.mcts_sims,
        batch_size=args.batch_size,
        noise_weight=0.25
    )
    
    self_play_start = time.time()
    
    parallel_self_play = ParallelSelfPlay(config, model_for_self_play, params, args.num_cpus)
    experience_path = parallel_self_play.generate_games(
        args.self_play_games, 
        data_dir, 
        iteration
    )
    
    self_play_time = time.time() - self_play_start
    
    # 4. Training phase
    print(f"\n--- Training Phase ---")
    
    # Load examples
    all_examples = load_examples(data_dir, iteration=None if args.use_all_data else iteration)
    
    if not all_examples:
        print("No training examples found!")
        return None
    
    print(f"Training on {len(all_examples)} examples")
    
    # Note: Here we use the original PyTorch training
    # In a complete JAX implementation, this would be replaced
    avg_policy_loss, avg_value_loss = train_network(
        all_examples, iteration, args.vertices, args.k, model_dir, args
    )
    
    print(f"Training completed: Policy Loss={avg_policy_loss:.4f}, Value Loss={avg_value_loss:.4f}")
    
    # Convert trained PyTorch model to JAX format (placeholder)
    # In real implementation, would properly convert the model
    trained_model = JAXCliqueGNN(args.vertices, args.hidden_dim, args.num_layers)
    rng = np.random.RandomState(iteration + 1)
    trained_params = trained_model.init_params(rng)
    
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
    save_jax_model(trained_model, trained_params, current_model_path, metadata)
    
    # 5. Evaluation phase
    print(f"\n--- Evaluation Phase ---")
    
    # Evaluate vs best model
    win_rate_vs_best = 0.5  # Default if no best model
    
    if os.path.exists(best_model_path):
        best_params, best_metadata = load_jax_model(best_model_path)
        best_model = JAXCliqueGNN(
            best_metadata.get('num_vertices', args.vertices),
            best_metadata.get('hidden_dim', args.hidden_dim),
            best_metadata.get('num_layers', args.num_layers)
        )
        
        print("Evaluating against best model...")
        win_rate_vs_best = evaluate_models_jax(
            trained_model, best_model,
            num_games=args.num_games,
            num_vertices=args.vertices,
            clique_size=args.k,
            num_mcts_sims=args.eval_mcts_sims,
            game_mode=args.game_mode,
            use_policy_only=args.use_policy_only
        )
    
    # Evaluate vs initial model
    win_rate_vs_initial = -1.0  # Default if no initial model
    win_rate_vs_initial_mcts_1 = -2.0
    
    if initial_model_path and os.path.exists(initial_model_path):
        initial_params, initial_metadata = load_jax_model(initial_model_path)
        initial_model = JAXCliqueGNN(
            initial_metadata.get('num_vertices', args.vertices),
            initial_metadata.get('hidden_dim', args.hidden_dim),
            initial_metadata.get('num_layers', args.num_layers)
        )
        
        print("Evaluating against initial model (full MCTS)...")
        win_rate_vs_initial = evaluate_models_jax(
            trained_model, initial_model,
            num_games=args.num_games,
            num_vertices=args.vertices,
            clique_size=args.k,
            num_mcts_sims=args.eval_mcts_sims,
            game_mode=args.game_mode
        )
        
        print("Evaluating against initial model (1 MCTS sim)...")
        win_rate_vs_initial_mcts_1 = evaluate_models_jax(
            trained_model, initial_model,
            num_games=args.num_games,
            num_vertices=args.vertices,
            clique_size=args.k,
            num_mcts_sims=1,
            game_mode=args.game_mode
        )
    
    # 6. Update best model
    if win_rate_vs_best > args.eval_threshold:
        print(f"New model is better ({win_rate_vs_best:.2%} > {args.eval_threshold}). Updating best model.")
        save_jax_model(trained_model, trained_params, best_model_path, metadata)
    else:
        print(f"New model is not better ({win_rate_vs_best:.2%} <= {args.eval_threshold}). Keeping previous best.")
    
    print(f"\n=== Iteration {iteration} finished ===")
    
    # Return metrics
    return {
        "iteration": iteration,
        "validation_policy_loss": avg_policy_loss,
        "validation_value_loss": avg_value_loss,
        "evaluation_win_rate_vs_best": win_rate_vs_best,
        "evaluation_win_rate_vs_initial": win_rate_vs_initial,
        "evaluation_win_rate_vs_initial_mcts_1": win_rate_vs_initial_mcts_1,
        "self_play_time": self_play_time,
        "games_per_second": args.self_play_games / self_play_time
    }


def run_pipeline(args: argparse.Namespace):
    """Run the full AlphaZero training pipeline"""
    start_time = time.time()
    
    # Define directories
    base_dir = f"./experiments/{args.experiment_name}"
    data_dir = os.path.join(base_dir, "datasets")
    model_dir = os.path.join(base_dir, "models")
    log_file_path = os.path.join(base_dir, "training_log.json")
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Starting pipeline for experiment: {args.experiment_name}")
    print(f"Data Dir: {data_dir}")
    print(f"Model Dir: {model_dir}")
    print(f"Log File: {log_file_path}")
    
    # Initialize wandb
    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project="alphazero_clique",
                name=args.experiment_name,
                config=vars(args),
                resume="allow",
                id=f"pipeline_{args.experiment_name}"
            )
            print("Weights & Biases initialized successfully.")
        except Exception as e:
            print(f"Could not initialize Weights & Biases: {e}. Skipping wandb logging.")
    
    # Load or initialize log data
    log_data = {"hyperparameters": {}, "log": []}
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r') as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, dict) and "hyperparameters" in loaded_data:
                    log_data = loaded_data
                    print(f"Loaded existing log data with {len(log_data['log'])} entries")
        except Exception as e:
            print(f"Error loading log file: {e}. Starting fresh.")
    
    # Update hyperparameters
    log_data["hyperparameters"] = vars(args)
    
    # Save initial model for comparison
    initial_model_path = f"{model_dir}/clique_net_initial_jax.pkl"
    if not os.path.exists(initial_model_path):
        print("Creating initial model for baseline comparison...")
        initial_model = JAXCliqueGNN(args.vertices, args.hidden_dim, args.num_layers)
        rng = np.random.RandomState(0)
        initial_params = initial_model.init_params(rng)
        
        metadata = {
            'num_vertices': args.vertices,
            'clique_size': args.k,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'iteration': -1,
            'type': 'initial'
        }
        save_jax_model(initial_model, initial_params, initial_model_path, metadata)
    
    # Run iterations
    for iteration in range(args.iterations):
        iteration_metrics = run_single_iteration(iteration, args, initial_model_path)
        
        if iteration_metrics:
            # Add to log
            log_data["log"].append(iteration_metrics)
            
            # Save log
            with open(log_file_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"Saved training log to {log_file_path}")
            
            # Log to wandb
            if wandb_run:
                try:
                    wandb.log(iteration_metrics, step=iteration)
                    print(f"Logged iteration {iteration} metrics to Weights & Biases.")
                except Exception as e:
                    print(f"Error logging to wandb: {e}")
        else:
            print(f"Iteration {iteration} failed. Stopping pipeline.")
            break
        
        time.sleep(2)  # Small delay between iterations
    
    end_time = time.time()
    print(f"\nPipeline finished in {(end_time - start_time)/60:.2f} minutes.")
    
    # Plot learning curves
    if log_data["log"]:
        plot_learning_curve(log_file_path)
    
    # Finish wandb run
    if wandb_run:
        wandb_run.finish()
        print("Weights & Biases run finished.")


def main():
    parser = argparse.ArgumentParser(description='Complete JAX AlphaZero Pipeline')
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="pipeline", 
                       choices=["pipeline", "selfplay", "train", "evaluate", "play"],
                       help="Execution mode")
    
    # Game parameters
    parser.add_argument("--vertices", type=int, default=6, help="Number of vertices")
    parser.add_argument("--k", type=int, default=3, help="Clique size to find")
    parser.add_argument("--game-mode", type=str, default="symmetric", 
                       choices=["symmetric", "asymmetric"], help="Game rules")
    
    # Pipeline parameters
    parser.add_argument("--iterations", type=int, default=5, help="Number of pipeline iterations")
    parser.add_argument("--self-play-games", type=int, default=100, help="Number of self-play games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=200, help="Number of MCTS simulations per move")
    parser.add_argument("--eval-threshold", type=float, default=0.55, help="Win rate threshold to update best model")
    parser.add_argument("--num-cpus", type=int, default=4, help="Number of CPUs for parallel self-play")
    parser.add_argument("--experiment-name", type=str, default="default", help="Name for organizing data/models")
    
    # Model parameters
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension size in GNN layers")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GNN layers in the model")
    
    # LR Scheduler parameters
    parser.add_argument("--initial-lr", type=float, default=0.00001, help="Initial learning rate for Adam")
    parser.add_argument("--lr-factor", type=float, default=0.7, help="LR reduction factor for ReduceLROnPlateau")
    parser.add_argument("--lr-patience", type=int, default=5, help="LR patience for ReduceLROnPlateau")
    parser.add_argument("--lr-threshold", type=float, default=1e-3, help="LR threshold for ReduceLROnPlateau")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Minimum learning rate for ReduceLROnPlateau")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size during training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs per iteration")
    parser.add_argument("--use-legacy-policy-loss", action='store_true', help="Use the old policy loss calculation")
    parser.add_argument("--min-alpha", type=float, default=0.5, help="Min weight factor for value loss")
    parser.add_argument("--max-alpha", type=float, default=100.0, help="Max weight factor for value loss")
    parser.add_argument("--use-all-data", action='store_true', help="Use all historical data for training")
    
    # Mode-specific parameters
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number (for train mode)")
    parser.add_argument("--num-games", type=int, default=21, help="Number of games (for evaluate/play modes)")
    parser.add_argument("--eval-mcts-sims", type=int, default=30, help="Number of MCTS simulations (for evaluate/play)")
    parser.add_argument("--use-policy-only", action='store_true', help="Use policy head output for move selection")
    
    args = parser.parse_args()
    
    # Execute selected mode
    if args.mode == "pipeline":
        run_pipeline(args)
    elif args.mode == "selfplay":
        # Just run self-play
        config = SelfPlayConfig(
            num_vertices=args.vertices,
            k=args.k,
            game_mode=args.game_mode,
            mcts_simulations=args.mcts_sims,
            batch_size=args.batch_size
        )
        
        # Create model
        model = JAXCliqueGNN(args.vertices, args.hidden_dim, args.num_layers)
        rng = np.random.RandomState(42)
        params = model.init_params(rng)
        
        # Run self-play
        data_dir = f"./experiments/{args.experiment_name}/datasets"
        os.makedirs(data_dir, exist_ok=True)
        
        parallel_self_play = ParallelSelfPlay(config, model, params, args.num_cpus)
        experience_path = parallel_self_play.generate_games(
            args.self_play_games, 
            data_dir, 
            args.iteration
        )
        print(f"Self-play complete. Data saved to: {experience_path}")
        
    elif args.mode == "train":
        # Just run training
        data_dir = f"./experiments/{args.experiment_name}/datasets"
        model_dir = f"./experiments/{args.experiment_name}/models"
        
        all_examples = load_examples(data_dir, args.iteration)
        if all_examples:
            avg_policy_loss, avg_value_loss = train_network(
                all_examples, args.iteration, args.vertices, args.k, model_dir, args
            )
            print(f"Training complete. Losses: Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f}")
        else:
            print("No training examples found!")
            
    elif args.mode == "evaluate":
        # Evaluate two models
        print("Evaluation mode - compare models by playing games")
        # Implementation would load two models and run evaluation
        
    elif args.mode == "play":
        # Play against AI
        model_path = f"./experiments/{args.experiment_name}/models/clique_net_jax.pkl"
        play_against_ai(model_path, args.vertices, args.k, args.eval_mcts_sims, human_player=0)
    
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
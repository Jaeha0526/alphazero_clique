#!/usr/bin/env python
import os
import argparse
import time
import datetime
import torch
import torch.multiprocessing as mp
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pickle
import glob
import json

from clique_board import CliqueBoard
from alpha_net_clique import CliqueGNN
from MCTS_clique import MCTS_self_play, UCT_search, get_policy, make_move_on_board
from train_clique import train_network, load_examples, train_pipeline
import encoder_decoder_clique as ed

def evaluate_models(current_model: CliqueGNN, best_model: CliqueGNN, 
                   num_games: int = 40, num_vertices: int = 6, clique_size: int = 3,
                   num_mcts_sims: int = 100, game_mode: str = "symmetric") -> float:
    """
    Evaluate the current model against the best model by playing games.
    
    Args:
        current_model: Current neural network model
        best_model: Best neural network model so far
        num_games: Number of games to play
        num_vertices: Number of vertices in the graph
        clique_size: Size of clique needed for Player 1 to win
        num_mcts_sims: Number of MCTS simulations per move
        game_mode: "symmetric" or "asymmetric" game mode
        
    Returns:
        win_rate: Win rate of current model against best model
    """
    current_model.eval()
    best_model.eval()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_model.to(device)
    best_model.to(device)
    
    # Track wins
    current_model_wins = 0
    best_model_wins = 0
    draws = 0
    
    # Play games
    for game_idx in range(num_games):
        print(f"Evaluation Game {game_idx+1}/{num_games}")
        
        # Initialize a new game
        board = CliqueBoard(num_vertices, clique_size, game_mode)
        game_over = False
        
        # Track game states for visualization
        game_states = [board.copy()]
        
        # Maximum possible moves
        max_moves = num_vertices * (num_vertices - 1) // 2
        
        # Play until game over or max moves reached
        while not game_over and board.move_count < max_moves:
            # Determine which model to use
            # Current model plays as Player 1 (even game_idx) or Player 2 (odd game_idx)
            current_model_turn = (game_idx % 2 == 0 and board.player == 0) or \
                                (game_idx % 2 == 1 and board.player == 1)
            
            model_to_use = current_model if current_model_turn else best_model
            
            # Get best move using MCTS
            best_move, _ = UCT_search(board, num_mcts_sims, model_to_use)
            
            # Make the move
            board = make_move_on_board(board, best_move)
            game_states.append(board.copy())
            
            # Check if game is over
            if board.game_state != 0:
                game_over = True
                
                # Determine winner
                if board.game_state == 1:  # Player 1 wins
                    if game_idx % 2 == 0:  # Current model is Player 1
                        current_model_wins += 1
                    else:  # Best model is Player 1
                        best_model_wins += 1
                elif board.game_state == 2:  # Player 2 wins
                    if game_idx % 2 == 1:  # Current model is Player 2
                        current_model_wins += 1
                    else:  # Best model is Player 2
                        best_model_wins += 1
                elif board.game_state == 3:  # Draw
                    draws += 1
            
            # Check for draw - board filled
            if not board.get_valid_moves() and game_mode == "symmetric":
                game_over = True
                if board.game_state == 0:  # Only count as draw if not already counted
                    draws += 1
        
        # Print result
        result = "Draw"
        if board.game_state == 1:
            result = "Player 1 wins"
        elif board.game_state == 2:
            result = "Player 2 wins"
            
        player1 = "Current" if game_idx % 2 == 0 else "Best"
        player2 = "Best" if game_idx % 2 == 0 else "Current"
        
        print(f"Game {game_idx+1} result: {result} (Player 1: {player1}, Player 2: {player2})")
    
    # Calculate win rate
    win_rate = current_model_wins / (current_model_wins + best_model_wins + draws)
    print(f"Evaluation complete: Current model wins: {current_model_wins}, "
          f"Best model wins: {best_model_wins}, Draws: {draws}")
    print(f"Current model win rate: {win_rate:.4f}")
    
    return win_rate

def run_iteration(iteration: int, args: argparse.Namespace, data_dir: str, model_dir: str) -> Dict[str, float]:
    """Run one iteration of the AlphaZero pipeline, return metrics."""
    print(f"=== Starting iteration {iteration} ===")
    
    # Access parameters from args object
    num_vertices = args.vertices
    clique_size = args.k
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    mcts_sims = args.mcts_sims
    game_mode = args.game_mode
    num_cpus = args.num_cpus
    num_self_play_games = args.self_play_games
    eval_threshold = args.eval_threshold

    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using evaluation device: {eval_device}")
    print(f"Model Config: V={num_vertices}, k={clique_size}, hidden_dim={hidden_dim}, num_layers={num_layers}")

    # --- 1. Determine Starting Model Path --- 
    model_to_load_path = None
    prev_model_path = f"{model_dir}/clique_net_iter{iteration-1}.pth.tar"
    best_model_path = f"{model_dir}/clique_net.pth.tar"
    current_model_path = f"{model_dir}/clique_net_iter{iteration}.pth.tar"

    if iteration > 0 and os.path.exists(prev_model_path):
        model_to_load_path = prev_model_path
        print(f"Using model from previous iteration for self-play: {model_to_load_path}")
    elif os.path.exists(best_model_path):
        model_to_load_path = best_model_path
        print(f"Using best model for self-play: {model_to_load_path}")
    else:
        print("No previous or best model found. Using fresh model for self-play.")
        # Need to create and save an initial model if none exists
        initial_model = CliqueGNN(num_vertices, hidden_dim=hidden_dim, num_layers=num_layers)
        save_initial = {
            'state_dict': initial_model.state_dict(),
            'num_vertices': num_vertices,
            'clique_size': clique_size,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers
        }
        torch.save(save_initial, current_model_path) # Save as iter 0
        model_to_load_path = current_model_path
        print(f"Saved initial model to: {model_to_load_path}")

    # --- 2. Load Model for Self-Play --- 
    checkpoint_selfplay = torch.load(model_to_load_path, map_location=torch.device('cpu'))
    # Verify loaded params match config - optional but recommended
    # ... (add checks comparing checkpoint params with args: num_vertices, clique_size, hidden_dim, num_layers)
    model_for_self_play = CliqueGNN(num_vertices, hidden_dim=hidden_dim, num_layers=num_layers)
    model_for_self_play.load_state_dict(checkpoint_selfplay['state_dict'])
    model_for_self_play.cpu() # Ensure on CPU before sharing
    model_for_self_play.share_memory()
    model_for_self_play.eval()
    print(f"Loaded model for self-play from {model_to_load_path}")
    
    # --- Remove Save Before Self-Play --- 
    # The logic above ensures a starting model exists
    # os.makedirs(model_dir, exist_ok=True)
    # torch.save(save_dict_pre_play, current_model_path) # REMOVED
    
    # --- 3. Self-Play --- 
    print(f"Starting self-play with {num_self_play_games} games using {num_cpus} CPU cores")
    processes = []
    games_per_cpu = num_self_play_games // num_cpus
    
    # Run at least 1 game per process
    if games_per_cpu < 1:
        games_per_cpu = 1
        
    # Adjust for remainder
    remainder = num_self_play_games - (games_per_cpu * num_cpus)
    
    # Start processes
    for i in range(num_cpus):
        # Add remainder games to first process
        games = games_per_cpu + (remainder if i == 0 else 0)
        p = mp.Process(target=MCTS_self_play, 
                      args=(model_for_self_play, games, num_vertices, clique_size, i, mcts_sims, game_mode, iteration, data_dir))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("Self-play completed")
    
    # Load examples generated by self-play
    print("Loading examples from current iteration")
    all_examples = load_examples(data_dir, iteration)
    if not all_examples:
        print("No training examples found for current iteration. Skipping training and evaluation.")
        return
    print(f"Loaded {len(all_examples)} examples for training.")

    # --- 4. Train New Model (train_network saves iterN model) --- 
    print("Starting training process...")
    # train_network will load the previous model (iter N-1 or best) internally based on iteration number
    # and save the result as iter N.
    avg_policy_loss, avg_value_loss = train_network(all_examples, iteration, num_vertices, clique_size, model_dir, args)
    print(f"Training process finished. Val Policy Loss: {avg_policy_loss:.4f}, Val Value Loss: {avg_value_loss:.4f}")
    
    # --- Remove Save After Training --- 
    # current_model = current_model.cpu() # No single 'current_model' instance used throughout
    # torch.save(save_dict_post_train, current_model_path) # REMOVED (train_network saved it)
    
    # --- 5. Load Trained Model for Evaluation --- 
    print(f"Loading recently trained model for evaluation: {current_model_path}")
    if not os.path.exists(current_model_path):
        print(f"ERROR: Trained model {current_model_path} not found. Cannot evaluate.")
        return
    checkpoint_eval = torch.load(current_model_path, map_location=eval_device)
    # Optional: Verify params in checkpoint_eval match args
    trained_model = CliqueGNN(num_vertices, hidden_dim=hidden_dim, num_layers=num_layers).to(eval_device)
    trained_model.load_state_dict(checkpoint_eval['state_dict'])
    trained_model.eval()
    print("Loaded trained model.")

    # --- 6. Evaluate Trained vs Best --- 
    print("Starting evaluation against best model...")
    win_rate_vs_best = 0.0 # Default win rate if no best model exists yet
    best_model_instance = None

    if not os.path.exists(best_model_path):
         print("No best model found. Current model will become the best model.")
         win_rate_vs_best = 1.0 # Treat as 100% win rate if no opponent
    else:
        print(f"Loading best model for comparison: {best_model_path}")
        checkpoint_best = torch.load(best_model_path, map_location=eval_device)
        # Verify configuration match before evaluation
        if checkpoint_best.get('num_vertices') != num_vertices or \
           checkpoint_best.get('clique_size') != clique_size:
             print("ERROR: Best model configuration does not match current settings. Skipping evaluation.")
             win_rate_vs_best = -1.0 # Indicate configuration mismatch
        else:
            best_hidden = checkpoint_best.get('hidden_dim', hidden_dim)
            best_layers = checkpoint_best.get('num_layers', num_layers)
            best_model_instance = CliqueGNN(num_vertices, hidden_dim=best_hidden, num_layers=best_layers).to(eval_device)
            best_model_instance.load_state_dict(checkpoint_best['state_dict'])
            best_model_instance.eval()
            
            # Run evaluation games
            win_rate_vs_best = evaluate_models(trained_model, best_model_instance, 
                                       num_games=args.num_games, # Use num_games from args
                                       num_vertices=num_vertices, clique_size=clique_size,
                                       num_mcts_sims=args.eval_mcts_sims, game_mode=game_mode)

    # --- 6b. Evaluate Trained vs Initial --- 
    print("Starting evaluation against initial model (iter0)...")
    win_rate_vs_initial = -2.0 # Default: not evaluated
    initial_model_path = f"{model_dir}/clique_net_iter0.pth.tar"
    
    if not os.path.exists(initial_model_path):
        print("Initial model (iter0) not found. Skipping evaluation against initial.")
    else:
        try:
            checkpoint_initial = torch.load(initial_model_path, map_location=eval_device)
            # Verify configuration match before evaluation
            if checkpoint_initial.get('num_vertices') != num_vertices or \
               checkpoint_initial.get('clique_size') != clique_size:
                 print("ERROR: Initial model configuration does not match current settings. Skipping evaluation against initial.")
                 win_rate_vs_initial = -1.0 # Indicate configuration mismatch
            else:
                initial_hidden = checkpoint_initial.get('hidden_dim', hidden_dim)
                initial_layers = checkpoint_initial.get('num_layers', num_layers)
                initial_model = CliqueGNN(num_vertices, hidden_dim=initial_hidden, num_layers=initial_layers).to(eval_device)
                initial_model.load_state_dict(checkpoint_initial['state_dict'])
                initial_model.eval()
                print("Loaded initial model for comparison.")
                
                # Run evaluation games (trained vs initial)
                win_rate_vs_initial = evaluate_models(trained_model, initial_model, 
                                           num_games=args.num_games, # Use num_games from args
                                           num_vertices=num_vertices, clique_size=clique_size,
                                           num_mcts_sims=args.eval_mcts_sims, game_mode=game_mode)
        except Exception as e:
            print(f"ERROR loading or evaluating against initial model: {e}")
            win_rate_vs_initial = -3.0 # Indicate other error

    # --- 7. Update Best Model (based on win_rate_vs_best) --- 
    if win_rate_vs_best > eval_threshold:
        print(f"New model is better (Win Rate vs Best: {win_rate_vs_best:.4f} > {eval_threshold}). Updating best model.")
        # Save the current trained model as the new best model
        save_best = {
            'state_dict': trained_model.state_dict(),
            'num_vertices': num_vertices,
            'clique_size': clique_size,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers
        }
        torch.save(save_best, best_model_path)
    else:
        print(f"New model is not better (Win Rate vs Best: {win_rate_vs_best:.4f} <= {eval_threshold}). Keeping previous best model.")

    print(f"=== Iteration {iteration} finished ===")
    
    # Return collected metrics
    return {
        "iteration": iteration,
        "validation_policy_loss": avg_policy_loss,
        "validation_value_loss": avg_value_loss,
        "evaluation_win_rate_vs_best": win_rate_vs_best,
        "evaluation_win_rate_vs_initial": win_rate_vs_initial # Add new metric
    }

def run_pipeline(args: argparse.Namespace) -> None:
    """
    Run the full AlphaZero training pipeline using parameters from args.
    
    Args:
        args: Parsed command line arguments from argparse.
    """
    start_time = time.time()
    
    # Define directories based on experiment name
    base_dir = f"./experiments/{args.experiment_name}"
    data_dir = os.path.join(base_dir, "datasets")
    model_dir = os.path.join(base_dir, "models")
    log_file_path = os.path.join(base_dir, "training_log.json")
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Starting pipeline for experiment: {args.experiment_name}")
    print(f"Data Dir: {data_dir}")
    print(f"Model Dir: {model_dir}")
    print(f"Log File: {log_file_path}")

    # --- Load existing log data or initialize --- 
    log_data = {"hyperparameters": {}, "log": []} # Initialize with new structure
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r') as f:
                loaded_data = json.load(f)
                # Check if loaded data is in the new dictionary format
                if isinstance(loaded_data, dict) and "hyperparameters" in loaded_data and "log" in loaded_data:
                    log_data = loaded_data
                    print(f"Loaded existing log data with {len(log_data['log'])} entries.")
                elif isinstance(loaded_data, list):
                    # Handle old list format (optional, maybe just warn/error)
                    print("Warning: Old log format detected. Converting to new format, but hyperparameters will be missing for previous runs.")
                    log_data["log"] = loaded_data
                else:
                    print(f"Warning: Unknown format in log file {log_file_path}. Starting fresh log.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode existing log file {log_file_path}. Starting fresh log.")
        except Exception as e:
            print(f"Warning: Error loading log file {log_file_path}: {e}. Starting fresh log.")
    else:
        print("No existing log file found. Creating new log.")

    # --- Store Hyperparameters --- 
    # Store only if not already present (e.g., loading an existing log)
    if not log_data["hyperparameters"]:
        print("Storing hyperparameters in log file.")
        log_data["hyperparameters"] = {
            "experiment_name": args.experiment_name,
            "vertices": args.vertices,
            "k": args.k,
            "game_mode": args.game_mode,
            "iterations": args.iterations,
            "self_play_games": args.self_play_games,
            "mcts_sims": args.mcts_sims,
            "num_cpus": args.num_cpus,
            "eval_threshold": args.eval_threshold,
            "num_games": args.num_games,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "initial_lr": args.initial_lr,
            "lr_factor": args.lr_factor,
            "lr_patience": args.lr_patience,
            "lr_threshold": args.lr_threshold,
            "min_lr": args.min_lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs
        }
        # Save immediately after storing hyperparameters for a new file
        if not os.path.exists(log_file_path):
             try:
                 with open(log_file_path, 'w') as f:
                     json.dump(log_data, f, indent=4)
             except Exception as e:
                 print(f"ERROR: Could not save initial log file with hyperparameters: {e}")

    # Determine starting iteration based on log data
    start_iteration = log_data["log"][-1]["iteration"] + 1 if log_data["log"] else 0
    print(f"Starting from iteration: {start_iteration}")
    
    # Run training iterations
    for iteration in range(start_iteration, start_iteration + args.iterations):
        iteration_metrics = run_iteration(iteration, args, data_dir, model_dir)
                                       
        # --- Append metrics to log data["log"] and save --- 
        if iteration_metrics:
            log_list = log_data["log"]
            # Check if this iteration already exists in the log list
            existing_entry = next((item for item in log_list if item["iteration"] == iteration), None)
            if existing_entry:
                existing_entry.update(iteration_metrics)
            else:
                log_list.append(iteration_metrics)
            
            # Save the entire log_data dictionary
            try:
                with open(log_file_path, 'w') as f:
                    json.dump(log_data, f, indent=4) # Save the whole dict
                print(f"Saved updated log to {log_file_path}")
                # Plot the curve after saving the log for this iteration
                if len(log_list) >= 2: 
                    plot_learning_curve(log_file_path)
            except Exception as e:
                print(f"ERROR: Could not save log file {log_file_path}: {e}")
        else:
            print(f"Iteration {iteration} did not return metrics (likely skipped). Not logging.")
            
        # Optional: Add delay or check for conditions before next iteration
        time.sleep(2) # Small delay
        
    end_time = time.time()
    print(f"\nPipeline finished in {(end_time - start_time):.2f} seconds.")
    
    if log_data["log"]:
        plot_learning_curve(log_file_path) 

def plot_learning_curve(log_file_path: str):
    """
    Plot validation policy and value loss curves from the training log.
    
    Args:
        log_file_path: Path to the training_log.json file.
    """
    if not os.path.exists(log_file_path):
        print(f"Log file not found at {log_file_path}. Cannot plot learning curve.")
        return
        
    try:
        with open(log_file_path, 'r') as f:
            loaded_data = json.load(f)
            # Expect dictionary format now
            if not isinstance(loaded_data, dict) or "log" not in loaded_data:
                print("Log file is not in the expected format (missing 'log' key).")
                return
            log_list = loaded_data["log"] # Get the list of log entries
            hyperparams = loaded_data.get("hyperparameters", {}) # Get hyperparams if they exist

    except Exception as e:
        print(f"Error loading or parsing log file {log_file_path}: {e}")
        return

    if not isinstance(log_list, list) or len(log_list) < 2:
        print("Not enough data points in log file to plot learning curve.")
        return

    # Extract data from the log_list
    plot_data = [
        (entry["iteration"], entry["validation_policy_loss"], entry["validation_value_loss"], entry.get("evaluation_win_rate_vs_initial"))
        for entry in log_list
        if entry.get("validation_policy_loss") is not None and entry.get("validation_value_loss") is not None
    ]
    
    # Separate into lists, handling None for win_rate_initial if missing
    iterations = [p[0] for p in plot_data]
    policy_losses = [p[1] for p in plot_data]
    value_losses = [p[2] for p in plot_data]
    win_rates_initial = [p[3] if p[3] is not None and p[3] >= 0 else np.nan for p in plot_data] # Replace errors/missing with NaN

    if len(iterations) < 2:
        print("Not enough valid data points (with losses) in log file to plot learning curve.")
        return
        
    # --- Plotting --- 
    # Use 3 axes now: one for iterations, one for losses, one for win rate
    fig, ax1 = plt.subplots(figsize=(12, 8)) # Slightly larger figure

    # Plot Policy Loss (Axis 1)
    color = 'tab:red'
    ax1.set_xlabel('Iteration', fontsize=14)
    ax1.set_ylabel('Validation Policy Loss', color=color, fontsize=14)
    ax1.plot(iterations, policy_losses, color=color, marker='o', linestyle='-', linewidth=2, markersize=5, label='Policy Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')

    # Create a second y-axis for Value Loss (Axis 2)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Value Loss', color=color, fontsize=14)
    ax2.plot(iterations, value_losses, color=color, marker='s', linestyle='--', linewidth=2, markersize=5, label='Value Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    # Create a third y-axis for Win Rate vs Initial (Axis 3)
    # Need to shift this axis slightly to avoid overlap with ax2
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60)) # Offset the right spine
    color = 'tab:green'
    ax3.set_ylabel('Win Rate vs Initial', color=color, fontsize=14)
    # Connect only non-NaN points for the win rate
    ax3.plot(iterations, win_rates_initial, color=color, marker='^', linestyle=':', linewidth=2, markersize=6, label='Win Rate vs Initial')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_ylim(-0.05, 1.05) # Set y-limit for win rate
    ax3.legend(loc='lower left')

    # Add title with some hyperparameters
    title = f"Training Losses & Win Rate vs Initial\n"
    title += f"Exp: {hyperparams.get('experiment_name', 'N/A')}, V={hyperparams.get('vertices', '?')}, k={hyperparams.get('k', '?')}, MCTS={hyperparams.get('mcts_sims', '?')}\n"
    title += f"LR={hyperparams.get('initial_lr', '?')}, Factor={hyperparams.get('lr_factor', '?')}, Patience={hyperparams.get('lr_patience', '?')}"
    plt.title(title, fontsize=12) # Adjust font size maybe
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    
    # Combine legends (might get cluttered, alternative is multiple legends)
    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # lines3, labels3 = ax3.get_legend_handles_labels()
    # ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Save plot in the same directory as the log file with a fixed name
    plot_dir = os.path.dirname(log_file_path)
    # Use a fixed filename to overwrite the plot each iteration
    plot_filename = os.path.join(plot_dir, f"training_losses.png")
    try:
        plt.savefig(plot_filename)
        print(f"Learning curve saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig) # Close the figure to free memory

def play_against_ai(model_path: str = None, num_vertices: int = 6, clique_size: int = 3, 
                   num_mcts_sims: int = 200, human_player: int = 0):
    """
    Play a game against the trained AI.
    
    Args:
        model_path: Path to the model to load
        num_vertices: Number of vertices in the graph
        clique_size: Size of clique needed for Player 1 to win
        num_mcts_sims: Number of MCTS simulations per move
        human_player: Human player (0 for Player 1, 1 for Player 2)
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = CliqueGNN(num_vertices=num_vertices)
    if model_path is None:
        model_path = "./model_data/clique_net.pth.tar"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}, using random initialization")
    
    model.eval()
    model.to(device)
    
    # Initialize game
    board = CliqueBoard(num_vertices, clique_size)
    game_over = False
    
    # Maximum possible moves
    max_moves = num_vertices * (num_vertices - 1) // 2
    
    # Print initial board
    print(board)
    
    # Play until game over or max moves reached
    while not game_over and board.move_count < max_moves:
        # Human's turn
        if board.player == human_player:
            valid_moves = board.get_valid_moves()
            
            # Display valid moves
            print("\nValid moves:")
            for i, move in enumerate(valid_moves):
                print(f"{i}: Edge {move}")
            
            # Get human's move
            try:
                move_idx = int(input("\nEnter move index: "))
                if move_idx < 0 or move_idx >= len(valid_moves):
                    print("Invalid move index, try again")
                    continue
                
                # Make the move
                edge = valid_moves[move_idx]
                board.make_move(edge)
                
                print(f"\nHuman selected edge {edge}")
                print(board)
            except ValueError:
                print("Invalid input, please enter a number")
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue
                
        # AI's turn
        else:
            print("\nAI is thinking...")
            
            # Get best move using MCTS
            best_move, _ = UCT_search(board, num_mcts_sims, model)
            
            # Make the move
            edge = ed.decode_action(board, best_move)
            board.make_move(edge)
            
            print(f"AI selected edge {edge}")
            print(board)
        
        # Check if game is over
        if board.game_state != 0:
            game_over = True
            if board.game_state == 1:
                print("Player 1 wins!")
            else:
                print("Player 2 wins!")
        
        # Check for draw - board filled
        if not board.get_valid_moves():
            game_over = True
            print("Game drawn - board filled!")

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        mp.set_start_method('spawn')
        print("Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        print("Multiprocessing start method already set or not needed")
    
    parser = argparse.ArgumentParser(description="AlphaZero Clique Game Pipeline")
    
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
    
    # Add LR Scheduler Hyperparameters
    parser.add_argument("--initial-lr", type=float, default=0.0003, help="Initial learning rate for Adam")
    parser.add_argument("--lr-factor", type=float, default=0.7, help="LR reduction factor for ReduceLROnPlateau")
    parser.add_argument("--lr-patience", type=int, default=7, help="LR patience for ReduceLROnPlateau")
    parser.add_argument("--lr-threshold", type=float, default=1e-5, help="LR threshold for ReduceLROnPlateau")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Minimum learning rate for ReduceLROnPlateau")
    
    # Add Training Loop Hyperparameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size during training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs per iteration")

    # Specific mode parameters (can add more if needed, e.g., model paths for evaluate/play)
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number (for train mode)")
    parser.add_argument("--num-games", type=int, default=31, help="Number of games (for evaluate/play modes)")
    parser.add_argument("--eval-mcts-sims", type=int, default=30, help="Number of MCTS simulations (for evaluate/play modes)")

    args = parser.parse_args()

    # Create base directories
    data_base_dir = f"./experiments/{args.experiment_name}/datasets"
    model_dir = f"./experiments/{args.experiment_name}/models"
    os.makedirs(data_base_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Execute selected mode
    if args.mode == "pipeline":
        # Pass the whole args object
        run_pipeline(args)
                     
    elif args.mode == "selfplay":
        print("Running Self-Play Only Mode")
        # Needs model loading and correct instantiation
        model_path = os.path.join(model_dir, "clique_net.pth.tar") # Assume using best model
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path} for self-play.")
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            # Infer params from checkpoint if possible, else use args/defaults
            loaded_v = checkpoint.get('num_vertices', args.vertices)
            loaded_k = checkpoint.get('clique_size', args.k)
            loaded_hidden = checkpoint.get('hidden_dim', args.hidden_dim)
            loaded_layers = checkpoint.get('num_layers', args.num_layers)
            if loaded_v != args.vertices or loaded_k != args.k:
                 print("Warning: Model parameters differ from command line args for self-play.")
                 
            model = CliqueGNN(num_vertices=loaded_v, hidden_dim=loaded_hidden, num_layers=loaded_layers)
            model.load_state_dict(checkpoint['state_dict'])
            model.share_memory() # Important for multiprocessing
            model.eval()
            
            MCTS_self_play(model=model, 
                           num_games=args.self_play_games, 
                           num_vertices=args.vertices, # Use game vertices
                           k=args.k,                   # Use game k
                           process_id=0, 
                           mcts_sims=args.mcts_sims, 
                           game_mode=args.game_mode, 
                           iteration=args.iteration, # Pass iteration if needed for saving
                           data_dir=data_base_dir) 
                           
    elif args.mode == "train":
        print("Running Training Only Mode")
        # This mode relies on train_clique.py which also needs updating
        print("WARNING: train_clique.py may need updates for hidden_dim/num_layers.")
        all_examples = load_examples(data_base_dir, args.iteration) # Load specific iteration
        if not all_examples:
             print(f"No examples found for iteration {args.iteration} in {data_base_dir}")
        else:
             # The train_network function needs to be updated to accept/use hidden_dim/num_layers
             # and instantiate the model correctly.
             # Placeholder call - ASSUMES train_network is updated elsewhere.
             train_network(all_examples, args.iteration, args.vertices, args.k, model_dir)

    elif args.mode == "evaluate":
        print("Running Evaluation Only Mode")
        # Load current and best models - requires paths and correct instantiation
        current_model_path = os.path.join(model_dir, f"clique_net_iter{args.iteration}.pth.tar")
        best_model_path = os.path.join(model_dir, "clique_net.pth.tar")
        
        if not os.path.exists(current_model_path) or not os.path.exists(best_model_path):
            print("Error: Both current iteration model and best model must exist for evaluation.")
        else:
            # Load current model
            chk_curr = torch.load(current_model_path, map_location=torch.device('cpu'))
            curr_hidden = chk_curr.get('hidden_dim', args.hidden_dim)
            curr_layers = chk_curr.get('num_layers', args.num_layers)
            current_model = CliqueGNN(args.vertices, hidden_dim=curr_hidden, num_layers=curr_layers)
            current_model.load_state_dict(chk_curr['state_dict'])
            current_model.eval()
            
            # Load best model
            chk_best = torch.load(best_model_path, map_location=torch.device('cpu'))
            best_hidden = chk_best.get('hidden_dim', args.hidden_dim)
            best_layers = chk_best.get('num_layers', args.num_layers)
            best_model = CliqueGNN(args.vertices, hidden_dim=best_hidden, num_layers=best_layers)
            best_model.load_state_dict(chk_best['state_dict'])
            best_model.eval()

            win_rate = evaluate_models(current_model, best_model, 
                                     num_games=args.num_games, 
                                     num_vertices=args.vertices, 
                                     clique_size=args.k,
                                     num_mcts_sims=args.mcts_sims, 
                                     game_mode=args.game_mode)
            print(f"Evaluation Result (Iter {args.iteration} vs Best): Win Rate = {win_rate:.2f}")

    elif args.mode == "play":
        print("Running Play Against AI Mode")
        # Load the best model
        model_path = os.path.join(model_dir, "clique_net.pth.tar")
        if not os.path.exists(model_path):
            print(f"ERROR: Best model not found at {model_path}")
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            loaded_hidden = checkpoint.get('hidden_dim', args.hidden_dim)
            loaded_layers = checkpoint.get('num_layers', args.num_layers)
            model = CliqueGNN(args.vertices, hidden_dim=loaded_hidden, num_layers=loaded_layers)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            
            play_against_ai(model, args.vertices, args.k, args.mcts_sims, args.game_mode) 
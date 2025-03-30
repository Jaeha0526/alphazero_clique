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

def run_iteration(iteration: int, num_self_play_games: int, num_vertices: int, 
                 clique_size: int, mcts_sims: int, eval_threshold: float,
                 hidden_dim: int, num_layers: int,
                 num_cpus: int = 4, game_mode: str = "symmetric",
                 data_dir: str = "./datasets/clique", model_dir: str = "./model_data") -> None:
    """Run one iteration of the AlphaZero pipeline"""
    print(f"=== Starting iteration {iteration} ===")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use passed hidden_dim and num_layers
    print(f"Model Config: hidden_dim={hidden_dim}, num_layers={num_layers}")
    
    # Initialize model correctly
    current_model = CliqueGNN(num_vertices, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    
    # Load current model from previous iteration if it exists
    current_model_path = f"{model_dir}/clique_net_iter{iteration}.pth.tar"
    prev_model_path = f"{model_dir}/clique_net_iter{iteration-1}.pth.tar"
    
    if iteration > 0 and os.path.exists(prev_model_path):
        print(f"Loading model from previous iteration: {prev_model_path}")
        checkpoint = torch.load(prev_model_path, map_location=device)
        # TODO: Optionally check if loaded model's params match current config before loading state_dict
        current_model.load_state_dict(checkpoint['state_dict'])
    else:
        # For first iteration, try to load best model if exists
        best_model_path = f"{model_dir}/clique_net.pth.tar"
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            # TODO: Optionally check parameters here too
            current_model.load_state_dict(checkpoint['state_dict'])
        else:
            print("No previous model found, starting from scratch")
    
    # Make sure model is on CPU before sharing
    current_model = current_model.cpu()
    current_model.share_memory()
    current_model.eval()
    
    # Save current model (before self-play)
    os.makedirs(model_dir, exist_ok=True)
    save_dict_pre_play = {
        'state_dict': current_model.state_dict(),
        'num_vertices': num_vertices,
        'clique_size': clique_size,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers
    }
    torch.save(save_dict_pre_play, current_model_path)
    
    # Run self-play games in parallel
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
                      args=(current_model, games, num_vertices, clique_size, i, mcts_sims, game_mode, iteration, data_dir))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    print("Self-play completed")
    
    # Load examples from current iteration only
    print("Loading examples from current iteration")
    all_examples = load_examples(data_dir, iteration)
    
    if not all_examples:
        print("No training examples found for current iteration. Aborting.")
        return
        
    print(f"Training on {len(all_examples)} examples from iteration {iteration}")
    
    # Train on current iteration's examples
    print("Training on examples")
    avg_policy_loss, avg_value_loss = train_network(all_examples, iteration, num_vertices, clique_size, model_dir)
    
    # Save current model (after training)
    os.makedirs(model_dir, exist_ok=True)
    save_dict_post_train = {
        'state_dict': current_model.state_dict(),
        'num_vertices': num_vertices,
        'clique_size': clique_size,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers
    }
    torch.save(save_dict_post_train, current_model_path)
    
    # Evaluate against best model
    print("=== Starting evaluation ===")
    best_model_path = f"{model_dir}/clique_net.pth.tar"
    
    # Initialize win rate for first iteration
    win_rate = 1.0 if not os.path.exists(best_model_path) else 0.0
    
    if os.path.exists(best_model_path):
        # Instantiate best_model correctly
        best_model = CliqueGNN(num_vertices, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        checkpoint = torch.load(best_model_path, map_location=device)
        # TODO: Optionally check parameters here too
        best_model.load_state_dict(checkpoint['state_dict'])
        best_model.eval()
        
        # Run evaluation using evaluate_models function
        win_rate = evaluate_models(current_model, best_model, num_games=10, 
                                 num_vertices=num_vertices, clique_size=clique_size,
                                 num_mcts_sims=mcts_sims, game_mode=game_mode)
        
        print(f"Evaluation win rate: {win_rate:.2f}")
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Load existing experiment log or create new one
    experiment_log_file = os.path.join(model_dir, "experiment_log.json")
    if os.path.exists(experiment_log_file):
        with open(experiment_log_file, 'r') as f:
            experiment_log = json.load(f)
    else:
        experiment_log = {
            'experiment_info': {
                'num_vertices': num_vertices,
                'clique_size': clique_size,
                'mcts_sims': mcts_sims,
                'num_cpus': num_cpus,
                'game_mode': game_mode,
                'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'iterations': []
        }
    
    # Add current iteration results
    iteration_results = {
        'iteration': iteration,
        'win_rate': win_rate,
        'num_examples': len(all_examples),
        'num_self_play_games': num_self_play_games,
        'mcts_sims': mcts_sims,
        'game_mode': game_mode,
        'validation_metrics': {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss
        }
    }
    
    # Append iteration results
    experiment_log['iterations'].append(iteration_results)
    
    # Save updated experiment log
    with open(experiment_log_file, 'w') as f:
        json.dump(experiment_log, f, indent=4)
    
    print(f"Experiment log saved to {experiment_log_file}")
    
    # If current model is better or no best model exists, update best model
    if win_rate > eval_threshold or not os.path.exists(best_model_path):
        print(f"Current model is better (win rate: {win_rate:.2f} > {eval_threshold:.2f})")
        print("Updating best model...")
        save_dict_best = {
            'state_dict': current_model.state_dict(),
            'num_vertices': num_vertices,
            'clique_size': clique_size,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers
        }
        torch.save(save_dict_best, best_model_path)
        print("Best model updated")
    else:
        print(f"Current model is not better (win rate: {win_rate:.2f} <= {eval_threshold:.2f})")
        print("Keeping best model")

def run_pipeline(iterations: int = 5, self_play_games: int = 10, 
                num_vertices: int = 6, clique_size: int = 3,
                mcts_sims: int = 500, hidden_dim: int = 64, num_layers: int = 2,
                num_cpus: int = 1, game_mode: str = "symmetric", 
                eval_threshold: float = 0.55,
                experiment_name: str = "default") -> None:
    """
    Run the full AlphaZero pipeline for the Clique Game.
    
    Args:
        iterations: Number of iterations to run
        self_play_games: Number of self-play games per iteration
        num_vertices: Number of vertices in the graph
        clique_size: Size of clique needed to win
        mcts_sims: Number of MCTS simulations per move
        num_cpus: Number of CPUs to use for parallel self-play
        game_mode: "symmetric" or "asymmetric" game mode
        experiment_name: Name of the experiment for organizing data and models
    """
    print(f"Starting pipeline with {iterations} iterations")
    print(f"Self-play games per iteration: {self_play_games}")
    print(f"Using {num_cpus} CPUs for parallel self-play")
    print(f"Experiment name: {experiment_name}")
    
    # Create experiment-specific directories
    data_dir = f"./datasets/clique/{experiment_name}"
    model_dir = f"./model_data/{experiment_name}"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Run iterations
    for iteration in range(iterations):
        print(f"\nStarting iteration {iteration+1}/{iterations}")
        
        # Run iteration with experiment-specific directories, passing new params
        run_iteration(iteration, self_play_games, num_vertices, 
                     clique_size, mcts_sims, eval_threshold,
                     hidden_dim, num_layers,
                     num_cpus, game_mode, data_dir, model_dir)
        
        # Move files to experiment-specific directories
        # Move game data
        for file in glob.glob("./datasets/clique/*.pkl"):
            if f"_iter{iteration}" in file:
                new_file = file.replace("./datasets/clique/", f"{data_dir}/")
                os.rename(file, new_file)
        
        # Move model files
        for file in glob.glob("./model_data/*.pth.tar"):
            if f"iter{iteration}" in file or "clique_net.pth.tar" in file:
                new_file = file.replace("./model_data/", f"{model_dir}/")
                os.rename(file, new_file)
        
        print(f"Iteration {iteration+1} completed")
        
        # Wait before next iteration
        if iteration < iterations - 1:
            print("Waiting 10 seconds before next iteration...")
            time.sleep(10)
    
    print("\nPipeline completed successfully!")

def plot_learning_curve(num_iterations: int):
    """
    Plot the learning curve based on evaluation results.
    
    Args:
        num_iterations: Number of iterations to include
    """
    iterations = []
    win_rates = []
    
    # Load evaluation results
    for i in range(num_iterations):
        results_file = f"./model_data/evaluation_results_iter{i}.txt"
        if not os.path.exists(results_file):
            continue
            
        with open(results_file, "r") as f:
            lines = f.readlines()
            win_rate = float(lines[0].split(": ")[1])
            iterations.append(i)
            win_rates.append(win_rate)
    
    if len(iterations) <= 1:
        print("Not enough data to plot learning curve")
        return
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, win_rates, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Win Rate vs Previous Best', fontsize=14)
    plt.title('AlphaZero Clique Game Learning Curve', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"./model_data/learning_curve_{datetime.datetime.today().strftime('%Y-%m-%d')}.png")
    plt.close()

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
    
    # Model parameters (New)
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension size in GNN layers")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GNN layers in the model")
    
    # Specific mode parameters (can add more if needed, e.g., model paths for evaluate/play)
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number (for train mode)")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games (for evaluate/play modes)")

    args = parser.parse_args()

    # Create base directories
    data_base_dir = f"./datasets/clique/{args.experiment_name}"
    model_dir = f"./model_data/{args.experiment_name}"
    os.makedirs(data_base_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Execute selected mode
    if args.mode == "pipeline":
        run_pipeline(iterations=args.iterations, 
                     self_play_games=args.self_play_games, 
                     num_vertices=args.vertices, 
                     clique_size=args.k,
                     mcts_sims=args.mcts_sims, 
                     hidden_dim=args.hidden_dim,
                     num_layers=args.num_layers,
                     num_cpus=args.num_cpus,
                     game_mode=args.game_mode, 
                     eval_threshold=args.eval_threshold,
                     experiment_name=args.experiment_name)
                     
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
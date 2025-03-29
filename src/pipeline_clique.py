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

from clique_board import CliqueBoard
from alpha_net_clique import CliqueGNN
from MCTS_clique import MCTS_self_play, UCT_search, get_policy, make_move_on_board
from train_clique import train_network, load_examples
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
                 num_cpus: int = 4, game_mode: str = "symmetric") -> None:
    """Run one iteration of the AlphaZero pipeline"""
    print(f"=== Starting iteration {iteration} ===")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    current_model = CliqueGNN(num_vertices, clique_size).to(device)
    
    # Load best model if it exists
    best_model_path = f"./model_data/clique_net.pth.tar"
    current_model_path = f"./model_data/clique_net_iter{iteration}.pth.tar"
    
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        current_model.load_state_dict(checkpoint['state_dict'])
    else:
        print("No best model found, starting from scratch")
    
    # Make sure model is on CPU before sharing
    current_model = current_model.cpu()
    current_model.share_memory()
    current_model.eval()
    
    # Save current model
    os.makedirs("./model_data", exist_ok=True)
    torch.save({'state_dict': current_model.state_dict()}, current_model_path)
    
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
                      args=(current_model, games, num_vertices, clique_size, i, mcts_sims, game_mode))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    print("Self-play completed")
    
    # Load all examples from datasets directory
    print("Loading all examples from datasets")
    all_examples = load_examples("./datasets/clique")
    
    if not all_examples:
        print("No training examples found. Aborting.")
        return
        
    print(f"Training on {len(all_examples)} examples")
    
    # Train on all examples
    print("Training on examples")
    train_network(all_examples, iteration, num_vertices, clique_size)
    
    # Save current model
    os.makedirs("./model_data", exist_ok=True)
    torch.save({'state_dict': current_model.state_dict()}, current_model_path)
    
    # Evaluate against best model
    print("=== Starting evaluation ===")
    if os.path.exists(best_model_path):
        best_model = CliqueGNN(num_vertices, clique_size).to(device)
        checkpoint = torch.load(best_model_path, map_location=device)
        best_model.load_state_dict(checkpoint['state_dict'])
        best_model.eval()
        
        # Run evaluation using evaluate_models function
        win_rate = evaluate_models(current_model, best_model, num_games=10, 
                                 num_vertices=num_vertices, clique_size=clique_size,
                                 num_mcts_sims=mcts_sims, game_mode=game_mode)
        
        print(f"Evaluation win rate: {win_rate:.2f}")
        
        # Update best model if current model is better
        if win_rate > eval_threshold:
            print(f"Current model is better (win rate: {win_rate:.2f} > {eval_threshold})")
            torch.save({'state_dict': current_model.state_dict()}, best_model_path)
    else:
        print("No best model exists, saving current model as best")
        torch.save({'state_dict': current_model.state_dict()}, best_model_path)

def run_pipeline(num_iterations: int = 5, num_self_play_games: int = 100,
                num_vertices: int = 6, clique_size: int = 3,
                num_mcts_sims: int = 500, evaluation_threshold: float = 0.55,
                num_cpus: int = 4, game_mode: str = "symmetric"):
    """
    Run the complete AlphaZero pipeline for Clique Game.
    
    Args:
        num_iterations: Number of iterations to run
        num_self_play_games: Number of self-play games per iteration
        num_vertices: Number of vertices in the graph
        clique_size: Size of clique needed for Player 1 to win
        num_mcts_sims: Number of MCTS simulations per move
        evaluation_threshold: Win rate threshold to update best model
        num_cpus: Number of CPU cores to use for parallel processing
        game_mode: "symmetric" or "asymmetric" game mode
    """
    # Create directories
    os.makedirs("./model_data", exist_ok=True)
    os.makedirs("./datasets/clique", exist_ok=True)
    
    # Run iterations
    for iteration in range(num_iterations):
        print(f"\n\n*** Starting iteration {iteration+1}/{num_iterations} ***\n")
        start_time = time.time()
        
        # Run iteration
        run_iteration(iteration, num_self_play_games, num_vertices, 
                    clique_size, num_mcts_sims, evaluation_threshold,
                    num_cpus, game_mode)
        
        # Print iteration stats
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Iteration {iteration} completed in {elapsed_time:.2f}s")
        
        # Plot learning curve if possible
        try:
            plot_learning_curve(iteration+1)
        except Exception as e:
            print(f"Error plotting learning curve: {e}")
    
    print("\nTraining pipeline completed successfully")

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
    
    parser = argparse.ArgumentParser(description="AlphaZero Pipeline for Clique Game")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="pipeline", 
                      choices=["pipeline", "selfplay", "train", "evaluate", "play"],
                      help="Mode to run")
    
    # General parameters
    parser.add_argument("--vertices", type=int, default=6,
                      help="Number of vertices in the graph")
    parser.add_argument("--clique-size", type=int, default=3,
                      help="Size of clique needed for Player 1 to win")
    parser.add_argument("--mcts-sims", type=int, default=500,
                      help="Number of MCTS simulations per move")
    parser.add_argument("--num-cpus", type=int, default=4,
                      help="Number of CPU cores to use for parallel processing (default: 4)")
    parser.add_argument("--game-mode", type=str, default="symmetric",
                      choices=["symmetric", "asymmetric"],
                      help="Game mode: symmetric (both players try to form k-clique) or asymmetric (Player 1 tries to form k-clique, Player 2 tries to prevent it)")
    
    # Pipeline parameters
    parser.add_argument("--iterations", type=int, default=5,
                      help="Number of iterations to run")
    parser.add_argument("--self-play-games", type=int, default=100,
                      help="Number of self-play games per iteration")
    parser.add_argument("--eval-threshold", type=float, default=0.55,
                      help="Win rate threshold to update best model")
    
    # Self-play parameters
    parser.add_argument("--num-games", type=int, default=100,
                      help="Number of games to play")
    parser.add_argument("--cpu", type=int, default=0,
                      help="CPU index for multiprocessing")
    
    # Training parameters
    parser.add_argument("--iteration", type=int, default=0,
                      help="Training iteration number")
    
    # Play parameters
    parser.add_argument("--model-path", type=str, default=None,
                      help="Path to model to load")
    parser.add_argument("--human-player", type=int, default=0,
                      help="Human player (0 for Player 1, 1 for Player 2)")
    
    args = parser.parse_args()
    
    # Run selected mode
    if args.mode == "pipeline":
        run_pipeline(args.iterations, args.self_play_games, args.vertices,
                   args.clique_size, args.mcts_sims, args.eval_threshold,
                   args.num_cpus, args.game_mode)
    
    elif args.mode == "selfplay":
        # Create model
        model = CliqueGNN(num_vertices=args.vertices)
        model_path = "./model_data/clique_net.pth.tar"
        
        # Load model if exists
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            print("No model found, using random initialization")
            
        # Make sure model is on CPU before sharing
        model = model.cpu()
        model.share_memory()
        model.eval()
        
        # Run self-play
        MCTS_self_play(model, args.num_games, args.vertices, args.clique_size, args.cpu, args.mcts_sims)
    
    elif args.mode == "train":
        # Load examples
        all_examples = load_examples("./datasets/clique")
        
        if not all_examples:
            print("No training examples found. Aborting.")
        else:
            # Train network
            train_network(all_examples, args.iteration, args.vertices)
    
    elif args.mode == "evaluate":
        # Create models
        current_model = CliqueGNN(num_vertices=args.vertices)
        best_model = CliqueGNN(num_vertices=args.vertices)
        
        # Load models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current_model_path = f"./model_data/clique_net_iter{args.iteration}.pth.tar"
        best_model_path = "./model_data/clique_net.pth.tar"
        
        if os.path.exists(current_model_path):
            current_model.load_state_dict(torch.load(current_model_path, map_location=device)['state_dict'])
            print(f"Loaded current model from {current_model_path}")
        else:
            print(f"Current model not found at {current_model_path}, using random initialization")
        
        if os.path.exists(best_model_path):
            best_model.load_state_dict(torch.load(best_model_path, map_location=device)['state_dict'])
            print(f"Loaded best model from {best_model_path}")
        else:
            print(f"Best model not found at {best_model_path}, using copy of current model")
            best_model.load_state_dict(current_model.state_dict())
        
        # Evaluate
        win_rate = evaluate_models(current_model, best_model, args.num_games, 
                                 args.vertices, args.clique_size, args.mcts_sims, args.game_mode)
        
        print(f"Win rate: {win_rate:.4f}")
    
    elif args.mode == "play":
        # Play against AI
        play_against_ai(args.model_path, args.vertices, args.clique_size, 
                      args.mcts_sims, args.human_player) 
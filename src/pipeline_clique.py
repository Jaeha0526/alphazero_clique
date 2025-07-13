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
import wandb
import random

from clique_board import CliqueBoard
from alpha_net_clique import CliqueGNN
from MCTS_clique import MCTS_self_play, UCT_search, get_policy, make_move_on_board
from train_clique import train_network, load_examples, train_pipeline
import encoder_decoder_clique as ed

def evaluate_models(current_model: CliqueGNN, best_model: CliqueGNN, 
                   num_games: int = 40, num_vertices: int = 6, clique_size: int = 3,
                   num_mcts_sims: int = 100, game_mode: str = "symmetric",
                   perspective_mode: str = "alternating",
                   use_policy_only: bool = False, 
                   decided_games_only: bool = False) -> float:
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
        perspective_mode: "fixed" or "alternating" for value perspective
        use_policy_only: If True, select moves directly from policy head output (no MCTS)
        decided_games_only: If True, calculate win rate from decided games only (exclude draws)
        
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
            
            # Get best move using MCTS or policy head directly
            if use_policy_only:
                # Prepare state for network
                state_dict = ed.prepare_state_for_network(board)
                edge_index = state_dict['edge_index'].to(device)
                edge_attr = state_dict['edge_attr'].to(device)
                
                with torch.no_grad():
                    # Pass player role for asymmetric models
                    if hasattr(model_to_use, 'asymmetric_mode') and model_to_use.asymmetric_mode:
                        player_role = board.player  # 0 for attacker (Player 1), 1 for defender (Player 2)
                        policy_output, _ = model_to_use(edge_index, edge_attr, player_role=player_role)
                    else:
                        policy_output, _ = model_to_use(edge_index, edge_attr)
                
                policy_output = policy_output.squeeze().cpu().numpy()
                
                # Apply valid moves mask
                valid_moves_mask = ed.get_valid_moves_mask(board)
                masked_policy = policy_output * valid_moves_mask
                
                # Check if there are any valid moves with non-zero probability
                if masked_policy.sum() > 1e-8:
                    best_move = np.argmax(masked_policy)
                else:
                    # Fallback: Choose a random valid move if policy assigns zero to all
                    valid_moves = board.get_valid_moves()
                    if valid_moves:
                        best_move = ed.encode_action(board, random.choice(valid_moves))
                    else:
                        # Should not happen if game loop condition is correct, but handle just in case
                        best_move = 0 # Or some other default/error handling
            else:
                # Original MCTS search
                best_move, _ = UCT_search(board, num_mcts_sims, model_to_use, 
                                        perspective_mode=perspective_mode)
            
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
    if decided_games_only:
        # Calculate win rate based only on decided games (exclude draws)
        decided_games = current_model_wins + best_model_wins
        if decided_games > 0:
            win_rate = current_model_wins / decided_games
            print(f"Evaluation complete: Current model wins: {current_model_wins}, "
                  f"Best model wins: {best_model_wins}, Draws: {draws}")
            print(f"Decided games: {decided_games}, Current model win rate (decided games only): {win_rate:.4f}")
        else:
            # All games were draws - treat as neutral (50% win rate)
            win_rate = 0.5
            print(f"Evaluation complete: All {draws} games ended in draws.")
            print(f"Using neutral win rate of 0.5 for model comparison.")
    else:
        # Traditional win rate calculation (including draws as losses)
        total_games = current_model_wins + best_model_wins + draws
        win_rate = current_model_wins / total_games
        print(f"Evaluation complete: Current model wins: {current_model_wins}, "
              f"Best model wins: {best_model_wins}, Draws: {draws}")
        if game_mode == "asymmetric":
            print(f"Current model win rate (all games): {win_rate:.4f}")
            print(f"  Note: In asymmetric mode, win rate reflects performance across both roles:")
            print(f"        - As attacker: How often current model forms cliques")
            print(f"        - As defender: How often current model prevents cliques")
        else:
            print(f"Current model win rate (all games): {win_rate:.4f}")
    
    return win_rate

def evaluate_models_asymmetric(current_model: CliqueGNN, other_model: CliqueGNN,
                              num_games: int = 20, num_vertices: int = 6, clique_size: int = 3,
                              num_mcts_sims: int = 100, perspective_mode: str = "alternating",
                              use_policy_only: bool = False, 
                              decided_games_only: bool = False) -> Dict[str, float]:
    """
    Evaluate asymmetric models with role-specific win rates.
    
    Returns:
        Dictionary with keys:
        - 'attacker_win_rate': How often current model (as attacker) beats other model (as defender)
        - 'defender_win_rate': How often current model (as defender) beats other model (as attacker)
    """
    current_model.eval()
    other_model.eval()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_model.to(device)
    other_model.to(device)
    
    results = {}
    
    # Test 1: Current model as attacker vs other model as defender
    print(f"Evaluating: Current model (attacker) vs Other model (defender)")
    attacker_wins = 0
    defender_wins = 0
    draws = 0
    
    for game_idx in range(num_games):
        # Initialize game
        board = CliqueBoard(num_vertices, clique_size, "asymmetric")
        game_over = False
        max_moves = num_vertices * (num_vertices - 1) // 2
        
        while not game_over and board.move_count < max_moves:
            # Current model always plays as Player 1 (attacker), other model as Player 2 (defender)
            model_to_use = current_model if board.player == 0 else other_model
            
            if use_policy_only:
                # Get move from policy head directly
                state_dict = ed.prepare_state_for_network(board)
                edge_index = state_dict['edge_index'].to(device)
                edge_attr = state_dict['edge_attr'].to(device)
                
                with torch.no_grad():
                    player_role = board.player
                    policy_output, _ = model_to_use(edge_index, edge_attr, player_role=player_role)
                
                policy_output = policy_output.squeeze().cpu().numpy()
                valid_moves_mask = ed.get_valid_moves_mask(board)
                masked_policy = policy_output * valid_moves_mask
                
                if masked_policy.sum() > 1e-8:
                    best_move = np.argmax(masked_policy)
                else:
                    valid_moves = board.get_valid_moves()
                    best_move = np.random.choice(valid_moves) if valid_moves else 0
            else:
                # Use MCTS
                best_move, _ = UCT_search(board, num_mcts_sims, model_to_use, 
                                        perspective_mode=perspective_mode)
            
            # Make the move
            board = make_move_on_board(board, best_move)
            game_over = board.game_state != 0
        
        # Record result
        if board.game_state == 1:  # Player 1 (Attacker) wins
            attacker_wins += 1
        elif board.game_state == 2:  # Player 2 (Defender) wins
            defender_wins += 1
        else:  # Draw (game_state == 3) or unfinished (game_state == 0)
            draws += 1
    
    # Calculate attacker win rate
    if decided_games_only:
        decided_games = attacker_wins + defender_wins
        results['attacker_win_rate'] = attacker_wins / decided_games if decided_games > 0 else 0.5
    else:
        results['attacker_win_rate'] = attacker_wins / num_games
    
    print(f"  Attacker wins: {attacker_wins}, Defender wins: {defender_wins}, Draws: {draws}")
    print(f"  Current model (attacker) win rate: {results['attacker_win_rate']:.4f}")
    
    # Test 2: Current model as defender vs other model as attacker
    print(f"Evaluating: Current model (defender) vs Other model (attacker)")
    attacker_wins = 0
    defender_wins = 0
    draws = 0
    
    for game_idx in range(num_games):
        # Initialize game
        board = CliqueBoard(num_vertices, clique_size, "asymmetric")
        game_over = False
        max_moves = num_vertices * (num_vertices - 1) // 2
        
        while not game_over and board.move_count < max_moves:
            # Other model always plays as Player 1 (attacker), current model as Player 2 (defender)
            model_to_use = other_model if board.player == 0 else current_model
            
            if use_policy_only:
                # Get move from policy head directly
                state_dict = ed.prepare_state_for_network(board)
                edge_index = state_dict['edge_index'].to(device)
                edge_attr = state_dict['edge_attr'].to(device)
                
                with torch.no_grad():
                    player_role = board.player
                    policy_output, _ = model_to_use(edge_index, edge_attr, player_role=player_role)
                
                policy_output = policy_output.squeeze().cpu().numpy()
                valid_moves_mask = ed.get_valid_moves_mask(board)
                masked_policy = policy_output * valid_moves_mask
                
                if masked_policy.sum() > 1e-8:
                    best_move = np.argmax(masked_policy)
                else:
                    valid_moves = board.get_valid_moves()
                    best_move = np.random.choice(valid_moves) if valid_moves else 0
            else:
                # Use MCTS
                best_move, _ = UCT_search(board, num_mcts_sims, model_to_use, 
                                        perspective_mode=perspective_mode)
            
            # Make the move
            board = make_move_on_board(board, best_move)
            game_over = board.game_state != 0
        
        # Record result
        if board.game_state == 1:  # Player 1 (Other model as Attacker) wins
            attacker_wins += 1
        elif board.game_state == 2:  # Player 2 (Current model as Defender) wins
            defender_wins += 1
        else:  # Draw (game_state == 3) or unfinished (game_state == 0)
            draws += 1
    
    # Calculate defender win rate
    if decided_games_only:
        decided_games = attacker_wins + defender_wins
        results['defender_win_rate'] = defender_wins / decided_games if decided_games > 0 else 0.5
    else:
        results['defender_win_rate'] = defender_wins / num_games
    
    print(f"  Attacker wins: {attacker_wins}, Defender wins: {defender_wins}, Draws: {draws}")
    print(f"  Current model (defender) win rate: {results['defender_win_rate']:.4f}")
    
    return results

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
        asymmetric_mode = (game_mode == "asymmetric")
        initial_model = CliqueGNN(num_vertices, hidden_dim=hidden_dim, num_layers=num_layers, asymmetric_mode=asymmetric_mode)
        save_initial = {
            'state_dict': initial_model.state_dict(),
            'num_vertices': num_vertices,
            'clique_size': clique_size,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'asymmetric_mode': asymmetric_mode
        }
        torch.save(save_initial, current_model_path) # Save as iter 0
        model_to_load_path = current_model_path
        print(f"Saved initial model to: {model_to_load_path}")

    # --- 2. Load Model for Self-Play --- 
    checkpoint_selfplay = torch.load(model_to_load_path, map_location=torch.device('cpu'))
    # Verify loaded params match config - optional but recommended
    # ... (add checks comparing checkpoint params with args: num_vertices, clique_size, hidden_dim, num_layers)
    # Use asymmetric_mode from saved checkpoint if available, otherwise infer from game_mode
    saved_asymmetric_mode = checkpoint_selfplay.get('asymmetric_mode', game_mode == "asymmetric")
    model_for_self_play = CliqueGNN(num_vertices, hidden_dim=hidden_dim, num_layers=num_layers, asymmetric_mode=saved_asymmetric_mode)
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
                      args=(model_for_self_play, games, num_vertices, clique_size, i, mcts_sims, game_mode, iteration, data_dir, args.perspective_mode, 0.25, args.skill_variation))
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
    # Handle both symmetric and asymmetric return formats
    train_result = train_network(all_examples, iteration, num_vertices, clique_size, model_dir, args)
    if len(train_result) == 4:  # Asymmetric mode
        avg_policy_loss, avg_value_loss, avg_attacker_loss, avg_defender_loss = train_result
        print(f"Training process finished. Val Policy Loss: {avg_policy_loss:.4f}, Val Value Loss: {avg_value_loss:.4f}")
        print(f"  Attacker Policy Loss: {avg_attacker_loss:.4f}, Defender Policy Loss: {avg_defender_loss:.4f}")
    else:  # Symmetric mode
        avg_policy_loss, avg_value_loss = train_result
        avg_attacker_loss, avg_defender_loss = None, None
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
    # Use asymmetric_mode from saved checkpoint if available, otherwise infer from game_mode
    saved_asymmetric_mode = checkpoint_eval.get('asymmetric_mode', game_mode == "asymmetric")
    trained_model = CliqueGNN(num_vertices, hidden_dim=hidden_dim, num_layers=num_layers, asymmetric_mode=saved_asymmetric_mode).to(eval_device)
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
         attacker_win_rate_vs_best = None
         defender_win_rate_vs_best = None
    else:
        print(f"Loading best model for comparison: {best_model_path}")
        checkpoint_best = torch.load(best_model_path, map_location=eval_device)
        # Verify configuration match before evaluation
        if checkpoint_best.get('num_vertices') != num_vertices or \
           checkpoint_best.get('clique_size') != clique_size:
             print("ERROR: Best model configuration does not match current settings. Skipping evaluation.")
             win_rate_vs_best = -1.0 # Indicate configuration mismatch
             attacker_win_rate_vs_best = None
             defender_win_rate_vs_best = None
        else:
            best_hidden = checkpoint_best.get('hidden_dim', hidden_dim)
            best_layers = checkpoint_best.get('num_layers', num_layers)
            asymmetric_mode = (game_mode == "asymmetric")
            best_model_instance = CliqueGNN(num_vertices, hidden_dim=best_hidden, num_layers=best_layers, asymmetric_mode=asymmetric_mode).to(eval_device)
            best_model_instance.load_state_dict(checkpoint_best['state_dict'])
            best_model_instance.eval()
            
            # Run evaluation games (decided games only for best model update)
            if game_mode == "asymmetric":
                # Use role-specific evaluation for asymmetric mode
                role_results = evaluate_models_asymmetric(trained_model, best_model_instance,
                                                        num_games=args.num_games // 2,  # Half games per role
                                                        num_vertices=num_vertices, clique_size=clique_size,
                                                        num_mcts_sims=args.eval_mcts_sims,
                                                        perspective_mode=args.perspective_mode,
                                                        use_policy_only=args.use_policy_only,
                                                        decided_games_only=True)
                # Overall win rate for model selection (average of both roles)
                win_rate_vs_best = (role_results['attacker_win_rate'] + role_results['defender_win_rate']) / 2
                attacker_win_rate_vs_best = role_results['attacker_win_rate']
                defender_win_rate_vs_best = role_results['defender_win_rate']
            else:
                # Use standard evaluation for symmetric mode
                win_rate_vs_best = evaluate_models(trained_model, best_model_instance, 
                                           num_games=args.num_games, # Use num_games from args
                                           num_vertices=num_vertices, clique_size=clique_size,
                                           num_mcts_sims=args.eval_mcts_sims, game_mode=game_mode,
                                           perspective_mode=args.perspective_mode,
                                           use_policy_only=args.use_policy_only,
                                           decided_games_only=True)
                attacker_win_rate_vs_best = None
                defender_win_rate_vs_best = None

    # --- 6b. Evaluate Trained vs Initial --- 
    print("Starting evaluation against initial model (iter0)...")
    win_rate_vs_initial = -2.0 # Default: not evaluated
    win_rate_vs_initial_mcts_1 = -2.0 # Default: not evaluated
    attacker_win_rate_vs_initial = None
    defender_win_rate_vs_initial = None
    attacker_win_rate_vs_initial_mcts_1 = None
    defender_win_rate_vs_initial_mcts_1 = None
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
                 win_rate_vs_initial_mcts_1 = -1.0 # Indicate configuration mismatch
            else:
                initial_hidden = checkpoint_initial.get('hidden_dim', hidden_dim)
                initial_layers = checkpoint_initial.get('num_layers', num_layers)
                asymmetric_mode = (game_mode == "asymmetric")
                initial_model = CliqueGNN(num_vertices, hidden_dim=initial_hidden, num_layers=initial_layers, asymmetric_mode=asymmetric_mode).to(eval_device)
                initial_model.load_state_dict(checkpoint_initial['state_dict'])
                initial_model.eval()
                print("Loaded initial model for comparison.")
                
                # Run evaluation games (trained vs initial) - include draws as losses
                if game_mode == "asymmetric":
                    # Use role-specific evaluation for asymmetric mode
                    initial_role_results = evaluate_models_asymmetric(trained_model, initial_model,
                                                                    num_games=args.num_games // 2,  # Half games per role
                                                                    num_vertices=num_vertices, clique_size=clique_size,
                                                                    num_mcts_sims=args.eval_mcts_sims,
                                                                    perspective_mode=args.perspective_mode,
                                                                    use_policy_only=args.use_policy_only,
                                                                    decided_games_only=False)
                    # Overall win rate (average of both roles)
                    win_rate_vs_initial = (initial_role_results['attacker_win_rate'] + initial_role_results['defender_win_rate']) / 2
                    attacker_win_rate_vs_initial = initial_role_results['attacker_win_rate']
                    defender_win_rate_vs_initial = initial_role_results['defender_win_rate']
                    
                    # MCTS=1 evaluation
                    initial_mcts1_results = evaluate_models_asymmetric(trained_model, initial_model,
                                                                     num_games=50,  # Half of 101 games per role
                                                                     num_vertices=num_vertices, clique_size=clique_size,
                                                                     num_mcts_sims=1,
                                                                     perspective_mode=args.perspective_mode,
                                                                     use_policy_only=True,
                                                                     decided_games_only=False)
                    win_rate_vs_initial_mcts_1 = (initial_mcts1_results['attacker_win_rate'] + initial_mcts1_results['defender_win_rate']) / 2
                    attacker_win_rate_vs_initial_mcts_1 = initial_mcts1_results['attacker_win_rate']
                    defender_win_rate_vs_initial_mcts_1 = initial_mcts1_results['defender_win_rate']
                else:
                    # Use standard evaluation for symmetric mode
                    win_rate_vs_initial = evaluate_models(trained_model, initial_model, 
                                               num_games=args.num_games, # Use num_games from args
                                               num_vertices=num_vertices, clique_size=clique_size,
                                               num_mcts_sims=args.eval_mcts_sims, game_mode=game_mode,
                                               perspective_mode=args.perspective_mode,
                                               use_policy_only=args.use_policy_only,
                                               decided_games_only=False)
                    
                    win_rate_vs_initial_mcts_1 = evaluate_models(trained_model, initial_model, 
                                               num_games=101, # Use num_games from args
                                               num_vertices=num_vertices, clique_size=clique_size,
                                               num_mcts_sims=1, game_mode=game_mode,
                                               perspective_mode=args.perspective_mode,
                                               use_policy_only=True,
                                               decided_games_only=False)
                    attacker_win_rate_vs_initial = None
                    defender_win_rate_vs_initial = None
                    attacker_win_rate_vs_initial_mcts_1 = None
                    defender_win_rate_vs_initial_mcts_1 = None
                
        except Exception as e:
            print(f"ERROR loading or evaluating against initial model: {e}")
            win_rate_vs_initial = -3.0 # Indicate other error
            win_rate_vs_initial_mcts_1 = -3.0 # Indicate other error

    # --- 7. Update Best Model (based on win_rate_vs_best) --- 
    if win_rate_vs_best > eval_threshold:
        print(f"New model is better (Win Rate vs Best: {win_rate_vs_best:.4f} > {eval_threshold}). Updating best model.")
        # Save the current trained model as the new best model
        save_best = {
            'state_dict': trained_model.state_dict(),
            'num_vertices': num_vertices,
            'clique_size': clique_size,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'asymmetric_mode': asymmetric_mode
        }
        torch.save(save_best, best_model_path)
    else:
        print(f"New model is not better (Win Rate vs Best: {win_rate_vs_best:.4f} <= {eval_threshold}). Keeping previous best model.")

    print(f"=== Iteration {iteration} finished ===")
    
    # Return collected metrics
    metrics = {
        "iteration": iteration,
        "validation_policy_loss": avg_policy_loss,
        "validation_value_loss": avg_value_loss,
        "evaluation_win_rate_vs_best": win_rate_vs_best,
        "evaluation_win_rate_vs_initial": win_rate_vs_initial,
        "evaluation_win_rate_vs_initial_mcts_1": win_rate_vs_initial_mcts_1
    }
    
    # Add separate policy losses for asymmetric mode
    if avg_attacker_loss is not None and avg_defender_loss is not None:
        metrics["validation_attacker_policy_loss"] = avg_attacker_loss
        metrics["validation_defender_policy_loss"] = avg_defender_loss
    
    # Add role-specific win rates for asymmetric mode
    if attacker_win_rate_vs_best is not None:
        metrics["evaluation_attacker_win_rate_vs_best"] = attacker_win_rate_vs_best
        metrics["evaluation_defender_win_rate_vs_best"] = defender_win_rate_vs_best
    
    if attacker_win_rate_vs_initial is not None:
        metrics["evaluation_attacker_win_rate_vs_initial"] = attacker_win_rate_vs_initial
        metrics["evaluation_defender_win_rate_vs_initial"] = defender_win_rate_vs_initial
        
    if attacker_win_rate_vs_initial_mcts_1 is not None:
        metrics["evaluation_attacker_win_rate_vs_initial_mcts_1"] = attacker_win_rate_vs_initial_mcts_1
        metrics["evaluation_defender_win_rate_vs_initial_mcts_1"] = defender_win_rate_vs_initial_mcts_1
    
    return metrics

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

    # Check if wandb is already initialized (for sweeps)
    wandb_run = wandb.run if wandb.run is not None else None
    
    if wandb_run is None:
        # Initialize wandb for standalone runs
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb_run = wandb.init(
                project="alphazero_clique", 
                name=f"{args.experiment_name}_{timestamp}",
                config=vars(args),
                resume="allow",
                id=f"pipeline_{args.experiment_name}_{timestamp}"
            )
            print("Weights & Biases initialized for standalone run.")
        except Exception as e:
            print(f"Could not initialize Weights & Biases: {e}. Skipping wandb logging.")
            wandb_run = None
    else:
        # Update config for sweep runs
        wandb.config.update(vars(args), allow_val_change=True)
        print("Using existing wandb run (sweep mode).")

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
            "epochs": args.epochs,
            "use_legacy_policy_loss": args.use_legacy_policy_loss,
            "min_alpha": args.min_alpha,
            "max_alpha": args.max_alpha,
            "value_weight": args.value_weight,
            "use_policy_only": args.use_policy_only,
            "perspective_mode": args.perspective_mode,
            "skill_variation": args.skill_variation,
            "early_stop_patience": args.early_stop_patience,
            "min_iterations": args.min_iterations
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
    
    # Early stopping parameters
    best_validation_loss = float('inf')
    iterations_without_improvement = 0
    early_stop_patience = args.early_stop_patience
    min_iterations = args.min_iterations
    
    print(f"Early stopping configuration:")
    print(f"  - Patience: {early_stop_patience} iterations without improvement")
    print(f"  - Minimum iterations: {min_iterations}")
    print(f"  - Monitoring: validation_policy_loss")
    
    # Initialize best loss from existing log if available
    if log_data["log"]:
        existing_losses = [entry.get("validation_policy_loss") for entry in log_data["log"] 
                         if entry.get("validation_policy_loss") is not None]
        if existing_losses:
            best_validation_loss = min(existing_losses)
            print(f"Best validation policy loss from existing log: {best_validation_loss:.6f}")
    
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

            # Log metrics to wandb if initialized
            if wandb_run and iteration_metrics:
                try:
                    # Make sure iteration is the step key
                    wandb.log(iteration_metrics, step=iteration_metrics.get("iteration", iteration))
                    print(f"Logged iteration {iteration} metrics to Weights & Biases.")
                except Exception as e:
                    print(f"Error logging metrics to Weights & Biases: {e}")
            
            # Early stopping check
            current_validation_loss = iteration_metrics.get("validation_policy_loss")
            if current_validation_loss is not None:
                if current_validation_loss < best_validation_loss:
                    best_validation_loss = current_validation_loss
                    iterations_without_improvement = 0
                    print(f"New best validation policy loss: {best_validation_loss:.6f}")
                else:
                    iterations_without_improvement += 1
                    print(f"No improvement in validation policy loss for {iterations_without_improvement} iterations")
                
                # Check early stopping conditions
                if (iteration >= min_iterations - 1 and  # -1 because iteration is 0-indexed
                    iterations_without_improvement >= early_stop_patience):
                    print(f"\nEarly stopping triggered!")
                    print(f"No improvement in validation policy loss for {iterations_without_improvement} iterations")
                    print(f"Best validation policy loss: {best_validation_loss:.6f}")
                    print(f"Completed {iteration + 1} iterations (minimum {min_iterations} satisfied)")
                    
                    # Log early stopping to wandb
                    if wandb_run:
                        try:
                            wandb.log({
                                "early_stopping_triggered": True,
                                "early_stopping_iteration": iteration + 1,
                                "best_validation_policy_loss": best_validation_loss,
                                "iterations_without_improvement": iterations_without_improvement
                            }, step=iteration)
                        except Exception as e:
                            print(f"Error logging early stopping to Weights & Biases: {e}")
                    
                    break
        else:
            print(f"Iteration {iteration} did not return metrics (likely skipped). Not logging.")
            
        # Optional: Add delay or check for conditions before next iteration
        time.sleep(2) # Small delay
        
    end_time = time.time()
    print(f"\nPipeline finished in {(end_time - start_time):.2f} seconds.")
    
    if log_data["log"]:
        plot_learning_curve(log_file_path) 

    # Finish the wandb run if initialized
    if wandb_run:
        wandb_run.finish()
        print("Weights & Biases run finished.")

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

    # Check if this is an asymmetric experiment
    has_asymmetric_data = any(entry.get("validation_attacker_policy_loss") is not None for entry in log_list)
    
    if has_asymmetric_data:
        # Extract asymmetric data with all role-specific metrics
        plot_data = [
            (entry["iteration"], 
             entry["validation_policy_loss"], 
             entry["validation_value_loss"], 
             entry.get("evaluation_win_rate_vs_initial"),
             entry.get("validation_attacker_policy_loss"),
             entry.get("validation_defender_policy_loss"),
             entry.get("evaluation_attacker_win_rate_vs_initial"),
             entry.get("evaluation_defender_win_rate_vs_initial"),
             entry.get("evaluation_attacker_win_rate_vs_best"),
             entry.get("evaluation_defender_win_rate_vs_best"))
            for entry in log_list
            if entry.get("validation_policy_loss") is not None and entry.get("validation_value_loss") is not None
        ]
        
        # Separate into lists for asymmetric mode
        iterations = [p[0] for p in plot_data]
        policy_losses = [p[1] for p in plot_data]
        value_losses = [p[2] for p in plot_data]
        win_rates_initial = [p[3] if p[3] is not None and p[3] >= 0 else np.nan for p in plot_data]
        attacker_policy_losses = [p[4] if p[4] is not None else np.nan for p in plot_data]
        defender_policy_losses = [p[5] if p[5] is not None else np.nan for p in plot_data]
        attacker_win_rates_initial = [p[6] if p[6] is not None and p[6] >= 0 else np.nan for p in plot_data]
        defender_win_rates_initial = [p[7] if p[7] is not None and p[7] >= 0 else np.nan for p in plot_data]
        attacker_win_rates_best = [p[8] if p[8] is not None and p[8] >= 0 else np.nan for p in plot_data]
        defender_win_rates_best = [p[9] if p[9] is not None and p[9] >= 0 else np.nan for p in plot_data]
    else:
        # Extract data for symmetric mode (original behavior)
        plot_data = [
            (entry["iteration"], entry["validation_policy_loss"], entry["validation_value_loss"], entry.get("evaluation_win_rate_vs_initial"))
            for entry in log_list
            if entry.get("validation_policy_loss") is not None and entry.get("validation_value_loss") is not None
        ]
        
        # Separate into lists, handling None for win_rate_initial if missing
        iterations = [p[0] for p in plot_data]
        policy_losses = [p[1] for p in plot_data]
        value_losses = [p[2] for p in plot_data]
        win_rates_initial = [p[3] if p[3] is not None and p[3] >= 0 else np.nan for p in plot_data]

    if len(iterations) < 2:
        print("Not enough valid data points (with losses) in log file to plot learning curve.")
        return
        
    # --- Plotting --- 
    if has_asymmetric_data:
        # Create comprehensive asymmetric plots with 2x3 subplots
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Policy Losses Comparison
        ax1.plot(iterations, policy_losses, 'k-', marker='o', linewidth=2, markersize=4, label='Combined Policy Loss')
        ax1.plot(iterations, attacker_policy_losses, 'r-', marker='s', linewidth=2, markersize=4, label='Attacker Policy Loss')
        ax1.plot(iterations, defender_policy_losses, 'b-', marker='^', linewidth=2, markersize=4, label='Defender Policy Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Validation Policy Loss')
        ax1.set_title('Policy Losses: Combined vs Role-Specific')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Value Loss
        ax2.plot(iterations, value_losses, 'g-', marker='o', linewidth=2, markersize=4, label='Value Loss')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Validation Value Loss')
        ax2.set_title('Value Loss (Shared Head)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Win Rates vs Initial
        ax3.plot(iterations, win_rates_initial, 'k:', marker='o', linewidth=2, markersize=4, label='Combined Win Rate')
        ax3.plot(iterations, attacker_win_rates_initial, 'r-', marker='s', linewidth=2, markersize=4, label='Attacker Win Rate')
        ax3.plot(iterations, defender_win_rates_initial, 'b-', marker='^', linewidth=2, markersize=4, label='Defender Win Rate')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Win Rate vs Initial')
        ax3.set_title('Performance vs Initial Model')
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Win Rates vs Best
        ax4.plot(iterations, attacker_win_rates_best, 'r-', marker='s', linewidth=2, markersize=4, label='Attacker vs Best Defender')
        ax4.plot(iterations, defender_win_rates_best, 'b-', marker='^', linewidth=2, markersize=4, label='Defender vs Best Attacker')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Win Rate vs Best')
        ax4.set_title('Performance vs Best Model')
        ax4.set_ylim(-0.05, 1.05)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Policy Loss Difference
        policy_diff = [a - d if not (np.isnan(a) or np.isnan(d)) else np.nan 
                      for a, d in zip(attacker_policy_losses, defender_policy_losses)]
        ax5.plot(iterations, policy_diff, 'purple', marker='o', linewidth=2, markersize=4, label='Attacker - Defender Loss')
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Policy Loss Difference')
        ax5.set_title('Learning Balance (Attacker - Defender Loss)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Win Rate Difference vs Initial
        win_diff = [a - d if not (np.isnan(a) or np.isnan(d)) else np.nan 
                   for a, d in zip(attacker_win_rates_initial, defender_win_rates_initial)]
        ax6.plot(iterations, win_diff, 'orange', marker='o', linewidth=2, markersize=4, label='Attacker - Defender Win Rate')
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Win Rate Difference')
        ax6.set_title('Performance Balance (Attacker - Defender)')
        ax6.set_ylim(-1.05, 1.05)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add overall title
        title = f"Asymmetric Training Analysis - {hyperparams.get('experiment_name', 'N/A')}\n"
        title += f"V={hyperparams.get('vertices', '?')}, k={hyperparams.get('k', '?')}, MCTS={hyperparams.get('mcts_sims', '?')}, "
        title += f"LR={hyperparams.get('initial_lr', '?')}"
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        
    else:
        # Original symmetric mode plotting (3 axes)
        fig, ax1 = plt.subplots(figsize=(12, 8))

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
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        color = 'tab:green'
        ax3.set_ylabel('Win Rate vs Initial', color=color, fontsize=14)
        ax3.plot(iterations, win_rates_initial, color=color, marker='^', linestyle=':', linewidth=2, markersize=6, label='Win Rate vs Initial')
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(loc='lower left')

        # Add title
        title = f"Training Losses & Win Rate vs Initial\n"
        title += f"Exp: {hyperparams.get('experiment_name', 'N/A')}, V={hyperparams.get('vertices', '?')}, k={hyperparams.get('k', '?')}, MCTS={hyperparams.get('mcts_sims', '?')}\n"
        title += f"LR={hyperparams.get('initial_lr', '?')}, Factor={hyperparams.get('lr_factor', '?')}, Patience={hyperparams.get('lr_patience', '?')}"
        plt.title(title, fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
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
                   num_mcts_sims: int = 200, perspective_mode: str = "alternating", 
                   human_player: int = 0):
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
            best_move, _ = UCT_search(board, num_mcts_sims, model, 
                                    perspective_mode=perspective_mode)
            
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
    
    # Initialize wandb first for sweep support
    wandb.init(project="alphazero_clique-src")
    
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
    parser.add_argument("--initial-lr", type=float, default=0.00001, help="Initial learning rate for Adam")
    parser.add_argument("--lr-factor", type=float, default=0.7, help="LR reduction factor for ReduceLROnPlateau")
    parser.add_argument("--lr-patience", type=int, default=5, help="LR patience for ReduceLROnPlateau")
    parser.add_argument("--lr-threshold", type=float, default=1e-3, help="LR threshold for ReduceLROnPlateau")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Minimum learning rate for ReduceLROnPlateau")
    
    # Add Training Loop Hyperparameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size during training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs per iteration")
    parser.add_argument("--use-legacy-policy-loss", action='store_true', help="Use the old (potentially problematic) policy loss calculation")
    parser.add_argument("--min-alpha", type=float, default=0.5, help="Weight factor for the value loss component")
    parser.add_argument("--max-alpha", type=float, default=100.0, help="Weight factor for the value loss component")
    parser.add_argument("--value-weight", type=float, default=1.0, help="Weight for value loss in the combined loss function")
    
    # Add perspective mode option
    parser.add_argument("--perspective-mode", type=str, default="alternating", 
                        choices=["fixed", "alternating"], 
                        help="Value perspective mode: 'fixed' (always from Player 1) or 'alternating' (from current player)")
    
    # Add skill variation option
    parser.add_argument("--skill-variation", type=float, default=0.0, 
                        help="Skill variation in MCTS simulation counts (0 = no variation, higher = more variation)")
    
    # Add early stopping parameters
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Number of iterations without improvement before early stopping")
    parser.add_argument("--min-iterations", type=int, default=10,
                        help="Minimum number of iterations before early stopping can be triggered")

    # Specific mode parameters (can add more if needed, e.g., model paths for evaluate/play)
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number (for train mode)")
    parser.add_argument("--num-games", type=int, default=21, help="Number of games (for evaluate/play modes)")
    parser.add_argument("--eval-mcts-sims", type=int, default=30, help="Number of MCTS simulations (for evaluate/play modes)")
    parser.add_argument("--use-policy-only", action='store_true', help="Use policy head output for move selection")

    args = parser.parse_args()
    
    # Override args with wandb config for sweeps
    if hasattr(wandb.config, 'keys') and len(wandb.config.keys()) > 0:
        print("Using wandb config for sweep parameters")
        for key, value in wandb.config.items():
            # Convert wandb config keys to match args attributes
            key = key.replace('-', '_')  # Convert hyphens to underscores
            setattr(args, key, value)
        
        # Set experiment name with wandb run id for uniqueness
        args.experiment_name = f"n7k4_32_8_sweep_{wandb.run.id}"
        print(f"Sweep experiment name: {args.experiment_name}")
    
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
                 
            asymmetric_mode = (args.game_mode == "asymmetric")
            model = CliqueGNN(num_vertices=loaded_v, hidden_dim=loaded_hidden, num_layers=loaded_layers, asymmetric_mode=asymmetric_mode)
            model.load_state_dict(checkpoint['state_dict'])
            model.share_memory() # Important for multiprocessing
            model.eval()
            
            MCTS_self_play(clique_net=model, 
                           num_games=args.self_play_games, 
                           num_vertices=args.vertices, # Use game vertices
                           clique_size=args.k,                   # Use game k
                           cpu=0, 
                           mcts_sims=args.mcts_sims, 
                           game_mode=args.game_mode, 
                           iteration=args.iteration, # Pass iteration if needed for saving
                           data_dir=data_base_dir,
                           perspective_mode=args.perspective_mode,
                           skill_variation=args.skill_variation) 
                           
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
            asymmetric_mode = (args.game_mode == "asymmetric")
            current_model = CliqueGNN(args.vertices, hidden_dim=curr_hidden, num_layers=curr_layers, asymmetric_mode=asymmetric_mode)
            current_model.load_state_dict(chk_curr['state_dict'])
            current_model.eval()
            
            # Load best model
            chk_best = torch.load(best_model_path, map_location=torch.device('cpu'))
            best_hidden = chk_best.get('hidden_dim', args.hidden_dim)
            best_layers = chk_best.get('num_layers', args.num_layers)
            best_model = CliqueGNN(args.vertices, hidden_dim=best_hidden, num_layers=best_layers, asymmetric_mode=asymmetric_mode)
            best_model.load_state_dict(chk_best['state_dict'])
            best_model.eval()

            win_rate = evaluate_models(current_model, best_model, 
                                     num_games=args.num_games, 
                                     num_vertices=args.vertices, 
                                     clique_size=args.k,
                                     num_mcts_sims=args.mcts_sims, 
                                     game_mode=args.game_mode,
                                     perspective_mode=args.perspective_mode,
                                     use_policy_only=args.use_policy_only,
                                     decided_games_only=True)
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
            asymmetric_mode = (args.game_mode == "asymmetric")
            model = CliqueGNN(args.vertices, hidden_dim=loaded_hidden, num_layers=loaded_layers, asymmetric_mode=asymmetric_mode)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            
            play_against_ai(model_path, args.vertices, args.k, args.mcts_sims, 
                          args.perspective_mode, human_player=0) 
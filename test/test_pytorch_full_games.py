#!/usr/bin/env python
"""Test PyTorch MCTS performance for complete games."""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

import time
import torch
import numpy as np
from clique_board import CliqueBoard
from MCTS_clique import UCT_search
from alpha_net_clique import CliqueGNN

# Test parameters
num_games = 10
num_simulations = 20  # Per move

# Test only n=6 for now to get quick results
test_configs = [
    (6, 3),   # n=6, k=3 (original size)
    # (14, 4),  # n=14, k=4 (larger size) - skip for speed
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("=" * 70)
print(f"PYTORCH FULL GAME PERFORMANCE TEST")
print("=" * 70)
print(f"Games: {num_games}")
print(f"MCTS simulations per move: {num_simulations}")
print()

for n, k in test_configs:
    print(f"\nTESTING n={n}, k={k}")
    print("-" * 50)
    print(f"Action space: {n*(n-1)//2} possible edges")
    
    # Create neural network
    print("Creating neural network...")
    net_start = time.time()
    net = CliqueGNN(
        num_vertices=n,
        hidden_dim=64,
        num_layers=3
    ).to(device)
    net.eval()
    print(f"Network created in {time.time() - net_start:.2f}s")
    
    # Warmup
    print("Warmup run...")
    board = CliqueBoard(n, k, game_mode="symmetric")
    UCT_search(board, 1, net, perspective_mode="alternating", noise_weight=0.0)
    
    # Play complete games
    print(f"\nPlaying {num_games} complete games...")
    
    game_times = []
    move_counts = []
    moves_per_second = []
    
    total_start = time.time()
    
    for game_idx in range(num_games):
        game_start = time.time()
        board = CliqueBoard(n, k, game_mode="symmetric")
        
        move_count = 0
        move_times = []
        
        # Play until game ends
        while board.game_state == 0:
            move_start = time.time()
            
            # Run MCTS
            best_move, root = UCT_search(
                board, 
                num_simulations, 
                net, 
                perspective_mode="alternating", 
                noise_weight=0.25  # Add some exploration
            )
            
            # Make the move
            board.make_move(best_move)
            
            move_time = time.time() - move_start
            move_times.append(move_time)
            move_count += 1
            
            # Safety check
            if move_count > 100:
                print(f"  Game {game_idx+1}: Exceeded 100 moves, ending game")
                break
        
        game_time = time.time() - game_start
        game_times.append(game_time)
        move_counts.append(move_count)
        moves_per_second.append(move_count / game_time)
        
        # Determine winner
        if board.game_state == 1:
            winner = "Player 1"
        elif board.game_state == 2:
            winner = "Player 2"
        else:
            winner = "Draw"
        
        print(f"  Game {game_idx+1}: {move_count} moves, {game_time:.2f}s, Winner: {winner}")
    
    total_time = time.time() - total_start
    
    # Statistics
    avg_game_time = np.mean(game_times)
    std_game_time = np.std(game_times)
    avg_moves = np.mean(move_counts)
    std_moves = np.std(move_counts)
    avg_move_rate = np.mean(moves_per_second)
    
    print(f"\nSUMMARY for n={n}, k={k}:")
    print(f"  Total time for {num_games} games: {total_time:.2f}s")
    print(f"  Average game time: {avg_game_time:.2f}s ± {std_game_time:.2f}s")
    print(f"  Average moves per game: {avg_moves:.1f} ± {std_moves:.1f}")
    print(f"  Average time per move: {avg_game_time/avg_moves:.3f}s")
    print(f"  Moves per second: {avg_move_rate:.1f}")
    
    # Extrapolation
    print(f"\nExtrapolation:")
    print(f"  Time for 100 games: {avg_game_time * 100:.1f}s ({avg_game_time * 100 / 60:.1f} minutes)")
    print(f"  Time for 500 games: {avg_game_time * 500:.1f}s ({avg_game_time * 500 / 60:.1f} minutes)")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
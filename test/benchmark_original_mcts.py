#!/usr/bin/env python
"""Benchmark original PyTorch MCTS for comparison."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
from src.clique_board import CliqueBoard
from src.MCTS_clique import MCTS
from src.alpha_net_clique import GNNPolicyValueNet_clique

def benchmark_original_mcts(num_games_list, num_sims_list):
    """Test original MCTS with different parameters."""
    
    # Fixed parameters
    num_vertices = 6
    k = 3
    
    # Create neural network once
    print("Creating neural network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    nn = GNNPolicyValueNet_clique(
        num_vertices=num_vertices,
        hidden_dim=64,
        num_gnn_layers=3,
        device=device
    )
    nn.eval()
    
    print("\n" + "="*70)
    print("ORIGINAL PYTORCH MCTS BENCHMARK - n=6, k=3")
    print("="*70)
    print(f"{'Games':>8} {'Sims':>8} {'Total Time':>12} {'Time/Sim':>12} {'Time/Game':>12} {'Time/Move':>12}")
    print("-"*70)
    
    results = []
    
    for num_games in num_games_list:
        for num_sims in num_sims_list:
            # Time multiple games sequentially (original doesn't batch)
            total_time = 0
            
            for game_idx in range(num_games):
                # Create game and MCTS
                board = CliqueBoard(num_vertices=num_vertices, k=k)
                mcts = MCTS(board, nn, c_puct=3.0)
                
                # Time single game MCTS
                start_time = time.time()
                try:
                    # Run MCTS simulations
                    for _ in range(num_sims):
                        mcts.run_simulation()
                    
                    # Get action probabilities
                    probs = mcts.get_action_probabilities(temperature=1.0)
                    
                    game_time = time.time() - start_time
                    total_time += game_time
                    
                except Exception as e:
                    print(f"Game {game_idx} failed: {e}")
                    break
            
            if total_time > 0:
                # Calculate metrics
                time_per_sim = total_time / num_sims
                time_per_game = total_time / num_games
                time_per_move = total_time / (num_games * num_sims)
                
                print(f"{num_games:8d} {num_sims:8d} {total_time:12.2f}s {time_per_sim:12.3f}s {time_per_game:12.3f}s {time_per_move:12.4f}s")
                
                results.append({
                    'games': num_games,
                    'sims': num_sims,
                    'total_time': total_time,
                    'time_per_sim': time_per_sim,
                    'time_per_game': time_per_game,
                    'time_per_move': time_per_move
                })
    
    print("-"*70)
    
    # Analysis
    print("\nKEY INSIGHTS:")
    if results:
        import numpy as np
        avg_time_per_move = np.mean([r['time_per_move'] for r in results])
        print(f"- Average time per move: {avg_time_per_move*1000:.1f}ms")
        print(f"- Estimated time for 200 sims, 100 games: {avg_time_per_move * 200 * 100:.1f}s ({avg_time_per_move * 200 * 100 / 60:.1f} minutes)")
        print(f"- Estimated time for 20 sims, 100 games: {avg_time_per_move * 20 * 100:.1f}s ({avg_time_per_move * 20 * 100 / 60:.1f} minutes)")
        
        # Compare first entry with JAX
        if results[0]['games'] == 1 and results[0]['sims'] == 5:
            print(f"\nComparison for 1 game, 5 sims:")
            print(f"- Original PyTorch: {results[0]['total_time']:.2f}s")
            print(f"- JAX SimpleTreeMCTS: 4.24s")
            print(f"- JAX is {results[0]['total_time']/4.24:.1f}x slower")

if __name__ == "__main__":
    # Test same combinations as JAX
    num_games_list = [1, 2, 5, 10, 20]
    num_sims_list = [5, 10, 20]
    
    print("Starting original PyTorch MCTS benchmark...")
    print("This will take several minutes...\n")
    
    benchmark_original_mcts(num_games_list, num_sims_list)
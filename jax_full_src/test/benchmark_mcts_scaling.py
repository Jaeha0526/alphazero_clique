#!/usr/bin/env python
"""Benchmark MCTS scaling with number of games and simulations."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from simple_tree_mcts import SimpleTreeMCTS

def benchmark_mcts(num_games_list, num_sims_list):
    """Test different combinations of games and simulations."""
    
    # Fixed parameters
    num_vertices = 6
    k = 3
    num_actions = 15
    
    # Create neural network once
    print("Creating neural network...")
    nn = ImprovedBatchedNeuralNetwork(
        num_vertices=num_vertices,
        hidden_dim=32,
        num_layers=2,
        asymmetric_mode=False
    )
    
    print("\n" + "="*70)
    print("MCTS SCALING BENCHMARK - n=6, k=3")
    print("="*70)
    print(f"{'Games':>8} {'Sims':>8} {'Total Time':>12} {'Time/Sim':>12} {'Time/Game':>12} {'Time/Move':>12}")
    print("-"*70)
    
    results = []
    
    for num_games in num_games_list:
        for num_sims in num_sims_list:
            # Create boards and MCTS for this batch size
            boards = VectorizedCliqueBoard(
                batch_size=num_games,
                num_vertices=num_vertices,
                k=k,
                game_mode="symmetric"
            )
            
            mcts = SimpleTreeMCTS(
                batch_size=num_games,
                num_actions=num_actions,
                c_puct=3.0,
                max_nodes_per_game=100
            )
            
            # Time the MCTS search
            start_time = time.time()
            try:
                probs = mcts.search(boards, nn, num_sims, temperature=1.0)
                elapsed = time.time() - start_time
                
                # Calculate metrics
                time_per_sim = elapsed / num_sims
                time_per_game = elapsed / num_games
                time_per_move = elapsed / (num_games * num_sims)
                
                print(f"{num_games:8d} {num_sims:8d} {elapsed:12.2f}s {time_per_sim:12.3f}s {time_per_game:12.3f}s {time_per_move:12.4f}s")
                
                results.append({
                    'games': num_games,
                    'sims': num_sims,
                    'total_time': elapsed,
                    'time_per_sim': time_per_sim,
                    'time_per_game': time_per_game,
                    'time_per_move': time_per_move
                })
                
            except Exception as e:
                print(f"{num_games:8d} {num_sims:8d} {'FAILED':>12} {str(e)[:40]}")
    
    print("-"*70)
    
    # Analysis
    print("\nANALYSIS:")
    print("---------")
    
    if len(results) > 1:
        # Check scaling with number of games
        fixed_sims = num_sims_list[0]
        games_results = [r for r in results if r['sims'] == fixed_sims]
        if len(games_results) > 1:
            print(f"\nScaling with number of games (fixed {fixed_sims} simulations):")
            base_time = games_results[0]['total_time']
            base_games = games_results[0]['games']
            for r in games_results:
                scaling_factor = r['total_time'] / base_time
                ideal_scaling = r['games'] / base_games
                efficiency = ideal_scaling / scaling_factor * 100 if scaling_factor > 0 else 0
                print(f"  {r['games']:3d} games: {r['total_time']:6.2f}s (scaling: {scaling_factor:.2f}x, ideal: {ideal_scaling:.2f}x, efficiency: {efficiency:.0f}%)")
        
        # Check scaling with number of simulations
        fixed_games = num_games_list[0]
        sims_results = [r for r in results if r['games'] == fixed_games]
        if len(sims_results) > 1:
            print(f"\nScaling with number of simulations (fixed {fixed_games} games):")
            base_time = sims_results[0]['total_time']
            base_sims = sims_results[0]['sims']
            for r in sims_results:
                scaling_factor = r['total_time'] / base_time
                ideal_scaling = r['sims'] / base_sims
                efficiency = ideal_scaling / scaling_factor * 100 if scaling_factor > 0 else 0
                print(f"  {r['sims']:3d} sims: {r['total_time']:6.2f}s (scaling: {scaling_factor:.2f}x, ideal: {ideal_scaling:.2f}x, efficiency: {efficiency:.0f}%)")
    
    print("\nKEY INSIGHTS:")
    if results:
        avg_time_per_move = np.mean([r['time_per_move'] for r in results])
        print(f"- Average time per move: {avg_time_per_move*1000:.1f}ms")
        print(f"- Estimated time for 200 sims, 100 games: {avg_time_per_move * 200 * 100:.1f}s ({avg_time_per_move * 200 * 100 / 60:.1f} minutes)")
        print(f"- Estimated time for 20 sims, 100 games: {avg_time_per_move * 20 * 100:.1f}s ({avg_time_per_move * 20 * 100 / 60:.1f} minutes)")

if __name__ == "__main__":
    # Test different combinations
    num_games_list = [1, 2, 5, 10, 20]
    num_sims_list = [5, 10, 20]
    
    print("Starting MCTS scaling benchmark...")
    print("This will take several minutes...\n")
    
    benchmark_mcts(num_games_list, num_sims_list)
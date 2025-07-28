#!/usr/bin/env python
"""Test original MCTS with detailed timing analysis."""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

import time
import torch
from clique_board import CliqueBoard
from MCTS_clique_timed import UCT_search_timed, MCTS_self_play_timed
from alpha_net_clique import CliqueGNN

# Parameters matching JAX test
n = 6
k = 3
num_games = 10  # Match JAX batch size
num_sims = 20

print("=" * 70)
print("ORIGINAL PYTORCH MCTS TIMING ANALYSIS")
print("=" * 70)
print(f"Setup: n={n}, k={k}, games={num_games}, simulations={num_sims}")

# Create neural network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

net = CliqueGNN(
    num_vertices=n,
    hidden_dim=64,
    num_layers=3
).to(device)
net.eval()

# Test single game first
print("\n1. SINGLE GAME TEST:")
print("-" * 40)

board = CliqueBoard(n, k, game_mode="symmetric")
best_move, root, timing = UCT_search_timed(board, num_sims, net, perspective_mode="alternating", noise_weight=0.0)

print(f"\nBest move: {best_move}")
print(f"Root visits: {root.number_visits}")

# Test multiple games sequentially
print("\n\n2. MULTIPLE GAMES TEST (Sequential):")
print("-" * 40)

total_start = time.time()
os.makedirs('./test_timing', exist_ok=True)

timing_results = MCTS_self_play_timed(
    net, 
    num_games=num_games, 
    vertices=n, 
    k=k, 
    cpu=0,
    num_simulations=num_sims,
    save_dir='./test_timing',
    noise_weight=0.0  # Disable noise to avoid shape issues
)

total_time = time.time() - total_start

print(f"\n\n3. COMPARISON WITH JAX:")
print("-" * 40)
print(f"Original PyTorch ({num_games} games sequential):")
print(f"  Total time: {total_time:.1f}s")
print(f"  Time per game: {total_time/num_games:.1f}s")
print(f"  Time per move: {np.mean(timing_results['avg_move_times']):.2f}s")

print(f"\nJAX SimpleTreeMCTS ({num_games} games parallel):")
print(f"  Total time per move: ~35s (from logs)")
print(f"  Time per game per move: ~3.5s")
import numpy as np

print(f"  Speedup: Original is {3.5/np.mean(timing_results['avg_move_times']):.1f}x faster per game")
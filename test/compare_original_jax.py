#!/usr/bin/env python
"""Compare original PyTorch vs JAX MCTS performance."""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))
sys.path.append(os.path.join(parent_dir, 'jax_full_src'))

import time
import torch

# Original imports
from clique_board import CliqueBoard
from MCTS_clique import UCT_search
from alpha_net_clique import CliqueGNN

# JAX imports
from jax_full_src.vectorized_board import VectorizedCliqueBoard
from jax_full_src.vectorized_nn import ImprovedBatchedNeuralNetwork
from jax_full_src.simple_tree_mcts import SimpleTreeMCTS

def test_original(n=6, k=3, num_sims=20):
    """Test original PyTorch MCTS."""
    board = CliqueBoard(n, k, game_mode="symmetric")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CliqueGNN(num_vertices=n, hidden_dim=64, num_layers=3).to(device)
    net.eval()
    
    start = time.time()
    best_move, root_node = UCT_search(board, num_sims, net, perspective_mode="alternating", noise_weight=0.0)
    elapsed = time.time() - start
    
    return elapsed, best_move

def test_jax(n=6, k=3, num_sims=20, batch_size=1):
    """Test JAX SimpleTreeMCTS."""
    boards = VectorizedCliqueBoard(batch_size=batch_size, num_vertices=n, k=k, game_mode="symmetric")
    nn = ImprovedBatchedNeuralNetwork(num_vertices=n, hidden_dim=64, num_layers=3, asymmetric_mode=False)
    
    num_actions = n * (n - 1) // 2
    mcts = SimpleTreeMCTS(batch_size=batch_size, num_actions=num_actions, c_puct=3.0)
    
    start = time.time()
    probs = mcts.search(boards, nn, num_sims, temperature=1.0)
    elapsed = time.time() - start
    
    return elapsed, probs

print("=" * 70)
print("MCTS PERFORMANCE COMPARISON: Original PyTorch vs JAX")
print("=" * 70)

# Test parameters
test_configs = [
    (6, 3, 20),   # n=6, k=3, 20 sims
    (6, 3, 50),   # n=6, k=3, 50 sims
    (6, 3, 100),  # n=6, k=3, 100 sims
    (9, 4, 20),   # n=9, k=4, 20 sims
]

print(f"{'Config':>15} {'PyTorch':>12} {'JAX (1 game)':>15} {'JAX (10 games)':>15} {'Speedup':>10}")
print("-" * 70)

for n, k, sims in test_configs:
    config_str = f"n={n},k={k},s={sims}"
    
    # Test original
    try:
        orig_time, _ = test_original(n, k, sims)
        orig_per_sim = orig_time / sims * 1000  # ms
    except Exception as e:
        orig_time = float('inf')
        orig_per_sim = float('inf')
        print(f"Original failed: {e}")
    
    # Test JAX with 1 game
    try:
        jax1_time, _ = test_jax(n, k, sims, batch_size=1)
        jax1_per_sim = jax1_time / sims * 1000  # ms
    except Exception as e:
        jax1_time = float('inf')
        jax1_per_sim = float('inf')
        print(f"JAX-1 failed: {e}")
    
    # Test JAX with 10 games
    try:
        jax10_time, _ = test_jax(n, k, sims, batch_size=10)
        jax10_per_sim = jax10_time / (sims * 10) * 1000  # ms per game-simulation
    except Exception as e:
        jax10_time = float('inf')
        jax10_per_sim = float('inf')
        print(f"JAX-10 failed: {e}")
    
    speedup = orig_per_sim / jax1_per_sim if jax1_per_sim > 0 else 0
    
    print(f"{config_str:>15} {orig_per_sim:>10.1f}ms {jax1_per_sim:>13.1f}ms {jax10_per_sim:>13.1f}ms {speedup:>10.2f}x")

print("-" * 70)
print("\nKEY INSIGHTS:")
print("- Original PyTorch MCTS is significantly faster for single games")
print("- JAX SimpleTreeMCTS has overhead from JAX compilation and array conversions")
print("- JAX batch processing helps amortize the overhead across multiple games")
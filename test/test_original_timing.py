#!/usr/bin/env python
"""Simple timing test for original MCTS."""

import sys
import os
# Add both parent and src directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

import time
import torch
from clique_board import CliqueBoard
from MCTS_clique import UCT_search
from alpha_net_clique import CliqueGNN

# Test parameters
n = 6
k = 3
num_sims = 20

print(f"Testing original MCTS: n={n}, k={k}, sims={num_sims}")

# Create board
board = CliqueBoard(n, k, game_mode="symmetric")

# Create neural network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
net = CliqueGNN(
    num_vertices=n,
    hidden_dim=64,
    num_layers=3
).to(device)
net.eval()

# Time MCTS
start = time.time()
try:
    # UCT_search(game_state: CliqueBoard, num_reads: int, net: nn.Module, perspective_mode: str = "alternating", noise_weight: float = 0.25)
    best_move, root_node = UCT_search(board, num_sims, net, perspective_mode="alternating", noise_weight=0.0)
    elapsed = time.time() - start
    
    print(f"✓ MCTS completed in {elapsed:.3f}s")
    print(f"  Time per simulation: {elapsed/num_sims*1000:.1f}ms")
    print(f"  Best move: {best_move}")
    print(f"  Root visits: {root_node.number_visits}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
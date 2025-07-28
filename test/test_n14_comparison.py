#!/usr/bin/env python
"""Compare JAX vs PyTorch MCTS performance with n=14, k=4."""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))
sys.path.append(os.path.join(parent_dir, 'jax_full_src'))

import time
import numpy as np
import torch

# Parameters
n = 14
k = 4
num_sims = 10  # Reduced for testing
batch_size = 5  # Reduced for testing

print("=" * 70)
print(f"PERFORMANCE COMPARISON: n={n}, k={k}")
print("=" * 70)
print(f"Action space: {n*(n-1)//2} edges")
print(f"Simulations: {num_sims}")
print(f"Batch size: {batch_size}")
print()

# Test 1: Original PyTorch
print("1. TESTING ORIGINAL PYTORCH MCTS")
print("-" * 40)

try:
    from clique_board import CliqueBoard
    from MCTS_clique import UCT_search
    from alpha_net_clique import CliqueGNN
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create network
    print("Creating PyTorch neural network...")
    start = time.time()
    pytorch_net = CliqueGNN(
        num_vertices=n,
        hidden_dim=64,
        num_layers=3
    ).to(device)
    pytorch_net.eval()
    print(f"Network created in {time.time() - start:.2f}s")
    
    # Test single game
    board = CliqueBoard(n, k, game_mode="symmetric")
    
    # Warmup
    print("Warmup run...")
    UCT_search(board, 1, pytorch_net, perspective_mode="alternating", noise_weight=0.0)
    
    # Timed run
    print(f"Running {num_sims} simulations...")
    start = time.time()
    best_move, root = UCT_search(board, num_sims, pytorch_net, perspective_mode="alternating", noise_weight=0.0)
    pytorch_time = time.time() - start
    
    print(f"✓ PyTorch MCTS completed in {pytorch_time:.3f}s")
    print(f"  Time per simulation: {pytorch_time/num_sims*1000:.1f}ms")
    print(f"  Root visits: {root.number_visits}")
    
    # Multiple games (sequential)
    print(f"\nTesting {batch_size} games sequentially...")
    start = time.time()
    for i in range(batch_size):
        board = CliqueBoard(n, k, game_mode="symmetric")
        UCT_search(board, num_sims, pytorch_net, perspective_mode="alternating", noise_weight=0.0)
    pytorch_batch_time = time.time() - start
    print(f"✓ {batch_size} games completed in {pytorch_batch_time:.3f}s")
    print(f"  Time per game: {pytorch_batch_time/batch_size:.3f}s")
    
except Exception as e:
    print(f"✗ PyTorch test failed: {e}")
    import traceback
    traceback.print_exc()
    pytorch_time = float('inf')
    pytorch_batch_time = float('inf')

print()

# Test 2: JAX SimpleTreeMCTS
print("2. TESTING JAX SIMPLETREEMCTS")
print("-" * 40)

try:
    from jax_full_src.vectorized_board import VectorizedCliqueBoard
    from jax_full_src.vectorized_nn import ImprovedBatchedNeuralNetwork
    from jax_full_src.simple_tree_mcts_timed import SimpleTreeMCTSTimed
    import jax
    
    print(f"JAX devices: {jax.devices()}")
    
    # Create network
    print("Creating JAX neural network...")
    start = time.time()
    jax_net = ImprovedBatchedNeuralNetwork(
        num_vertices=n,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=False
    )
    print(f"Network created in {time.time() - start:.2f}s")
    
    # Test batch of games
    print(f"\nTesting {batch_size} games in parallel...")
    boards = VectorizedCliqueBoard(
        batch_size=batch_size,
        num_vertices=n,
        k=k,
        game_mode="symmetric"
    )
    
    num_actions = n * (n - 1) // 2
    mcts = SimpleTreeMCTSTimed(
        batch_size=batch_size,
        num_actions=num_actions,
        c_puct=3.0,
        max_nodes_per_game=100
    )
    
    # Warmup
    print("Warmup run...")
    boards_warmup = VectorizedCliqueBoard(batch_size=1, num_vertices=n, k=k, game_mode="symmetric")
    mcts_warmup = SimpleTreeMCTSTimed(batch_size=1, num_actions=num_actions, c_puct=3.0)
    mcts_warmup.search(boards_warmup, jax_net, 1, temperature=1.0)
    
    # Timed run
    print(f"Running {num_sims} simulations for {batch_size} games...")
    start = time.time()
    probs = mcts.search(boards, jax_net, num_sims, temperature=1.0)
    jax_batch_time = time.time() - start
    
    print(f"\n✓ JAX MCTS completed in {jax_batch_time:.3f}s")
    print(f"  Time per game: {jax_batch_time/batch_size:.3f}s")
    print(f"  Time per simulation per game: {jax_batch_time/(num_sims*batch_size)*1000:.1f}ms")
    
except Exception as e:
    print(f"✗ JAX test failed: {e}")
    import traceback
    traceback.print_exc()
    jax_batch_time = float('inf')

print()

# Comparison
print("3. COMPARISON RESULTS")
print("-" * 40)

if pytorch_batch_time < float('inf') and jax_batch_time < float('inf'):
    speedup = jax_batch_time / pytorch_batch_time
    print(f"PyTorch ({batch_size} games sequential): {pytorch_batch_time:.3f}s")
    print(f"JAX ({batch_size} games parallel): {jax_batch_time:.3f}s")
    print(f"PyTorch is {speedup:.1f}x faster")
    
    print(f"\nPer-game comparison:")
    print(f"  PyTorch: {pytorch_batch_time/batch_size:.3f}s per game")
    print(f"  JAX: {jax_batch_time/batch_size:.3f}s per game")
    
    print(f"\nPer-simulation comparison:")
    print(f"  PyTorch: {pytorch_batch_time/(batch_size*num_sims)*1000:.1f}ms")
    print(f"  JAX: {jax_batch_time/(batch_size*num_sims)*1000:.1f}ms")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
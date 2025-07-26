#!/usr/bin/env python
"""
Compare MCTS speed: PyTorch vs JAX implementations.
"""

import time
import sys
import os
import torch
import numpy as np

# Add paths
sys.path.append('/workspace/alphazero_clique/src')
sys.path.append('/workspace/alphazero_clique/jax_full_src')

# PyTorch imports
from clique_board import CliqueBoard
from alpha_net_clique import CliqueGNN
from MCTS_clique import UCT_search

# JAX imports
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from tree_based_mcts import TreeBasedMCTS
from vectorized_mcts_jit import JITVectorizedMCTS


def test_pytorch_mcts():
    """Test PyTorch MCTS speed."""
    print("PyTorch MCTS Performance")
    print("-"*40)
    
    # Initialize
    board = CliqueBoard(num_vertices=6, k=3, game_mode="symmetric")
    model = CliqueGNN(num_vertices=6)
    model.eval()
    
    # Test with different simulation counts
    for num_sims in [5, 20, 50]:
        start = time.time()
        action, root = UCT_search(board, num_sims, model, perspective_mode="alternating")
        elapsed = time.time() - start
        
        print(f"{num_sims} simulations: {elapsed:.3f}s ({elapsed/num_sims:.3f}s per sim)")


def test_jax_implementations():
    """Test JAX MCTS implementations."""
    # Initialize
    boards = VectorizedCliqueBoard(1, num_vertices=6, k=3, game_mode="symmetric")
    model = ImprovedBatchedNeuralNetwork(num_vertices=6)
    
    # 1. Broken JIT MCTS (fast but wrong)
    print("\n\nJAX JIT MCTS (Broken - No Tree Search)")
    print("-"*40)
    
    jit_mcts = JITVectorizedMCTS(1, num_actions=15, c_puct=3.0)
    
    for num_sims in [5, 20, 50]:
        # Warm up JIT
        _ = jit_mcts.search(boards, model, np.array([num_sims]), temperature=1.0)
        
        start = time.time()
        _ = jit_mcts.search(boards, model, np.array([num_sims]), temperature=1.0)
        elapsed = time.time() - start
        
        print(f"{num_sims} simulations: {elapsed:.3f}s (but doesn't actually search!)")
    
    # 2. Fixed Tree MCTS (correct but slow)
    print("\n\nJAX Tree MCTS (Fixed - Proper Search)")
    print("-"*40)
    
    tree_mcts = TreeBasedMCTS(num_actions=15, c_puct=3.0)
    
    for num_sims in [5, 20, 50]:
        start = time.time()
        _ = tree_mcts.search(boards, model, num_sims, game_idx=0)
        elapsed = time.time() - start
        
        print(f"{num_sims} simulations: {elapsed:.3f}s ({elapsed/num_sims:.3f}s per sim)")


def analyze_bottlenecks():
    """Analyze where the time is spent."""
    print("\n\nBottleneck Analysis")
    print("="*60)
    
    boards = VectorizedCliqueBoard(1, num_vertices=6, k=3)
    model = ImprovedBatchedNeuralNetwork(num_vertices=6)
    tree_mcts = TreeBasedMCTS(num_actions=15)
    
    # Time different operations
    print("\n1. Neural Network Evaluation:")
    edge_indices, edge_features = boards.get_features_for_nn_undirected()
    valid_mask = boards.get_valid_moves_mask()
    
    # Single evaluation
    start = time.time()
    for _ in range(10):
        _, _ = model.evaluate_batch(edge_indices, edge_features, valid_mask)
    nn_time = (time.time() - start) / 10
    print(f"   Single NN evaluation: {nn_time*1000:.1f}ms")
    
    # Batch evaluation
    batch_indices = np.tile(edge_indices, (16, 1, 1))
    batch_features = np.tile(edge_features, (16, 1, 1))
    batch_mask = np.tile(valid_mask, (16, 1))
    
    start = time.time()
    _, _ = model.evaluate_batch(batch_indices, batch_features, batch_mask)
    batch_time = time.time() - start
    print(f"   Batch-16 NN evaluation: {batch_time*1000:.1f}ms ({batch_time/16*1000:.1f}ms per position)")
    
    print("\n2. Board Operations:")
    start = time.time()
    for _ in range(100):
        boards.make_moves(np.array([0]))
        boards = VectorizedCliqueBoard(1, num_vertices=6, k=3)  # Reset
    board_time = (time.time() - start) / 100
    print(f"   Make move + reset: {board_time*1000:.1f}ms")
    
    print("\n3. Tree Operations (Python objects):")
    from tree_based_mcts import MCTSNode
    
    start = time.time()
    for _ in range(1000):
        node = MCTSNode(board_state=boards)
        node.visit_count += 1
        node.value_sum += 0.5
    tree_time = (time.time() - start) / 1000
    print(f"   Create node + update: {tree_time*1000:.1f}ms")
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"- NN evaluation is the main bottleneck: {nn_time*1000:.1f}ms per call")
    print(f"- With 20 simulations: ~{nn_time*20:.1f}s just for NN calls")
    print(f"- Python object creation adds overhead")
    print(f"- Need to batch NN evaluations across multiple tree nodes")
    print("="*60)


if __name__ == "__main__":
    test_pytorch_mcts()
    test_jax_implementations()
    analyze_bottlenecks()
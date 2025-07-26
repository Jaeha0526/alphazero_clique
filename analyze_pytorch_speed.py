#!/usr/bin/env python
"""
Analyze why PyTorch MCTS is so fast.
"""

import time
import sys
import torch
import numpy as np

sys.path.append('/workspace/alphazero_clique/src')

from clique_board import CliqueBoard
from alpha_net_clique import CliqueGNN
from MCTS_clique import UCT_search, UCTNode
import encoder_decoder_clique as ed


def profile_pytorch_mcts():
    """Profile PyTorch MCTS to understand its speed."""
    print("PyTorch MCTS Profiling")
    print("="*60)
    
    # Initialize
    board = CliqueBoard(num_vertices=6, k=3)
    model = CliqueGNN(num_vertices=6)
    model.eval()
    
    # Track NN calls
    nn_call_count = 0
    nn_total_time = 0
    
    original_forward = model.forward
    def counting_forward(*args, **kwargs):
        nonlocal nn_call_count, nn_total_time
        nn_call_count += 1
        start = time.time()
        result = original_forward(*args, **kwargs)
        nn_total_time += time.time() - start
        return result
    model.forward = counting_forward
    
    # Run MCTS
    print("\n1. Running MCTS with tracking...")
    start_time = time.time()
    action, root = UCT_search(board, 20, model)
    total_time = time.time() - start_time
    
    print(f"Total time: {total_time:.3f}s")
    print(f"NN calls: {nn_call_count}")
    print(f"NN total time: {nn_total_time:.3f}s")
    print(f"NN avg time: {nn_total_time/nn_call_count*1000:.1f}ms per call")
    print(f"Other operations: {(total_time - nn_total_time):.3f}s")
    
    # Check if PyTorch is caching
    print("\n2. Checking for caching/reuse...")
    
    # Count unique positions evaluated
    positions_evaluated = set()
    
    def tracking_forward(edge_index, edge_attr, **kwargs):
        # Create hash of position
        pos_hash = hash((tuple(edge_index.flatten().tolist()), 
                        tuple(edge_attr.flatten().tolist())))
        positions_evaluated.add(pos_hash)
        return original_forward(edge_index, edge_attr, **kwargs)
    
    model.forward = tracking_forward
    nn_call_count = 0
    
    action, root = UCT_search(board, 20, model)
    
    print(f"Unique positions evaluated: {len(positions_evaluated)}")
    print(f"Total NN calls: {nn_call_count}")
    
    # Analyze tree structure
    print("\n3. Analyzing tree structure...")
    
    def count_nodes(node):
        count = 1
        for child in node.children.values():
            count += count_nodes(child)
        return count
    
    total_nodes = count_nodes(root)
    print(f"Total nodes in tree: {total_nodes}")
    print(f"Root visit count: {root.number_visits}")
    
    # Check PyTorch device
    print("\n4. Device and optimization checks...")
    print(f"PyTorch device: {next(model.parameters()).device}")
    print(f"Model in eval mode: {not model.training}")
    print(f"Torch compiled: {torch.compiled_with_cxx11_abi()}")
    
    # Time individual operations
    print("\n5. Individual operation timing...")
    
    # NN forward pass
    state_dict = ed.prepare_state_for_network(board)
    edge_index = state_dict['edge_index']
    edge_attr = state_dict['edge_attr']
    
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            policy, value = model(edge_index, edge_attr)
        nn_time = (time.time() - start) / 100
    
    print(f"NN forward pass: {nn_time*1000:.2f}ms")
    
    # Board copy
    start = time.time()
    for _ in range(100):
        copy_board = board.copy()
    copy_time = (time.time() - start) / 100
    print(f"Board copy: {copy_time*1000:.2f}ms")
    
    # Tree operations
    start = time.time()
    for _ in range(1000):
        node = UCTNode(board, move=None, parent=None)
        node.number_visits += 1
        node.is_expanded = True
    tree_time = (time.time() - start) / 1000
    print(f"Node creation + update: {tree_time*1000:.3f}ms")
    
    print("\n" + "="*60)
    print("Key findings:")
    print("- PyTorch NN is much faster (~1ms vs JAX's 35ms)")
    print("- PyTorch might be using CPU which is faster for small models")
    print("- Tree operations are negligible compared to NN calls")
    print("="*60)


if __name__ == "__main__":
    profile_pytorch_mcts()
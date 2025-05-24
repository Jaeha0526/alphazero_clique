#!/usr/bin/env python3
"""
Test script to verify the fixed architecture handles 15 edges correctly
"""

import torch
import numpy as np
import pickle
import sys
import os

# Add src to path
sys.path.append('src')

from alpha_net_clique import CliqueGNN, CliqueGameData

def test_fixed_architecture():
    print("=" * 60)
    print("TESTING FIXED ARCHITECTURE: 15 EDGES ONLY")
    print("=" * 60)
    
    # Load a real data example
    data_path = "experiments/improved_n6k3_h128_l3_mcts1000_20250518_222614/datasets/game_20250518_222640_cpu2_game0_iter0.pkl"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    with open(data_path, 'rb') as f:
        examples = pickle.load(f)
    
    print(f"Loaded {len(examples)} examples")
    
    # Test first example
    example = examples[0]
    print(f"Example keys: {example.keys()}")
    print(f"Policy shape: {example['policy'].shape}")
    print(f"Policy: {example['policy']}")
    print()
    
    # Create dataset
    dataset = CliqueGameData([example])
    data = dataset[0]
    
    print("DATASET OUTPUT:")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Edge attr shape: {data.edge_attr.shape}")
    print(f"Policy shape: {data.policy.shape}")
    print(f"Value: {data.value}")
    print(f"Num nodes: {data.num_nodes}")
    print()
    
    # Expected number of edges for 6 nodes
    expected_edges = 6 * 5 // 2
    print(f"Expected edges for 6 nodes: {expected_edges}")
    print(f"Actual edges from dataset: {data.edge_index.shape[1]}")
    print(f"Policy size: {data.policy.shape[0]}")
    print()
    
    # Test model forward pass
    print("TESTING MODEL FORWARD PASS:")
    model = CliqueGNN(num_vertices=6, hidden_dim=64, num_layers=2)
    model.eval()  # Put in eval mode to avoid BatchNorm issues
    
    # Single example test
    with torch.no_grad():
        policy_output, value_output = model(data.edge_index, data.edge_attr, debug=True)
    
    print(f"Policy output shape: {policy_output.shape}")
    print(f"Value output shape: {value_output.shape}")
    print(f"Policy output: {policy_output}")
    print(f"Value output: {value_output}")
    print()
    
    # Check if sizes match
    print("SIZE MATCHING CHECK:")
    print(f"Policy target size: {data.policy.shape}")
    print(f"Policy output size: {policy_output.shape}")
    print(f"MATCH: {data.policy.shape == policy_output.shape}")
    print()
    
    # Test batch processing
    print("TESTING BATCH PROCESSING:")
    from torch_geometric.data import DataLoader as PyGDataLoader
    
    # Create a small batch
    batch_dataset = CliqueGameData(examples[:3])
    batch_loader = PyGDataLoader(batch_dataset, batch_size=3, shuffle=False)
    
    for batch_data in batch_loader:
        print(f"Batch edge_index shape: {batch_data.edge_index.shape}")
        print(f"Batch edge_attr shape: {batch_data.edge_attr.shape}")
        print(f"Batch policy shape: {batch_data.policy.shape}")
        print(f"Batch value shape: {batch_data.value.shape}")
        print(f"Batch num_graphs: {batch_data.num_graphs}")
        
        with torch.no_grad():
            batch_policy_output, batch_value_output = model(batch_data.edge_index, batch_data.edge_attr, batch=batch_data.batch)
        
        print(f"Batch policy output shape: {batch_policy_output.shape}")
        print(f"Batch value output shape: {batch_value_output.shape}")
        
        # Check shapes match
        expected_policy_shape = (batch_data.num_graphs, 15)
        expected_value_shape = (batch_data.num_graphs, 1)
        
        print(f"Expected policy shape: {expected_policy_shape}")
        print(f"Expected value shape: {expected_value_shape}")
        print(f"Policy shape MATCH: {batch_policy_output.shape == expected_policy_shape}")
        print(f"Value shape MATCH: {batch_value_output.shape == expected_value_shape}")
        
        break  # Only test first batch
    
    print("=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    test_fixed_architecture() 
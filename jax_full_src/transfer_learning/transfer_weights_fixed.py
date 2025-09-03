#!/usr/bin/env python
"""
Fixed transfer learning for AlphaZero Clique models.
Properly handles JAX parameter tree structure.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

from vectorized_nn import ImprovedBatchedNeuralNetwork


def transfer_weights_simple(
    source_checkpoint_path: str,
    target_vertices: int,
    target_k: int,
    output_path: str = None
):
    """
    Simple transfer that keeps the exact parameter structure.
    Since the GNN is size-agnostic, we just copy the parameters directly.
    """
    print(f"\n🚀 Creating transfer checkpoint")
    print(f"  Source: {source_checkpoint_path}")
    print(f"  Target: n={target_vertices}, k={target_k}")
    
    # Load source checkpoint
    with open(source_checkpoint_path, 'rb') as f:
        source_checkpoint = pickle.load(f)
    
    # The parameters should work directly for different graph sizes
    # because the GNN architecture doesn't depend on the number of vertices
    # (it processes edges and nodes dynamically)
    
    # Create new checkpoint with same parameters
    new_checkpoint = {
        'params': source_checkpoint['params'],  # Direct copy!
        'iteration': 0,
        'source_checkpoint': source_checkpoint_path,
        'transfer_info': {
            'source_vertices': source_checkpoint.get('model_config', {}).get('num_vertices', 9),
            'target_vertices': target_vertices,
            'source_k': source_checkpoint.get('model_config', {}).get('k', 4),
            'target_k': target_k,
            'transfer_method': 'direct_copy'
        },
        'model_config': {
            'num_vertices': target_vertices,
            'hidden_dim': 64,
            'num_gnn_layers': 3,
            'asymmetric_mode': False,
        }
    }
    
    # Save checkpoint
    if output_path is None:
        output_path = f"checkpoint_n{target_vertices}k{target_k}_transferred.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(new_checkpoint, f)
    
    print(f"\n✅ Transfer checkpoint saved to: {output_path}")
    
    # Test the transferred model
    print(f"\n🧪 Testing transferred model...")
    test_success = test_transferred_model(output_path, target_vertices)
    
    if test_success:
        print(f"✅ Model verified - transfer successful!")
    else:
        print(f"⚠️ Model test failed - may need adjustment")
    
    return output_path


def test_transferred_model(checkpoint_path: str, num_vertices: int) -> bool:
    """Test that transferred model can do forward pass."""
    
    try:
        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Create model 
        model = ImprovedBatchedNeuralNetwork(
            num_vertices=num_vertices,
            hidden_dim=64,
            num_layers=3,
            asymmetric_mode=False
        )
        
        # Replace the initialized params with transferred ones
        # The key insight: the params structure from n=9 should work for n=13
        # because the GNN doesn't have vertex-specific parameters
        model.params = checkpoint['params']
        
        # Test forward pass
        num_edges = num_vertices * (num_vertices - 1) // 2
        
        # Create dummy input - proper edge indices
        edge_list = []
        for i in range(num_vertices):
            for j in range(i+1, num_vertices):
                edge_list.append([i, j])
        edge_index = jnp.array(edge_list, dtype=jnp.int32).T
        edge_index = edge_index[None, :, :]  # Add batch dimension
        
        edge_features = jnp.ones((1, num_edges, 3)) / 3.0
        
        # Try evaluation
        policies, values = model.evaluate_batch(edge_index, edge_features)
        
        print(f"  ✓ Forward pass successful!")
        print(f"    Policy shape: {policies.shape} (expected: (1, {num_edges}))")
        print(f"    Value: {values[0, 0]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Transfer AlphaZero Clique model weights')
    
    parser.add_argument('source_checkpoint', type=str,
                        help='Path to source model checkpoint')
    parser.add_argument('--target_vertices', type=int, required=True,
                        help='Number of vertices for target model')
    parser.add_argument('--target_k', type=int, required=True,
                        help='Clique size for target model')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for transferred checkpoint')
    
    args = parser.parse_args()
    
    transfer_weights_simple(
        args.source_checkpoint,
        args.target_vertices,
        args.target_k,
        args.output
    )


if __name__ == "__main__":
    main()
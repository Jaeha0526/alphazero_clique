#!/usr/bin/env python
"""
Simple test to verify transferred model works.
"""

import jax
import jax.numpy as jnp
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from vectorized_nn import ImprovedBatchedNeuralNetwork

def test_transferred_model():
    """Test that transferred model can do forward pass."""
    
    print("Testing transferred model...")
    
    # Load checkpoint
    with open("checkpoint_n13k4_transferred.pkl", 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"Checkpoint keys: {checkpoint.keys()}")
    print(f"Transfer info: {checkpoint.get('transfer_info', 'None')}")
    
    # Create model
    model = ImprovedBatchedNeuralNetwork(
        num_vertices=13,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=False
    )
    
    # Set parameters directly
    model.params = checkpoint['params']
    
    # Test forward pass
    n = 13
    num_edges = n * (n - 1) // 2  # 78 edges
    
    # Create dummy input
    edge_index = jnp.zeros((1, 2, num_edges), dtype=jnp.int32)
    edge_features = jnp.ones((1, num_edges, 3)) / 3.0
    
    try:
        policies, values = model.evaluate_batch(edge_index, edge_features)
        print(f"✓ Forward pass successful!")
        print(f"  Policy shape: {policies.shape} (expected: (1, 78))")
        print(f"  Value shape: {values.shape} (expected: (1, 1))")
        print(f"  Policy sum: {jnp.sum(policies[0]):.4f}")
        print(f"  Value: {values[0, 0]:.4f}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_random_model():
    """Test that random model works for comparison."""
    
    print("\nTesting random model...")
    
    # Create model
    model = ImprovedBatchedNeuralNetwork(
        num_vertices=13,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=False
    )
    
    # Test forward pass
    n = 13
    num_edges = n * (n - 1) // 2  # 78 edges
    
    # Create dummy input
    edge_index = jnp.zeros((1, 2, num_edges), dtype=jnp.int32)
    edge_features = jnp.ones((1, num_edges, 3)) / 3.0
    
    try:
        policies, values = model.evaluate_batch(edge_index, edge_features)
        print(f"✓ Forward pass successful!")
        print(f"  Policy shape: {policies.shape}")
        print(f"  Value shape: {values.shape}")
        print(f"  Policy sum: {jnp.sum(policies[0]):.4f}")
        print(f"  Value: {values[0, 0]:.4f}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Transfer Learning Test")
    print("="*60)
    
    success1 = test_transferred_model()
    success2 = test_random_model()
    
    if success1 and success2:
        print("\n✅ Both models work! Transfer successful.")
    else:
        print("\n❌ Model test failed. Check transfer process.")
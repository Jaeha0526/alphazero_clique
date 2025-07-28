#!/usr/bin/env python
"""
Test direct import and usage of the existing neural network
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

print("Testing direct import...")

# Test 1: Import the module
print("\n1. Importing vectorized_nn...")
import time
start = time.time()
from vectorized_nn import ImprovedBatchedNeuralNetwork
print(f"✓ Import successful in {time.time()-start:.3f}s")

# Test 2: Create the model
print("\n2. Creating model...")
start = time.time()
model = ImprovedBatchedNeuralNetwork(
    num_vertices=6,
    hidden_dim=32,
    num_layers=1,  # Reduce layers for faster init
    asymmetric_mode=False
)
print(f"✓ Model created in {time.time()-start:.3f}s")

print("\nDirect import test completed!")
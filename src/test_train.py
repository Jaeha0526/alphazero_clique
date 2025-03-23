#!/usr/bin/env python
"""
Simple test script to verify that the training process works correctly.
"""
import os
import numpy as np
import torch
from alpha_net import ChessNet, train
from pathlib import Path

# Create a small test dataset
def create_test_dataset(num_examples=10):
    dataset = []
    for i in range(num_examples):
        # Create a sample board state (22x8x8)
        board_state = np.random.rand(22, 8, 8).astype(np.float32)
        
        # Create a sample policy (4672)
        policy = np.zeros(4672, dtype=np.float32)
        policy[np.random.randint(0, 4672, 10)] = 1.0
        policy = policy / policy.sum()
        
        # Create a sample value (-1 to 1)
        value = np.random.uniform(-1, 1)
        
        dataset.append([board_state, policy, value])
    
    return dataset

def test_train():
    # Create the model_data directory if it doesn't exist
    Path("./model_data/").mkdir(exist_ok=True)
    
    print("Creating test dataset...")
    test_dataset = create_test_dataset(30)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ChessNet()
    net.to(device)
    
    print("Starting test training...")
    # Run training for a few epochs
    train(net, test_dataset, epoch_start=0, epoch_stop=2, cpu=0)
    print("Test training completed successfully!")

if __name__ == "__main__":
    test_train() 
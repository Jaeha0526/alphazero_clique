#!/usr/bin/env python
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from alpha_net_clique import CliqueGNN
import pickle
import glob
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import datetime
import time
import random
from torch_geometric.data import DataLoader as PyGDataLoader
from alpha_net_clique import CliqueGameData
import torch.nn.functional as F

def load_examples(folder_path: str) -> List:
    """
    Load all example files from a folder.
    
    Args:
        folder_path: Path to the folder containing example files
        
    Returns:
        all_examples: List of all examples
    """
    all_examples = []
    
    # Get all pickle files in the folder
    pickle_files = glob.glob(os.path.join(folder_path, "*.pkl"))
    # pickle_files += glob.glob(os.path.join(folder_path, "clique_game_*.pkl"))
    
    if not pickle_files:
        print(f"No training examples found in {folder_path}")
        return []
    
    print(f"Found {len(pickle_files)} pickle files")
    
    for pickle_file in pickle_files:
        try:
            with open(pickle_file, 'rb') as f:
                examples = pickle.load(f)
                all_examples.extend(examples)
        except Exception as e:
            print(f"Error loading {pickle_file}: {e}")
            continue
    
    print(f"Loaded {len(all_examples)} examples")
    return all_examples

def train_network(all_examples: List, iteration: int, num_vertices: int, clique_size: int) -> None:
    """
    Train the network on the given examples.
    
    Args:
        all_examples: List of all training examples
        iteration: Current iteration number
        num_vertices: Number of vertices in the graph
    """
    # Split examples into training and validation sets
    train_size = int(0.9 * len(all_examples))
    train_examples = all_examples[:train_size]
    val_examples = all_examples[train_size:]
    
    print(f"Training on {len(train_examples)} examples, validating on {len(val_examples)} examples")
    
    # Initialize network
    net = CliqueGNN(num_vertices, clique_size)
    
    # Load previous model if exists
    model_path = f"./model_data/clique_net_iter{iteration}.pth.tar"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model from {model_path}")
    
    # Train the network
    print("Starting training...")
    # Calculate number of iterations based on dataset size
    # Use 30 epochs with batch size of 16
    num_iterations = max(1, (len(train_examples) // 16) * 30)
    # num_iterations = 1
    
    # First try training on CPU if we hit CUDA errors
    try:
        net.train_network(train_examples, 0, num_iterations)
    except RuntimeError as e:
        if "CUDA" in str(e):
            print("CUDA error occurred, trying to train on CPU instead...")
            cpu_device = torch.device("cpu")
            net.to(cpu_device)
            net.train_network(train_examples, 0, num_iterations, device=cpu_device)
        else:
            raise e
    
    # Compute validation metrics
    print("\nComputing validation metrics...")
    net.eval()
    total_policy_loss = 0
    total_value_loss = 0
    num_examples = 0
    
    # Try validation on CPU to avoid CUDA errors
    device = torch.device("cpu")
    net.to(device)
    
    with torch.no_grad():
        for i, example in enumerate(val_examples):
            try:
                # Convert example to graph data
                data = CliqueGameData([example])[0]
                
                # Validate on CPU
                data = data.to(device)
                
                # Forward pass
                policy_output, value_output = net(data.edge_index, data.edge_attr)
                
                # Add small epsilon to avoid log(0)
                policy_output = policy_output + 1e-8
                policy_output = policy_output / policy_output.sum()
                
                # Calculate losses
                policy_loss = -torch.sum(data.policy * torch.log(policy_output))
                value_loss = F.mse_loss(value_output.squeeze(), data.value)
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_examples += 1
                
                if (i+1) % 50 == 0:
                    print(f"Validated {i+1}/{len(val_examples)} examples")
                
            except Exception as e:
                print(f"Error validating example {i}: {e}")
                continue
    
    if num_examples > 0:
        avg_policy_loss = total_policy_loss / num_examples
        avg_value_loss = total_value_loss / num_examples
        print(f"Validation Policy Loss: {avg_policy_loss:.4f}")
        print(f"Validation Value Loss: {avg_value_loss:.4f}")
    else:
        print("No valid examples for validation!")
    
    # Save the trained model
    torch.save({'state_dict': net.state_dict()}, model_path)
    print(f"Model saved to {model_path}")

def train_pipeline(iterations: int = 5, num_vertices: int = 6) -> None:
    """
    Run the full training pipeline for the given number of iterations.
    
    Args:
        iterations: Number of iterations to run
        num_vertices: Number of vertices in the graph
    """
    for iteration in range(iterations):
        print(f"Starting iteration {iteration+1}/{iterations}")
        
        # 1. Load all examples from previous iterations
        all_examples = []
        
        # Get all subdirectories in the datasets folder
        dataset_dir = "./datasets/clique"
        if os.path.exists(dataset_dir):
            all_examples = load_examples(dataset_dir)
        
        # If no examples found, generate initial examples
        if not all_examples:
            print("No examples found. Please run MCTS_clique.py first to generate examples.")
            return
        
        print(f"Loaded {len(all_examples)} examples total")
        
        # 2. Train network on all examples
        train_network(all_examples, iteration, num_vertices)
        
        # 3. Wait before next iteration to allow for self-play
        if iteration < iterations - 1:
            print(f"Iteration {iteration+1} completed. "
                  f"Starting iteration {iteration+2} in 10 seconds...")
            time.sleep(10)
    
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train AlphaZero for Clique Game")
    parser.add_argument("--iterations", type=int, default=5, 
                       help="Number of iterations to run")
    parser.add_argument("--vertices", type=int, default=6, 
                       help="Number of vertices in the graph")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs("./model_data", exist_ok=True)
    os.makedirs("./datasets/clique", exist_ok=True)
    
    # Start the training pipeline
    train_pipeline(args.iterations, args.vertices) 
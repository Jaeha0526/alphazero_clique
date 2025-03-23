#!/usr/bin/env python
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from alpha_net_clique import CliqueGNN, train
import pickle
import glob
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import datetime
import time
import random
from torch.utils.data import DataLoader
from alpha_net_clique import CliqueGameData

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
    pickle_files += glob.glob(os.path.join(folder_path, "clique_game_*.pkl"))
    
    # Loop through all files and load examples
    for file_path in pickle_files:
        try:
            with open(file_path, 'rb') as f:
                examples = pickle.load(f)
                all_examples.extend(examples)
                print(f"Loaded {len(examples)} examples from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_examples

def train_network(examples: List, iteration: int = 0, num_vertices: int = 6) -> None:
    """
    Train a neural network on the given examples.
    
    Args:
        examples: List of examples to train on
        iteration: Training iteration number
        num_vertices: Number of vertices in the graph
    """
    # Create output directories
    os.makedirs("./model_data", exist_ok=True)
    
    # Initialize the neural network
    net = CliqueGNN(num_vertices=num_vertices)
    
    # Load previous iteration if available
    prev_model_path = f"./model_data/clique_net_iter{iteration-1}.pth.tar"
    current_model_path = f"./model_data/clique_net_iter{iteration}.pth.tar"
    best_model_path = "./model_data/clique_net.pth.tar"
    
    if iteration > 0 and os.path.exists(prev_model_path):
        print(f"Loading previous model from {prev_model_path}")
        try:
            checkpoint = torch.load(prev_model_path, map_location='cpu')
            net.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print(f"Error loading previous model: {e}")
            print("Starting training from scratch")
    
    # Split examples into training and validation sets
    random.shuffle(examples)
    split_idx = int(0.9 * len(examples))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    print(f"Training on {len(train_examples)} examples, "
          f"validating on {len(val_examples)} examples")
    
    # Train the network
    print("Starting training...")
    # Save the original network for comparison
    if iteration == 0:
        torch.save({'state_dict': net.state_dict()}, 
                  f"./model_data/clique_net_iter0_init.pth.tar")
    
    # Create a GPU list
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found, using CPU only")
        device_list = [torch.device("cpu")]
    else:
        device_list = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        print(f"Found {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    
    # Train on multiple CPUs/GPUs if available
    processes = []
    if len(train_examples) > 1000:
        # Split examples for multi-process training
        split_size = len(train_examples) // len(device_list)
        splits = [train_examples[i:i+split_size] for i in range(0, len(train_examples), split_size)]
        
        # Make sure we use all examples
        if len(splits) > len(device_list):
            splits[-2].extend(splits[-1])
            splits.pop()
        
        # Start a process for each device
        mp.set_start_method("spawn", force=True)
        for i, (device, examples_split) in enumerate(zip(device_list, splits)):
            p = mp.Process(target=train, args=(net, examples_split, 0, 50, i))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        # Train on a single process if few examples
        train(net, train_examples, 0, 50, 0)
    
    # Save the trained model
    torch.save({'state_dict': net.state_dict()}, current_model_path)
    torch.save({'state_dict': net.state_dict()}, best_model_path)
    print(f"Model saved to {current_model_path} and {best_model_path}")
    
    # Validation
    print("Validating model...")
    device = device_list[0]
    net.to(device)
    net.eval()
    
    val_dataset = CliqueGameData(val_examples)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                          collate_fn=val_dataset.collate_fn)
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_examples = 0
    
    with torch.no_grad():
        for edge_indices, edge_attrs, policies, values in val_loader:
            batch_size = len(edge_indices)
            total_examples += batch_size
            
            batch_policy_loss = 0.0
            batch_value_loss = 0.0
            
            for edge_index, edge_attr, policy, value in zip(edge_indices, edge_attrs, policies, values):
                # Move data to device
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                policy = policy.to(device)
                value = value.to(device)
                
                # Forward pass
                policy_pred, value_pred = net(edge_index, edge_attr)
                
                # Compute losses
                # Policy loss (cross-entropy)
                policy_loss = -torch.sum(policy * torch.log(policy_pred + 1e-8))
                
                # Value loss (MSE)
                value_loss = (value - value_pred)**2
                
                batch_policy_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
            
            # Average losses for the batch
            total_policy_loss += batch_policy_loss / batch_size
            total_value_loss += batch_value_loss / batch_size
    
    # Calculate average losses
    avg_policy_loss = total_policy_loss / len(val_loader)
    avg_value_loss = total_value_loss / len(val_loader)
    
    print(f"Validation results - Policy loss: {avg_policy_loss:.4f}, Value loss: {avg_value_loss:.4f}")
    
    # Save validation results
    with open(f"./model_data/validation_results_iter{iteration}.txt", "w") as f:
        f.write(f"Policy loss: {avg_policy_loss:.4f}\n")
        f.write(f"Value loss: {avg_value_loss:.4f}\n")
        f.write(f"Total examples: {total_examples}\n")

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
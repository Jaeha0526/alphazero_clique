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
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import datetime
import time
import random
from torch_geometric.data import DataLoader as PyGDataLoader
from alpha_net_clique import CliqueGameData
import torch.nn.functional as F
import argparse

def load_examples(folder_path: str, iteration: int = None) -> List:
    """
    Load all example files from a folder.
    
    Args:
        folder_path: Path to the folder containing example files
        iteration: If provided, only load examples from this iteration
        
    Returns:
        all_examples: List of all examples
    """
    all_examples = []
    
    # Get all pickle files in the folder
    if iteration is not None:
        # Only load files from the current iteration
        pickle_files = glob.glob(os.path.join(folder_path, f"game_*_iter{iteration}.pkl"))
    else:
        # Load all files
        pickle_files = glob.glob(os.path.join(folder_path, "*.pkl"))
    
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

def train_network(all_examples: List, iteration: int, num_vertices: int, clique_size: int, model_dir: str, 
                   args: argparse.Namespace) -> Tuple[float, float]:
    """
    Train the network on the given examples.
    
    Args:
        all_examples: List of all training examples
        iteration: Current iteration number
        num_vertices: Number of vertices in the graph
        clique_size: Size of clique needed to win
        model_dir: Directory to save models
        args: Parsed command line arguments (contains hidden_dim, num_layers, LR params)
        
    Returns:
        Tuple of (avg_policy_loss, avg_value_loss) from validation
    """
    # Split examples into training and validation sets
    train_size = int(0.9 * len(all_examples))
    train_examples = all_examples[:train_size]
    val_examples = all_examples[train_size:]
    
    # Access model parameters from args
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    print(f"Training on {len(train_examples)} examples, validating on {len(val_examples)} examples")
    print(f"Model Config: hidden_dim={hidden_dim}, num_layers={num_layers}")
    
    # Initialize network correctly
    net = CliqueGNN(num_vertices, hidden_dim=hidden_dim, num_layers=num_layers)
    
    # Load model from previous iteration
    prev_model_path = f"{model_dir}/clique_net_iter{iteration-1}.pth.tar"
    if iteration > 0 and os.path.exists(prev_model_path):
        checkpoint = torch.load(prev_model_path)
        net.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model from previous iteration: {prev_model_path}")
    else:
        # For first iteration, try to load best model if exists
        best_model_path = f"{model_dir}/clique_net.pth.tar"
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            net.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded best model from {best_model_path}")
        else:
            print("No previous model found, starting from scratch")
    
    # Train the network
    print("Starting training...")
    # Calculate number of iterations (training steps) based on dataset size, batch size, and epochs
    batch_size = args.batch_size
    epochs = args.epochs
    num_training_steps = max(1, (len(train_examples) // batch_size) * epochs)
    print(f"Calculated training steps: {num_training_steps} ({epochs} epochs, batch size {batch_size})")
    
    # Add early stopping to prevent overfitting
    patience = 5  # Number of epochs with no improvement after which training will be stopped
    min_delta = 0.001  # Minimum change to qualify as an improvement
    best_loss = float('inf')
    patience_counter = 0
    
    # Updated args with additional parameters
    if not hasattr(args, 'value_weight'):
        args.value_weight = 1.0  # Default value weight if not specified
    
    # First try training on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    
    try:
        # Move model to selected device
        net.to(device)
        
        # Pass the whole args object to the network's training method
        train_losses = []  # Track losses for early stopping
        
        # Call train_network with early stopping logic
        best_model_state = None
        for epoch in range(epochs):
            try:
                epoch_loss = net.train_network(train_examples, 0, num_training_steps // epochs, 
                                             device=device, args=args)
                train_losses.append(epoch_loss)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")
                
                # Check if this is the best model so far
                if epoch_loss < best_loss - min_delta:
                    best_loss = epoch_loss
                    patience_counter = 0
                    # Save the best model state
                    best_model_state = copy.deepcopy(net.state_dict())
                    print(f"New best model! Loss: {best_loss:.6f}")
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} epochs (best: {best_loss:.6f}, current: {epoch_loss:.6f})")
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    # Restore best model
                    if best_model_state is not None:
                        net.load_state_dict(best_model_state)
                    break
                    
            except Exception as e:
                print(f"Error during epoch {epoch+1}: {e}")
                if "CUDA" in str(e):
                    print("CUDA error occurred, trying to continue on CPU...")
                    device = torch.device("cpu")
                    net.to(device)
                else:
                    raise e
    except RuntimeError as e:
        print(f"Error during training: {e}")
        if "CUDA" in str(e):
            print("CUDA error occurred, trying to train on CPU instead...")
            device = torch.device("cpu")
            net.to(device)
            net.train_network(train_examples, 0, num_training_steps, device=device, args=args)
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
        avg_policy_loss = 0.0
        avg_value_loss = 0.0
    
    # Save the trained model
    model_path = f"{model_dir}/clique_net_iter{iteration}.pth.tar"
    save_dict = {
        'state_dict': net.state_dict(),
        'num_vertices': num_vertices,
        'clique_size': clique_size,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers
    }
    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")
    
    return avg_policy_loss, avg_value_loss

def train_pipeline(iterations: int = 5, num_vertices: int = 6, data_dir: str = "./datasets/clique", 
                   model_dir: str = "./model_data", clique_size: int = 3, 
                   hidden_dim: int = 64, num_layers: int = 2) -> None:
    """
    Run the full training pipeline for the given number of iterations.
    
    Args:
        iterations: Number of iterations to run
        num_vertices: Number of vertices in the graph
        data_dir: Directory containing training data
        model_dir: Directory to save models
        clique_size: Size of clique needed to win
        hidden_dim: Hidden dimension size for GNN layers
        num_layers: Number of GNN layers
    """
    for iteration in range(iterations):
        print(f"Starting iteration {iteration+1}/{iterations}")
        
        # 1. Load only the current iteration's examples
        if not os.path.exists(data_dir):
            print("No examples found. Please run MCTS_clique.py first to generate examples.")
            return
            
        # Get all pickle files from the current iteration
        pickle_files = glob.glob(os.path.join(data_dir, f"game_*_iter{iteration}.pkl"))
        if not pickle_files:
            print(f"No examples found for iteration {iteration}. Please run MCTS_clique.py first.")
            return
            
        all_examples = []
        for pickle_file in pickle_files:
            try:
                with open(pickle_file, 'rb') as f:
                    examples = pickle.load(f)
                    all_examples.extend(examples)
            except Exception as e:
                print(f"Error loading {pickle_file}: {e}")
                continue
        
        print(f"Loaded {len(all_examples)} examples for iteration {iteration}")
        
        # 2. Train network on current iteration's examples
        train_network(all_examples, iteration, num_vertices, clique_size, model_dir, args)
        
        # 3. Wait before next iteration to allow for self-play
        if iteration < iterations - 1:
            print(f"Iteration {iteration+1} completed. "
                  f"Starting iteration {iteration+2} in 10 seconds...")
            time.sleep(10)
    
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    # If this script is run directly, it needs its own argparse setup
    # matching the parameters expected by train_network (or use defaults)
    # Example:
    parser = argparse.ArgumentParser(description="Train AlphaZero for Clique Game (Standalone)")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number to train")
    parser.add_argument("--vertices", type=int, default=6, help="Number of vertices")
    parser.add_argument("--k", type=int, default=3, help="Clique size")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GNN hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--data-dir", type=str, default="./datasets/clique", help="Dir for datasets")
    parser.add_argument("--model-dir", type=str, default="./model_data", help="Dir for models")
    # Add necessary LR args if running standalone
    parser.add_argument("--initial-lr", type=float, default=0.0001)
    parser.add_argument("--lr-factor", type=float, default=0.95)
    parser.add_argument("--lr-patience", type=int, default=7)
    parser.add_argument("--lr-threshold", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    print(f"Running Standalone Training for Iteration {args.iteration}")
    all_examples = load_examples(args.data_dir, args.iteration)
    if all_examples:
        train_network(all_examples, args.iteration, args.vertices, args.k, args.model_dir, args)
    else:
        print("No examples found, skipping training.") 
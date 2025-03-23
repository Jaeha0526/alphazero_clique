#!/usr/bin/env python

from alpha_net import ChessNet, train
from MCTS_chess import MCTS_self_play
import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import List, Dict, Any
from pathlib import Path

def load_datasets(data_paths: List[str]) -> List:
    datasets = []
    for path in data_paths:
        for file in os.listdir(path):
            filename = os.path.join(path, file)
            with open(filename, 'rb') as fo:
                datasets.extend(pickle.load(fo, encoding='bytes'))
    return datasets

def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    torch.save({'state_dict': model.state_dict()}, path)

def main():
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Create necessary directories
    Path("./model_data/").mkdir(exist_ok=True)
    Path("./datasets/iter0/").mkdir(exist_ok=True)
    Path("./datasets/iter1/").mkdir(exist_ok=True)
    Path("./datasets/iter2/").mkdir(exist_ok=True)
    
    # Initialize model and move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ChessNet()
    net.to(device)
    net.share_memory()
    
    # Save initial model
    initial_model_path = os.path.join("./model_data/", "current_net_trained8_iter1.pth.tar")
    save_checkpoint(net, initial_model_path)
    
    # Training loop
    for iteration in range(1):
        print(f"\nStarting iteration {iteration + 1}")
        
        # MCTS self-play phase
        net.eval()
        processes1 = []
        for i in range(6):
            p1 = mp.Process(target=MCTS_self_play, args=(net, 50, i))
            p1.start()
            processes1.append(p1)
        for p1 in processes1:
            p1.join()
            
        # Neural network training phase
        # Load training data
        data_paths = [
            "./datasets/iter0/",
            "./datasets/iter1/",
            "./datasets/iter2/"
        ]
        datasets = load_datasets(data_paths)
        
        # Train the network
        net.train()
        processes2 = []
        for i in range(1):
            p2 = mp.Process(target=train, args=(net, datasets, 0, 200, i))
            p2.start()
            processes2.append(p2)
        for p2 in processes2:
            p2.join()
            
        # Save the trained model
        save_checkpoint(net, initial_model_path)
        print(f"Completed iteration {iteration + 1}")

if __name__ == "__main__":
    main()
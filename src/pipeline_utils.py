#!/usr/bin/env python
"""
Utility functions for AlphaZero pipeline
"""

import os
import json
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any


def get_game_args(num_vertices: int = 6, hidden_dim: int = 64) -> Dict[str, Any]:
    """Get game-specific arguments for neural network initialization."""
    return {
        'num_vertices': num_vertices,
        'hidden_dim': hidden_dim,
        'edge_feat_dim': 3,
        'num_actions': num_vertices * (num_vertices - 1) // 2
    }


def setup_directories(experiment_name: str) -> Dict[str, str]:
    """Create directory structure for experiment."""
    root_dir = f"experiments/{experiment_name}"
    
    dirs = {
        'root': root_dir,
        'models': os.path.join(root_dir, 'models'),
        'self_play': os.path.join(root_dir, 'self_play_data'),
        'plots': os.path.join(root_dir, 'plots'),
        'logs': os.path.join(root_dir, 'logs')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def save_config(config: Dict[str, Any], save_dir: str):
    """Save configuration to JSON file."""
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")


def save_checkpoint(model: torch.nn.Module, iteration: int, save_dir: str) -> str:
    """Save model checkpoint."""
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"model_iter_{iteration}.pt"
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    
    return filepath


def load_latest_model(model: torch.nn.Module, model_dir: str) -> int:
    """Load the latest model checkpoint."""
    checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not checkpoints:
        return -1
    
    # Get latest by iteration number
    latest = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
    filepath = os.path.join(model_dir, latest)
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return checkpoint['iteration']


def plot_training_curves(history: Dict[str, list], save_dir: str):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Games per second
    ax = axes[0, 0]
    ax.plot(history['iteration'], history['games_per_second'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Games/Second')
    ax.set_title('Self-Play Performance')
    ax.grid(True, alpha=0.3)
    
    # Win rate vs MCTS
    ax = axes[0, 1]
    ax.plot(history['iteration'], history['win_rate_vs_mcts'], 'g-', linewidth=2)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Win Rate')
    ax.set_title('Performance vs Pure MCTS')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Win rate vs previous
    ax = axes[1, 0]
    ax.plot(history['iteration'], history['win_rate_vs_previous'], 'orange', linewidth=2)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Win Rate')
    ax.set_title('Performance vs Previous Version')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Time breakdown
    ax = axes[1, 1]
    self_play_time = history['self_play_time']
    training_time = history['training_time']
    ax.bar(history['iteration'], self_play_time, label='Self-Play', alpha=0.7)
    ax.bar(history['iteration'], training_time, bottom=self_play_time, 
           label='Training', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time per Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_curves_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath
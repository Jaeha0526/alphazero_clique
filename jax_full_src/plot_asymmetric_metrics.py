#!/usr/bin/env python
"""Enhanced plotting for asymmetric training metrics."""

import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict


def plot_asymmetric_learning_curves(log_file_path: str):
    """
    Enhanced plotting for asymmetric games showing role-specific performance.
    Creates two plots:
    1. Standard training metrics (losses + overall win rate)
    2. Role-specific win rates (attacker vs defender performance)
    """
    if not os.path.exists(log_file_path):
        print(f"Log file not found at {log_file_path}")
        return
    
    try:
        with open(log_file_path, 'r') as f:
            log_data = json.load(f)
    except Exception as e:
        print(f"Error loading log file: {e}")
        return
    
    # Extract data
    iterations = []
    policy_losses = []
    value_losses = []
    overall_win_rates = []
    attacker_win_rates = []
    defender_win_rates = []
    
    for entry in log_data:
        if entry.get("validation_policy_loss") is not None:
            iterations.append(entry["iteration"])
            policy_losses.append(entry["validation_policy_loss"])
            value_losses.append(entry["validation_value_loss"])
            overall_win_rates.append(entry.get("evaluation_win_rate_vs_initial", 0.5))
            
            # Get asymmetric-specific metrics if available
            if "attacker_win_rate_vs_initial" in entry:
                attacker_win_rates.append(entry["attacker_win_rate_vs_initial"])
                defender_win_rates.append(entry["defender_win_rate_vs_initial"])
    
    if len(iterations) < 1:
        print("Not enough data points to plot")
        return
    
    # Create figure with subplots
    if attacker_win_rates:  # Asymmetric mode
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    else:  # Symmetric mode
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax2 = None
    
    # Plot 1: Standard metrics (same as before)
    color = 'tab:red'
    ax1.set_xlabel('Iteration', fontsize=14)
    ax1.set_ylabel('Policy Loss', color=color, fontsize=14)
    ax1.plot(iterations, policy_losses, color=color, marker='o', linestyle='-', 
             linewidth=2, markersize=5, label='Policy Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Value Loss on second y-axis
    ax1_2 = ax1.twinx()
    color = 'tab:blue'
    ax1_2.set_ylabel('Value Loss', color=color, fontsize=14)
    ax1_2.plot(iterations, value_losses, color=color, marker='s', linestyle='--',
               linewidth=2, markersize=5, label='Value Loss')
    ax1_2.tick_params(axis='y', labelcolor=color)
    ax1_2.legend(loc='upper center')
    
    # Overall Win Rate on third y-axis
    ax1_3 = ax1.twinx()
    ax1_3.spines['right'].set_position(('outward', 60))
    color = 'tab:green'
    ax1_3.set_ylabel('Overall Win Rate vs Initial', color=color, fontsize=14)
    ax1_3.plot(iterations, overall_win_rates, color=color, marker='^', linestyle=':',
               linewidth=2, markersize=6, label='Overall Win Rate')
    ax1_3.tick_params(axis='y', labelcolor=color)
    ax1_3.set_ylim(-0.05, 1.05)
    ax1_3.legend(loc='upper right')
    
    ax1.set_title('Training Metrics', fontsize=16, pad=20)
    
    # Plot 2: Role-specific win rates (only for asymmetric)
    if ax2 and attacker_win_rates:
        ax2.set_xlabel('Iteration', fontsize=14)
        ax2.set_ylabel('Win Rate vs Initial', fontsize=14)
        
        # Plot attacker performance
        ax2.plot(iterations, attacker_win_rates, color='tab:orange', marker='o',
                linestyle='-', linewidth=2, markersize=6, 
                label='As Attacker (Form Clique)')
        
        # Plot defender performance
        ax2.plot(iterations, defender_win_rates, color='tab:purple', marker='s',
                linestyle='-', linewidth=2, markersize=6,
                label='As Defender (Prevent Clique)')
        
        # Plot overall for comparison
        ax2.plot(iterations, overall_win_rates, color='tab:gray', marker='^',
                linestyle='--', linewidth=1.5, markersize=4, alpha=0.7,
                label='Overall (Average)')
        
        ax2.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, label='50% baseline')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax2.legend(loc='best', fontsize=12)
        ax2.set_title('Role-Specific Performance', fontsize=16, pad=20)
        
        # Add annotations for the latest values
        if len(iterations) > 0:
            latest_idx = -1
            ax2.annotate(f'{attacker_win_rates[latest_idx]:.1%}',
                        xy=(iterations[latest_idx], attacker_win_rates[latest_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, color='tab:orange')
            ax2.annotate(f'{defender_win_rates[latest_idx]:.1%}',
                        xy=(iterations[latest_idx], defender_win_rates[latest_idx]),
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=10, color='tab:purple')
    
    # Add overall title
    experiment_name = os.path.basename(os.path.dirname(log_file_path))
    mode = "Asymmetric" if attacker_win_rates else "Symmetric"
    fig.suptitle(f'{mode} Training Progress - {experiment_name}', fontsize=18)
    
    plt.tight_layout()
    
    # Save plots
    plot_dir = os.path.dirname(log_file_path)
    if attacker_win_rates:
        plot_path = os.path.join(plot_dir, "learning_curves_asymmetric.png")
    else:
        plot_path = os.path.join(plot_dir, "learning_curves.png")
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close()


def plot_role_balance(log_file_path: str):
    """
    Plot the balance between attacker and defender performance over time.
    Shows the difference and ratio between the two roles.
    """
    try:
        with open(log_file_path, 'r') as f:
            log_data = json.load(f)
    except:
        return
    
    iterations = []
    differences = []  # Attacker rate - Defender rate
    ratios = []       # Attacker rate / Defender rate
    
    for entry in log_data:
        if "attacker_win_rate_vs_initial" in entry:
            iterations.append(entry["iteration"])
            att_rate = entry["attacker_win_rate_vs_initial"]
            def_rate = entry["defender_win_rate_vs_initial"]
            
            differences.append(att_rate - def_rate)
            if def_rate > 0:
                ratios.append(att_rate / def_rate)
            else:
                ratios.append(1.0)
    
    if not iterations:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot difference
    ax1.plot(iterations, differences, color='tab:red', marker='o', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.fill_between(iterations, differences, 0, where=[d > 0 for d in differences],
                     alpha=0.3, color='tab:orange', label='Attacker Advantage')
    ax1.fill_between(iterations, differences, 0, where=[d <= 0 for d in differences],
                     alpha=0.3, color='tab:purple', label='Defender Advantage')
    ax1.set_ylabel('Win Rate Difference\n(Attacker - Defender)', fontsize=12)
    ax1.set_title('Role Performance Balance', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot ratio
    ax2.plot(iterations, ratios, color='tab:blue', marker='s', linewidth=2)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Win Rate Ratio\n(Attacker / Defender)', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle('Attacker vs Defender Balance Analysis', fontsize=16)
    plt.tight_layout()
    
    plot_dir = os.path.dirname(log_file_path)
    plot_path = os.path.join(plot_dir, "role_balance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved role balance plot to {plot_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
        plot_asymmetric_learning_curves(log_file)
        plot_role_balance(log_file)
    else:
        print("Usage: python plot_asymmetric_metrics.py <log_file_path>")
#!/usr/bin/env python
"""Compare the results of fixed vs alternating perspective experiments."""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_experiment_data(experiment_name):
    """Load training log for an experiment."""
    with open(f"./experiments/{experiment_name}/training_log.json", 'r') as f:
        data = json.load(f)
    return data

def compare_experiments():
    """Compare fixed vs alternating perspective experiments."""
    # Load data
    fixed_data = load_experiment_data("perspective_fixed_minimal")
    alternating_data = load_experiment_data("perspective_alternating_minimal")
    
    # Extract metrics
    fixed_log = fixed_data["log"]
    alternating_log = alternating_data["log"]
    
    # Get iterations - use the minimum length
    min_iterations = min(len(fixed_log), len(alternating_log))
    iterations = list(range(min_iterations))
    
    # Extract metrics for comparison (truncate to same length)
    fixed_policy_loss = [fixed_log[i]["validation_policy_loss"] for i in range(min_iterations)]
    alternating_policy_loss = [alternating_log[i]["validation_policy_loss"] for i in range(min_iterations)]
    
    fixed_value_loss = [fixed_log[i]["validation_value_loss"] for i in range(min_iterations)]
    alternating_value_loss = [alternating_log[i]["validation_value_loss"] for i in range(min_iterations)]
    
    fixed_win_rate = [fixed_log[i]["evaluation_win_rate_vs_initial"] for i in range(min_iterations)]
    alternating_win_rate = [alternating_log[i]["evaluation_win_rate_vs_initial"] for i in range(min_iterations)]
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fixed vs Alternating Perspective Comparison', fontsize=16)
    
    # Policy Loss
    ax1 = axes[0, 0]
    ax1.plot(iterations, fixed_policy_loss, 'b-o', label='Fixed', linewidth=2, markersize=6)
    ax1.plot(iterations, alternating_policy_loss, 'r-s', label='Alternating', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Validation Policy Loss')
    ax1.set_title('Policy Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Value Loss
    ax2 = axes[0, 1]
    ax2.plot(iterations, fixed_value_loss, 'b-o', label='Fixed', linewidth=2, markersize=6)
    ax2.plot(iterations, alternating_value_loss, 'r-s', label='Alternating', linewidth=2, markersize=6)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Validation Value Loss')
    ax2.set_title('Value Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Win Rate vs Initial
    ax3 = axes[1, 0]
    ax3.plot(iterations, fixed_win_rate, 'b-o', label='Fixed', linewidth=2, markersize=6)
    ax3.plot(iterations, alternating_win_rate, 'r-s', label='Alternating', linewidth=2, markersize=6)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Win Rate vs Initial Model')
    ax3.set_title('Performance Against Initial Model')
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary stats
    fixed_final_policy = fixed_policy_loss[-1]
    alternating_final_policy = alternating_policy_loss[-1]
    fixed_final_value = fixed_value_loss[-1]
    alternating_final_value = alternating_value_loss[-1]
    fixed_avg_win = np.mean(fixed_win_rate)
    alternating_avg_win = np.mean(alternating_win_rate)
    
    summary_text = f"""Summary Statistics (After {min_iterations} iterations):

Fixed Perspective:
  Final Policy Loss: {fixed_final_policy:.4f}
  Final Value Loss: {fixed_final_value:.4f}
  Avg Win Rate vs Initial: {fixed_avg_win:.4f}

Alternating Perspective:
  Final Policy Loss: {alternating_final_policy:.4f}
  Final Value Loss: {alternating_final_value:.4f}
  Avg Win Rate vs Initial: {alternating_avg_win:.4f}

Differences:
  Policy Loss: {alternating_final_policy - fixed_final_policy:+.4f}
  Value Loss: {alternating_final_value - fixed_final_value:+.4f}
  Win Rate: {alternating_avg_win - fixed_avg_win:+.4f}
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('./perspective_comparison.png', dpi=150)
    plt.show()
    
    # Print detailed comparison
    print("=" * 60)
    print("DETAILED COMPARISON: Fixed vs Alternating Perspective")
    print("=" * 60)
    
    print("\nTraining Configuration:")
    print(f"  Vertices: {fixed_data['hyperparameters']['vertices']}")
    print(f"  Clique Size (k): {fixed_data['hyperparameters']['k']}")
    print(f"  Hidden Dim: {fixed_data['hyperparameters']['hidden_dim']}")
    print(f"  Num Layers: {fixed_data['hyperparameters']['num_layers']}")
    print(f"  MCTS Sims: {fixed_data['hyperparameters']['mcts_sims']}")
    print(f"  Self-play Games: {fixed_data['hyperparameters']['self_play_games']}")
    
    print("\nLearning Progress:")
    print(f"  Fixed - Policy Loss: {fixed_policy_loss[0]:.4f} → {fixed_final_policy:.4f} "
          f"(Δ = {fixed_final_policy - fixed_policy_loss[0]:.4f})")
    print(f"  Alternating - Policy Loss: {alternating_policy_loss[0]:.4f} → {alternating_final_policy:.4f} "
          f"(Δ = {alternating_final_policy - alternating_policy_loss[0]:.4f})")
    
    print(f"\n  Fixed - Value Loss: {fixed_value_loss[0]:.4f} → {fixed_final_value:.4f} "
          f"(Δ = {fixed_final_value - fixed_value_loss[0]:.4f})")
    print(f"  Alternating - Value Loss: {alternating_value_loss[0]:.4f} → {alternating_final_value:.4f} "
          f"(Δ = {alternating_final_value - alternating_value_loss[0]:.4f})")
    
    print("\nPerformance Metrics:")
    print(f"  Fixed - Best Win Rate vs Initial: {max(fixed_win_rate):.4f}")
    print(f"  Alternating - Best Win Rate vs Initial: {max(alternating_win_rate):.4f}")
    
    # Check which model was promoted to best more often
    fixed_promotions = sum(1 for i in range(1, len(fixed_log)) 
                          if fixed_log[i]["evaluation_win_rate_vs_best"] > 0.4)
    alternating_promotions = sum(1 for i in range(1, len(alternating_log)) 
                                if alternating_log[i]["evaluation_win_rate_vs_best"] > 0.4)
    
    print(f"\n  Fixed - Times promoted to best: {fixed_promotions}/{min_iterations-1}")
    print(f"  Alternating - Times promoted to best: {alternating_promotions}/{min_iterations-1}")
    
    print("\nConclusion:")
    if alternating_avg_win > fixed_avg_win:
        print(f"  Alternating perspective shows better average performance "
              f"({alternating_avg_win:.4f} vs {fixed_avg_win:.4f})")
    else:
        print(f"  Fixed perspective shows better average performance "
              f"({fixed_avg_win:.4f} vs {alternating_avg_win:.4f})")
    
    if alternating_final_value < fixed_final_value:
        print(f"  Alternating perspective has lower final value loss "
              f"({alternating_final_value:.4f} vs {fixed_final_value:.4f})")
    else:
        print(f"  Fixed perspective has lower final value loss "
              f"({fixed_final_value:.4f} vs {alternating_final_value:.4f})")

if __name__ == "__main__":
    compare_experiments()
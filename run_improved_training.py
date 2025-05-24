#!/usr/bin/env python
import os
import argparse
import subprocess
import datetime
import sys

def main():
    """Run improved AlphaZero training for the Clique Game."""
    parser = argparse.ArgumentParser(description="Run improved AlphaZero training for Clique Game")
    
    # Game parameters
    parser.add_argument("--vertices", type=int, default=6, help="Number of vertices in the graph")
    parser.add_argument("--k", type=int, default=3, help="Size of clique needed to win")
    parser.add_argument("--game-mode", type=str, default="symmetric", 
                        choices=["symmetric", "asymmetric"], help="Game rules")
    
    # Model parameters
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for GNN layers")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of GNN layers")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=30, help="Number of pipeline iterations")
    parser.add_argument("--self-play-games", type=int, default=100, help="Self-play games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=1000, help="MCTS simulations per move")
    parser.add_argument("--num-cpus", type=int, default=6, help="Number of CPUs for parallel execution")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per iteration")
    parser.add_argument("--initial-lr", type=float, default=0.0003, help="Initial learning rate")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor")
    parser.add_argument("--lr-patience", type=int, default=3, help="LR scheduler patience")
    parser.add_argument("--value-weight", type=float, default=1.0, help="Weight for value loss")
    
    # Evaluation parameters
    parser.add_argument("--eval-threshold", type=float, default=0.52, help="Win rate threshold for best model update")
    parser.add_argument("--num-games", type=int, default=21, help="Number of evaluation games")
    parser.add_argument("--mcts-sims-eval", type=int, default=30, help="MCTS simulations for evaluation")
    
    args = parser.parse_args()
    
    # Generate a unique experiment name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (f"improved_attn_value_n{args.vertices}k{args.k}_h{args.hidden_dim}_l{args.num_layers}_"
                       f"mcts{args.mcts_sims}_{timestamp}")
    
    print(f"Starting improved AlphaZero training with experiment name: {experiment_name}")
    
    # Build the command with all parameters
    cmd = [
        "python", "src/pipeline_clique.py",
        "--mode", "pipeline",
        "--vertices", str(args.vertices),
        "--k", str(args.k),
        "--game-mode", args.game_mode,
        "--hidden-dim", str(args.hidden_dim),
        "--num-layers", str(args.num_layers),
        "--iterations", str(args.iterations),
        "--self-play-games", str(args.self_play_games),
        "--mcts-sims", str(args.mcts_sims),
        "--num-cpus", str(args.num_cpus),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--initial-lr", str(args.initial_lr),
        "--lr-factor", str(args.lr_factor),
        "--lr-patience", str(args.lr_patience),
        "--eval-threshold", str(args.eval_threshold),
        "--num-games", str(args.num_games),
        "--experiment-name", experiment_name,
        "--value-weight", str(args.value_weight)
    ]
    
    # Print the command for reference
    print("Running command:", " ".join(cmd))
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running training pipeline: {e}")
        return 1
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return 1
    
    print(f"Training complete. Results saved in experiments/{experiment_name}/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
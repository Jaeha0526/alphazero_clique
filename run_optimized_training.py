#!/usr/bin/env python
import os
import argparse
import subprocess
import datetime
import sys

def main():
    """Run optimized AlphaZero training for the Clique Game with carefully tuned hyperparameters."""
    parser = argparse.ArgumentParser(description="Run optimized AlphaZero training for Clique Game")
    
    # Game parameters
    parser.add_argument("--vertices", type=int, default=6, help="Number of vertices in the graph")
    parser.add_argument("--k", type=int, default=3, help="Size of clique needed to win")
    parser.add_argument("--game-mode", type=str, default="symmetric", 
                        choices=["symmetric", "asymmetric"], help="Game rules")
    
    # Model parameters
    parser.add_argument("--hidden-dim", type=int, default=192, help="Hidden dimension for GNN layers")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of GNN layers")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=50, help="Number of pipeline iterations")
    parser.add_argument("--self-play-games", type=int, default=250, help="Self-play games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=1000, help="MCTS simulations per move")
    parser.add_argument("--num-cpus", type=int, default=6, help="Number of CPUs for parallel execution")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs per iteration")
    parser.add_argument("--initial-lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lr-factor", type=float, default=0.3, help="LR reduction factor")
    parser.add_argument("--lr-patience", type=int, default=7, help="LR scheduler patience")
    parser.add_argument("--value-weight", type=float, default=5.0, help="Weight for value loss")
    
    # MCTS parameters (these will require code modifications to use)
    parser.add_argument("--c-puct", type=float, default=3.0, help="Exploration constant c_puct for MCTS")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet noise alpha parameter")
    parser.add_argument("--dirichlet-weight", type=float, default=0.25, help="Weight of Dirichlet noise")
    
    # Evaluation parameters
    parser.add_argument("--eval-threshold", type=float, default=0.52, help="Win rate threshold for best model update")
    parser.add_argument("--num-games", type=int, default=31, help="Number of evaluation games")
    parser.add_argument("--mcts-sims-eval", type=int, default=50, help="MCTS simulations for evaluation")
    
    args = parser.parse_args()
    
    # Generate a unique experiment name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (f"optimized_n{args.vertices}k{args.k}_h{args.hidden_dim}_l{args.num_layers}_"
                       f"mcts{args.mcts_sims}_{timestamp}")
    
    print(f"Starting optimized AlphaZero training with experiment name: {experiment_name}")
    
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
    
    # Create a file to document the hyperparameter settings
    readme_path = f"experiments/{experiment_name}/README.md"
    os.makedirs(os.path.dirname(readme_path), exist_ok=True)
    with open(readme_path, "w") as f:
        f.write(f"# AlphaZero Clique Game Optimized Training\n\n")
        f.write(f"Experiment: {experiment_name}\n\n")
        f.write("## Optimized Hyperparameters\n\n")
        f.write("### Game Parameters\n")
        f.write(f"- Vertices: {args.vertices}\n")
        f.write(f"- Clique size (k): {args.k}\n")
        f.write(f"- Game mode: {args.game_mode}\n\n")
        
        f.write("### Model Architecture\n")
        f.write(f"- Hidden dimension: {args.hidden_dim} (increased for better expressivity)\n")
        f.write(f"- Number of GNN layers: {args.num_layers} (deeper for better graph understanding)\n\n")
        
        f.write("### Training Parameters\n")
        f.write(f"- Iterations: {args.iterations} (increased for longer training)\n")
        f.write(f"- Self-play games per iteration: {args.self_play_games} (more games for better learning)\n")
        f.write(f"- MCTS simulations: {args.mcts_sims} (more simulations for better planning)\n")
        f.write(f"- Batch size: {args.batch_size} (larger for better gradient estimates)\n")
        f.write(f"- Epochs per iteration: {args.epochs} (reduced to prevent overfitting)\n\n")
        
        f.write("### Learning Parameters\n")
        f.write(f"- Initial learning rate: {args.initial_lr} (higher for faster initial learning)\n")
        f.write(f"- LR reduction factor: {args.lr_factor} (more aggressive LR reduction)\n")
        f.write(f"- LR scheduler patience: {args.lr_patience} (wait longer before reducing LR)\n")
        f.write(f"- Value loss weight: {args.value_weight} (increased to prevent value collapse)\n\n")
        
        f.write("### Desired MCTS Parameters (requires code modification)\n")
        f.write(f"- c_puct: {args.c_puct} (higher exploration constant)\n")
        f.write(f"- Dirichlet alpha: {args.dirichlet_alpha} (tuned for 6-vertex graph)\n")
        f.write(f"- Dirichlet noise weight: {args.dirichlet_weight}\n\n")
        
        f.write("### Evaluation Parameters\n")
        f.write(f"- Win rate threshold: {args.eval_threshold}\n")
        f.write(f"- Evaluation games: {args.num_games} (increased for more reliable evaluation)\n")
        f.write(f"- MCTS simulations for evaluation: {args.mcts_sims_eval}\n\n")
        
        f.write("## Expected Improvements\n\n")
        f.write("1. **Higher value weight (5.0)** should prevent value loss collapse\n")
        f.write("2. **Deeper network (4 layers)** with **larger hidden dimension (192)** should improve representation power\n")
        f.write("3. **More self-play games (250)** with **more MCTS simulations (1000)** should generate better training data\n")
        f.write("4. **Higher initial learning rate (0.001)** with **aggressive decay (0.3)** should help escape local minima\n")
        f.write("5. **More iterations (50)** allows longer training time to see improvements\n")
    
    print(f"Created hyperparameter documentation at {readme_path}")
    
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
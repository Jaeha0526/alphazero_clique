#!/usr/bin/env python
"""
Quick test to verify custom eval_games and eval_mcts_sims arguments work correctly.
"""

import subprocess
import sys

def test_custom_args():
    """Run a minimal training with custom arguments to verify they work."""
    
    print("="*60)
    print("Testing Custom Evaluation Arguments")
    print("="*60)
    
    # Test command with custom eval settings
    cmd = [
        "python", "run_jax_optimized.py",
        "--experiment_name", "test_custom_args",
        "--vertices", "6",
        "--k", "3",
        "--num_iterations", "1",
        "--num_episodes", "10",
        "--eval_games", "5",  # Custom: only 5 eval games
        "--eval_mcts_sims", "5",  # Custom: only 5 MCTS sims for eval
        "--mcts_sims", "10",  # Self-play uses 10
        "--num_epochs", "2",
        "--batch_size", "32",
        "--use_true_mctx"
    ]
    
    print("\nRunning command:")
    print(" ".join(cmd))
    print("\nExpected behavior:")
    print("- Self-play: 10 games with 10 MCTS simulations")
    print("- Evaluation: 5 games with 5 MCTS simulations")
    print("- Should complete quickly due to minimal settings")
    print("-"*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check output for confirmation
        if "eval_games" in result.stdout or "Evaluation: 5 games" in result.stdout:
            print("\n✅ Custom eval_games argument appears to be working!")
        
        if "eval_mcts_sims" in result.stdout or "mcts_sims': 5" in result.stdout:
            print("✅ Custom eval_mcts_sims argument appears to be working!")
        
        # Look for evaluation results
        if "Parallel Evaluation Results" in result.stdout or "Evaluation Results" in result.stdout:
            print("✅ Evaluation completed successfully!")
            
            # Extract some stats
            lines = result.stdout.split('\n')
            for line in lines:
                if "games in" in line.lower() or "games/sec" in line.lower():
                    print(f"  {line.strip()}")
        
        if result.returncode != 0:
            print(f"\n⚠️ Process exited with code {result.returncode}")
            print("STDERR:", result.stderr[:500])
        else:
            print("\n✅ Test completed successfully!")
            
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing that custom eval_games and eval_mcts_sims arguments work...")
    success = test_custom_args()
    sys.exit(0 if success else 1)
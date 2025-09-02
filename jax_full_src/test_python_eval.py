#!/usr/bin/env python
"""
Test the --python_eval flag to verify it uses Python MCTS for evaluation.
"""

import subprocess
import time

def test_python_eval():
    """Test that python_eval flag forces Python MCTS for evaluation."""
    
    print("="*60)
    print("Testing --python_eval Flag")
    print("="*60)
    
    # Test 1: With python_eval flag
    print("\n1. Testing WITH --python_eval flag...")
    cmd1 = [
        "python", "run_jax_optimized.py",
        "--experiment_name", "test_python_eval",
        "--vertices", "4", "--k", "3",
        "--num_iterations", "1",
        "--num_episodes", "2",
        "--eval_games", "2",
        "--eval_mcts_sims", "3",
        "--mcts_sims", "5",
        "--num_epochs", "1",
        "--training_batch_size", "2",
        "--game_batch_size", "2",
        "--use_true_mctx",  # Use JAX for self-play
        "--python_eval",    # But Python for evaluation
        "--parallel_evaluation"
    ]
    
    print("Command:", " ".join(cmd1))
    print("\nExpected behavior:")
    print("- Self-play: Uses True MCTX (JAX)")
    print("- Evaluation: Uses Python MCTS (no compilation)")
    print("-"*60)
    
    start_time = time.time()
    result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=120)
    elapsed1 = time.time() - start_time
    
    # Check for key indicators
    has_true_mctx_selfplay = "Creating True MCTX" in result1.stdout or "True MCTX Implementation" in result1.stdout
    has_python_eval = "Using Python MCTS for evaluation" in result1.stdout
    has_compilation = "compilation" in result1.stdout.lower()
    
    print(f"\nResults:")
    print(f"  Time taken: {elapsed1:.1f}s")
    print(f"  ✅ Self-play uses True MCTX: {has_true_mctx_selfplay}")
    print(f"  ✅ Evaluation uses Python: {has_python_eval}")
    print(f"  Compilation mentioned: {has_compilation}")
    
    # Test 2: Without python_eval flag (for comparison)
    print("\n2. Testing WITHOUT --python_eval flag...")
    cmd2 = cmd1[:-2]  # Remove --python_eval and --parallel_evaluation
    cmd2.append("--parallel_evaluation")  # Add back parallel_evaluation
    
    print("Command:", " ".join(cmd2))
    print("\nExpected behavior:")
    print("- Self-play: Uses True MCTX (JAX)")
    print("- Evaluation: Also uses True MCTX (JAX) - may compile")
    print("-"*60)
    
    start_time = time.time()
    result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)
    elapsed2 = time.time() - start_time
    
    has_python_eval2 = "Using Python MCTS for evaluation" in result2.stdout
    
    print(f"\nResults:")
    print(f"  Time taken: {elapsed2:.1f}s")
    print(f"  Evaluation uses Python: {has_python_eval2} (should be False)")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    
    if has_python_eval and not has_python_eval2:
        print("✅ SUCCESS: --python_eval flag correctly forces Python MCTS for evaluation")
        print(f"   With flag: {elapsed1:.1f}s")
        print(f"   Without flag: {elapsed2:.1f}s")
        if elapsed1 < elapsed2:
            print(f"   Python eval was {elapsed2-elapsed1:.1f}s faster (avoided compilation)")
    else:
        print("❌ FAILURE: --python_eval flag not working correctly")
        
    return has_python_eval and not has_python_eval2

if __name__ == "__main__":
    success = test_python_eval()
    exit(0 if success else 1)
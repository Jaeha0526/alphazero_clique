#!/usr/bin/env python
"""
Test script to verify --skip_evaluation flag works correctly.
"""

import subprocess
import time
import sys
from pathlib import Path

def test_skip_evaluation():
    """Test that training can run with evaluation skipped."""
    
    print("="*60)
    print("Testing --skip_evaluation flag")
    print("="*60)
    
    # Create test experiment directory
    test_exp_name = f"test_skip_eval_{int(time.time())}"
    
    # Run training with skip_evaluation flag
    cmd = [
        "python", "run_jax_optimized.py",
        "--experiment_name", test_exp_name,
        "--num_iterations", "1",
        "--num_episodes", "5",
        "--mcts_sims", "10",
        "--vertices", "4",
        "--k", "3",
        "--num_epochs", "2",
        "--skip_evaluation",  # This is what we're testing
        "--python_eval"  # Use Python MCTS for speed
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    print("\nExpecting to see:")
    print("  - 'Skipping evaluation' message")
    print("  - 'Skipping best model update' message")
    print("  - No evaluation results")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check output
    output = result.stdout
    
    # Verify skip messages appear
    if "Skipping evaluation (--skip_evaluation flag set)" in output:
        print("✅ Found 'Skipping evaluation' message")
    else:
        print("❌ Missing 'Skipping evaluation' message")
        return False
    
    if "Skipping best model update (evaluation was skipped)" in output:
        print("✅ Found 'Skipping best model update' message")
    else:
        print("❌ Missing 'Skipping best model update' message")
        return False
    
    # Verify no evaluation results
    if "Win rate vs initial:" not in output:
        print("✅ No evaluation results shown (as expected)")
    else:
        print("❌ Evaluation results found (should be skipped)")
        return False
    
    # Check that training still completed
    if "Iteration 1/" in output:
        print("✅ Training completed successfully")
    else:
        print("❌ Training did not complete")
        return False
    
    # Check training log
    exp_dir = Path(f"experiments/{test_exp_name}")
    log_file = exp_dir / "training_log.json"
    
    if log_file.exists():
        import json
        with open(log_file) as f:
            log = json.load(f)
        
        if log and log[0].get('evaluation_win_rate_vs_initial') == -1:
            print("✅ Training log shows evaluation was skipped (-1 values)")
        else:
            print("❌ Training log doesn't show skipped evaluation")
            return False
    
    print("\n" + "="*60)
    print("--skip_evaluation flag test PASSED!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_skip_evaluation()
    sys.exit(0 if success else 1)
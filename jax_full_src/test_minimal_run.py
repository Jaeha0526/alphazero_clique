#!/usr/bin/env python
"""
Minimal test to verify the training pipeline works with custom eval settings.
"""

import subprocess
import sys
import time
import os

def run_minimal_test():
    """Run a very minimal training to test functionality."""
    
    print("="*60)
    print("Running Minimal Training Test")
    print("="*60)
    
    # Use minimal settings for quick test
    cmd = [
        "python", "run_jax_optimized.py",
        "--experiment_name", "test_minimal",
        "--vertices", "4",  # Very small graph
        "--k", "3",
        "--num_iterations", "1",  # Just 1 iteration
        "--num_episodes", "2",  # Only 2 self-play games
        "--eval_games", "2",  # Only 2 eval games
        "--eval_mcts_sims", "2",  # Very shallow MCTS for eval
        "--mcts_sims", "3",  # Shallow MCTS for self-play
        "--num_epochs", "1",  # Minimal training
        "--training_batch_size", "2",
        "--game_batch_size", "2",
        # Don't use true_mctx for this test as it might be slower to compile
    ]
    
    print("\nCommand:")
    print(" ".join(cmd))
    print("\nSettings:")
    print("- 4 vertices, 3-cliques (very small graph)")
    print("- 2 self-play games with 3 MCTS sims each")
    print("- 2 evaluation games with 2 MCTS sims each")
    print("- 1 training epoch")
    print("-"*60)
    
    start_time = time.time()
    
    try:
        # Run with timeout
        print("\nStarting training (timeout: 60 seconds)...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Read output in real-time
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                output_lines.append(line)
                
                # Check for key milestones
                if "Self-play games generated:" in line:
                    print("  ✅ Self-play completed")
                if "Training completed in" in line:
                    print("  ✅ Training completed")
                if "Evaluation Results" in line or "Parallel Evaluation Results" in line:
                    print("  ✅ Evaluation started")
                if "Win rate vs initial:" in line:
                    print("  ✅ Evaluation completed")
                if "Iteration 1 completed" in line:
                    print("  ✅ Full iteration completed!")
                    
                # Check if process has ended
                if process.poll() is not None:
                    break
        
        # Wait for process to complete (with timeout)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            print("\n⏱️ Process timed out after 60 seconds")
            
        elapsed = time.time() - start_time
        
        # Check for success indicators in output
        output_text = ''.join(output_lines)
        
        success_indicators = {
            'self_play': 'Self-play games generated:' in output_text,
            'training': 'Training completed in' in output_text or 'Final losses' in output_text,
            'evaluation': 'Evaluation Results' in output_text or 'Parallel Evaluation Results' in output_text,
            'custom_games': 'num_games\': 2' in output_text or '2 games' in output_text,
            'custom_sims': 'mcts_sims\': 2' in output_text or 'sims: 2' in output_text,
        }
        
        print("\n" + "="*60)
        print("Test Results:")
        print("="*60)
        print(f"Elapsed time: {elapsed:.1f} seconds")
        
        for key, found in success_indicators.items():
            status = "✅" if found else "❌"
            print(f"{status} {key.replace('_', ' ').title()}")
        
        # Check if experiment directory was created
        exp_dir = f"experiments/test_minimal"
        if os.path.exists(exp_dir):
            print(f"✅ Experiment directory created: {exp_dir}")
            
            # Check for specific files
            if os.path.exists(f"{exp_dir}/training_log.json"):
                print(f"  ✅ Training log created")
            if os.path.exists(f"{exp_dir}/checkpoints"):
                print(f"  ✅ Checkpoints directory created")
        
        # Overall success
        if all(success_indicators.values()):
            print("\n🎉 All tests PASSED! Custom eval settings work correctly.")
            return True
        else:
            print("\n⚠️ Some tests failed. Check output above for details.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_minimal_test()
    sys.exit(0 if success else 1)
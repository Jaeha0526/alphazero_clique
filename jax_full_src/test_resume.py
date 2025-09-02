#!/usr/bin/env python
"""
Test that resume functionality properly loads initial and best models.
"""

import os
import json
import pickle
import shutil
import subprocess
import time

def test_resume_functionality():
    """Test resume with proper model loading."""
    
    print("="*60)
    print("Testing Resume Functionality")
    print("="*60)
    
    exp_name = "test_resume_models"
    exp_dir = f"experiments/{exp_name}"
    
    # Clean up any existing test
    if os.path.exists(exp_dir):
        print(f"Cleaning up existing {exp_dir}")
        shutil.rmtree(exp_dir)
    
    print("\n1. Running initial training (2 iterations)...")
    cmd1 = [
        "python", "run_jax_optimized.py",
        "--experiment_name", exp_name,
        "--vertices", "4", "--k", "3",
        "--num_iterations", "2",
        "--num_episodes", "4",
        "--eval_games", "2",
        "--eval_mcts_sims", "2",
        "--mcts_sims", "3",
        "--num_epochs", "1",
        "--training_batch_size", "2",
        "--game_batch_size", "2"
    ]
    
    result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=120)
    
    if result1.returncode != 0:
        print(f"❌ Initial training failed: {result1.stderr[:500]}")
        return False
    
    # Check what was created
    print("\n2. Checking initial training results...")
    
    # Check models directory
    models_dir = f"{exp_dir}/models"
    if os.path.exists(models_dir):
        models = os.listdir(models_dir)
        print(f"   Models created: {models}")
        
        has_initial = "initial_model.pkl" in models
        has_best = "best_model.pkl" in models
        
        print(f"   ✅ initial_model.pkl: {'Found' if has_initial else 'MISSING!'}")
        print(f"   ✅ best_model.pkl: {'Found' if has_best else 'MISSING!'}")
    else:
        print("   ❌ No models directory!")
        return False
    
    # Check training log
    log_path = f"{exp_dir}/training_log.json"
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
        print(f"   Training log has {len(log)} entries")
        if log:
            last = log[-1]
            print(f"   Last iteration: {last.get('iteration', 'N/A')}")
            print(f"   Best model iteration: {last.get('best_model_iteration', 'N/A')}")
    
    # Get checkpoint
    checkpoint_path = f"{exp_dir}/checkpoints/checkpoint_iter_2.pkl"
    if not os.path.exists(checkpoint_path):
        print(f"   ❌ Checkpoint not found at {checkpoint_path}")
        return False
    print(f"   ✅ Checkpoint found: {checkpoint_path}")
    
    print("\n3. Resuming training from checkpoint...")
    cmd2 = [
        "python", "run_jax_optimized.py",
        "--resume_from", checkpoint_path,
        "--experiment_name", exp_name,
        "--vertices", "4", "--k", "3",
        "--num_iterations", "3",  # Will run iteration 3 only
        "--num_episodes", "4",
        "--eval_games", "2",
        "--eval_mcts_sims", "2",
        "--mcts_sims", "3",
        "--num_epochs", "1",
        "--training_batch_size", "2",
        "--game_batch_size", "2"
    ]
    
    result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)
    
    # Check for key messages in output
    output = result2.stdout
    
    success_indicators = {
        'checkpoint_loaded': 'Resumed from iteration' in output,
        'log_loaded': 'Loaded existing log' in output,
        'best_model_loaded': '✅ Best model loaded successfully' in output,
        'initial_model_loaded': '✅ Initial model loaded successfully' in output,
        'best_model_info': 'Best model is from iteration' in output,
    }
    
    print("\n4. Resume Results:")
    for key, found in success_indicators.items():
        status = "✅" if found else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    # Check for evaluation against best
    if 'Evaluating against best model' in output or 'vs best' in output:
        print("   ✅ Evaluation against best model occurred")
    else:
        print("   ⚠️ No clear indication of best model evaluation")
    
    # Final verdict
    all_good = all(success_indicators.values())
    
    print("\n" + "="*60)
    if all_good:
        print("✅ Resume functionality working correctly!")
        print("   - Initial model preserved")
        print("   - Best model loaded") 
        print("   - Training continued from correct iteration")
    else:
        print("⚠️ Some issues with resume functionality")
        print("   Check the indicators above for problems")
    print("="*60)
    
    return all_good

if __name__ == "__main__":
    success = test_resume_functionality()
    exit(0 if success else 1)
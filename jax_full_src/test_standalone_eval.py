#!/usr/bin/env python
"""
Test script for standalone evaluation functionality.
Verifies that models can be saved and evaluated independently.
"""

import os
import pickle
import time
from pathlib import Path
from vectorized_nn import ImprovedBatchedNeuralNetwork
from standalone_evaluation import load_model, evaluate_two_models

def test_standalone_evaluation():
    """Test that standalone evaluation works correctly."""
    
    print("="*60)
    print("Testing Standalone Evaluation")
    print("="*60)
    
    # Look for an existing experiment with models
    experiments_dir = Path("experiments")
    
    # Find experiments with checkpoints
    valid_experiments = []
    for exp_dir in experiments_dir.glob("*"):
        if exp_dir.is_dir():
            checkpoints = list(exp_dir.glob("checkpoints/checkpoint_iter_*.pkl"))
            initial_model = exp_dir / "models" / "initial_model.pkl"
            if checkpoints and initial_model.exists():
                valid_experiments.append({
                    'name': exp_dir.name,
                    'checkpoints': checkpoints,
                    'initial_model': initial_model,
                    'best_model': exp_dir / "models" / "best_model.pkl"
                })
    
    if not valid_experiments:
        print("No experiments with saved models found!")
        print("Please run training first to generate models.")
        return False
    
    # Use the first valid experiment
    exp = valid_experiments[0]
    print(f"\nUsing experiment: {exp['name']}")
    print(f"  Checkpoints found: {len(exp['checkpoints'])}")
    print(f"  Initial model: {exp['initial_model'].exists()}")
    print(f"  Best model: {exp['best_model'].exists()}")
    
    # Load a checkpoint to get configuration
    latest_checkpoint = max(exp['checkpoints'], key=lambda p: int(p.stem.split('_')[-1]))
    print(f"\nLoading latest checkpoint: {latest_checkpoint.name}")
    
    try:
        # Test loading model
        model = load_model(str(latest_checkpoint))
        print(f"✅ Model loaded successfully")
        print(f"   Vertices: {model.num_vertices}")
        print(f"   Hidden dim: {model.hidden_dim}")
        print(f"   Layers: {model.num_layers}")
        
        # Test evaluation if we have initial model
        if exp['initial_model'].exists():
            print(f"\nTesting evaluation: {latest_checkpoint.name} vs initial_model.pkl")
            
            config = {
                'num_games': 5,  # Small number for quick test
                'num_vertices': model.num_vertices,
                'k': 3,  # Assuming k=3, adjust if needed
                'mcts_sims': 10,  # Small for quick test
                'c_puct': 3.0,
                'game_mode': 'symmetric',
                'python_eval': True  # Use Python MCTS for speed
            }
            
            start_time = time.time()
            results = evaluate_two_models(
                str(latest_checkpoint),
                str(exp['initial_model']),
                config
            )
            eval_time = time.time() - start_time
            
            print(f"\n✅ Evaluation completed in {eval_time:.1f}s")
            print(f"   Model1 wins: {results['model1_wins']}")
            print(f"   Model2 wins: {results['model2_wins']}")
            print(f"   Draws: {results['draws']}")
            print(f"   Win rate: {results['model1_win_rate']:.1%}")
            
        print("\n" + "="*60)
        print("Standalone evaluation test PASSED!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_standalone_evaluation()
    exit(0 if success else 1)
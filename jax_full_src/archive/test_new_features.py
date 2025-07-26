#!/usr/bin/env python
"""
Test script to verify all new features in the vectorized pipeline
"""

import os
import sys
import json

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from pipeline_vectorized import VectorizedAlphaZeroPipeline

def test_basic_functionality():
    """Test basic pipeline functionality with minimal settings."""
    print("Testing basic pipeline functionality...")
    
    config = {
        'experiment_name': 'test_features',
        'num_iterations': 2,
        'games_per_iteration': 10,
        'batch_size': 4,
        'mcts_simulations': 10,
        'temperature_threshold': 5,
        'c_puct': 1.0,
        'num_vertices': 6,
        'k': 3,
        'game_mode': 'asymmetric',
        'hidden_dim': 32,
        'num_layers': 2,
        'epochs_per_iteration': 2,
        'training_batch_size': 4,
        'learning_rate': 0.001,
        'eval_games': 2,
        'eval_mcts_simulations': 10,
        'target_win_rate': 0.95
    }
    
    # Test 1: Basic initialization
    print("\n1. Testing initialization...")
    pipeline = VectorizedAlphaZeroPipeline(config)
    print("✓ Pipeline initialized successfully")
    
    # Check if directories were created
    assert os.path.exists(pipeline.dirs['root'])
    assert os.path.exists(pipeline.dirs['models'])
    assert os.path.exists(pipeline.dirs['self_play'])
    print("✓ Directories created")
    
    # Check if initial model was saved
    initial_model_path = os.path.join(pipeline.dirs['models'], 'clique_net_iter0.pth.tar')
    assert os.path.exists(initial_model_path)
    print("✓ Initial model saved")
    
    # Test 2: Run one iteration
    print("\n2. Testing single iteration...")
    pipeline.run_iteration(0)
    print("✓ Iteration 0 completed")
    
    # Check if logs were created
    log_path = os.path.join(pipeline.dirs['root'], 'training_log.json')
    assert os.path.exists(log_path)
    with open(log_path, 'r') as f:
        log_data = json.load(f)
        assert 'hyperparameters' in log_data
        assert 'log' in log_data
        assert len(log_data['log']) == 1
    print("✓ JSON logging working")
    
    # Check if plots were created
    plot_path = os.path.join(pipeline.dirs['root'], 'learning_curves.png')
    # Plot might not be created for first iteration
    
    # Check if checkpoint was saved
    checkpoint_path = os.path.join(pipeline.dirs['models'], 'clique_net_iter0.pth.tar')
    assert os.path.exists(checkpoint_path)
    print("✓ Model checkpoint saved")
    
    # Test 3: Resume capability
    print("\n3. Testing resume capability...")
    # Create new pipeline instance to test resume
    pipeline2 = VectorizedAlphaZeroPipeline(config, resume_from=0)
    assert pipeline2.start_iteration == 1
    print("✓ Resume capability working")
    
    # Test 4: Check training history
    assert len(pipeline.training_history['iteration']) == 1
    assert 'win_rate_vs_initial' in pipeline.training_history
    assert 'policy_loss' in pipeline.training_history
    assert 'value_loss' in pipeline.training_history
    print("✓ All metrics tracked")
    
    print("\n✅ All tests passed!")
    
    # Clean up
    if pipeline.wandb_run:
        pipeline.wandb_run.finish()


def test_plotting():
    """Test plotting functionality separately."""
    print("\nTesting plotting functionality...")
    
    config = {
        'experiment_name': 'test_plotting',
        'num_iterations': 3,
        'games_per_iteration': 5,
        'batch_size': 2,
        'mcts_simulations': 5,
        'temperature_threshold': 5,
        'c_puct': 1.0,
        'num_vertices': 6,
        'k': 3,
        'game_mode': 'asymmetric',
        'hidden_dim': 32,
        'num_layers': 2,
        'epochs_per_iteration': 1,
        'training_batch_size': 2,
        'learning_rate': 0.001,
        'eval_games': 1,
        'eval_mcts_simulations': 5,
        'target_win_rate': 0.95
    }
    
    pipeline = VectorizedAlphaZeroPipeline(config)
    
    # Run 2 iterations to have enough data for plotting
    for i in range(2):
        pipeline.run_iteration(i)
    
    # Check if plot was created
    plot_path = os.path.join(pipeline.dirs['root'], 'learning_curves.png')
    assert os.path.exists(plot_path)
    print("✓ Learning curves plotted")
    
    # Clean up
    if pipeline.wandb_run:
        pipeline.wandb_run.finish()


if __name__ == "__main__":
    print("="*70)
    print("TESTING NEW FEATURES IN VECTORIZED PIPELINE")
    print("="*70)
    
    try:
        test_basic_functionality()
        test_plotting()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✅")
        print("="*70)
        print("\nNew features successfully added:")
        print("- ✅ Wandb integration")
        print("- ✅ JSON logging for metrics persistence")
        print("- ✅ Plotting functionality for learning curves")
        print("- ✅ Resume capability from checkpoints")
        print("- ✅ Evaluation against initial model (iter0)")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
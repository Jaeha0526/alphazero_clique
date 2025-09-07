#!/usr/bin/env python3
"""Quick test to verify evaluation game saving works."""

import sys
import os
sys.path.append('/workspace/alphazero_clique/jax_full_src')

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from vectorized_nn import ImprovedBatchedNeuralNetwork
from evaluation_jax_fixed import evaluate_models_jax

def test_eval_save():
    """Test that evaluation games are saved correctly."""
    
    # Create simple test models
    num_vertices = 6
    k = 3
    num_actions = num_vertices * (num_vertices - 1) // 2  # 15 for K_6
    
    # Initialize two simple models
    current_model = ImprovedBatchedNeuralNetwork(
        num_vertices=num_vertices,
        hidden_dim=64,
        num_layers=2,
        asymmetric_mode=False
    )
    
    baseline_model = ImprovedBatchedNeuralNetwork(
        num_vertices=num_vertices,
        hidden_dim=64,
        num_layers=2,
        asymmetric_mode=False
    )
    
    # Test directory
    test_dir = Path("/tmp/test_eval_games")
    test_dir.mkdir(parents=True, exist_ok=True)
    save_path = test_dir / "test_eval.pkl"
    
    print("Testing evaluation game saving...")
    print(f"Save path: {save_path}")
    
    # Run evaluation with game saving
    results = evaluate_models_jax(
        current_model=current_model,
        baseline_model=baseline_model,
        num_games=3,  # Just 3 games for quick test
        num_vertices=num_vertices,
        k=k,
        game_mode='avoid_clique',  # Test with avoid_clique mode
        mcts_sims=5,  # Very few simulations for speed
        c_puct=3.0,
        temperature=0.0,
        verbose=True,
        save_games=True,
        save_path=str(save_path)
    )
    
    print(f"\nEvaluation results: {results}")
    
    # Check if file was created
    if save_path.exists():
        print(f"✅ Evaluation games saved successfully to {save_path}")
        
        # Try to load and verify the data
        import pickle
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nSaved data contains:")
        print(f"  - {data['num_games']} games")
        print(f"  - {len(data['games_data'])} total moves")
        print(f"  - {len(data['games_info'])} game info entries")
        print(f"  - Game mode: {data['game_mode']}")
        print(f"  - Graph: K_{data['vertices']} avoiding {data['k']}-cliques")
        
        # Check first game
        if data['games_info']:
            first_game = data['games_info'][0]
            print(f"\nFirst game:")
            print(f"  - Current starts: {first_game['current_starts']}")
            print(f"  - Winner: {first_game['winner']}")
            print(f"  - Moves: {first_game['num_moves']}")
            
            # Check first move
            if data['games_data']:
                first_move = data['games_data'][0]
                print(f"\nFirst move data:")
                print(f"  - Player: {first_move['player']}")
                print(f"  - Action: {first_move['action']}")
                print(f"  - Model used: {first_move['model_used']}")
                print(f"  - Policy shape: {first_move['policy'].shape}")
    else:
        print(f"❌ Evaluation games file not created at {save_path}")
    
    return save_path

if __name__ == "__main__":
    test_eval_save()
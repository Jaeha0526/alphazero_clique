#!/usr/bin/env python
"""
Minimal test to debug pipeline issues
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from vectorized_self_play_fixed import FixedVectorizedSelfPlay, FixedSelfPlayConfig


def test_self_play():
    """Test self-play component specifically"""
    print("Testing self-play component...")
    print(f"JAX backend: {jax.default_backend()}")
    
    # Initialize model
    print("\n1. Initializing model...")
    model = ImprovedBatchedNeuralNetwork(
        num_vertices=6,
        hidden_dim=32,
        num_layers=2,
        asymmetric_mode=False
    )
    print("✓ Model initialized")
    
    # Configure self-play
    print("\n2. Configuring self-play...")
    config = FixedSelfPlayConfig(
        batch_size=1,
        num_vertices=6,
        k=3,
        game_mode='symmetric',
        mcts_simulations=5,
        c_puct=1.0,
        noise_weight=0.25,
        perspective_mode='alternating',
        skill_variation=0.0
    )
    
    self_play = FixedVectorizedSelfPlay(config, model)
    print("✓ Self-play configured")
    
    # Play one game
    print("\n3. Playing one game...")
    start_time = time.time()
    
    try:
        games = self_play.play_games(1)
        elapsed = time.time() - start_time
        
        print(f"✓ Game completed in {elapsed:.2f}s")
        print(f"  Game length: {len(games[0])} moves")
        
        # Show first few moves
        print("\n  First 3 moves:")
        for i, (board_state, action, policy, value) in enumerate(games[0][:3]):
            print(f"    Move {i}: action={action}, value={value:.3f}")
            
    except Exception as e:
        print(f"✗ Self-play failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_self_play()
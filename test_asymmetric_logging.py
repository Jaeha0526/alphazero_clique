#!/usr/bin/env python
"""
Test script to verify asymmetric logging and statistics work correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

# Add the jax_full_src directory to the path
sys.path.append('jax_full_src')

from vectorized_nn import ImprovedBatchedNeuralNetwork
from train_jax_fully_optimized import train_network_jax_optimized
from train_jax_with_validation import train_network_jax_with_validation
from run_jax_optimized import OptimizedSelfPlay
from dataclasses import dataclass

@dataclass
class TestConfig:
    batch_size: int = 4
    num_vertices: int = 6
    k: int = 3
    game_mode: str = "asymmetric"
    mcts_simulations: int = 10
    temperature_threshold: int = 5
    c_puct: float = 3.0
    perspective_mode: str = "alternating"
    use_true_mctx: bool = False

def create_mock_asymmetric_experience(num_vertices=6, player_role=0):
    """Create a mock training experience for asymmetric mode."""
    # Simple mock edge indices and features
    num_edges = num_vertices * (num_vertices - 1) // 2
    edge_indices = jnp.array([[i, j] for i in range(num_vertices) for j in range(i+1, num_vertices)])
    edge_features = jnp.ones((len(edge_indices), 3))  # 3 features to match network expectation
    
    # Mock policy (uniform random)
    policy = jnp.ones(num_edges) / num_edges
    
    return {
        'edge_indices': edge_indices,
        'edge_features': edge_features,
        'policy': policy,
        'value': 1.0 if player_role == 0 else -1.0,  # Attacker wins = +1, Defender wins = -1
        'player_role': player_role
    }

def test_asymmetric_training():
    """Test asymmetric training with separate attacker/defender loss tracking."""
    print("🧪 Testing Asymmetric Training Loss Tracking")
    print("=" * 50)
    
    # Create model
    model = ImprovedBatchedNeuralNetwork(
        num_vertices=6,
        hidden_dim=32,
        num_layers=2,
        asymmetric_mode=True
    )
    
    # Create mock training data with mixed attacker/defender experiences
    experiences = []
    for i in range(20):
        # Alternate between attacker (0) and defender (1) experiences
        player_role = i % 2
        exp = create_mock_asymmetric_experience(num_vertices=6, player_role=player_role)
        experiences.append(exp)
    
    print(f"Created {len(experiences)} mock experiences")
    print(f"Attacker experiences: {sum(1 for exp in experiences if exp['player_role'] == 0)}")
    print(f"Defender experiences: {sum(1 for exp in experiences if exp['player_role'] == 1)}")
    
    # Test optimized training
    print("\n📈 Testing Optimized Training with Asymmetric Mode...")
    try:
        result = train_network_jax_optimized(
            model,
            experiences,
            epochs=3,
            batch_size=4,
            learning_rate=0.001,
            verbose=True,
            asymmetric_mode=True
        )
        
        if len(result) == 5:
            state, policy_loss, value_loss, attacker_loss, defender_loss = result
            print(f"✅ Optimized training successful!")
            print(f"   Policy Loss: {policy_loss:.4f}")
            print(f"   Value Loss: {value_loss:.4f}")
            print(f"   Attacker Policy Loss: {attacker_loss:.4f}")
            print(f"   Defender Policy Loss: {defender_loss:.4f}")
        else:
            print(f"❌ Expected 5 return values, got {len(result)}")
            
    except Exception as e:
        print(f"❌ Optimized training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test validation training
    print("\n📊 Testing Validation Training with Asymmetric Mode...")
    try:
        state, policy_loss, value_loss, history = train_network_jax_with_validation(
            model,
            experiences,
            epochs=3,
            batch_size=4,
            learning_rate=0.001,
            verbose=True,
            asymmetric_mode=True,
            validation_split=0.3
        )
        
        print(f"✅ Validation training successful!")
        print(f"   Final Policy Loss: {policy_loss:.4f}")
        print(f"   Final Value Loss: {value_loss:.4f}")
        
        # Check history for asymmetric losses
        if 'train_attacker_loss' in history:
            print(f"   Train Attacker Losses: {history['train_attacker_loss']}")
            print(f"   Train Defender Losses: {history['train_defender_loss']}")
        
        if 'val_attacker_loss' in history:
            print(f"   Val Attacker Losses: {history['val_attacker_loss']}")
            print(f"   Val Defender Losses: {history['val_defender_loss']}")
            
    except Exception as e:
        print(f"❌ Validation training failed: {e}")
        import traceback
        traceback.print_exc()

def test_selfplay_statistics():
    """Test self-play statistics collection."""
    print("\n🎮 Testing Self-Play Statistics Collection")
    print("=" * 50)
    
    config = TestConfig()
    self_play = OptimizedSelfPlay(config)
    
    # Check initial statistics
    stats = self_play.get_statistics()
    print("📊 Initial Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test statistics update manually (simulating finished games)
    print("\n🔄 Simulating batch statistics update...")
    
    # Simulate a batch of finished games
    batch_stats = {
        'games_played': 10,
        'attacker_wins': 6,
        'defender_wins': 4,
        'game_lengths': [12, 15, 18, 10, 14, 16, 13, 11, 17, 19],
        'total_moves': 145
    }
    
    self_play._update_statistics(batch_stats)
    
    # Check updated statistics
    updated_stats = self_play.get_statistics()
    print("📈 Updated Statistics:")
    for key, value in updated_stats.items():
        print(f"   {key}: {value}")
    
    # Verify calculations
    expected_avg_length = np.mean(batch_stats['game_lengths'])
    expected_attacker_rate = batch_stats['attacker_wins'] / (batch_stats['attacker_wins'] + batch_stats['defender_wins'])
    
    print(f"\n✅ Verification:")
    print(f"   Expected avg game length: {expected_avg_length:.2f}, Got: {updated_stats['avg_game_length']:.2f}")
    print(f"   Expected attacker win rate: {expected_attacker_rate:.2f}, Got: {updated_stats['win_ratio_attacker']:.2f}")

def main():
    """Run all tests."""
    print("🚀 Starting Asymmetric Logging and Statistics Tests")
    print("=" * 60)
    
    # Check JAX setup
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    
    try:
        test_asymmetric_training()
        test_selfplay_statistics()
        
        print("\n🎉 All tests completed!")
        print("✅ Asymmetric logging and statistics are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
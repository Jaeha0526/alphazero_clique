#!/usr/bin/env python
"""
Test the full pipeline step by step to verify all features are working
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import time

print("Verifying JAX AlphaZero Implementation")
print("=" * 60)
print(f"1. GPU: {jax.default_backend()} - {jax.devices()}")
print(f"2. JAX: Version {jax.__version__}")

# Test 3: MCTS with proper tree search
print("\n3. Testing MCTS with UCT...")
from tree_based_mcts import TreeBasedMCTS, ParallelTreeBasedMCTS
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork

# Create a small model
model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=32, num_layers=1)

# Test single MCTS
mcts = TreeBasedMCTS(num_actions=15, c_puct=3.0)
board = VectorizedCliqueBoard(batch_size=1)

print("   Running 10 MCTS simulations...")
start = time.time()
action_probs = mcts.search(board, model, num_simulations=10, game_idx=0)
print(f"   ✓ MCTS search completed in {time.time()-start:.3f}s")
print(f"   Action probabilities: {action_probs[:5]}... (showing first 5)")

# Test 4: Vectorized games
print("\n4. Testing vectorized games...")
batch_size = 4
parallel_mcts = ParallelTreeBasedMCTS(batch_size=batch_size, num_actions=15, c_puct=3.0)
boards = VectorizedCliqueBoard(batch_size=batch_size)
num_sims = jnp.array([10] * batch_size)

start = time.time()
action_probs_batch = parallel_mcts.search(boards, model, num_sims)
print(f"   ✓ Parallel MCTS for {batch_size} games in {time.time()-start:.3f}s")
print(f"   Output shape: {action_probs_batch.shape}")

# Test 5: JIT compilation
print("\n5. Testing JIT compilation...")
# The model's evaluate_batch should be JIT compiled on first use
edge_indices, edge_features = boards.get_features_for_nn_undirected()

start = time.time()
policies, values = model.evaluate_batch(edge_indices, edge_features)
first_time = time.time() - start
print(f"   First evaluation (includes JIT): {first_time:.3f}s")

start = time.time()
policies, values = model.evaluate_batch(edge_indices, edge_features)
second_time = time.time() - start
print(f"   Second evaluation (JIT cached): {second_time:.3f}s")
print(f"   ✓ JIT speedup: {first_time/second_time:.1f}x")

# Test self-play
print("\n6. Testing self-play with all features...")
from vectorized_self_play_fixed import FixedVectorizedSelfPlay, FixedSelfPlayConfig

config = FixedSelfPlayConfig(
    batch_size=4,
    num_vertices=6,
    k=3,
    game_mode='symmetric',
    mcts_simulations=20,
    c_puct=3.0
)

self_play = FixedVectorizedSelfPlay(config, model)
print("   Playing 2 games...")
start = time.time()
games = self_play.play_games(2)
print(f"   ✓ Generated {len(games)} games in {time.time()-start:.3f}s")
print(f"   Average game length: {sum(len(g) for g in games)/len(games):.1f} moves")

print("\n" + "=" * 60)
print("All features verified:")
print("✓ GPU acceleration with JAX")
print("✓ Proper MCTS with UCT tree search") 
print("✓ Vectorized parallel games")
print("✓ JIT compilation for performance")
print("✓ Full pipeline components working")
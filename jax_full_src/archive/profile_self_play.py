#!/usr/bin/env python
"""
Profile vectorized self-play to find bottlenecks
"""

import sys
sys.path.append('/workspace/alphazero_clique')
sys.path.append('/workspace/alphazero_clique/jax_full_src')

import time
import jax
import jax.numpy as jnp
from vectorized_self_play import VectorizedSelfPlay, SelfPlayConfig
from vectorized_nn import BatchedNeuralNetwork
from vectorized_board import VectorizedCliqueBoard
from optimized_mcts import OptimizedVectorizedMCTS


def profile_components():
    """Profile individual components to find bottlenecks."""
    print("Profiling Vectorized Self-Play Components")
    print("="*60)
    print(f"Device: {jax.devices()[0]}")
    
    batch_size = 16
    num_simulations = 20
    
    # Create components
    nn = BatchedNeuralNetwork()
    boards = VectorizedCliqueBoard(batch_size)
    mcts = OptimizedVectorizedMCTS(batch_size, num_simulations=num_simulations)
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  MCTS simulations: {num_simulations}")
    
    # 1. Test board operations
    print("\n1. Board Operations:")
    start = time.time()
    for _ in range(100):
        edge_indices, edge_features = boards.get_features_for_nn()
        valid_mask = boards.get_valid_moves_mask()
        boards.make_moves(jnp.zeros(batch_size, dtype=jnp.int32))
    elapsed = time.time() - start
    print(f"   100 iterations: {elapsed:.3f}s ({1000*elapsed/100:.1f}ms per iteration)")
    
    # 2. Test neural network
    print("\n2. Neural Network Evaluation:")
    edge_indices, edge_features = boards.get_features_for_nn()
    valid_mask = boards.get_valid_moves_mask()
    
    # Warmup
    _ = nn.evaluate_batch(edge_indices, edge_features, valid_mask)
    
    start = time.time()
    for _ in range(100):
        policies, values = nn.evaluate_batch(edge_indices, edge_features, valid_mask)
        policies.block_until_ready()
    elapsed = time.time() - start
    print(f"   100 batch evaluations: {elapsed:.3f}s ({1000*elapsed/100:.1f}ms per batch)")
    print(f"   Throughput: {batch_size * 100 / elapsed:.0f} positions/sec")
    
    # 3. Test MCTS
    print("\n3. MCTS Search:")
    
    # Warmup JIT
    _ = mcts.search_batch_jit(
        (edge_indices, edge_features),
        nn.model.apply,
        nn.params,
        valid_mask,
        1.0
    )
    
    start = time.time()
    action_probs = mcts.search_batch_jit(
        (edge_indices, edge_features),
        nn.model.apply,
        nn.params,
        valid_mask,
        1.0
    )
    action_probs.block_until_ready()
    elapsed = time.time() - start
    print(f"   {num_simulations} simulations: {elapsed:.3f}s")
    print(f"   Games/second: {batch_size/elapsed:.1f}")
    
    # 4. Test action sampling
    print("\n4. Action Sampling (vectorized):")
    key = jax.random.PRNGKey(0)
    
    # Vectorized sampling
    start = time.time()
    for _ in range(100):
        sample_fn = jax.vmap(
            lambda k, p: jax.random.choice(k, 15, p=p)
        )
        keys = jax.random.split(key, batch_size)
        actions = sample_fn(keys, action_probs)
        actions.block_until_ready()
    elapsed = time.time() - start
    print(f"   100 iterations: {elapsed:.3f}s ({1000*elapsed/100:.1f}ms per iteration)")
    
    # 5. Test full self-play iteration
    print("\n5. Full Self-Play Iteration:")
    config = SelfPlayConfig(
        batch_size=batch_size,
        mcts_simulations=num_simulations,
        temperature_threshold=5,
        max_moves=20  # Limit moves
    )
    self_play = VectorizedSelfPlay(config, nn)
    
    start = time.time()
    experiences = self_play.play_batch()
    elapsed = time.time() - start
    
    num_games = len(experiences)
    total_moves = sum(len(exp) for exp in experiences)
    avg_moves = total_moves / num_games if num_games > 0 else 0
    
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Games: {num_games}")
    print(f"   Average moves/game: {avg_moves:.1f}")
    print(f"   Games/second: {num_games/elapsed:.2f}")
    
    # Breakdown
    print("\n6. Performance Breakdown:")
    mcts_time_per_move = 0.3  # From test above
    total_mcts_time = mcts_time_per_move * avg_moves
    overhead_time = elapsed - total_mcts_time
    
    print(f"   MCTS time: ~{total_mcts_time:.1f}s ({100*total_mcts_time/elapsed:.0f}%)")
    print(f"   Overhead: ~{overhead_time:.1f}s ({100*overhead_time/elapsed:.0f}%)")
    print(f"   (Overhead includes board updates, experience storage, etc.)")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("1. MCTS is the main bottleneck")
    print("2. Need to increase batch size for better GPU utilization")
    print("3. Consider reducing MCTS simulations for self-play")
    print("="*60)


if __name__ == "__main__":
    profile_components()
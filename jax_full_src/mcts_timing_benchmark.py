#!/usr/bin/env python
"""
Benchmark to measure time for each phase of MCTS
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import List, Tuple

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


def benchmark_mcts_phases(num_games: int = 100, num_simulations: int = 50):
    """
    Measure time for each phase of MCTS to identify bottlenecks.
    """
    print(f"\n=== MCTS Phase Timing Benchmark ===")
    print(f"Games: {num_games}, Simulations per move: {num_simulations}")
    print(f"Board: n=9, k=4 (36 possible actions)")
    
    # Initialize
    boards = VectorizedCliqueBoard(
        batch_size=num_games,
        num_vertices=9,
        k=4,
        game_mode="symmetric"
    )
    
    model = ImprovedBatchedNeuralNetwork(
        num_vertices=9,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=False
    )
    
    # Warm up JIT compilation
    print("\nWarming up JIT compilation...")
    edge_indices, edge_features = boards.get_features_for_nn_undirected()
    valid_masks = boards.get_valid_moves_mask()
    _ = model.evaluate_batch(edge_indices, edge_features, valid_masks)
    
    # Simulate tree data structures
    # In real implementation, these would be more complex
    N = jnp.zeros((num_games, 1000, 36))  # Visit counts (1000 nodes per game)
    W = jnp.zeros((num_games, 1000, 36))  # Total values
    P = jnp.zeros((num_games, 1000, 36))  # Priors
    
    # Initialize with some data to make it realistic
    N = N.at[:, 0, :].set(jnp.ones((num_games, 36)) * 10)  # Root visited 10 times
    key = jax.random.PRNGKey(42)
    W = W.at[:, 0, :].set(jax.random.normal(key, (num_games, 36)))
    P = P.at[:, 0, :].set(jnp.ones((num_games, 36)) / 36)  # Uniform priors
    
    print("\nRunning benchmark...\n")
    
    # Track timings
    selection_times = []
    nn_eval_times = []
    backup_times = []
    total_times = []
    
    # Run several MCTS simulations
    for sim in range(num_simulations):
        total_start = time.time()
        
        # Phase 1: SELECTION
        selection_start = time.time()
        
        # Simulate parallel tree traversal
        current_nodes = jnp.zeros(num_games, dtype=jnp.int32)  # Start at root
        paths = []
        
        # Simulate 3-4 steps down the tree
        for depth in range(4):
            # Get data for current nodes
            batch_idx = jnp.arange(num_games)
            current_N = N[batch_idx, current_nodes]  # (num_games, 36)
            current_W = W[batch_idx, current_nodes]
            current_P = P[batch_idx, current_nodes]
            
            # Calculate UCB
            Q = current_W / jnp.maximum(current_N, 1)
            sqrt_parent = jnp.sqrt(current_N.sum(axis=1, keepdims=True))
            U = 3.0 * current_P * sqrt_parent / (1 + current_N)
            UCB = Q + U
            
            # Mask invalid actions (simplified)
            UCB = jnp.where(valid_masks, UCB, -jnp.inf)
            
            # Select best actions
            best_actions = jnp.argmax(UCB, axis=1)
            paths.append((current_nodes.copy(), best_actions))
            
            # Move to children (simulate)
            # In real implementation, this would check if children exist
            current_nodes = (current_nodes * 36 + best_actions) % 1000  # Fake child index
        
        selection_time = time.time() - selection_start
        selection_times.append(selection_time)
        
        # Phase 2: NEURAL NETWORK EVALUATION
        nn_start = time.time()
        
        # Simulate that 80% of games need NN eval (others hit terminal)
        games_needing_nn = int(num_games * 0.8)
        
        if games_needing_nn > 0:
            # Get features for leaf nodes
            # In real implementation, we'd get features for specific nodes
            leaf_indices, leaf_features = boards.get_features_for_nn_undirected()
            leaf_masks = boards.get_valid_moves_mask()
            
            # Batch NN evaluation
            leaf_policies, leaf_values = model.evaluate_batch(
                leaf_indices[:games_needing_nn],
                leaf_features[:games_needing_nn],
                leaf_masks[:games_needing_nn]
            )
        
        nn_time = time.time() - nn_start
        nn_eval_times.append(nn_time)
        
        # Phase 3: BACKUP
        backup_start = time.time()
        
        # Simulate backing up values through the paths
        values = jnp.ones(num_games) * 0.5  # Dummy values
        
        for node_indices, actions in reversed(paths):
            # Update N and W
            batch_idx = jnp.arange(num_games)
            
            # These operations would update the tree
            N = N.at[batch_idx, node_indices, actions].add(1)
            W = W.at[batch_idx, node_indices, actions].add(values)
            
            # Flip values for opponent
            values = -values
        
        backup_time = time.time() - backup_start
        backup_times.append(backup_time)
        
        total_time = time.time() - total_start
        total_times.append(total_time)
        
        if sim % 10 == 0:
            print(f"Simulation {sim}: Selection={selection_time*1000:.1f}ms, "
                  f"NN={nn_time*1000:.1f}ms, Backup={backup_time*1000:.1f}ms, "
                  f"Total={total_time*1000:.1f}ms")
    
    # Calculate averages
    avg_selection = np.mean(selection_times) * 1000
    avg_nn = np.mean(nn_eval_times) * 1000
    avg_backup = np.mean(backup_times) * 1000
    avg_total = np.mean(total_times) * 1000
    
    print(f"\n=== Average Times per Simulation ===")
    print(f"Selection:  {avg_selection:6.2f}ms ({avg_selection/avg_total*100:4.1f}%)")
    print(f"NN Eval:    {avg_nn:6.2f}ms ({avg_nn/avg_total*100:4.1f}%)")
    print(f"Backup:     {avg_backup:6.2f}ms ({avg_backup/avg_total*100:4.1f}%)")
    print(f"Total:      {avg_total:6.2f}ms")
    print(f"\nOther (overhead): {avg_total - avg_selection - avg_nn - avg_backup:6.2f}ms")
    
    # Extrapolate to full MCTS
    print(f"\n=== Extrapolated for 300 simulations ===")
    print(f"Total time per move: {avg_total * 300:.1f}ms = {avg_total * 300 / 1000:.2f}s")
    print(f"For 30 moves: {avg_total * 300 * 30 / 1000:.1f}s")
    print(f"For 500 games: Same time (fully parallelized)")
    
    # Compare with different batch sizes
    print(f"\n=== Batch Size Impact ===")
    for batch_size in [10, 50, 100, 500]:
        # NN time scales with batch size (but sublinearly on GPU)
        nn_time_scaled = avg_nn * (batch_size / num_games) ** 0.3  # GPU scaling
        total_scaled = avg_selection + nn_time_scaled + avg_backup
        print(f"{batch_size:3d} games: {total_scaled:6.2f}ms per sim, "
              f"{total_scaled * 300 / 1000:5.2f}s per move")


if __name__ == "__main__":
    # Run with different configurations
    print("Testing with 100 games:")
    benchmark_mcts_phases(num_games=100, num_simulations=50)
    
    print("\n" + "="*60 + "\n")
    print("Testing with 500 games:")
    benchmark_mcts_phases(num_games=500, num_simulations=20)
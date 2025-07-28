#!/usr/bin/env python
"""
Analyze MCTS timing by examining the algorithm structure
"""

import time
import numpy as np


def analyze_mcts_phases():
    """
    Analyze expected timing for each MCTS phase based on operations
    """
    print("=== MCTS Phase Analysis for n=9, k=4 ===")
    print("\nBoard: 9 vertices, 4-clique, 36 possible edges")
    print("Comparing: Sequential vs Vectorized (500 games)\n")
    
    # Based on our experiments and typical GPU/CPU performance
    
    # 1. NN Evaluation timing (most important)
    print("1. NEURAL NETWORK EVALUATION")
    print("-" * 40)
    
    # Single NN eval (from our JAX tests)
    single_nn_time = 0.001  # 1ms after JIT
    batch_nn_time_500 = 0.003  # 3ms for 500 games (GPU parallelism)
    
    print(f"Single game:     {single_nn_time*1000:.1f}ms per evaluation")
    print(f"Batch 500 games: {batch_nn_time_500*1000:.1f}ms per evaluation")
    print(f"Speedup:         {single_nn_time*500/batch_nn_time_500:.0f}x")
    
    # 2. Tree operations
    print("\n2. TREE OPERATIONS (Selection, Expansion, Backup)")
    print("-" * 40)
    
    # Python dict operations (from SimpleTreeMCTS)
    python_tree_ops = 0.5  # 0.5ms per simulation per game
    
    # Vectorized array operations
    vectorized_ops = 0.01  # 0.01ms for all 500 games
    
    print(f"Python (per game):  {python_tree_ops:.1f}ms")
    print(f"Vectorized (batch): {vectorized_ops:.2f}ms for all 500 games")
    print(f"Speedup:            {python_tree_ops*500/vectorized_ops:.0f}x")
    
    # 3. Full MCTS simulation breakdown
    print("\n3. FULL MCTS SIMULATION (1 simulation)")
    print("-" * 40)
    
    print("\nSequential (1 game):")
    seq_selection = 0.1
    seq_nn_eval = single_nn_time * 1000
    seq_expansion = 0.1
    seq_backup = 0.1
    seq_total = seq_selection + seq_nn_eval + seq_expansion + seq_backup
    
    print(f"  Selection:  {seq_selection:6.2f}ms ({seq_selection/seq_total*100:4.1f}%)")
    print(f"  NN Eval:    {seq_nn_eval:6.2f}ms ({seq_nn_eval/seq_total*100:4.1f}%)")
    print(f"  Expansion:  {seq_expansion:6.2f}ms ({seq_expansion/seq_total*100:4.1f}%)")
    print(f"  Backup:     {seq_backup:6.2f}ms ({seq_backup/seq_total*100:4.1f}%)")
    print(f"  Total:      {seq_total:6.2f}ms")
    
    print("\nVectorized (500 games at once):")
    vec_selection = 0.02  # Vectorized ops
    vec_nn_eval = batch_nn_time_500 * 1000
    vec_expansion = 0.02
    vec_backup = 0.02
    vec_total = vec_selection + vec_nn_eval + vec_expansion + vec_backup
    
    print(f"  Selection:  {vec_selection:6.2f}ms ({vec_selection/vec_total*100:4.1f}%)")
    print(f"  NN Eval:    {vec_nn_eval:6.2f}ms ({vec_nn_eval/vec_total*100:4.1f}%)")
    print(f"  Expansion:  {vec_expansion:6.2f}ms ({vec_expansion/vec_total*100:4.1f}%)")
    print(f"  Backup:     {vec_backup:6.2f}ms ({vec_backup/vec_total*100:4.1f}%)")
    print(f"  Total:      {vec_total:6.2f}ms")
    
    print(f"\nPer-game time: {vec_total/500:.3f}ms (vs {seq_total:.2f}ms sequential)")
    print(f"Speedup: {seq_total/(vec_total/500):.0f}x")
    
    # 4. Full move timing (300 simulations)
    print("\n4. FULL MOVE TIMING (300 MCTS simulations)")
    print("-" * 40)
    
    seq_move_time = seq_total * 300 / 1000  # seconds
    vec_move_time = vec_total * 300 / 1000  # seconds
    
    print(f"Sequential (per game): {seq_move_time:.2f}s")
    print(f"Vectorized (500 games): {vec_move_time:.2f}s")
    print(f"Speedup: {seq_move_time*500/vec_move_time:.0f}x")
    
    # 5. Full game projection
    print("\n5. FULL GAME PROJECTION (30 moves, 500 games)")
    print("-" * 40)
    
    moves_per_game = 30
    
    # Sequential
    seq_total_time = seq_move_time * moves_per_game * 500
    print(f"Sequential: {seq_total_time:.0f}s = {seq_total_time/60:.1f} minutes = {seq_total_time/3600:.2f} hours")
    
    # Vectorized  
    vec_total_time = vec_move_time * moves_per_game
    print(f"Vectorized: {vec_total_time:.0f}s = {vec_total_time/60:.1f} minutes")
    
    print(f"\nTotal speedup: {seq_total_time/vec_total_time:.0f}x")
    
    # 6. Bottleneck analysis
    print("\n6. BOTTLENECK ANALYSIS")
    print("-" * 40)
    
    print("\nIn vectorized implementation:")
    print(f"- NN evaluation: {vec_nn_eval/vec_total*100:.1f}% of time")
    print(f"- Tree operations: {(vec_total-vec_nn_eval)/vec_total*100:.1f}% of time")
    
    print("\nKey insights:")
    print("1. NN batching gives ~167x speedup (500 games / 3ms overhead)")
    print("2. Tree vectorization gives ~1000x+ speedup")
    print("3. Overall speedup limited by Amdahl's law")
    print("4. Main bottleneck becomes NN evaluation (even batched)")
    
    # 7. Memory considerations
    print("\n7. MEMORY REQUIREMENTS")
    print("-" * 40)
    
    nodes_per_tree = 200  # Average
    node_size = 36 * 4 * 2  # N, W arrays (float32)
    total_memory = 500 * nodes_per_tree * node_size / 1024 / 1024
    
    print(f"Tree storage: {total_memory:.1f} MB for 500 games")
    print(f"GPU memory for NN: ~100 MB for batch of 500")
    print(f"Total: ~{total_memory + 100:.0f} MB (easily fits in modern GPU)")


def compare_implementations():
    """
    Compare our different MCTS implementations
    """
    print("\n\n=== IMPLEMENTATION COMPARISON ===")
    print("-" * 50)
    
    implementations = [
        ("Original (src/)", "Sequential", "Full tree", "Python loops", 1.3, "hours"),
        ("SimpleTreeMCTS", "Semi-parallel", "Full tree", "Python + batch NN", 0.5, "hours"),
        ("VectorizedJITMCTS", "Fully parallel", "No tree", "JAX arrays", 0.1, "hours"),
        ("Ideal VectorizedTree", "Fully parallel", "Full tree", "JAX arrays", 0.15, "hours")
    ]
    
    print(f"{'Implementation':<20} {'Parallelism':<15} {'Tree?':<10} {'Backend':<15} {'Time':<10}")
    print("-" * 80)
    
    for name, parallel, tree, backend, time, unit in implementations:
        print(f"{name:<20} {parallel:<15} {tree:<10} {backend:<15} {time:>4.1f} {unit:<5}")
    
    print("\nRecommendations:")
    print("1. For research/experimentation: Use VectorizedJITMCTS (fast but approximate)")
    print("2. For final training: Implement full VectorizedTree MCTS")
    print("3. Key optimization: Batch NN evaluations across all games")


if __name__ == "__main__":
    analyze_mcts_phases()
    compare_implementations()
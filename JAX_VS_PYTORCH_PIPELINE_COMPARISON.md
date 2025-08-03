# JAX vs PyTorch Pipeline Comparison (Updated August 2025)

## Overview

Both pipelines implement AlphaZero, but with different approaches to parallelization and optimization. Recent optimizations have significantly improved JAX performance.

## 1. Pipeline Structure Comparison

### PyTorch Pipeline (`src/pipeline_clique.py`)
```
Iteration Structure:
1. Model Loading → 2. Self-Play → 3. Training → 4. Evaluation → 5. Model Update
```

**Key characteristics:**
- Multiprocessing for parallel self-play (CPU-based)
- Individual MCTS trees per game
- Sequential game processing within each process
- Real-time plotting with matplotlib

### JAX Pipeline (`jax_full_src/run_jax_optimized.py`)
```
Iteration Structure:
1. Self-Play (batched) → 2. Training → 3. Evaluation → 4. Checkpoint → 5. Logging
```

**Key characteristics:**
- Vectorized batch processing (GPU-accelerated)
- Memory-optimized MCTS (90% reduction)
- JIT-compiled training (5x speedup)
- Optional True MCTX with JAX primitives
- Vectorized batch preparation (10x speedup)
- Similar plotting but simplified structure

## 2. MCTS Implementation Comparison

### PyTorch MCTS
- **Location**: `src/MCTS_clique.py`
- **Class**: `UCTNode` with tree traversal
- **Key features**:
  - Traditional tree structure with node objects
  - Sequential UCB calculation: `for action in range(num_actions)`
  - Dirichlet noise at root
  - Python dictionaries for child nodes

### JAX MCTS - Current Implementation
- **Primary**: `simple_tree_mcts.py` (SimpleTreeMCTS)
- **Backup**: `tree_based_mcts.py` (ParallelTreeBasedMCTS)
- **NOT using**: The optimized MCTX implementations (mctx_final_optimized.py, etc.)

**Key features of SimpleTreeMCTS:**
- Pre-allocated arrays for tree nodes
- JIT-compiled UCB calculation
- Vectorized operations within nodes
- Batch neural network evaluation

## 3. Key Differences

### A. Parallelization Strategy

**PyTorch:**
```python
# CPU multiprocessing
for i in range(num_cpus):
    p = mp.Process(target=MCTS_self_play, args=(...))
    processes.append(p)
```

**JAX:**
```python
# Vectorized batch processing
boards = VectorizedCliqueBoard(batch_size=batch_size, ...)
mcts_probs = mcts.search(boards, neural_network, ...)
```

### B. MCTS Computation

**PyTorch (Sequential):**
```python
# O(n) for n actions
for action in range(num_actions):
    q = W[action] / (1 + N[action])
    u = c_puct * sqrt(parent_N) * P[action] / (1 + N[action])
    ucb[action] = q + u
```

**JAX (Vectorized):**
```python
# O(1) with SIMD
Q = W / (1.0 + N)
U = c_puct * sqrt_visits * (P / (1.0 + N))
ucb = Q + U
```

### C. Data Storage

**PyTorch:**
- Individual pickle files per game: `game_{timestamp}_cpu{i}_game{j}_iter{N}.pkl`
- Model files: `clique_net_iter{N}.pth.tar`

**JAX:**
- Batched data collection in memory
- Checkpoint files: `checkpoint_iter_{N}.pkl`
- No individual game files

### D. Training Process

**PyTorch:**
- 90/10 train/validation split
- Early stopping with patience
- Learning rate scheduling
- Separate policy losses for asymmetric mode

**JAX:**
- Full dataset training (no explicit validation split in shown code)
- Fixed number of epochs
- Simpler loss tracking
- Optax optimizer with JIT compilation

## 4. MCTX Implementation Status

### Available MCTX Implementations (NOT CURRENTLY USED):

1. **mctx_final_optimized.py** - MCTXFinalOptimized
   - Full vectorization with pre-allocated arrays
   - Optimized for large batches
   - Best performance for n=9, k=4

2. **true_mctx_implementation.py** - TrueMCTXImplementation
   - Follows MCTX design principles
   - Complex but fully optimized

3. **mctx_style_mcts_v2.py** - MCTXStyleMCTSV2
   - Uses jax.lax.while_loop
   - Better JIT compilation

4. **simple_true_mctx.py** - SimpleTrueMCTX
   - Simplified version for demonstration
   - Easier to understand

### Current Status:
- **JAX pipeline uses SimpleTreeMCTS**, not the optimized MCTX versions
- Based on our benchmarks, for optimal performance:
  - Small games (n=6, k=3): Current implementation is fine
  - Large games (n=9, k=4): Should switch to MCTX implementations

## 5. Performance Implications

From our analysis:
- **Small games, small batches**: PyTorch wins (lower overhead)
- **Large games, any batch**: JAX should win IF using MCTX
- **Current JAX pipeline**: Not utilizing full potential

## 6. Recommendations

1. **Update JAX pipeline** to use MCTX implementations:
   ```python
   # Replace
   from simple_tree_mcts import SimpleTreeMCTS
   # With
   from mctx_final_optimized import MCTXFinalOptimized
   ```

2. **Add configuration** to select MCTS implementation:
   ```python
   if args.game_size == "large":
       mcts = MCTXFinalOptimized(...)
   else:
       mcts = SimpleTreeMCTS(...)
   ```

3. **Port missing features** from PyTorch to JAX:
   - Early stopping
   - Learning rate scheduling  
   - Validation split
   - Skill variation (already partially implemented)

4. **Benchmark the updated pipeline** to verify improvements

## Summary

The JAX pipeline has the infrastructure for high performance but is not using the optimized MCTX implementations we developed. Switching to MCTXFinalOptimized for large games should provide significant speedups based on our benchmarking results.
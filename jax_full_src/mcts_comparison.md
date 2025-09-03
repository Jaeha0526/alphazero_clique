# MCTS Implementation Comparison

## Current Working Implementations

### 1. TreeBasedMCTS
- **What it does**: Full MCTS with proper tree structure (MCTSNode class)
- **Features**: SELECT → EXPAND → EVALUATE → BACKUP phases
- **Parallel support**: Yes, via ParallelTreeBasedMCTS class
- **Used by**: vectorized_self_play_fixed.py

### 2. SimpleTreeMCTS
- **What it does**: Simplified tree MCTS with clean implementation
- **Features**: Full tree search with native batch support
- **Parallel support**: Yes, processes batch_size games in parallel
- **Used by**: run_jax_optimized.py

### 3. SimpleTreeMCTSTimed
- **What it does**: Same as SimpleTreeMCTS with timing/profiling
- **Features**: Adds performance monitoring
- **Used by**: run_jax_optimized.py for performance analysis

## Key Characteristics

### Why Tree MCTS is Slower on CPU:
1. **Python overhead**: Managing tree structures in Python
2. **Memory allocation**: Creating nodes dynamically
3. **Board copying**: Each node needs its own board state
4. **Sequential nature**: Hard to fully vectorize tree traversal

### Performance on CPU:
- PyTorch: ~19ms per MCTS search
- JAX TreeBasedMCTS: ~515ms per search
- JAX SimpleTreeMCTS: ~413ms per game (with parallelism)

### All Implementations Now:
- ✅ Perform real tree search
- ✅ Support parallel game processing
- ✅ Are algorithmically correct
- ❌ Removed all "fake" MCTS that just reweighted policies
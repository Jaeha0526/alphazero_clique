# MCTX Pipeline Update Summary

## Changes Made

### 1. Updated `run_jax_optimized.py`
- **Removed imports**: `SimpleTreeMCTS` and `SimpleTreeMCTSTimed`
- **Added import**: `from mctx_final_optimized import MCTXFinalOptimized`
- **Updated `OptimizedSelfPlay.play_games()`**: Now creates `MCTXFinalOptimized` instead of `SimpleTreeMCTSTimed`
- **Added command line arguments**: `--vertices`, `--k`, `--mcts_sims`
- **Updated Config**: Uses command line arguments instead of hardcoded values

### 2. Updated `run_jax_improved.py`
- **Added import**: `from mctx_final_optimized import MCTXFinalOptimized`
- Ready for future updates if needed

### 3. Updated `vectorized_self_play_fixed.py`
- **Removed import**: `from tree_based_mcts import ParallelTreeBasedMCTS`
- **Added import**: `from mctx_final_optimized import MCTXFinalOptimized`
- **Updated `play_batch()`**: Now creates `MCTXFinalOptimized` instead of `ParallelTreeBasedMCTS`

## Performance Impact

Based on our benchmarks, using `MCTXFinalOptimized`:

### For n=6, k=3 (15 actions):
- Before: ~30ms per game
- After: Similar performance (MCTX optimizations don't help much for small games)

### For n=9, k=4 (36 actions):
- Before: ~660ms per game (if using SimpleTreeMCTS)
- After: ~119ms per game with MCTXFinalOptimized
- **5.6x speedup!**

## How to Run

### Small game (n=6, k=3):
```bash
python run_jax_optimized.py \
    --num_iterations 10 \
    --num_episodes 100 \
    --batch_size 32 \
    --vertices 6 \
    --k 3 \
    --mcts_sims 50
```

### Large game (n=9, k=4) - where MCTX shines:
```bash
python run_jax_optimized.py \
    --num_iterations 10 \
    --num_episodes 100 \
    --batch_size 100 \
    --vertices 9 \
    --k 4 \
    --mcts_sims 100
```

### Key parameters:
- `--batch_size`: MCTX scales better with larger batches (try 100+)
- `--mcts_sims`: Can afford more simulations with MCTX efficiency
- `--vertices` and `--k`: Game size parameters

## Testing

Run the test script to verify everything works:
```bash
cd jax_full_src
python test_updated_pipeline.py
```

## Notes

1. **Probability normalization**: The test revealed that MCTX outputs might need normalization. This is handled in the pipeline.

2. **JIT compilation**: First run will be slower due to JAX compilation. Subsequent runs will be much faster.

3. **GPU memory**: MCTX pre-allocates arrays, so watch GPU memory usage for very large batch sizes.

4. **Future improvements** (as mentioned by user):
   - Port early stopping from PyTorch pipeline
   - Add learning rate scheduling
   - Add validation split
   - Add more logging/metrics

## Summary

The JAX pipeline now uses the fully optimized MCTX implementation, which should provide significant speedups for larger games. The change is transparent - all interfaces remain the same, but the underlying MCTS is now highly optimized with:
- Pre-allocated arrays
- Vectorized operations
- JIT compilation
- Efficient tree traversal
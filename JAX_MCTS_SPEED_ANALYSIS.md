# JAX MCTS Speed Analysis

## The Real Problem

It's not the tree search algorithm - it's the neural network inference!

### Performance Breakdown

**PyTorch MCTS (20 simulations)**:
- Total time: 0.14s
- NN inference: 0.7ms per call × 20 = 14ms
- Tree operations: ~6ms
- **Result**: Fast and correct

**JAX Tree MCTS (20 simulations)**:
- Total time: 1.12s  
- NN inference: 35ms per call × 20 = 700ms
- Tree operations: ~400ms
- **Result**: Correct but 80x slower

**JAX JIT MCTS (20 simulations)**:
- Total time: 0.035s
- NN inference: 35ms × 1 call = 35ms
- No tree operations
- **Result**: Fast but completely broken

## Why is JAX NN so slow?

1. **JAX is optimized for large batches on GPU/TPU**, not single examples on CPU
2. **PyTorch has better CPU optimizations** for small models
3. **JAX's JIT compilation overhead** doesn't pay off for single forward passes
4. **JAX's functional style** has overhead for small operations

## The Dilemma

1. **Broken JIT MCTS**: Fast but doesn't work (can't learn)
2. **Fixed Tree MCTS**: Works but too slow (80x slower than PyTorch)
3. **Need**: A solution that is both correct AND reasonably fast

## Potential Solutions

### 1. **Batch Tree Search** (Best approach)
Instead of evaluating one position at a time:
```python
# Collect positions from multiple simulations
positions_to_evaluate = []
for sim in range(batch_size):
    # Tree traversal
    positions_to_evaluate.append(leaf_position)

# Evaluate all at once
all_values = model(batch_positions)  # Much faster!
```

### 2. **Virtual Loss**
Allow multiple simulations to run in parallel through the tree:
```python
# Multiple threads traverse tree simultaneously
# Each assumes others will lose (virtual loss)
# Batch evaluate when all reach leaves
```

### 3. **Hybrid Approach**
- Use PyTorch for MCTS during self-play (fast)
- Use JAX for training (better for large batches)
- Convert between formats as needed

### 4. **Root Parallelization** 
Run completely independent MCTS trees for different games:
```python
# Instead of 1 game with 100 simulations
# Run 10 games with 10 simulations each in parallel
```

## Benchmarks

| Implementation | Time (20 sims) | Correct? | Practical? |
|----------------|----------------|----------|------------|
| PyTorch MCTS | 0.014s | ✓ | ✓ |
| JAX Broken | 0.035s | ✗ | ✗ |
| JAX Fixed | 1.117s | ✓ | ✗ (too slow) |
| JAX Batched* | ~0.1s | ✓ | ✓ |

*Estimated with proper batching

## Conclusion

The fixed tree-based MCTS is algorithmically correct but impractically slow due to JAX's poor single-example CPU performance. The solution is not to abandon correct MCTS, but to implement it in a way that leverages JAX's strengths (batching) while maintaining algorithmic correctness.

The most practical approach is likely a hybrid:
1. Use batched tree search for self-play
2. Or use PyTorch for self-play and JAX for training
3. Or implement virtual loss for parallel simulations
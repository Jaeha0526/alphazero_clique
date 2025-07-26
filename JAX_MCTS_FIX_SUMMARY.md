# JAX AlphaZero MCTS Fix Summary

## Critical Issue Discovered

The JAX implementation had a **fundamental algorithmic error** in the MCTS implementation. The original `vectorized_mcts_jit.py` was not actually performing Monte Carlo Tree Search at all.

### Original JAX MCTS (Broken)
- ❌ No tree structure
- ❌ Only evaluated the root position once
- ❌ Never explored new game states
- ❌ Just redistributed visit counts based on PUCT scores
- ❌ Essentially a weighted random selection with slight bias

### PyTorch MCTS (Correct)
- ✅ Builds an actual search tree with nodes
- ✅ Each simulation traverses from root to leaf
- ✅ Expands new nodes and explores new positions
- ✅ Evaluates multiple game states with the neural network
- ✅ Backs up values through the tree

## Why This Broke Training

1. **No Exploration**: MCTS should find better moves than the raw neural network policy, but the JAX version couldn't explore beyond the initial position
2. **No Lookahead**: Real MCTS evaluates future positions to estimate move quality, JAX only saw the current position
3. **Policy Can't Improve**: Training data was just a slightly noisy version of the current policy
4. **Values Are Meaningless**: Without actual search, value estimates couldn't improve

## Solution Implemented

Created a new `tree_based_mcts.py` that implements proper MCTS:

```python
class TreeBasedMCTS:
    """Proper tree-based MCTS implementation that actually searches."""
    
    def search(self, root_board, neural_network, num_simulations, ...):
        # 1. Selection - traverse tree to leaf using UCB
        # 2. Expansion - add new node when reaching unexplored position  
        # 3. Evaluation - use neural network on NEW position
        # 4. Backup - propagate value back up the tree
```

### Key Features of Fixed Implementation
- ✅ Maintains actual tree structure with nodes and children
- ✅ Explores different game states by making moves
- ✅ Evaluates multiple positions per search (not just root)
- ✅ Properly implements UCB selection formula
- ✅ Supports both fixed and alternating perspective modes
- ✅ Adds Dirichlet noise to root for exploration

## Verification Test Results

Running the test (`test_tree_mcts.py`) shows the difference:

### Tree-Based MCTS (Fixed)
- 10 simulations → 11 neural network calls (evaluates multiple positions)
- 50 simulations → 65 neural network calls  
- 100 simulations → 116 neural network calls
- Non-uniform visit distribution (selective search)

### JIT MCTS (Broken)
- Any number of simulations → 1 neural network call (only root)
- Uniform-ish visit distribution

## Performance Trade-offs

The fixed tree-based MCTS is slower than the broken JIT version because:
1. It actually builds and searches a tree structure
2. It evaluates many positions instead of just one
3. It can't be fully JIT-compiled due to dynamic tree structure

However, this is necessary for AlphaZero to work correctly. The performance can be improved by:
- Running multiple independent MCTS searches in parallel
- Using smaller batch sizes for self-play
- Implementing a more efficient tree structure
- Potentially using a hybrid approach with some JIT optimization

## Other Fixes Applied

While fixing MCTS, I also aligned other parameters with PyTorch:
- ✅ Set c_puct to 3.0 (was 1.0) for better exploration
- ✅ Added proper temperature annealing schedule
- ✅ Added noise weight decay during games
- ✅ Fixed L2 regularization in training
- ✅ Added gradient clipping
- ✅ Added learning rate warmup

## Next Steps

1. Run full training with the fixed MCTS to verify improvement
2. Optimize the tree-based implementation for better performance
3. Consider implementing a batched tree search for efficiency
4. Add checkpointing and model evaluation during training

## Conclusion

The JAX implementation's poor training performance was due to a fundamental misunderstanding of how MCTS works. The "vectorized" version optimized away the actual tree search, leaving only a policy reweighting mechanism. With the proper tree-based implementation, the JAX version should now be able to learn effectively like the PyTorch version.
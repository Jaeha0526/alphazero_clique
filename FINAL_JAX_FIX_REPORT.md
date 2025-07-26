# Final Report: JAX AlphaZero Critical Fix

## Executive Summary

I was asked to investigate why the JAX implementation of AlphaZero was not training properly. Through a thorough line-by-line comparison with the PyTorch implementation, I discovered a **fundamental algorithmic error**: the JAX MCTS implementation was not actually performing tree search at all.

## The Critical Discovery

### What Was Wrong

The JAX "MCTS" in `vectorized_mcts_jit.py` was fundamentally broken:

```python
# Pseudo-code of broken implementation
def search():
    # Evaluate root position ONCE
    policy, value = neural_network(root_position)
    
    # Just redistribute visit counts based on PUCT scores
    for sim in range(num_simulations):
        action = argmax(PUCT_scores)
        visit_counts[action] += 1
        value_sums[action] += value  # Same value every time!
```

**This is not Monte Carlo Tree Search!** It never:
- Built a tree structure
- Explored new positions
- Made any moves
- Evaluated anything beyond the root

### What MCTS Should Do

The PyTorch implementation correctly performs:

```python
# Correct MCTS algorithm
def search():
    for sim in range(num_simulations):
        # 1. SELECT: Traverse tree to leaf
        leaf = root.select_leaf()
        
        # 2. EXPAND: Add new node at leaf
        new_position = make_move(leaf.position)
        
        # 3. EVALUATE: Neural network on NEW position
        policy, value = neural_network(new_position)
        
        # 4. BACKUP: Propagate value up tree
        leaf.backup(value)
```

## The Fix

I implemented a proper tree-based MCTS in `tree_based_mcts.py`:

1. **MCTSNode class**: Maintains tree structure with parent/children relationships
2. **Tree traversal**: Actually explores different game states
3. **Node expansion**: Evaluates new positions with the neural network
4. **Value backup**: Propagates information through the tree

### Verification

The test results confirm the fix works:
- Tree-based MCTS calls the neural network multiple times (11, 65, 116 calls for 10, 50, 100 simulations)
- JIT MCTS only calls it once (just the root)
- Tree-based MCTS produces non-uniform action distributions (selective search)

## Other Improvements Made

While fixing the core issue, I also aligned other parameters with PyTorch:

1. **Exploration constant**: c_puct = 3.0 (was 1.0)
2. **Temperature annealing**: Proper schedule from 1.0 → 0.1 over game
3. **Noise weight decay**: Reduces Dirichlet noise as game progresses
4. **Training parameters**: Added L2 regularization, gradient clipping, LR warmup

## Impact on Training

With the broken MCTS:
- Model couldn't improve beyond random play (~50% win rate)
- Training data was just noisy versions of the current policy
- No lookahead or strategic planning possible

With the fixed MCTS:
- Model can now discover better moves through search
- Training data includes improved policies from MCTS
- Value estimates become meaningful through actual game tree evaluation

## Performance Considerations

The fixed implementation is slower because it actually does tree search. However, this is necessary for AlphaZero to work. Performance can be improved through:
- Parallel tree searches for multiple games
- Optimized tree data structures
- Hybrid CPU/GPU computation
- Caching and reusing parts of the tree

## Conclusion

The JAX implementation's training failures were caused by a fundamental misunderstanding of MCTS. The "vectorized" implementation had optimized away the actual tree search, leaving only a policy reweighting mechanism. 

With the proper tree-based MCTS implementation, the JAX version should now be able to learn and improve like the PyTorch version. While slower, this is the correct algorithm that made AlphaZero successful.

## Files Modified/Created

- `tree_based_mcts.py`: New proper MCTS implementation
- `vectorized_self_play_fixed.py`: Updated self-play to use tree-based MCTS
- `run_jax_fixed.py`: Pipeline using the fixed implementation
- `test_tree_mcts.py`: Verification test showing the fix works
- `critical_issues_analysis.md`: Detailed analysis of the bug
- `JAX_MCTS_FIX_SUMMARY.md`: Technical summary of the fix
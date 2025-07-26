# Critical Issues Found in JAX Implementation

## 1. MCTS Does Not Do Tree Search! ⚠️

### PyTorch MCTS (Correct Implementation)
- Builds an actual search tree with nodes
- Each simulation:
  1. **Select**: Traverse tree from root to leaf using PUCT
  2. **Expand**: Add new node when reaching unexplored position
  3. **Evaluate**: Use neural network to evaluate the new position
  4. **Backup**: Propagate value back up the tree
- Makes actual moves and explores different game states
- Evaluates many different positions with the neural network

### JAX MCTS (Incorrect Implementation)
- NO tree structure at all
- Only evaluates the root position ONCE
- Each "simulation" just:
  1. Updates visit counts based on PUCT scores
  2. Accumulates the same root value
- Never makes any moves or explores new positions
- Essentially just reweights the initial policy using PUCT formula

### Impact
This is why training fails! The MCTS is supposed to improve upon the raw neural network policy by searching, but the JAX version doesn't search at all. It's just using a slightly modified version of the raw policy.

## 2. Fundamental Architecture Difference

### PyTorch Flow
```python
# Actual tree search
for simulation in range(num_simulations):
    leaf = root.select_leaf()  # Traverse tree to leaf
    board_state = leaf.game  # Different position each time!
    policy, value = neural_network(board_state)  # Evaluate NEW position
    leaf.expand(policy)  # Add children
    leaf.backup(value)  # Propagate up tree
```

### JAX Flow  
```python
# Fake "search" - no tree!
policy, value = neural_network(root_position)  # Only evaluate ONCE
for simulation in range(num_simulations):
    action = argmax(PUCT_scores)  # Based on same policy
    visit_counts[action] += 1
    value_sums[action] += value  # Same value every time!
```

## 3. Why This Breaks Training

1. **No Exploration**: MCTS should find better moves than raw policy, but JAX version can't
2. **No Lookahead**: Real MCTS sees future positions, JAX only sees current
3. **Policy Can't Improve**: Training data is just slightly noisy version of current policy
4. **Values Are Meaningless**: Without search, value estimates don't improve

## 4. The Fix Required

The JAX MCTS needs a complete rewrite to:
1. Maintain a tree structure (nodes with children)
2. Actually make moves and track game states
3. Evaluate new positions with the neural network
4. Properly backup values through the tree

This is not a small fix - it requires implementing a proper MCTS algorithm, not just statistics tracking.

## 5. Other Issues Found

### Training Issues (Already Fixed)
- ✅ c_puct was 1.0 instead of 3.0
- ✅ Missing temperature annealing
- ✅ Missing L2 regularization
- ✅ Missing gradient clipping
- ✅ Missing learning rate warmup

### Data Format Issues
- ✅ Value assignment logic (fixed)
- ✅ Perspective mode handling (fixed)

But these fixes don't matter if MCTS doesn't work!

## Conclusion

The JAX implementation has a **fundamental algorithmic error**. It's not doing Monte Carlo Tree Search at all - it's just doing a weighted random selection based on the neural network's initial policy. This explains why the model can't learn effectively.

The implementation needs a proper MCTS that:
1. Builds and maintains a search tree
2. Explores different game states
3. Evaluates multiple positions per search
4. Aggregates information from actual lookahead

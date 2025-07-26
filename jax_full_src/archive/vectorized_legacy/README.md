# Legacy Vectorized MCTS Implementation

These files contain the original "vectorized" MCTS implementation that had a fundamental flaw: it didn't actually perform tree search.

## The Problem

The vectorized MCTS implementations (all files in this directory) attempted to parallelize the MCTS algorithm itself. However, this approach only:
1. Evaluated the root position once with the neural network
2. Added noise to the policy
3. Redistributed visit counts based on the policy
4. Never actually explored child nodes or built a tree

This made it impossible for the model to improve beyond random play (~50% win rate).

## Files

### Broken MCTS Implementations
- `vectorized_mcts.py` - Original broken implementation
- `vectorized_mcts_improved.py` - Added features but still broken
- `vectorized_mcts_jit.py` - JIT-compiled version of broken MCTS

### Self-Play Using Broken MCTS
- `vectorized_self_play.py` - Uses broken MCTS
- `vectorized_self_play_improved.py` - Uses broken MCTS with more features
- `vectorized_self_play_jit.py` - Uses JIT-compiled broken MCTS

## Current Solution

The fixed implementation uses:
- `tree_based_mcts.py` - Proper tree search that builds and explores nodes
- `vectorized_self_play_fixed.py` - Self-play using the fixed MCTS

The key insight: Don't vectorize the tree search itself. Instead, run many independent tree searches in parallel, one for each game.
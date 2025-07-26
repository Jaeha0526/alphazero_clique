# Batched Tree MCTS

`batched_tree_mcts.py` was an experimental optimization that attempted to batch neural network evaluations across multiple positions within the MCTS tree.

## Why it's not used:

1. **Added Complexity**: The batching logic made the code significantly more complex
2. **Limited Benefit**: In practice, the overhead of collecting positions to batch often outweighed the benefits
3. **Simpler Solution Works**: The current approach (`ParallelTreeBasedMCTS` in `tree_based_mcts.py`) achieves good performance by running multiple independent trees in parallel

## Current approach:

Instead of batching evaluations within a tree, we:
- Run N independent games in parallel
- Each game has its own MCTS tree
- Neural network already processes multiple games in a batch
- Simpler and more effective

The file is preserved here for reference but is not part of the active implementation.
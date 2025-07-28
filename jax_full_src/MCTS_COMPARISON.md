# MCTS Implementation Comparison

## 1. Simple "MCTS" (VectorizedJITMCTS)
- **What it does**: Just reweights the neural network policy using UCB formula
- **Tree depth**: 1 (only root node)
- **Speed**: ~1 second per move for 50 games
- **Accuracy**: Limited - doesn't explore future positions

## 2. Proper Tree MCTS (SimpleTreeMCTS)
- **What it does**: Full MCTS with tree building, expansion, and backup
- **Tree depth**: Variable, builds actual game trees
- **Speed**: Much slower - ~30+ seconds per move for 20 games
- **Accuracy**: Much better - explores future positions

## Key Findings

### Why Tree MCTS is Slower:
1. **Python overhead**: Managing tree structures in Python
2. **Memory allocation**: Creating nodes dynamically
3. **Board copying**: Each node needs its own board state
4. **Sequential nature**: Hard to fully vectorize tree traversal

### Performance Numbers:
- Simple MCTS: 500 games with 5 sims = ~6 minutes
- Tree MCTS: 20 games with 50 sims = >3 minutes (for just one move!)

### Recommendations:
1. For fast experimentation: Use simple MCTS with more simulations (300-500)
2. For best quality: Use tree MCTS with fewer games in parallel
3. Hybrid approach: Use tree MCTS for evaluation, simple for self-play

## Next Steps:
- Optimize tree MCTS with Cython or C++ backend
- Use JAX's tree utilities for better performance
- Implement virtual loss for parallel tree traversal
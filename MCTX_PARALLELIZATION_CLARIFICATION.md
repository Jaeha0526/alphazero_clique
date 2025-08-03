# MCTX Parallelization Model - Clarification

## Key Understanding: Parallelization is ACROSS Games, Not Within MCTS

### What Our Implementation Actually Does

Our MCTX implementations (both `MCTXFinalOptimized` and `MCTXTrueJAX`) parallelize across multiple games, not within individual MCTS trees.

```
Batch Processing Model:
┌─────────────┐ ┌─────────────┐     ┌─────────────┐
│   Game 1    │ │   Game 2    │ ... │  Game 150   │
│ Sequential  │ │ Sequential  │     │ Sequential  │
│    MCTS     │ │    MCTS     │     │    MCTS     │
└─────────────┘ └─────────────┘     └─────────────┘
       ↑               ↑                    ↑
       └───────────────┴────────────────────┘
              Processed in parallel
```

### Within Each Game: Standard Sequential MCTS

Each game runs traditional MCTS with sequential simulations:
1. Simulation 1 completes and updates visit counts
2. Simulation 2 sees updated counts from Simulation 1
3. And so on...

This means:
- **No virtual loss** within a game
- **No parallel exploration artifacts**
- **Standard UCB behavior** with accurate statistics
- **No need for extra simulations**

### Batch Dimension Breakdown

When you set `--game_batch_size 150` and `--mcts_sims 50`:
- 150 different games play simultaneously
- Each game gets 50 sequential MCTS simulations
- Total: 150 × 50 = 7,500 simulations per move across all games

### Array Shapes in MCTX

```python
# MCTS arrays have shape [batch_size, max_nodes, num_actions]
N = jnp.zeros((150, 51, 91))  # Visit counts
W = jnp.zeros((150, 51, 91))  # Total values

# Where:
# 150 = number of parallel games
# 51 = max nodes per game (50 simulations + 1 root)
# 91 = number of possible actions (for n=14)
```

### What "True MCTX" Means in Our Context

The `MCTXTrueJAX` implementation:
- Uses JAX primitives (`jax.lax.while_loop`, `jax.lax.scan`) instead of Python loops
- Achieves ~5x speedup through better GPU utilization
- **Still runs sequential MCTS within each game**
- Same algorithm, just faster execution

### Implications for Hyperparameters

Since we're using sequential MCTS within each game:
- `--mcts_sims 50` provides standard MCTS quality
- No need to increase simulations for "parallel exploration compensation"
- Each game gets full-quality MCTS search

### Comparison with True Parallel MCTS

Some advanced implementations (not ours) do parallelize within a single game:

```
True Parallel MCTS (NOT our implementation):
Single Game:
├─ Simulation 1 ──┐
├─ Simulation 2   ├── All start simultaneously
├─ ...            ├── Can't see each other's visits
└─ Simulation 50 ─┘   (requires virtual loss)
```

This would require:
- Virtual loss to prevent all simulations taking the same path
- More simulations for comparable quality
- Different implementation architecture

### Summary

Our MCTX implementation provides:
1. **Efficient batch processing** of multiple games
2. **Standard sequential MCTS** within each game
3. **No algorithmic changes** from traditional MCTS
4. **Optimal hyperparameters** remain the same as single-game MCTS

The speedup comes from processing many games simultaneously on GPU, not from changing the MCTS algorithm itself.
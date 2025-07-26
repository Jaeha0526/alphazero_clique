# MCTS Performance Analysis: Fixed vs Broken

## Timing Results

### Single MCTS Call (20 simulations, 4 games batch)
- **Broken JIT MCTS**: 0.092s per game
- **Fixed Tree MCTS**: 6.468s per game  
- **Slowdown**: 70x slower

### Full Game Self-Play
- **Broken MCTS**: 12.5s for 1 complete game (13 moves)
- **Fixed MCTS**: 27.0s for 1 complete game (7 moves)
- **Slowdown**: 2.2x slower

## Why Such a Big Difference?

### Broken JIT MCTS (Fast but Wrong)
```python
# What it actually does:
1. Call NN once on root position -> 0.01s
2. Calculate PUCT scores -> 0.001s  
3. Update visit counts -> 0.001s
Total: ~0.01s regardless of simulation count!
```

### Fixed Tree MCTS (Correct Algorithm)
```python
# What it does for 20 simulations:
For each simulation:
    1. Traverse tree (multiple nodes) -> 0.05s
    2. Expand node (create children) -> 0.1s
    3. Call NN on NEW position -> 0.2s
    4. Backup through tree path -> 0.05s
Total: ~0.4s × 20 = 8s
```

## The Fundamental Issue

MCTS is inherently sequential because:
1. Each simulation updates tree statistics
2. Next simulation depends on updated statistics
3. Can't parallelize within a single tree

## Performance Optimization Strategies

### 1. **Batch Parallel Games** (Most Effective)
Instead of:
```python
# Sequential: 16 games × 10s each = 160s
for game in games:
    run_mcts(game)
```

Do:
```python
# Parallel: All 16 games at once = ~15s
run_mcts_parallel(all_16_games)
```

### 2. **Reduce Simulations During Self-Play**
- Training games: 50-100 simulations
- Evaluation games: 200-500 simulations
- Tournament games: 800+ simulations

### 3. **Neural Network Batching**
Collect positions from multiple trees:
```python
# Instead of 100 individual NN calls
positions = collect_from_all_trees()
values = model(batch_positions)  # 1 batched call
```

### 4. **Virtual Loss** (Advanced)
Allow multiple threads to traverse tree simultaneously:
```python
# Thread 1: Traversing path A
# Thread 2: Can traverse path B at same time
# Both update tree when done
```

## Realistic Training Times

With proper optimization:
- **Self-play**: 16-32 games in parallel
- **Time per iteration**: 20-30 minutes for 100 games
- **Total training**: 5-10 hours for strong play

## The Trade-off

- **Broken MCTS**: Fast but can't improve beyond random play
- **Fixed MCTS**: 70x slower but actually learns to play well

The slowdown is the price of doing actual tree search, which is essential for AlphaZero to work!

## Bottom Line

Yes, it's much slower, but:
1. It's the correct algorithm that made AlphaZero successful
2. The broken version could never learn to play well
3. With parallel games and optimizations, it's still practical
4. This is why DeepMind used 5000 TPUs for AlphaZero!
# Game Data Saving Methods Comparison

## Overview
The JAX AlphaZero implementation offers two distinct modes for saving game data during training, controlled by the `--save_full_game_data` flag.

## Comparison Table

| Aspect | **With** `--save_full_game_data` | **Without** `--save_full_game_data` (Default) |
|--------|-----------------------------------|------------------------------------------------|
| **Frequency** | Every iteration | Every 5 iterations |
| **Data Volume** | ALL training examples | Sample (10% or 1000 examples, whichever is smaller) |
| **Disk Usage** | ~3-4 MB per iteration (n=9, k=4, 100 games) | ~400-500 KB per saved file |
| **Use Case** | Detailed analysis, debugging | Normal training, periodic monitoring |
| **Storage Growth** | Linear with iterations | 5x slower growth |
| **Data Completeness** | 100% of moves | ~10% representative sample |
| **Game Reconstruction** | All games fully reconstructable | Only sampled games reconstructable |

## Detailed Analysis

### With `--save_full_game_data`

**What's Saved:**
- Every single training example from every game
- Complete move-by-move data for all games
- Full game boundaries and reconstruction info
- Iteration metadata and statistics

**Example (100 games, n=9, k=4):**
```python
{
    'iteration': 0,
    'total_training_examples': 2783,  # ALL examples
    'num_games_played': 100,
    'training_data': [...],  # 2783 move examples
    'games_info': [...],     # 100 game boundaries
    'is_full_data': True,
    'num_examples_saved': 2783,
    'game_mode': 'avoid_clique',
    'vertices': 9,
    'k': 4,
    # ... additional metadata
}
```

**Storage Calculation:**
- Average ~28 moves per game (n=9, k=4)
- 100 games = ~2800 training examples
- Each example ~1.4 KB (board state + features + policy + visit_counts)
- **Total: ~3.9 MB per iteration**
- 100 iterations = **390 MB**

### Without `--save_full_game_data` (Default)

**What's Saved:**
- First 10% of training examples (max 1000)
- Subset of games (typically 3-10 complete games)
- Same structure but reduced volume
- Only saved every 5 iterations

**Example (100 games, n=9, k=4):**
```python
{
    'iteration': 5,
    'total_training_examples': 2783,  # Total generated
    'num_games_played': 100,
    'training_data': [...],  # Only 278 examples (10%)
    'games_info': [...],     # ~10 game boundaries
    'is_full_data': False,
    'num_examples_saved': 278,
    # ... same metadata
}
```

**Storage Calculation:**
- 10% of 2800 examples = 280 examples
- Each example ~1.4 KB
- **Total: ~400 KB per saved file**
- 100 iterations = 20 files = **8 MB**

## Practical Examples

### Training Command Comparison

**Full Data Collection (Research/Debugging):**
```bash
python jax_full_src/run_jax_optimized.py \
    --experiment_name detailed_analysis \
    --vertices 9 --k 4 \
    --num_iterations 50 \
    --num_episodes 100 \
    --save_full_game_data  # ← Saves everything
```
- Storage: ~195 MB for 50 iterations
- Use when: Analyzing learning progression, debugging, research

**Standard Training (Production):**
```bash
python jax_full_src/run_jax_optimized.py \
    --experiment_name production_run \
    --vertices 9 --k 4 \
    --num_iterations 200 \
    --num_episodes 100
    # No --save_full_game_data flag
```
- Storage: ~16 MB for 200 iterations (40 files)
- Use when: Normal training, long runs, limited disk space

## What You Can Analyze

### With Full Data
- **Complete learning curves**: Track every move's policy evolution
- **Per-game analysis**: Reconstruct and visualize any game
- **Statistical significance**: Large sample for confidence intervals
- **Debugging**: Find exact moves where errors occur
- **Visit count analysis**: Full MCTS exploration patterns

### With Sampled Data
- **Trend monitoring**: See general improvement patterns
- **Sample games**: Analyze representative games
- **Key checkpoints**: Every 5 iterations gives good overview
- **Efficiency**: Minimal storage overhead
- **Quick insights**: Sufficient for most training monitoring

## Recommendations

### Use Full Data When:
1. **Research**: Publishing results requiring complete data
2. **Debugging**: Tracking down specific training issues
3. **Short runs**: <50 iterations with plenty of disk space
4. **Novel experiments**: Testing new architectures or game modes
5. **Ramsey search**: Finding and analyzing counterexamples

### Use Default (Sampled) When:
1. **Production training**: Long runs (100+ iterations)
2. **Hyperparameter tuning**: Multiple parallel experiments
3. **Limited storage**: Cloud instances or shared systems
4. **Standard monitoring**: Regular training progress checks
5. **Established configurations**: Known working parameters

## Implementation Details

### Code Location
File: `jax_full_src/run_jax_optimized.py`, lines 662-708

### Sampling Strategy (Default Mode)
```python
# Line 676-677
sample_size = min(1000, len(game_data) // 10)
data_to_save = game_data[:sample_size]
```
- Takes first 10% of examples (not random)
- Capped at 1000 to prevent huge files
- Preserves game order (early games in iteration)

### Game Boundary Preservation
Both modes save `games_info` which tracks:
- Game boundaries (start_idx, end_idx)
- Winners and game outcomes
- Number of moves per game
- Allows proper game reconstruction from saved moves

## Disk Space Planning

### Typical Scenarios

**Small Graph (n=6, k=3):**
- ~15 moves/game average
- Full: ~2 MB/iteration
- Sampled: ~200 KB/file

**Medium Graph (n=9, k=4):**
- ~28 moves/game average
- Full: ~4 MB/iteration
- Sampled: ~400 KB/file

**Large Graph (n=13, k=5):**
- ~50 moves/game average
- Full: ~7 MB/iteration
- Sampled: ~700 KB/file

**Ramsey Search (n=17, k=5):**
- ~100+ moves/game average
- Full: ~14 MB/iteration
- Sampled: ~1.4 MB/file

## Analysis Tools

Both data formats work with the same analysis tools:

```bash
# Visualize games in browser
cd game_data_analyze
python app.py
# Visit http://localhost:8080

# Analyze from command line
python jax_full_src/analyze_game_data.py \
    experiments/your_exp/game_data/iteration_10.pkl

# Compare across iterations
python jax_full_src/analyze_game_data.py \
    experiments/your_exp/game_data/ --compare
```

## Conclusion

The choice between full and sampled data saving is a trade-off between:
- **Completeness vs Efficiency**
- **Storage cost vs Analysis depth**
- **Every iteration vs Key checkpoints**

For most training runs, the default sampled mode provides sufficient visibility while keeping storage manageable. Reserve full data collection for specific research needs or debugging scenarios.
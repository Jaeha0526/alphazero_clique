# How to Upgrade JAX Pipeline to Use Optimized MCTX

## Quick Fix

In `run_jax_optimized.py`, make these changes:

### 1. Import the optimized MCTX

Replace:
```python
from simple_tree_mcts import SimpleTreeMCTS
from simple_tree_mcts_timed import SimpleTreeMCTSTimed
```

With:
```python
from simple_tree_mcts import SimpleTreeMCTS
from mctx_final_optimized import MCTXFinalOptimized
```

### 2. Update OptimizedSelfPlay.play_games()

Replace lines 50-57:
```python
# Old code
mcts = SimpleTreeMCTSTimed(
    batch_size=batch_size,
    num_actions=num_actions,
    c_puct=self.config.c_puct,
    max_nodes_per_game=200
)
```

With:
```python
# New code - choose based on game size
if self.config.num_vertices >= 9:
    # Use optimized MCTX for large games
    print(f"  Creating Optimized MCTX: {num_actions} actions, batch_size={batch_size}")
    mcts = MCTXFinalOptimized(batch_size=batch_size)
else:
    # Use SimpleTreeMCTS for small games
    print(f"  Creating Simple Tree MCTS: {num_actions} actions, batch_size={batch_size}")
    mcts = SimpleTreeMCTSTimed(
        batch_size=batch_size,
        num_actions=num_actions,
        c_puct=self.config.c_puct,
        max_nodes_per_game=200
    )
```

### 3. Update Configuration

In the `Config` dataclass (line 301), add:
```python
@dataclass
class Config:
    batch_size: int = args.batch_size
    num_vertices: int = 6  
    k: int = 3
    game_mode: str = "asymmetric" if args.asymmetric else "symmetric"
    mcts_simulations: int = 20
    temperature_threshold: int = 10
    c_puct: float = 3.0
    perspective_mode: str = "alternating"
    use_optimized_mctx: bool = True  # Add this flag
```

## Full Example

Here's a complete updated play_games method:

```python
def play_games(self, neural_network, num_games):
    """Play games using optimized MCTS."""
    all_game_data = []
    games_played = 0
    
    while games_played < num_games:
        batch_size = min(self.config.batch_size, num_games - games_played)
        print(f"\nStarting batch of {batch_size} games (Total: {games_played}/{num_games})")
        
        # Create MCTS with correct batch size for this iteration
        num_actions = self.config.num_vertices * (self.config.num_vertices - 1) // 2
        
        # Choose MCTS implementation based on game size
        if self.config.num_vertices >= 9 and self.config.use_optimized_mctx:
            print(f"  Creating Optimized MCTX for large game: {num_actions} actions")
            mcts = MCTXFinalOptimized(batch_size=batch_size)
        else:
            print(f"  Creating Simple Tree MCTS: {num_actions} actions")
            mcts = SimpleTreeMCTSTimed(
                batch_size=batch_size,
                num_actions=num_actions,
                c_puct=self.config.c_puct,
                max_nodes_per_game=200
            )
        
        # ... rest of the method remains the same
```

## Performance Impact

Based on our benchmarks:

### For n=6, k=3:
- SimpleTreeMCTS: ~30ms per game (batch=8)
- Keep using SimpleTreeMCTS

### For n=9, k=4:
- SimpleTreeMCTS: ~660ms per game (batch=8)
- MCTXFinalOptimized: ~119ms per game (batch=8)
- **5.6x speedup!**

## Testing the Upgrade

Run with different game sizes:
```bash
# Small game - uses SimpleTreeMCTS
python run_jax_optimized.py --vertices 6 --k 3

# Large game - uses MCTXFinalOptimized  
python run_jax_optimized.py --vertices 9 --k 4
```

## Additional Optimizations

1. **Increase batch size for MCTX**: It scales better with larger batches
   ```bash
   python run_jax_optimized.py --vertices 9 --k 4 --batch_size 100
   ```

2. **Adjust MCTS simulations**: MCTX is more efficient, so you can afford more
   ```bash
   python run_jax_optimized.py --vertices 9 --k 4 --mcts_sims 100
   ```

3. **Monitor performance**: The pipeline already prints timing information
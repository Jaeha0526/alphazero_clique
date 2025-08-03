# Running Commands for Pure JAX AlphaZero

## Main Pipeline Entry Point

```bash
# Standard run with command-line interface (identical to PyTorch)
python jax_full_src/run_jax_optimized.py \
    --experiment-name my_experiment \
    --iterations 10 \
    --self-play-games 100 \
    --mcts-sims 50
```

This command:
- Uses the same CLI interface as PyTorch pipeline
- Generates games using vectorized self-play
- Trains with JAX/Flax/Optax
- Evaluates against initial and previous models
- Creates plots matching PyTorch style
- Saves to experiments/my_experiment/

## Command-Line Options

```bash
# Full example with all common options
python jax_full_src/run_jax_optimized.py \
    --experiment-name advanced_run \
    --vertices 7 \
    --k 4 \
    --game-mode asymmetric \
    --iterations 20 \
    --self-play-games 500 \
    --batch-size 256 \
    --mcts-sims 100 \
    --epochs 10 \
    --hidden-dim 128 \
    --num-layers 3 \
    --learning-rate 0.001 \
    --perspective-mode alternating \
    --skill-variation 0.3 \
    --eval-games 50

# Minimal run for testing
python jax_full_src/run_jax_optimized.py \
    --experiment-name test_run \
    --iterations 2 \
    --self-play-games 20 \
    --mcts-sims 10 \
    --epochs 2
```

## Performance Testing

```bash
# Quick test to verify JAX GPU acceleration
python -c "
import jax
print(f'JAX devices: {jax.devices()}')
print(f'Default backend: {jax.default_backend()}')
"

# Test JIT compilation speedup
python -c "
import sys
sys.path.insert(0, '.')
from jax_full_src.vectorized_mcts_jit import test_mcts_speed
test_mcts_speed()
"
```

## Check Results

```bash
# View experiment results
ls -la experiments/

# Check training progress (replace my_experiment with your experiment name)
cat experiments/my_experiment/training_log.json | python -m json.tool

# View learning curves
# On Linux: xdg-open experiments/my_experiment/training_losses.png
# On Mac: open experiments/my_experiment/training_losses.png
# On Windows: start experiments/my_experiment/training_losses.png
```

## GPU Memory Optimization

```bash
# Set memory allocation options
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Run with smaller batch size for limited memory
python jax_full_src/run_jax_optimized.py \
    --experiment-name memory_efficient \
    --batch-size 16 \
    --mcts-sims 25
```

## Debugging

```bash
# Enable JAX debugging
JAX_DEBUG_NANS=True python jax_full_src/run_jax_optimized.py --experiment-name debug_run

# Check JAX device placement
python -c "import jax; print(jax.devices())"

# Force specific device
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORM_NAME=gpu python jax_full_src/run_jax_optimized.py

# Profile performance
python -m cProfile -o profile.stats jax_full_src/run_jax_optimized.py --iterations 1
```

## Expected Output

Successful run shows:
```
Starting JAX AlphaZero Pipeline
Experiment: my_experiment
Device: cuda:0
==================================================

Iteration 0/10
==================================================
Generating 100 self-play games...
Using JIT-compiled MCTS
Generated 100 games in 13.4 seconds (7.5 games/sec)
Average game length: 8.2 moves

Training network...
Epoch 1/10, Policy Loss: 2.1234, Value Loss: 0.6789
...
Training completed in 3.2 seconds

Evaluating model...
Win rate vs initial: 65.0%
Win rate vs previous: 65.0%
```

## Performance Expectations

| Component | PyTorch | JAX | JAX+JIT |
|-----------|---------|-----|---------|
| Self-Play | ~0.25 games/s | ~3 games/s | ~7-25 games/s |
| MCTS | Sequential | Vectorized | 30x faster |
| Training | Standard | JAX/Optax | ~10x faster |
| Memory | Moderate | Efficient | Scales with batch |

## Monitoring Training

### Real-time Progress
```bash
# Watch training metrics (replace my_experiment with your name)
tail -f experiments/my_experiment/training_log.json

# Parse JSON output
cat experiments/my_experiment/training_log.json | python -m json.tool | tail -20
```

### Check Training History
```python
import json
import matplotlib.pyplot as plt

# Load and analyze results
with open('experiments/my_experiment/training_log.json') as f:
    log = json.load(f)
    
print(f"Total iterations: {log['total_iterations']}")
print(f"Final win rate vs initial: {log['final_win_rate_vs_initial']:.1%}")
print(f"Average self-play speed: {log['avg_games_per_second']:.1f} games/sec")

# View iteration details
for i, iteration in enumerate(log['iterations']):
    print(f"\nIteration {i}:")
    print(f"  Policy loss: {iteration['avg_policy_loss']:.4f}")
    print(f"  Value loss: {iteration['avg_value_loss']:.4f}")
    print(f"  Win rate vs initial: {iteration['win_rate_vs_initial']:.1%}")
```

## Troubleshooting

### JAX Not Using GPU
```bash
# Check CUDA availability
python -c "import jax; print(jax.devices())"
# Should show: [cuda:0] or [gpu:0]

# Force GPU usage
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORM_NAME=gpu
python jax_full_src/run_jax_optimized.py --experiment-name gpu_test
```

### Out of Memory Errors
```bash
# Clear GPU memory and limit allocation
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

# Run with smaller batch
python jax_full_src/run_jax_optimized.py --batch-size 16 --mcts-sims 25
```

### Slow Initial Compilation
The first iteration may be slower due to JAX JIT compilation. Subsequent iterations will be much faster. This is normal and expected.

## Integration with PyTorch Pipeline

Both implementations save to the same directory structure:
```
experiments/
├── pytorch_experiment/
│   ├── models/
│   ├── datasets/
│   └── training_log.json
└── jax_experiment/
    ├── models/
    ├── datasets/
    └── training_log.json
```

This allows easy comparison between PyTorch and JAX implementations.
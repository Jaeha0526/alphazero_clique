# Working Commands for AlphaZero Clique

## Quick Start

### 1. Setup Environment
```bash
# Run automated setup (detects CUDA and installs appropriate JAX)
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. PyTorch Pipeline (Original)
```bash
# Basic training run
python src/pipeline_clique.py \
    --experiment-name pytorch_demo \
    --iterations 10 \
    --self-play-games 100 \
    --mcts-sims 50

# Advanced configuration for n=7, k=4
python src/pipeline_clique.py \
    --experiment-name n7k4_advanced \
    --vertices 7 \
    --k 4 \
    --iterations 20 \
    --self-play-games 200 \
    --mcts-sims 100 \
    --skill-variation 0.3 \
    --perspective-mode alternating
```

### 3. JAX Pipeline (GPU Accelerated)
```bash
# Basic JAX run (30x faster with JIT)
python jax_full_src/run_jax_improved.py \
    --experiment-name jax_demo \
    --iterations 10 \
    --self-play-games 100 \
    --mcts-sims 50

# JAX with GPU memory optimization
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

python jax_full_src/run_jax_improved.py \
    --experiment-name jax_gpu_demo \
    --iterations 20 \
    --self-play-games 500 \
    --mcts-sims 100 \
    --batch-size 256
```

## Interactive Game Interface

```bash
# Play against trained models
python src/interactive_clique_game.py

# Copy trained models to playable directory
cp experiments/*/models/clique_net.pth.tar playable_models/
```

## Performance Comparison

### Running Experiments
```bash
# Compare PyTorch vs JAX performance
# Terminal 1: PyTorch
python src/pipeline_clique.py --experiment-name comparison_pytorch --iterations 5

# Terminal 2: JAX  
python jax_full_src/run_jax_improved.py --experiment-name comparison_jax --iterations 5
```

### Key Performance Metrics
| Implementation | Self-Play Speed | Training Speed | Memory Usage |
|----------------|----------------|----------------|--------------|
| PyTorch (CPU)  | ~0.25 games/s  | Baseline       | Moderate     |
| PyTorch (GPU)  | ~0.25 games/s  | GPU accelerated| High         |
| JAX (GPU)      | ~7.5 games/s   | GPU accelerated| Efficient    |
| JAX (JIT)      | Up to 30x faster| GPU accelerated| Efficient   |

## Troubleshooting

### JAX GPU Issues
```bash
# Check JAX can see GPU
python -c "import jax; print(jax.devices())"

# Force GPU usage
export JAX_PLATFORM_NAME=gpu

# Limit memory pre-allocation
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### PyTorch Geometric Issues
```bash
# Reinstall with specific CUDA version
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Experiment Organization

All experiments save to `experiments/experiment_name/` with:
- `models/` - Trained model checkpoints
- `datasets/` - Self-play game data
- `training_log.json` - Metrics and progress
- `plots/` - Training curves and analysis

Both PyTorch and JAX implementations use identical directory structure for easy comparison.
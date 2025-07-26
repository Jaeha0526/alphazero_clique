# AlphaZero Clique Setup Guide

## Quick Setup

Run the automated setup script:
```bash
./setup.sh
```

This will:
1. Create a Python virtual environment (if not exists)
2. Install PyTorch and PyTorch Geometric
3. Install JAX with appropriate CUDA support
4. Create necessary directories
5. Verify installations

## Manual Setup

If you prefer manual setup or the script fails:

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install PyTorch Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install JAX Dependencies

For GPU (CUDA 12):
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For GPU (CUDA 11):
```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For CPU only:
```bash
pip install --upgrade jax jaxlib
```

Then install JAX ecosystem:
```bash
pip install --upgrade flax optax
pip install -r jax_full_src/requirements_jax.txt
```

### 4. Create Directories
```bash
mkdir -p experiments logs playable_models
```

## Verify Installation

Run the test script:
```bash
python test_setup.py
```

## Troubleshooting

### CUDA/GPU Issues

If JAX doesn't detect your GPU:
```bash
# Check CUDA is available
nvidia-smi

# Set memory allocation
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Force GPU
export JAX_PLATFORM_NAME=gpu
```

### PyTorch Geometric Issues

If PyTorch Geometric fails to install:
```bash
# Install with specific CUDA version
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Memory Issues

For large batch sizes:
```bash
# Limit GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

## Running the Pipelines

After setup, you can run:

**PyTorch Pipeline:**
```bash
python src/pipeline_clique.py \
    --experiment-name my_pytorch_exp \
    --iterations 10 \
    --self-play-games 100 \
    --mcts-sims 50
```

**JAX Pipeline (faster):**
```bash
python jax_full_src/run_jax_improved.py \
    --experiment-name my_jax_exp \
    --iterations 10 \
    --self-play-games 100 \
    --mcts-sims 50
```

Both will save results to `experiments/your_experiment_name/`
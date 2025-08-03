# JAX Implementation Files Summary

## Main Entry Point
- **`run_jax_optimized.py`** - The current, optimized pipeline with all features

## Core Components
- **`vectorized_board.py`** - Vectorized game logic for parallel processing
- **`vectorized_nn.py`** - Neural network with symmetric/asymmetric support
- **`mctx_final_optimized.py`** - Memory-efficient MCTS (Python loops)
- **`mctx_true_jax.py`** - JIT-compiled MCTS (pure JAX, fastest)

## Training Modules
- **`train_jax.py`** - Basic JAX training functions
- **`train_jax_fully_optimized.py`** - JIT-compiled training (5x faster)
- **`train_jax_with_validation.py`** - Training with validation split & early stopping

## Evaluation Modules
- **`evaluation_jax_fixed.py`** - Sequential symmetric evaluation
- **`evaluation_jax_asymmetric.py`** - Sequential asymmetric evaluation  
- **`evaluation_jax_parallel.py`** - Parallel symmetric evaluation (faster)
- **`evaluation_jax_asymmetric_parallel.py`** - Parallel asymmetric evaluation (faster)

## Utilities
- **`plot_asymmetric_metrics.py`** - Plotting functions for asymmetric games
- **`requirements_jax.txt`** - JAX dependencies

## Usage
```bash
# Run the optimized pipeline
python run_jax_optimized.py \
    --num_iterations 10 \
    --num_episodes 100 \
    --vertices 6 \
    --k 3 \
    --use_validation \        # Use 90/10 split with early stopping
    --parallel_evaluation \   # Use fast parallel evaluation
    --use_true_mctx          # Use JIT-compiled MCTS
```

## Deleted Files (were buggy or obsolete)
- ~~`run_jax_improved.py`~~ - Old pipeline using buggy evaluation
- ~~`evaluation_jax.py`~~ - Buggy evaluation with wrong MCTS interface
- ~~`train_jax_optimized.py`~~ - Superseded by train_jax_fully_optimized.py
- ~~`vectorized_nn_fixed_asymmetric.py`~~ - Merged into vectorized_nn.py
- ~~`vectorized_self_play_fixed.py`~~ - Integrated into run_jax_optimized.py

All remaining files are actively used by the current pipeline.
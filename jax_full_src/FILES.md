# JAX AlphaZero File Structure

## Main Files

### Pipeline Entry Point
- `run_jax_optimized.py` - Main pipeline script with command-line interface

### Core Implementation
- `vectorized_board.py` - Vectorized Clique game implementation
- `vectorized_nn.py` - Graph Neural Network (GNN) using Flax
- `vectorized_mcts_improved.py` - MCTS with perspective modes and skill variation
- `vectorized_mcts_jit.py` - JIT-compiled MCTS for performance
- `vectorized_self_play_improved.py` - Self-play with advanced features
- `vectorized_self_play_jit.py` - JIT-compiled self-play
- `train_jax.py` - Training loop with JAX/Optax
- `evaluation_jax.py` - Model evaluation utilities

### Supporting Files
- `vectorized_mcts.py` - Basic MCTS implementation (legacy)
- `vectorized_self_play.py` - Basic self-play (legacy)

### Documentation
- `README.md` - Main documentation
- `IMPROVED_JAX_SUMMARY.md` - Summary of improvements
- `JAX_IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `RUN_COMMANDS.md` - Example commands

### Configuration
- `requirements_jax.txt` - Python dependencies

### Archive
- `archive/` - Previous iterations and experimental code

## Usage

Run the pipeline with:
```bash
python run_jax_optimized.py --experiment-name my_experiment --num_iterations 10
```

All outputs will be saved to `../experiments/my_experiment/` matching PyTorch structure.
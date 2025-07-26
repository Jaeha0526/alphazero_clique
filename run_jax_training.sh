#!/bin/bash
# Script to run JAX training with proper settings

echo "Starting JAX Vectorized AlphaZero Training"
echo "=========================================="

# Set environment variables
export DISABLE_WANDB=true
export JAX_PLATFORM_NAME=gpu
export CUDA_VISIBLE_DEVICES=0

# Navigate to the project directory
cd /workspace/alphazero_clique

# Run the training with standard settings for 3 iterations
echo "Configuration:"
echo "- Iterations: 3"
echo "- Games per iteration: 50"
echo "- Batch size: 16 (parallel games)"
echo "- MCTS simulations: 25"
echo ""

python jax_full_src/run_training_simple.py

echo ""
echo "Training completed!"
echo "Check results in: experiments/standard_training_3iter/"
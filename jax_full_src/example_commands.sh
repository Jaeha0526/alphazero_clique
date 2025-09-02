#!/bin/bash

# Example Commands for JAX AlphaZero Training with Custom Game Numbers
# =====================================================================

# 1. AVOID_CLIQUE MODE - Searching for Ramsey Counterexamples
# -------------------------------------------------------------
# This mode searches for complete 2-colorings without monochromatic k-cliques
# Draws are saved as potential Ramsey counterexamples

# Small graph (n=13, k=4) with custom evaluation settings
python run_jax_optimized.py \
    --experiment_name ramsey_n13_k4_custom \
    --vertices 13 --k 4 \
    --game_mode avoid_clique \
    --num_iterations 50 \
    --num_episodes 100 \
    --eval_games 50 \
    --eval_mcts_sims 20 \
    --mcts_sims 30 \
    --num_epochs 10 \
    --game_batch_size 32 \
    --training_batch_size 512 \
    --use_true_mctx \
    --parallel_evaluation

# Larger graph (n=17, k=5) - more challenging
python run_jax_optimized.py \
    --experiment_name ramsey_n17_k5 \
    --vertices 17 --k 5 \
    --game_mode avoid_clique \
    --num_iterations 100 \
    --num_episodes 200 \
    --eval_games 100 \
    --eval_mcts_sims 50 \
    --mcts_sims 100 \
    --num_epochs 20 \
    --game_batch_size 64 \
    --training_batch_size 1024 \
    --use_true_mctx \
    --parallel_evaluation


# 2. SYMMETRIC MODE - Standard Clique Game
# -----------------------------------------
# Both players try to form k-cliques

# Quick test with minimal settings
python run_jax_optimized.py \
    --experiment_name test_symmetric \
    --vertices 6 --k 3 \
    --num_iterations 5 \
    --num_episodes 50 \
    --eval_games 21 \
    --eval_mcts_sims 10 \
    --mcts_sims 20 \
    --num_epochs 5 \
    --game_batch_size 16 \
    --training_batch_size 64 \
    --use_true_mctx

# Production training with more games
python run_jax_optimized.py \
    --experiment_name clique_n9_k4_prod \
    --vertices 9 --k 4 \
    --num_iterations 100 \
    --num_episodes 500 \
    --eval_games 42 \
    --eval_mcts_sims 50 \
    --mcts_sims 200 \
    --num_epochs 50 \
    --game_batch_size 128 \
    --training_batch_size 2048 \
    --use_true_mctx \
    --parallel_evaluation


# 3. ASYMMETRIC MODE - Attacker vs Defender
# ------------------------------------------
# One player forms cliques, other prevents

python run_jax_optimized.py \
    --experiment_name asymmetric_n8_k4 \
    --vertices 8 --k 4 \
    --asymmetric \
    --num_iterations 50 \
    --num_episodes 200 \
    --eval_games 40 \
    --eval_mcts_sims 30 \
    --mcts_sims 100 \
    --num_epochs 20 \
    --game_batch_size 64 \
    --training_batch_size 256 \
    --use_true_mctx \
    --parallel_evaluation


# 4. PARAMETER EXPLANATIONS
# --------------------------
# --num_episodes: Number of self-play games per iteration (for training data)
# --eval_games: Number of evaluation games (default: 21 for symmetric, 40 for asymmetric)
# --eval_mcts_sims: MCTS simulations for evaluation (default: 30)
# --mcts_sims: MCTS simulations for self-play (typically higher than eval)
# --num_epochs: Number of training epochs per iteration
# --game_batch_size: Number of games to play in parallel during self-play (default: 32)
# --training_batch_size: Batch size for neural network training (default: 32)
# --use_true_mctx: Use pure JAX MCTS implementation (5x faster)
# --parallel_evaluation: Run all evaluation games in single batch
# --game_mode: {symmetric, asymmetric, avoid_clique}


# 5. DIFFERENT SELF-PLAY vs EVALUATION CONFIGURATIONS
# ----------------------------------------------------

# High-quality self-play, quick evaluation
python run_jax_optimized.py \
    --experiment_name high_selfplay_quick_eval \
    --vertices 7 --k 4 \
    --num_iterations 20 \
    --num_episodes 1000 \
    --eval_games 21 \
    --eval_mcts_sims 10 \
    --mcts_sims 500 \
    --num_epochs 50 \
    --game_batch_size 256 \
    --training_batch_size 4096 \
    --use_true_mctx \
    --parallel_evaluation

# Quick self-play, thorough evaluation
python run_jax_optimized.py \
    --experiment_name quick_selfplay_thorough_eval \
    --vertices 7 --k 4 \
    --num_iterations 50 \
    --num_episodes 100 \
    --eval_games 100 \
    --eval_mcts_sims 100 \
    --mcts_sims 50 \
    --num_epochs 10 \
    --game_batch_size 32 \
    --training_batch_size 128 \
    --use_true_mctx \
    --parallel_evaluation

# Balanced approach
python run_jax_optimized.py \
    --experiment_name balanced_training \
    --vertices 8 --k 4 \
    --avoid_clique \
    --num_iterations 30 \
    --num_episodes 300 \
    --eval_games 42 \
    --eval_mcts_sims 30 \
    --mcts_sims 100 \
    --num_epochs 20 \
    --game_batch_size 64 \
    --training_batch_size 512 \
    --use_true_mctx \
    --parallel_evaluation


# 6. MONITORING TRAINING
# ----------------------
# Check training progress:
tail -f experiments/YOUR_EXPERIMENT_NAME/training_log.json

# View training plots:
ls experiments/YOUR_EXPERIMENT_NAME/*.png

# Check for Ramsey counterexamples (avoid_clique mode):
ls experiments/YOUR_EXPERIMENT_NAME/ramsey_counterexamples/

# Monitor GPU usage (if using GPU):
watch -n 1 nvidia-smi
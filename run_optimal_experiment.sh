#!/bin/bash

# Run AlphaZero experiment with win-rate optimized hyperparameters
# Changed from n=7 to n=16 vertices
# Using optimal hyperparameters from win rate analysis

echo "🚀 Running AlphaZero experiment with win-rate optimized hyperparameters"
echo "📊 Configuration: n=16, k=4, hidden-dim=32, num-layers=8"
echo "🎯 Optimized for: evaluation_win_rate_vs_initial"
echo ""

# Win-rate optimized hyperparameters
BATCH_SIZE=64
INITIAL_LR=0.0003
EPOCHS=16
SELF_PLAY_GAMES=78  # User requested 78 instead of 77
MCTS_SIMS=400
SKILL_VARIATION=0.67
LR_FACTOR=0.5
LR_PATIENCE=5

# Fixed parameters (same as sweep_config.yaml but with n=16)
MODE="pipeline"
VERTICES=16  # Changed from 7 to 16
K=4
HIDDEN_DIM=32
NUM_LAYERS=8
PERSPECTIVE_MODE="alternating"
GAME_MODE="symmetric"
ITERATIONS=60
NUM_CPUS=6
EVAL_THRESHOLD=0.45
NUM_GAMES=51
LR_THRESHOLD=0.001
MIN_LR=0.0000001
VALUE_WEIGHT=1.0
MIN_ALPHA=0.5
MAX_ALPHA=100.0

# Early stopping parameters
EARLY_STOP_PATIENCE=5  # Stop if no improvement for 5 iterations
MIN_ITERATIONS=10      # Minimum iterations before early stopping can trigger

# Create experiment name
EXPERIMENT_NAME="n${VERTICES}k${K}_${HIDDEN_DIM}_${NUM_LAYERS}_winrate_optimal"

echo "🔧 Hyperparameters:"
echo "  batch-size: ${BATCH_SIZE}"
echo "  initial-lr: ${INITIAL_LR}"
echo "  epochs: ${EPOCHS}"
echo "  self-play-games: ${SELF_PLAY_GAMES}"
echo "  mcts-sims: ${MCTS_SIMS}"
echo "  skill-variation: ${SKILL_VARIATION}"
echo "  lr-factor: ${LR_FACTOR}"
echo "  lr-patience: ${LR_PATIENCE}"
echo ""
echo "🛑 Early Stopping:"
echo "  early-stop-patience: ${EARLY_STOP_PATIENCE}"
echo "  min-iterations: ${MIN_ITERATIONS}"
echo ""
echo "📝 Experiment name: ${EXPERIMENT_NAME}"
echo ""

# Run the experiment
python src/pipeline_clique.py \
    --mode="${MODE}" \
    --vertices="${VERTICES}" \
    --k="${K}" \
    --hidden-dim="${HIDDEN_DIM}" \
    --num-layers="${NUM_LAYERS}" \
    --perspective-mode="${PERSPECTIVE_MODE}" \
    --game-mode="${GAME_MODE}" \
    --iterations="${ITERATIONS}" \
    --num-cpus="${NUM_CPUS}" \
    --eval-threshold="${EVAL_THRESHOLD}" \
    --num-games="${NUM_GAMES}" \
    --lr-threshold="${LR_THRESHOLD}" \
    --min-lr="${MIN_LR}" \
    --value-weight="${VALUE_WEIGHT}" \
    --min-alpha="${MIN_ALPHA}" \
    --max-alpha="${MAX_ALPHA}" \
    --batch-size="${BATCH_SIZE}" \
    --initial-lr="${INITIAL_LR}" \
    --epochs="${EPOCHS}" \
    --self-play-games="${SELF_PLAY_GAMES}" \
    --mcts-sims="${MCTS_SIMS}" \
    --skill-variation="${SKILL_VARIATION}" \
    --lr-factor="${LR_FACTOR}" \
    --lr-patience="${LR_PATIENCE}" \
    --early-stop-patience="${EARLY_STOP_PATIENCE}" \
    --min-iterations="${MIN_ITERATIONS}" \
    --experiment-name="${EXPERIMENT_NAME}"

echo ""
echo "✅ Experiment completed!"
echo "📊 Results should be logged to wandb under experiment: ${EXPERIMENT_NAME}"
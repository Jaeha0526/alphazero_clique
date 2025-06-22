#!/bin/bash

# Run AlphaZero experiments with win-rate optimized hyperparameters
# Automatically run from n=16 down to n=7 vertices
# Using optimal hyperparameters from win rate analysis

echo "🚀 Running AlphaZero experiments with win-rate optimized hyperparameters"
echo "📊 Configuration: n=16→7, k=4, hidden-dim=32, num-layers=8"
echo "🎯 Optimized for: evaluation_win_rate_vs_initial"
echo "🛑 Early stopping: patience=5, min_iterations=10"
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

# Fixed parameters
MODE="pipeline"
K=4
HIDDEN_DIM=32
NUM_LAYERS=8
PERSPECTIVE_MODE="alternating"
GAME_MODE="symmetric"
ITERATIONS=60
NUM_CPUS=6
EVAL_THRESHOLD=0.45
NUM_GAMES=61
LR_THRESHOLD=0.001
MIN_LR=0.0000001
VALUE_WEIGHT=1.0
MIN_ALPHA=0.5
MAX_ALPHA=100.0

# Early stopping parameters
EARLY_STOP_PATIENCE=5  # Stop if no improvement for 5 iterations
MIN_ITERATIONS=15      # Minimum iterations before early stopping can trigger

echo "🔧 Hyperparameters (fixed for all experiments):"
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

# Track successful and failed experiments
SUCCESSFUL_EXPERIMENTS=()
FAILED_EXPERIMENTS=()
TOTAL_EXPERIMENTS=0

# Function to run experiment for given n
run_experiment() {
    local VERTICES=$1
    local EXPERIMENT_NAME="n${VERTICES}k${K}_${HIDDEN_DIM}_${NUM_LAYERS}_winrate_optimal"
    
    echo "="*80
    echo "🎯 Starting experiment: ${EXPERIMENT_NAME}"
    echo "📊 Vertices: ${VERTICES}, K: ${K}"
    echo "⏰ Started at: $(date)"
    echo "="*80
    
    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
    
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
    
    # Check if experiment succeeded
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Experiment ${EXPERIMENT_NAME} completed successfully!"
        SUCCESSFUL_EXPERIMENTS+=("${EXPERIMENT_NAME}")
    else
        echo "❌ Experiment ${EXPERIMENT_NAME} failed with exit code ${EXIT_CODE}"
        FAILED_EXPERIMENTS+=("${EXPERIMENT_NAME}")
    fi
    
    echo "⏰ Finished at: $(date)"
    echo ""
}

# Main execution: Run experiments from n=16 down to n=7
echo "🏁 Starting batch experiments..."
echo ""

for n in 15 14 13 12 11 10 9 8 7; do
    run_experiment $n
    
    # Add a small delay between experiments to avoid potential resource conflicts
    if [ $n -gt 7 ]; then
        echo "⏳ Waiting 10 seconds before next experiment..."
        sleep 10
    fi
done

# Final summary
echo "="*80
echo "📊 BATCH EXPERIMENT SUMMARY"
echo "="*80
echo "⏰ Completed at: $(date)"
echo "📈 Total experiments: ${TOTAL_EXPERIMENTS}"
echo "✅ Successful: ${#SUCCESSFUL_EXPERIMENTS[@]}"
echo "❌ Failed: ${#FAILED_EXPERIMENTS[@]}"
echo ""

if [ ${#SUCCESSFUL_EXPERIMENTS[@]} -gt 0 ]; then
    echo "✅ Successful experiments:"
    for exp in "${SUCCESSFUL_EXPERIMENTS[@]}"; do
        echo "  - ${exp}"
    done
    echo ""
fi

if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
    echo "❌ Failed experiments:"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - ${exp}"
    done
    echo ""
fi

echo "🎯 All results should be logged to wandb"
echo "📊 Check wandb dashboard for detailed results and comparisons"
echo ""
echo "🏁 Batch experiments completed!"
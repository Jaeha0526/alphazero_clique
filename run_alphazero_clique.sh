#!/bin/bash

#SBATCH --job-name=alphazero_clique_n7k4
#SBATCH --partition=expansion
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --time=15:00:00
#SBATCH --output=logs/alphazero_clique_%j.out
#SBATCH --error=logs/alphazero_clique_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@caltech.edu

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 50G"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (adjust based on Caltech HPC setup)
# Uncomment and modify these lines based on your HPC environment
# module load python/3.9
# module load cuda/11.8
# module load gcc/9.3.0

# Activate your Python environment
# Adjust this path to your actual environment
if [ -d "venv" ]; then
    echo "Activating Python virtual environment..."
    source venv/bin/activate
elif [ -d "$HOME/miniconda3" ]; then
    echo "Activating conda environment..."
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate alphazero  # Adjust environment name as needed
elif [ -d "$HOME/anaconda3" ]; then
    echo "Activating conda environment..."
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate alphazero  # Adjust environment name as needed
else
    echo "Warning: No Python environment found. Using system Python."
fi

# Verify Python and required packages
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch not found')"

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set PyTorch to use available CPUs efficiently
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Starting AlphaZero Clique training pipeline..."
echo "Command: python src/pipeline_clique.py --mode pipeline --vertices 7 --k 4 --game-mode symmetric --hidden-dim 128 --num-layers 3 --iterations 50 --self-play-games 60 --mcts-sims 1500 --num-cpus 20 --batch-size 64 --epochs 10 --initial-lr 0.0003 --lr-factor 0.5 --lr-patience 3 --eval-threshold 0.52 --num-games 21 --experiment-name improved_attn_value_n7k4_h128_l3_mcts1500_2 --value-weight 1.0"

# Run the AlphaZero pipeline
python src/pipeline_clique.py \
    --mode pipeline \
    --vertices 7 \
    --k 4 \
    --game-mode symmetric \
    --hidden-dim 128 \
    --num-layers 3 \
    --iterations 50 \
    --self-play-games 60 \
    --mcts-sims 1500 \
    --num-cpus 20 \
    --batch-size 64 \
    --epochs 10 \
    --initial-lr 0.0003 \
    --lr-factor 0.5 \
    --lr-patience 3 \
    --eval-threshold 0.52 \
    --num-games 21 \
    --experiment-name improved_attn_value_n7k4_h128_l3_mcts1500_2 \
    --value-weight 1.0

# Capture exit code
EXIT_CODE=$?

echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

# Print some final statistics
echo "=== Job Statistics ==="
echo "Job ID: $SLURM_JOB_ID"
echo "User: $USER"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Archive important logs if training completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    # Create a summary of results
    if [ -d "experiments/improved_attn_value_n7k4_h128_l3_mcts1500_2" ]; then
        echo "Experiment results saved in: experiments/improved_attn_value_n7k4_h128_l3_mcts1500_2"
        ls -la experiments/improved_attn_value_n7k4_h128_l3_mcts1500_2/
    fi
else
    echo "Training failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE 
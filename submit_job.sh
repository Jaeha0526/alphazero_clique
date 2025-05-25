#!/bin/bash

# Simple helper script to submit the AlphaZero Clique job to SLURM

echo "=== AlphaZero Clique Job Submission ==="
echo "This will submit a job with:"
echo "  - 20 CPUs"
echo "  - 50GB memory" 
echo "  - 15 hour time limit"
echo "  - 7-vertex, 4-clique game"
echo "  - 50 iterations with enhanced training"
echo ""

# Check if script exists
if [ ! -f "run_alphazero_clique.sh" ]; then
    echo "Error: run_alphazero_clique.sh not found!"
    exit 1
fi

# Create logs directory
mkdir -p logs
echo "Created logs directory: $(pwd)/logs"

# Check if Python environment is available
if command -v python &> /dev/null; then
    echo "Python version: $(python --version)"
else
    echo "Warning: Python not found in PATH"
fi

# Check if required source files exist
if [ ! -f "src/pipeline_clique.py" ]; then
    echo "Error: src/pipeline_clique.py not found!"
    echo "Please make sure you're in the correct directory."
    exit 1
fi

echo ""
echo "Submitting job to SLURM..."

# Submit the job
sbatch run_alphazero_clique.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "Job submitted successfully!"
    echo ""
    echo "To monitor your job:"
    echo "  squeue -u $USER                    # Check job status"
    echo "  tail -f logs/alphazero_clique_*.out # Follow output log"
    echo "  tail -f logs/alphazero_clique_*.err # Follow error log"
    echo "  scancel <job_id>                   # Cancel job if needed"
    echo ""
    echo "Current jobs in queue:"
    squeue -u $USER 2>/dev/null || echo "  (squeue not available)"
else
    echo "Error: Failed to submit job!"
    exit 1
fi 
# Missing Features Checklist - JAX vs Original Pipeline

**Note: This directory contains an older experimental JAX implementation. The current production-ready JAX implementation in `jax_full_src/` has ALL features implemented with full PyTorch compatibility, including:**

✅ Complete JSON logging with hyperparameters  
✅ PyTorch-style 3-axis visualization  
✅ Full model evaluation (vs initial/previous)  
✅ Weights & Biases integration  
✅ Model saving/loading  
✅ Identical CLI interface  
✅ JIT compilation for 30x speedup  

For the complete implementation, use:
```bash
python jax_full_src/run_jax_optimized.py --experiment-name my_experiment
```

---

## Legacy Implementation - Missing Features

Below are features that were missing in this experimental JAX implementation:

### 1. ❌ Logging System
- **Original**: Detailed JSON logging with hyperparameters and iteration metrics
- **Original**: Logs to `training_log.json` with structure:
  ```json
  {
    "hyperparameters": {...},
    "log": [
      {
        "iteration": 0,
        "validation_policy_loss": 0.5,
        "validation_value_loss": 0.3,
        "evaluation_win_rate_vs_best": 0.55,
        "evaluation_win_rate_vs_initial": 0.8,
        "evaluation_win_rate_vs_initial_mcts_1": 0.9
      }
    ]
  }
  ```
- **JAX**: Basic logging, missing hyperparameters section

### 2. ❌ Visualization/Graphs
- **Original**: `plot_learning_curve()` function that creates:
  - Policy loss curve (red)
  - Value loss curve (blue) 
  - Win rate vs initial model (green)
  - Saves as `training_losses.png`
  - Three Y-axes for different metrics
  - Title with hyperparameters
- **JAX**: No visualization implemented

### 3. ❌ Model Evaluation
- **Original**: Three types of evaluation:
  1. New model vs Best model
  2. New model vs Initial model (full MCTS)
  3. New model vs Initial model (1 MCTS sim)
- **JAX**: Only placeholder evaluation

### 4. ❌ Weights & Biases Integration
- **Original**: Full W&B integration with:
  - Project/run initialization
  - Config logging
  - Metric logging each iteration
  - Run finishing
- **JAX**: No W&B support

### 5. ❌ Model Saving/Loading
- **Original**: Saves models with metadata:
  ```python
  {
    'state_dict': model.state_dict(),
    'num_vertices': num_vertices,
    'clique_size': clique_size,
    'hidden_dim': hidden_dim,
    'num_layers': num_layers
  }
  ```
- **JAX**: No proper model saving

### 6. ❌ Play Against AI Mode
- **Original**: `play_against_ai()` function for human vs AI games
- **JAX**: Not implemented

### 7. ❌ Different Execution Modes
- **Original**: Supports modes:
  - `pipeline` - Full training pipeline
  - `selfplay` - Just self-play
  - `train` - Just training
  - `evaluate` - Just evaluation
  - `play` - Play against AI
- **JAX**: Only pipeline mode

### 8. ❌ Command-line Arguments
Missing arguments in JAX:
- `--mode` - Execution mode selection
- `--use-legacy-policy-loss` - Legacy loss calculation
- `--min-alpha`, `--max-alpha` - Loss weighting
- `--lr-factor`, `--lr-patience`, `--lr-threshold`, `--min-lr` - LR scheduler params
- `--iteration` - For single iteration training
- `--num-games` - For evaluation games
- `--eval-mcts-sims` - MCTS sims for evaluation
- `--use-policy-only` - Policy-only evaluation

### 9. ❌ Initial Model Evaluation
- **Original**: Keeps initial model for comparison throughout training
- **JAX**: Not implemented

### 10. ❌ Training Integration
- **Original**: Uses actual PyTorch training with:
  - LR scheduling (ReduceLROnPlateau)
  - Configurable batch size
  - Configurable epochs
  - Loss weighting (alpha)
- **JAX**: Just placeholder training

### 11. ❌ Data Loading
- **Original**: Can load all historical data or just current iteration
- **JAX**: Not properly integrated

### 12. ❌ Directory Structure
- **Original**: Uses `./experiments/{name}/` structure
- **JAX**: Uses different structure

## Implementation Priority:
1. **Critical**: Logging, Model Save/Load, Training Integration
2. **Important**: Evaluation, Visualization 
3. **Nice to have**: W&B, Play mode, Multiple execution modes
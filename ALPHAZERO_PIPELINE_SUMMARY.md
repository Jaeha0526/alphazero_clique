# AlphaZero Pipeline Summary

## Overview
The AlphaZero training pipeline in the `src` directory follows a self-play reinforcement learning approach for training neural networks to play the Clique game.

## 1. Process of One Iteration

Each iteration consists of the following steps:

### Step 1: Model Loading
- **Location**: `pipeline_clique.py:359-399`
- Determines which model to use for self-play:
  - If `iteration > 0`: Load previous iteration model (`clique_net_iter{N-1}.pth.tar`)
  - Else: Load best model (`clique_net.pth.tar`) or create fresh model
- Model is loaded to CPU and shared memory for multiprocessing

### Step 2: Self-Play Phase
- **Location**: `pipeline_clique.py:405-428`, `MCTS_clique.py:435-607`
- Runs parallel self-play games using multiprocessing
- Each CPU process plays `games_per_cpu` games
- For each game:
  - Initialize new `CliqueBoard`
  - Run MCTS with neural network guidance
  - Temperature annealing: starts at 1.0 (exploration) → 0.1 (exploitation)
  - Noise weight decreases as game progresses
  - Store (state, policy, value) tuples for each move
  - Value assignment based on game outcome and perspective mode

### Step 3: Training Phase
- **Location**: `train_clique.py:61-292`
- Load all examples from current iteration
- Split into 90% train / 10% validation
- Train neural network:
  - Optimizer: Adam with learning rate scheduler
  - Loss: Policy loss (cross-entropy) + Value loss (MSE)
  - Early stopping with patience=5 epochs
  - Save model as `clique_net_iter{N}.pth.tar`

### Step 4: Evaluation Phase
- **Location**: `pipeline_clique.py:470-606`
- Two types of evaluation:
  1. **vs Best Model**: Decides if new model replaces best
  2. **vs Initial Model**: Tracks overall improvement
- Play evaluation games with reduced MCTS simulations
- Win rate threshold (default 0.55) determines model update

### Step 5: Model Update & Logging
- **Location**: `pipeline_clique.py:607-623`
- If win_rate > threshold: Update best model
- Save metrics to `training_log.json`
- Generate plots with `plot_learning_curve()`

## 2. What Gets Saved

### A. Model Files (`models/` directory)
```
clique_net_iter{N}.pth.tar    # Model after iteration N
clique_net.pth.tar            # Current best model
```

Each model file contains:
```python
{
    'state_dict': model.state_dict(),    # Model weights
    'num_vertices': num_vertices,         # Game parameters
    'clique_size': clique_size,
    'hidden_dim': hidden_dim,             # Model architecture
    'num_layers': num_layers,
    'asymmetric_mode': asymmetric_mode   # Game mode
}
```

### B. Self-Play Data (`datasets/` directory)
```
game_{timestamp}_cpu{i}_game{j}_iter{N}.pkl
```

Each pickle file contains a list of examples:
```python
{
    'board_state': {
        'edge_index': numpy_array,    # Graph connectivity
        'edge_attr': numpy_array      # Edge features
    },
    'policy': numpy_array,            # MCTS visit distribution
    'value': float,                   # Game outcome (-1, 0, 1)
    'player_role': int                # 0=attacker, 1=defender
}
```

### C. Training Logs (`training_log.json`)
```json
{
    "hyperparameters": {
        "experiment_name": "...",
        "vertices": 6,
        "k": 3,
        "mcts_sims": 200,
        "batch_size": 32,
        // ... all hyperparameters
    },
    "log": [
        {
            "iteration": 0,
            "validation_policy_loss": 2.715,
            "validation_value_loss": 0.773,
            "evaluation_win_rate_vs_best": 1.0,
            "evaluation_win_rate_vs_initial": 0.4,
            "evaluation_win_rate_vs_initial_mcts_1": 0.505
        },
        // ... more iterations
    ]
}
```

## 3. Plotting and Logging

### A. Real-time Plotting
- **Function**: `plot_learning_curve()` in `pipeline_clique.py:873-1078`
- **Output**: `training_losses.png` (overwritten each iteration)
- **Symmetric Mode** (3 axes):
  - Validation Policy Loss (red)
  - Validation Value Loss (blue)
  - Win Rate vs Initial (green)
- **Asymmetric Mode** (2x3 subplots):
  - Policy losses: combined vs role-specific
  - Value loss
  - Win rates vs initial/best
  - Learning balance metrics

### B. Logging Systems

1. **Console Logging**
   - Progress updates for each phase
   - Game states during self-play
   - Training epoch losses
   - Evaluation results

2. **JSON Logging**
   - Persistent storage of all metrics
   - Hyperparameter tracking
   - Incremental updates after each iteration

3. **Weights & Biases Integration**
   - **Location**: `pipeline_clique.py:679-701`
   - Real-time metric tracking
   - Sweep support for hyperparameter tuning
   - Remote monitoring capabilities

### C. Early Stopping
- **Location**: `pipeline_clique.py:771-856`
- Monitors validation policy loss
- Patience: 5 iterations without improvement
- Minimum iterations before stopping allowed

## Key Features

1. **Multiprocessing**: Parallel self-play across CPUs
2. **Temperature Annealing**: Exploration → Exploitation during games
3. **Skill Variation**: Optional MCTS simulation count variation
4. **Perspective Modes**: Fixed (Player 1) or Alternating (current player)
5. **Game Modes**: Symmetric or Asymmetric rules
6. **Model Architecture**: Graph Neural Network (GNN) with configurable layers

## Typical Usage

```bash
# Run full pipeline
python pipeline_clique.py --iterations 10 --self-play-games 100 --num-cpus 4

# Run specific mode
python pipeline_clique.py --mode evaluate --iteration 5

# With custom parameters
python pipeline_clique.py --vertices 9 --k 4 --mcts-sims 500 --hidden-dim 128
```
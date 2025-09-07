# AlphaZero Clique Game

This project implements the AlphaZero algorithm for the Clique Game, using Graph Neural Networks (GNNs) and Monte Carlo Tree Search (MCTS).

**Key Features:**
- **Undirected Graph GNN:** Optimized GNN architecture working directly with undirected edges
- **Flexible Value Learning:** Supports both fixed and alternating perspective modes for better value learning
- **Draw-Heavy Scenario Support:** Skill variation and specialized evaluation metrics for challenging game configurations
- **Experiment Management:** Organized experiment tracking with comprehensive logging and analysis tools

Main structure of the code originated from https://github.com/geochri/AlphaZero_Chess

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/alphazero_clique.git
cd alphazero_clique
```

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

The requirements.txt file includes the following key dependencies:
- numpy>=1.19.5: For numerical computations
- matplotlib>=3.5.1: For visualization
- networkx>=2.6.3: For graph operations
- flask>=2.0.1: For web interface
- torch>=2.0.0: For deep learning
- torchvision>=0.15.0: For computer vision utilities
- tqdm>=4.65.0: For progress bars
- torch-geometric>=2.0.4: For graph neural networks
- pandas>=1.3.5: For data manipulation
- wandb: For experiment tracking (optional)

## Game Rules

The Clique Game is played on an undirected graph with N vertices. There are two game modes:

### Symmetric Mode (Default)
- Both players try to form a k-clique (a complete subgraph with k vertices)
- Players take turns adding edges to the graph
- First player to form a k-clique wins
- If no more moves are possible and no k-clique is formed, the game is a draw
- Example: With 6 vertices and k=3, players try to form triangles

### Asymmetric Mode
- Player 1 tries to form a k-clique
- Player 2 tries to prevent Player 1 from forming a k-clique
- Players take turns adding edges to the graph
- Player 1 wins by forming a k-clique
- Player 2 wins by preventing Player 1 from forming a k-clique
- If no more moves are possible and no k-clique is formed, Player 2 wins

## Interactive Game Interface

Test the Clique Game, visualize the board, and interact with trained AI models using the web interface:

```bash
python src/interactive_clique_game.py
```

This starts a Flask web server (usually at `http://127.0.0.1:8080` or `http://localhost:8080`). Open this address in your web browser.

**Features:**

*   Select the number of vertices (N) and clique size (k).
*   Choose between Symmetric and Asymmetric game modes.
*   Start new games and reset the current game.
*   Make moves by clicking available edges on the board or from the valid moves list.
*   View game state, move history, and valid moves.
*   Visualize the game board with colored edges (blue for Player 1, red for Player 2).
*   **AI Prediction (New!):**
    *   Toggle the "Show AI Predictions" checkbox.
    *   If enabled, a dropdown lists compatible trained models found in the `playable_models/` directory.
    *   **Compatibility:** A model is compatible if the `num_vertices` and `clique_size` (k) saved within the model file match the current game's settings.
    *   Select a model and click "Load Model".
    *   If loaded successfully, the interface will display the model's predicted game outcome (Value) and the probability distribution over valid moves (Policy) for the current state. This shows the model's raw evaluation *without* running MCTS search.
    *   Use the "Show All Found Models" button for debugging to list all `.pth.tar` files in `playable_models/` and their detected V/k parameters.

**Using Your Own Models:**

1.  Ensure your models were trained with the corrected pipeline/training scripts that save `num_vertices`, `clique_size`, and `hidden_dim` within the `.pth.tar` file.
2.  Create a directory named `playable_models` in the project's root directory (`alphazero_clique/playable_models/`).
3.  Copy your trained model files (e.g., `clique_net.pth.tar`, `clique_net_iterN.pth.tar`) into this `playable_models` directory.
4.  Relaunch the interactive game server.

## Running the AlphaZero Pipeline

Train your own models using the full AlphaZero pipeline:

**Example Command:**

```bash
python src/pipeline_clique.py --mode pipeline \
                             --vertices 6 \
                             --k 3 \
                             --iterations 20 \
                             --self-play-games 48 \
                             --mcts-sims 200 \
                             --num-cpus 6 \
                             --hidden-dim 128 \
                             --num-layers 3 \
                             --initial-lr 0.0003 \
                             --lr-factor 0.7 \
                             --lr-patience 5 \
                             --lr-threshold 0.003 \
                             --batch-size 128 \
                             --epochs 10 \
                             --perspective-mode alternating \
                             --skill-variation 0.0 \
                             --value-weight 1.0 \
                             --experiment-name n6k3_h128_l3
```

**Quick Start Workflow:**

1. **Start with small experiments** (n=6, k=3) to verify setup
2. **Use skill variation** for draw-heavy scenarios (n=7, k=4)
3. **Monitor progress** with training logs and optional wandb
4. **Compare experiments** using analysis scripts

**Pipeline Arguments Detailed:**

Above are main arguments we want to change for the experiment. You can find details on arguments in Command-Line Arguments section. 

*(Other modes like `selfplay`, `train`, `evaluate` are available via `--mode` for running specific parts of the pipeline - see `src/pipeline_clique.py` for details)*

The pipeline will:

1.  Create experiment-specific directories.
2.  Run multiple iterations, generating self-play data, training the network, and evaluating.
3.  Save the best model (e.g., `clique_net.pth.tar`) and per-iteration models (e.g., `clique_net_iterN.pth.tar`) in `./experiments/<experiment_name>/models/`.
    *   **Model Compatibility:** Models saved by the pipeline now include `num_vertices`, `clique_size`, `hidden_dim`, and `num_layers`, allowing them to be loaded by the interactive game's AI feature.
4.  Store training examples (`.pkl` files) in `./experiments/<experiment_name>/datasets/`.
5.  Log training progress and results in `./experiments/<experiment_name>/training_log.json`.

## Handling Draw-Heavy Scenarios

For challenging game configurations (e.g., n=7, k=4) where many games end in draws, the pipeline includes several features to improve training:

**Skill Variation (`--skill-variation`):**
- Randomly varies MCTS simulation counts between players
- Creates skill imbalances that lead to more decisive games
- Provides better value learning signal by reducing draw rates

**Dual Evaluation Metrics:**
- **Best Model Updates:** Uses win rate from decided games only (excludes draws)
- **Progress Monitoring:** Uses traditional win rate including draws as losses
- This allows model progression in draw-heavy scenarios while still tracking overall performance

**Perspective Mode (`--perspective-mode`):**
- `alternating` (recommended): Values from current player's perspective
- `fixed`: Values always from Player 1's perspective
- Alternating mode typically provides better value learning

**Example for Draw-Heavy Scenario (n=7, k=4):**
```bash
python src/pipeline_clique.py --mode pipeline \
                             --vertices 7 \
                             --k 4 \
                             --skill-variation 0.3 \
                             --perspective-mode alternating \
                             --eval-threshold 0.6 \
                             --experiment-name n7k4_skill_var
```

## Graph Neural Network Architecture

Our AlphaZero implementation uses a specialized Graph Neural Network (GNN) designed for the Clique Game:

### **Graph Representation**
- **Nodes**: Represent vertices in the graph (initialized with zero features to preserve permutation symmetry)
- **Edges**: Represent potential connections between vertices with 3-dimensional features:
  - `[1,0,0]`: Unselected edge (available for play)
  - `[0,1,0]`: Edge selected by Player 1
  - `[0,0,1]`: Edge selected by Player 2

### **Core Architecture Components**

**1. Input Embeddings:**
- Node embedding: Maps 1D node indices to `hidden_dim` features
- Edge embedding: Maps 3D edge states to `hidden_dim` features

**2. GNN Layers (configurable with `--num-layers`):**
- **EdgeAwareGNNBlock**: Message passing that combines node and edge features
  - Uses `mean` aggregation for undirected graphs
  - Includes residual connections and layer normalization
  - Messages: `ReLU(Linear(concat(node_features, edge_features)))`
- **EdgeBlock**: Updates edge features based on connected node pairs
  - Processes both directions (i→j and j→i) then averages
  - Residual connections preserve information flow

**3. Dual Output Heads:**
- **Policy Head**: Predicts move probabilities for each possible edge
  - Applied per-edge to generate action probabilities
  - Uses dropout and multiple linear layers
- **Value Head**: Predicts game outcome from current position
  - Global attention pooling combines node and edge information
  - Outputs value ∈ [-1, +1] representing expected outcome

### **Key Design Decisions**

**Undirected Edge Processing:**
- Works directly with undirected edges (n*(n-1)/2 edges for n vertices)
- Handles bidirectional message passing automatically
- More efficient than creating explicit bidirectional edges

**Permutation Symmetry:**
- All nodes initialized with identical zero features
- Symmetry preserved throughout the network
- Allows model to generalize across vertex relabelings

**Attention-Based Pooling:**
- Separate attention mechanisms for nodes and edges
- Learns to focus on relevant graph regions
- Combines global node and edge representations for value prediction

### **Architecture Flexibility**
- `--hidden-dim`: Controls feature dimensionality (default: 64)
- `--num-layers`: Number of GNN layer pairs (default: 2)
- Scalable to different graph sizes (tested on n=6,7 vertices)

This architecture efficiently captures both local edge relationships and global graph structure essential for strategic game play.

## Project Structure

```
alphazero_clique/
├── experiments/            # Experiment results (35 experiments with data preserved)
│   └── <experiment_name>/
│       ├── models/         # Trained models
│       ├── datasets/       # Self-play game data
│       ├── checkpoints/    # JAX model checkpoints
│       └── training_log.json # Training metrics
├── src/                    # Original PyTorch implementation
│   ├── pipeline_clique.py      # Main AlphaZero pipeline
│   ├── interactive_clique_game.py  # Web interface
│   ├── alpha_net_clique.py     # GNN architecture
│   ├── clique_board.py         # Game logic
│   ├── MCTS_clique.py          # MCTS implementation
│   └── train_clique.py         # Training utilities
├── jax_full_src/           # JAX implementation with MCTX
│   ├── run_jax_optimized.py    # Main entry point with all optimizations
│   ├── mctx_final_optimized.py # Memory-efficient MCTS (num_sims+1 nodes)
│   ├── mctx_true_jax.py        # Optional True MCTX with JAX primitives
│   ├── train_jax_fully_optimized.py # JIT-compiled training pipeline
│   ├── vectorized_board.py     # Vectorized game logic
│   ├── vectorized_nn.py        # JAX neural network
│   ├── train_jax.py            # Standard JAX training
│   └── archive/                # Older implementations preserved
├── templates/              # Web interface templates
├── requirements.txt        # PyTorch dependencies
├── requirements_jax.txt    # JAX dependencies
├── README.md              # This file
├── MCTX_ANALYSIS_SUMMARY.md    # MCTX performance analysis
├── MCTX_PIPELINE_UPDATE_SUMMARY.md # Integration guide
└── JAX_VS_PYTORCH_PIPELINE_COMPARISON.md # Detailed comparison
```

## Pure JAX Implementation (GPU Accelerated)

We have developed a **pure JAX** implementation with an optimized MCTX (Monte Carlo Tree Search in JAX) that achieves significant speedup through GPU-accelerated vectorized computation. This implementation processes multiple games in parallel and uses JAX's JIT compilation for maximum performance.

### Recent Optimizations (September 2025)

- **Memory Efficiency**: MCTX now allocates only `num_simulations + 1` nodes instead of fixed 500 (90% memory reduction)
- **JIT-Compiled Training**: 5x faster training with fully JIT-compiled train step
- **Vectorized Batch Preparation**: 10x faster batch preparation using JAX arrays
- **Optional True MCTX**: Use `--use_true_mctx` flag for 5x faster self-play with JAX primitives (still sequential MCTS)
- **Parallel Evaluation**: Use `--parallel_evaluation` flag to run all evaluation games in ONE batch (truly parallel)
- **Dirichlet Noise**: Proper exploration with 25% noise for self-play, 10% for evaluation (AlphaZero standard)
- **UCT Exploration**: Correctly implemented UCB formula with c_puct=3.0 for balanced exploration/exploitation
- **Best Model Tracking**: Maintains best model throughout training with 55% win threshold for updates
- **Flexible Evaluation**: Control evaluation games (`--eval_games`) and MCTS simulations (`--eval_mcts_sims`) separately
- **Overall Performance**: ~35% reduction in total training time

### Quick Start with JAX

```bash
# Setup JAX environment (run from root directory)
./setup.sh

# Run the JAX pipeline with all optimizations (from root directory)
python jax_full_src/run_jax_optimized.py \
    --experiment_name my_jax_exp \
    --num_iterations 10 \
    --num_episodes 100 \
    --mcts_sims 50 \
    --game_batch_size 32 \
    --training_batch_size 256 \
    --num_epochs 20 \
    --vertices 6 \
    --k 3 \
    --eval_games 21 \
    --eval_mcts_sims 30 \
    --use_true_mctx  # Optional: 5x faster MCTS (causes compilation overhead)

# Resume from checkpoint
python jax_full_src/run_jax_optimized.py \
    --resume_from experiments/my_jax_exp/checkpoints/checkpoint_iter_5.pkl \
    --experiment_name my_jax_exp \
    # ... same other parameters

# CPU-optimized evaluation with subprocess parallelization
python jax_full_src/run_jax_optimized.py \
    --experiment_name cpu_optimized \
    --subprocess_eval \
    --eval_num_cpus 8 \
    --python_eval \
    --eval_games 100 \
    --eval_mcts_sims 50 \
    # ... other parameters

# GPU-optimized with parallel evaluation
python jax_full_src/run_jax_optimized.py \
    --experiment_name gpu_optimized \
    --parallel_evaluation \
    --use_true_mctx \
    --eval_games 100 \
    # ... other parameters

# Standalone evaluation of saved models
python jax_full_src/standalone_evaluation.py \
    --experiment my_experiment \
    --subprocess --num_cpus 8 \
    --num_games 100 --mcts_sims 50
```

This will:
- Use the memory-optimized MCTXFinalOptimized implementation
- Run vectorized self-play with configurable batch sizes
- Train using JIT-compiled JAX/Flax/Optax pipeline
- Generate learning curves and checkpoints
- Achieve significant speedup with optimized memory allocation and JIT compilation

### JAX Implementation Features

#### Core Optimizations
- **Memory-Optimized MCTX**: Dynamic node allocation based on simulation count
- **JIT-Compiled Training**: Full training step compiled for GPU execution
- **True MCTX Option**: Pure JAX primitives implementation (same sequential MCTS, just faster)

#### Evaluation Options
The JAX implementation provides multiple evaluation strategies during training:

1. **Skip Evaluation** (`--skip_evaluation`): Skip evaluation for quick iterations
2. **Default**: Sequential evaluation with vectorized batches
3. **Parallel Evaluation** (`--parallel_evaluation`): All games in single batch (GPU-friendly)
4. **Subprocess Evaluation** (`--subprocess_eval`): CPU multiprocessing via separate processes
   - Use with `--eval_num_cpus N` to control parallelization
   - Recommended with `--python_eval` to avoid JAX compilation
5. **Python MCTS** (`--python_eval`): Use Python MCTS instead of JAX (avoids compilation overhead)

All evaluation modes respect:
- `--eval_games`: Number of evaluation games (default: 21)
- `--eval_mcts_sims`: MCTS simulations per move (default: 30)
- Game mode from training (symmetric/asymmetric/avoid_clique)
- **Vectorized Operations**: Batch preparation and processing fully vectorized
- **Pure JAX**: No PyTorch dependencies - everything runs in JAX
- **Scalable Performance**: Optimal for different game sizes (n,k)

#### AlphaZero Enhancements
- **Proper UCT/PUCT**: Correctly implemented Upper Confidence Bound with c_puct=3.0
- **Dirichlet Noise**: Exploration noise at root (25% self-play, 10% evaluation) following AlphaZero paper
- **Best Model Tracking**: Maintains champion model, updates when win rate > 55%
- **Dual Evaluation**: Evaluates against both initial and best models for progress tracking
- **First Iteration Optimization**: Skips redundant best model evaluation in first iteration
- **Truly Parallel Evaluation**: All evaluation games run in single batch (not sequential)

#### Training Control
- **Flexible Batch Sizes**: `--game_batch_size` for self-play, `--training_batch_size` for NN training
- **Separate Evaluation Settings**: `--eval_games` and `--eval_mcts_sims` for evaluation control
- **Resume Training**: `--resume_from checkpoint.pkl` to continue from saved state (properly loads initial/best models)
- **Game Modes**: symmetric, asymmetric, avoid_clique (for Ramsey counterexamples)
- **Neural Network Architecture**: `--hidden_dim` (default: 64) and `--num_layers` (default: 3) for GNN configuration
- **Game Data Saving**: `--save_full_game_data` to save complete game data every iteration (not just every 5)

### ⚠️ Important Performance Considerations

#### JAX Compilation Overhead
JAX recompiles functions when input shapes change. This causes significant delays when:
- Evaluation batch size differs from self-play batch size
- Evaluation MCTS simulations differ from self-play simulations
- First time running after code changes

**Recommendation**: To avoid recompilation delays, use matching parameters:
```bash
--game_batch_size 50 --eval_games 50  # Same batch size
--mcts_sims 100 --eval_mcts_sims 100  # Same MCTS depth
```

### Requirements for JAX Version

The setup script automatically detects your CUDA version and installs the appropriate JAX:

```bash
# Automated setup (recommended)
./setup.sh

# Or manual installation for specific CUDA versions:
# For CUDA 12:
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# For CUDA 11:
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# For CPU only:
pip install --upgrade jax jaxlib

# Install JAX ecosystem
pip install --upgrade flax optax
```

### Comparison: PyTorch vs JAX Implementation

| Feature | PyTorch (Original) | JAX (MCTX Optimized) |
|---------|-------------------|-----------------------|
| **Framework** | PyTorch + torch-geometric | JAX + Flax |
| **Self-Play** | CPU multiprocessing | Vectorized GPU batches |
| **MCTS** | Tree-based with dictionaries | Memory-optimized arrays (only num_sims+1 nodes) |
| **Training** | Standard Adam | JIT-compiled Optax (5x faster) |
| **Batch Preparation** | Python loops | Vectorized JAX arrays (10x faster) |
| **Performance (n=6,k=3)** | ~30ms/game | ~6ms/game with True MCTX |
| **Performance (n=14,k=4)** | ~30 min/iteration | ~5-10 min/iteration |
| **GPU Utilization** | Training only (~30%) | Full pipeline (~80%) |
| **Memory Usage** | Low | Optimized (90% reduction vs initial) |
| **Interface** | Original CLI | Enhanced CLI with optimization flags |

### When to Use Each Implementation

**Use PyTorch version when:**
- Running small games (n≤6) with small batches
- Need CPU multiprocessing
- Debugging or understanding the algorithm
- Want the most stable, tested implementation

**Use JAX version when:**
- Running large games (n≥9) - always faster
- Processing large batches (>50 games)
- Have GPU available with sufficient memory
- Want to leverage JAX ecosystem features

See `jax_full_src/README.md` for detailed documentation on the JAX implementation.

### Performance Optimization Results

Our optimized JAX implementation achieves significant performance improvements:

**Training Performance (August 2025 Optimizations):**
- **Memory Efficiency**: 90% reduction in MCTX memory usage
- **Training Speed**: 5x faster with JIT compilation
- **Batch Preparation**: 10x faster with vectorized operations
- **Overall Speedup**: ~35% reduction in total training time

**Game Performance Comparison:**
- **Small games (n=6, k=3)**: 30ms → 6ms per game (5x faster)
- **Large games (n=14, k=4)**: 30 min → 5-10 min per iteration (3-6x faster)
- **GPU Utilization**: Increased from ~30% to ~80%

**Key Optimizations:**
- Dynamic memory allocation (only num_sims+1 nodes instead of fixed 500)
- JIT-compiled training step with gradient operations
- Vectorized batch preparation using JAX arrays
- Optional True MCTX with JAX primitives for maximum speed

See `TRAINING_OPTIMIZATION_SUMMARY.md` and `MCTX_MEMORY_OPTIMIZATION.md` for implementation details.

### Transfer Learning (Experimental)

We attempted transfer learning from n=9,k=4 to n=13,k=4 models (August 2025). While GNNs are theoretically size-agnostic, we discovered architectural limitations:

- **What worked**: Weight transfer technically successful, model loads without errors
- **What didn't**: Policy head outputs fixed size, performance benefits unverified due to evaluation challenges
- **Key learning**: Current GNN architecture not truly portable across graph sizes

See `jax_full_src/transfer_learning/` for implementation and `TRANSFER_LEARNING_ATTEMPT.md` for detailed findings.

## Analysis and Utilities

The main analysis capabilities are built into the pipeline and training scripts:

- **Training Logs**: Each experiment generates `training_log.json` with detailed metrics
- **Learning Curves**: Automatically generated `training_losses.png` plots
- **Model Checkpoints**: Saved at each iteration for analysis
- **Self-Play Data**: Pickled game records for replay analysis

For JAX experiments, additional metrics are saved in `metrics_iter_*.json` files.

## Experiment Tracking

The pipeline includes comprehensive integration with [Weights & Biases (wandb)](https://wandb.ai/) for experiment tracking and hyperparameter optimization.

### **Basic Experiment Tracking**

- Training metrics are automatically logged to wandb if available
- Each experiment gets a unique run name with timestamp
- All hyperparameters and results are tracked
- If wandb is not available or fails to initialize, training continues normally with local logging only

### **Hyperparameter Optimization with Wandb Sweeps**

For systematic hyperparameter optimization, the pipeline supports wandb sweeps with Bayesian optimization:

**1. Setup Sweep Configuration:**

The repository includes `sweep_config.yaml` for optimizing training hyperparameters with a fixed 32-layer, 8-depth architecture:

```yaml
# Example sweep configuration (see sweep_config.yaml)
method: bayes
metric:
  goal: minimize
  name: validation_policy_loss
parameters:
  initial_lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.002
  batch_size:
    values: [16, 32, 64, 96]
  epochs:
    min: 10
    max: 30
  self_play_games:
    min: 60
    max: 150
  mcts_sims:
    values: [200, 300, 400, 600, 800]
  skill_variation:
    min: 0.2
    max: 0.8
```

**2. Launch Hyperparameter Sweep:**

```bash
# Create the sweep
wandb sweep sweep_config.yaml

# Run sweep agents (use sweep ID from previous command)
wandb agent <your-sweep-id>

# For parallel optimization (recommended):
# Terminal 1:
wandb agent <your-sweep-id>
# Terminal 2:
wandb agent <your-sweep-id>
# Terminal 3:
wandb agent <your-sweep-id>
```

**3. Sweep Features:**

- **Bayesian Optimization**: Intelligently explores hyperparameter space
- **Early Termination**: Stops underperforming runs to save compute
- **Parallel Execution**: Run multiple experiments simultaneously
- **Real-time Monitoring**: Track progress and compare results in wandb dashboard
- **Automatic Best Model Selection**: Find optimal settings for challenging scenarios like n7k4

**4. Recommended Sweep Strategy:**

For optimal results with limited compute:
- **Single Agent**: Run one agent on MacBook Air M3 to avoid thermal throttling
- **Monitor Temperature**: Keep CPU usage under 80% for sustained performance
- **Expected Runtime**: 20-30 runs over 2-3 days to find optimal hyperparameters
- **Target Improvement**: Achieve policy loss < 1.1 consistently

**5. Interpreting Results:**

The sweep optimizes for `validation_policy_loss` minimization. Key metrics to monitor:
- **Policy Loss Convergence**: Target < 1.2 for good performance
- **Training Stability**: Avoid value loss collapse (< 1e-6)
- **Win Rate vs Initial**: Should improve to > 0.3 for effective learning

## Game Data Analysis

The JAX implementation automatically saves sample game data every 5 iterations for analyzing learning progress.

### Saved Data

Game data is saved at iterations 0, 5, 10, 15, etc. in the experiment directory:
```
experiments/your_experiment/
├── game_data/
│   ├── iteration_0.pkl     # Initial random play
│   ├── iteration_5.pkl     # After 5 iterations
│   ├── iteration_10.pkl    # After 10 iterations
│   └── ...
```

Each file contains:
- Sample of 10 games from that iteration
- Move-by-move data including:
  - Player who made each move
  - MCTS policy (top 10 action probabilities)
  - Final value assignment for training
- Metadata (iteration, timestamp, game configuration)

### Analyzing Game Data

Use the included analysis script to examine learning progress:

```bash
# Analyze a single iteration
python jax_full_src/analyze_game_data.py experiments/your_experiment/game_data/iteration_10.pkl

# Compare across all iterations to track learning
python jax_full_src/analyze_game_data.py experiments/your_experiment/game_data/ --compare
```

The analysis shows:
- **Game Length Trends**: Whether games get shorter (finding wins faster) or longer (more defensive play)
- **Policy Entropy**: Confidence in move selection (lower entropy = more confident)
- **Move-by-Move Analysis**: Detailed view of first few games
- **Learning Indicators**: Automated assessment of whether the model is improving

### Interpreting Results

Good learning indicators:
- ✓ Decreasing policy entropy over iterations (model becoming more confident)
- ✓ Consistent game lengths or slight decrease (finding efficient strategies)
- ✓ Clear preference for certain moves in opening positions

Warning signs:
- ⚠ Increasing entropy (model becoming less certain)
- ⚠ Dramatic changes in game length (unstable learning)
- ⚠ Random-looking policies even after many iterations

This analysis helps verify the AI is learning meaningful strategies rather than just memorizing patterns.

## License

[Add your license information here]

## Command-Line Arguments

The `pipeline_clique.py` script accepts various command-line arguments to configure the training process:

| Argument                      | Type    | Default     | Description                                                                      |
|-------------------------------|---------|-------------|----------------------------------------------------------------------------------|
| `--mode`                      | str     | `pipeline`  | Execution mode (`pipeline`, `selfplay`, `train`, `evaluate`, `play`).            |
| `--vertices`                  | int     | 6           | Number of vertices in the graph.                                                 |
| `--k`                         | int     | 3           | Size of the clique (`k`) required to win.                                        |
| `--game-mode`                 | str     | `symmetric` | Game rules (`symmetric` or `asymmetric`).                                        |
| `--iterations`                | int     | 5           | Total number of training iterations to run in pipeline mode.                     |
| `--self-play-games`           | int     | 100         | Number of self-play games generated per iteration.                               |
| `--mcts-sims`                 | int     | 200         | Number of MCTS simulations per move during self-play.                            |
| `--eval-threshold`            | float   | 0.55        | Win rate threshold against the previous best model to accept the new model.        |
| `--num-cpus`                  | int     | 4           | Number of CPU processes for parallel self-play game generation.                  |
| `--experiment-name`           | str     | `default`   | Directory name under `./experiments/` to store models, data, and logs.           |
| `--hidden-dim`                | int     | 64          | Hidden dimension size within the GNN layers.                                     |
| `--num-layers`                | int     | 2           | Number of GNN layers (each consisting of a node and edge update block).        |
| `--initial-lr`                | float   | 1e-5        | Initial learning rate for the Adam optimizer.                                    |
| `--lr-factor`                 | float   | 0.7         | Factor by which the learning rate is reduced by the ReduceLROnPlateau scheduler. | 
| `--lr-patience`               | int     | 5           | Number of epochs with no improvement after which learning rate will be reduced.  |
| `--lr-threshold`              | float   | 1e-3        | Threshold for measuring the new optimum, to only focus on significant changes.   |
| `--min-lr`                    | float   | 1e-7        | Lower bound on the learning rate for ReduceLROnPlateau.                          |
| `--batch-size`                | int     | 32          | Batch size used during the training phase.                                       |
| `--epochs`                    | int     | 30          | Number of training epochs performed on the collected data each iteration.      |
| `--use-legacy-policy-loss`    | flag    | `False`     | If set, use the older (potentially unstable) policy loss calculation method.     |
| `--min-alpha`                 | float   | 0.5         | Minimum clipping value for the dynamically calculated value loss weight (`alpha`). | 
| `--max-alpha`                 | float   | 100.0       | Maximum clipping value for the dynamically calculated value loss weight (`alpha`). | 
| `--perspective-mode`          | str     | `alternating` | Value perspective mode: `fixed` (always from Player 1) or `alternating` (from current player). |
| `--skill-variation`           | float   | 0.0         | Variation in MCTS simulation counts (0 = no variation, higher = more variation for reducing draws). |
| `--value-weight`              | float   | 1.0         | Weight for value loss in the combined loss function (higher = more emphasis on value learning). |
| `--use-policy-only`           | flag    | `False`     | If set, select moves directly from policy head output during evaluation (no MCTS). |
| `--iteration`                 | int     | 0           | Iteration number (used specifically for `train` mode).                           |
| `--num-games`                 | int     | 21          | Number of games to play (used for `evaluate` and `play` modes).                  |
| `--eval-mcts-sims`            | int     | 30          | Number of MCTS simulations per move during evaluation/play modes.                |

**Notes:** 
- The `min-alpha` and `max-alpha` arguments define the clipping range for the value loss weight, which is dynamically calculated as `alpha = policy_loss.detach() / (value_loss.detach() + 1e-6)` during training in `alpha_net_clique.py`.
- **Perspective Mode:** `alternating` mode (default) provides better value learning by using the current player's perspective, while `fixed` mode always uses Player 1's perspective.
- **Skill Variation:** Useful for draw-heavy scenarios (e.g., n=7, k=4) where random MCTS simulation count differences between players create skill imbalances, leading to more decisive games and better value learning signal.
- **Model Evaluation:** Best model updates use win rate calculated from decided games only (excluding draws), while initial model comparisons include draws as losses to show overall performance.
- **Policy-Only Mode:** When `--use-policy-only` is enabled, moves are selected directly from the neural network's policy head without MCTS search, useful for faster evaluation and testing raw policy quality.

## Game Data Analysis and Visualization

### Game Data Saving

The training pipeline saves game data periodically for analysis and debugging:

- **Default**: Saves game data every 5 iterations (`iteration_0.pkl`, `iteration_5.pkl`, etc.)
- **Full Save Mode**: Use `--save_full_game_data` flag to save data every iteration
- **Location**: `experiments/<experiment_name>/game_data/`

Each saved file includes:
- Complete move sequences with board states
- **MCTS visit counts** (NEW): Raw visit distribution before temperature transformation
- **Final action probabilities**: After temperature-based selection
- Neural network value predictions
- Game metadata (winner, move count, etc.)

### MCTS Visit Counts (NEW Feature)

As of September 2025, the system now saves MCTS visit counts alongside action probabilities:

```python
# Each move in saved game data now includes:
{
    'policy': [...],        # Final action probabilities (after temperature)
    'visit_counts': [...],  # Raw MCTS visit counts for each action
    'value': 0.65,         # Neural network's position evaluation
    'action': 7,           # Action actually taken
    ...
}
```

This enables analysis of:
- **Exploration vs Exploitation**: How MCTS distributes search effort
- **Temperature Effect**: Comparing raw visits (exploration) vs final probabilities (selection)
- **Search Depth Impact**: How deeper search changes action preferences
- **Network vs Search**: Differences between neural network priors and MCTS conclusions

### Web-Based Game Visualizer

Interactive visualization tool for analyzing saved games:

```bash
cd game_data_analyze
python app.py
# Open browser to http://localhost:5001
```

Features:
- **Interactive Graph Display**: Visualize the clique game as a graph
- **Move-by-Move Replay**: Step through games seeing each decision
- **MCTS Visit Distribution**: View raw visit counts per action (green bars)
- **Final Action Probabilities**: See temperature-modified selection (yellow bars)
- **Game Statistics**: Win rates, game lengths, patterns across iterations
- **Comparison Tools**: Load multiple iterations to track learning progress

The visualizer clearly shows the temperature transition:
- **Moves 1-10**: Smooth probability distributions (temperature = 1.0)
- **Moves 11+**: One-hot selections (temperature = 0.0)

### Analyzing Learning Progress

Use the provided analysis scripts to track model improvement:

```bash
# Analyze single iteration
python jax_full_src/analyze_game_data.py experiments/my_exp/game_data/iteration_10.pkl

# Compare across iterations
python jax_full_src/analyze_game_data.py experiments/my_exp/game_data/ --compare
```

This shows:
- Average game length trends
- Win rate evolution
- Move quality improvements
- Strategy pattern changes

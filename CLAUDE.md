# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AlphaZero implementation for the Clique Game - a graph-based combinatorial game where players strategically add edges to form k-cliques. The project implements deep reinforcement learning using Graph Neural Networks (GNNs) and Monte Carlo Tree Search (MCTS).

### Key Documents to Read First
- `README.md` - Complete documentation of features, installation, and usage
- `project_history.md` - Detailed development history with lessons learned about JAX vs PyTorch performance

## Build and Run Commands

### PyTorch Version (Stable/Production)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the full AlphaZero pipeline
python src/pipeline_clique.py --mode pipeline \
    --vertices 6 --k 3 \
    --iterations 20 --self-play-games 48 \
    --mcts-sims 200 --num-cpus 6 \
    --experiment-name my_experiment

# Run interactive web interface (port 8080)
python src/interactive_clique_game.py

# Evaluate trained models
python src/pipeline_clique.py --mode evaluate \
    --vertices 6 --k 3 \
    --num-games 21 --eval-mcts-sims 30
```

### JAX Version (GPU-Optimized)
```bash
# Install JAX dependencies
pip install -r requirements_jax.txt

# Run JAX pipeline with all optimizations
cd jax_full_src
python run_jax_optimized.py \
    --experiment_name my_jax_exp \
    --num_iterations 10 --num_episodes 100 \
    --mcts_sims 50 --game_batch_size 32 \
    --training_batch_size 256 \
    --vertices 6 --k 3 \
    --num_epochs 20 \
    --eval_games 42 --eval_mcts_sims 30 \
    --use_true_mctx \  # 5x faster MCTS with JAX primitives
    --parallel_evaluation  # All eval games in single batch

# Resume from checkpoint
python run_jax_optimized.py \
    --resume_from experiments/exp_name/checkpoints/checkpoint_iter_5.pkl \
    --experiment_name continued_exp \
    --num_iterations 20 \
    # ... other parameters
```

### Common Development Tasks
```bash
# Run tests on a trained model
python src/test_clique_model.py

# Visualize game boards
python src/visualize_clique.py

# Analyze training results
python src/analyze_games.py

# Analyze JAX training game data (saved every 5 iterations)
python jax_full_src/analyze_game_data.py experiments/your_experiment/game_data/iteration_10.pkl

# Compare learning progress across iterations
python jax_full_src/analyze_game_data.py experiments/your_experiment/game_data/ --compare
```

## High-Level Architecture

### Two Parallel Implementations

1. **PyTorch Implementation** (`/src/`)
   - Production-ready, stable, well-tested
   - Uses torch-geometric for GNN operations
   - CPU-based multiprocessing for self-play
   - Tree-based MCTS with dictionary node storage
   - ~30ms per game (n=6, k=3)

2. **JAX Implementation** (`/jax_full_src/`)
   - GPU-accelerated version with full AlphaZero features
   - Pure JAX/Flax neural networks
   - Vectorized batch processing
   - Memory-optimized MCTX (allocates only num_sims+1 nodes)
   - Proper UCT exploration with Dirichlet noise (25% self-play, 10% eval)
   - Best model tracking with competitive evaluation
   - Performance varies: faster on GPU for large batches, slower on CPU

### Core Components Architecture

#### Game Logic (`clique_board.py` / `vectorized_board.py`)
- Represents game state as undirected graph
- Two game modes: symmetric (both form cliques) vs asymmetric (one forms, one prevents)
- Action space: edges between vertices (n*(n-1)/2 possible actions)
- Terminal conditions: k-clique formed or no moves left

#### Neural Network (`alpha_net_clique.py` / `vectorized_nn.py`)
- **Graph Neural Network** with edge-aware message passing
- **Input**: Graph with node features (zeros) and edge features (3D: unselected/player1/player2)
- **Architecture**: 
  - EdgeAwareGNNBlock: Combines node and edge features for message passing
  - EdgeBlock: Updates edge features based on connected nodes
  - Dual heads: Policy (action probabilities) and Value (position evaluation)
- **Key design**: Preserves permutation symmetry, works with undirected edges

#### MCTS Implementation
- **PyTorch**: Traditional tree with UCB selection, expansion, evaluation, backup
- **JAX**: Multiple implementations with varying optimization levels:
  - `mctx_final_optimized.py`: Memory-efficient, allocates only needed nodes, proper UCT formula
  - `mctx_true_jax.py`: Pure JAX primitives for 5x speedup (optional)
- Both implement proper MCTS (SELECT → EXPAND → EVALUATE → BACKUP)
- **Exploration**: UCT with c_puct=3.0 + Dirichlet noise (α=0.3, ε=0.25 for self-play, ε=0.1 for eval)

#### Training Pipeline (`pipeline_clique.py` / `run_jax_optimized.py`)
1. **Self-Play**: Generate games using current model + MCTS with Dirichlet noise
2. **Training**: Update network on collected game data
3. **Evaluation**: Dual evaluation against initial AND best models
   - First iteration: Only vs initial (no redundant self-evaluation)
   - Later iterations: vs both opponents (in single batch with `--parallel_evaluation`)
4. **Model Selection**: Update best model if win rate > 55%
5. **Iterate**: Repeat for N iterations

### Critical Performance Considerations

#### JAX vs PyTorch Trade-offs
- **JAX excels**: Large batch processing, GPU utilization, vectorized operations
- **PyTorch excels**: Tree-based algorithms, CPU efficiency, single-game performance
- **Key insight**: JAX struggles with tree algorithms due to dynamic structure vs static compilation

#### JAX Performance Optimization Flags
- **`--use_true_mctx`**: Enables pure JAX MCTS implementation with JAX primitives (5x faster self-play)
- **`--parallel_evaluation`**: Runs all 21 evaluation games simultaneously instead of sequentially (10x faster evaluation)
- Both flags can be used together for maximum performance on GPU

#### Draw-Heavy Scenarios (e.g., n=7, k=4)
- Use `--skill-variation` to create skill imbalances between players
- Use `--perspective-mode alternating` for better value learning
- Evaluation uses two metrics: decided-games-only for model updates, all-games for monitoring

### Experiment Management
- Results stored in `/experiments/<experiment_name>/`
- Models saved as `clique_net.pth.tar` (best) and `clique_net_iter{N}.pth.tar`
- Training logs in `training_log.json`
- **Game data saved every 5 iterations** in `game_data/` subdirectory
  - Files: `iteration_0.pkl`, `iteration_5.pkl`, etc.
  - Contains sample games with move-by-move data for analysis
  - Use `analyze_game_data.py` to track learning progress
- Optional Weights & Biases integration for tracking

## Important Implementation Notes

1. **MCTS Correctness**: Ensure any MCTS implementation performs actual tree search. Some removed JAX implementations achieved "speedup" by not searching.

2. **GPU Availability**: JAX version requires GPU for claimed performance. CPU-only JAX is ~27x slower than PyTorch.

3. **Model Compatibility**: Models include `num_vertices`, `clique_size`, and `hidden_dim` for compatibility checking in interactive interface.

4. **Asymmetric Mode**: Requires special handling in neural network (dual policy heads) and training (role-specific values).

5. **Batch Sizes**: JAX benefits from larger batches (>50 games) while PyTorch works well with smaller batches.

6. **Transfer Learning**: Attempted n=9→n=13 transfer (August 2025). While weights transfer successfully, GNN architecture not truly size-agnostic - policy head outputs fixed size. See `jax_full_src/transfer_learning/TRANSFER_LEARNING_ATTEMPT.md` for detailed analysis and lessons learned.

## Testing and Validation

- **Unit Tests**: Located in `src/test_*.py`
- **Integration Test**: `src/minimal_pipeline_test.py` - Quick pipeline validation
- **Performance Benchmarks**: `test/speed_comparison*.py` - Compare implementations
- **Interactive Testing**: Web interface at `http://localhost:8080`

## Common Issues and Solutions

1. **JAX CUDA Errors**: Ensure correct CUDA/JAX version match (see requirements_jax.txt)
2. **High Draw Rate**: Use skill variation and alternating perspective mode (PyTorch only)
3. **Memory Issues**: Reduce batch size or MCTS simulations
4. **Slow Evaluation**: JAX recompiles when batch sizes change. Use matching parameters:
   ```bash
   --game_batch_size 50 --eval_games 50  # Same batch size
   --mcts_sims 100 --eval_mcts_sims 100  # Same MCTS depth
   ```
5. **Resume Training Issues**: Fixed in Sept 2025 - now properly loads initial/best models
6. **Decreasing Win Rate in avoid_clique**: This is expected! As models improve at avoiding cliques, more games end in draws

## Ramsey Number R(5,5) Counterexample Search

### Objective
Find a 2-coloring of K₄₂ (complete graph with 42 vertices) that avoids all monochromatic 5-cliques, proving R(5,5) ≥ 43. Currently, we only know 43 ≤ R(5,5) ≤ 48.

### Implemented Approach: avoid_clique Mode

**Status**: ✅ Fully implemented and working (September 2025)

Use `--avoid_clique` flag to enable this mode where:
- **Goal**: Avoid forming k-cliques (forming one causes you to LOSE)
- **Win**: Opponent forms a k-clique in their color
- **Draw**: All edges filled without any k-cliques
- **Ramsey Saving**: Draws automatically saved as potential counterexamples

```bash
python jax_full_src/run_jax_optimized.py \
    --experiment_name ramsey_search \
    --vertices 17 --k 5 \
    --avoid_clique \
    --num_episodes 200 \
    --num_iterations 100
```

Counterexamples saved to: `experiments/YOUR_EXP/ramsey_counterexamples/`

### Alternative Approaches (Not Implemented)

#### Approach 1: Cooperative (Both players avoid k-cliques)
- Would require modifying AlphaZero for cooperative play

#### Approach 2: Progressive Difficulty
- Start with smaller graphs and gradually increase
- Could be implemented with curriculum learning

### Training Signal Solutions

1. **Progress-based rewards**: Reward = number of edges colored before first 5-clique
2. **Density minimization**: Minimize total number of monochromatic 5-cliques
3. **Curriculum learning**: Start with easier graphs where avoidance is possible

### Strategy Transfer Across Graph Sizes

#### Graph-Invariant Features (Recommended)
```python
# Features that work across different graph sizes:
- Local density metrics per vertex neighborhood
- Distribution of 3-cliques and 4-cliques
- Color balance ratios (red vs blue edges)
- Spectral graph properties
- Vertex degree distributions by color
```

#### Subgraph Sampling Strategy
- Always use K₄₂ structure in neural network
- Randomly activate 20-30 vertices initially
- Gradually activate more vertices as training progresses
- Network learns scalable defensive principles

#### Attention-Based Architecture
- Use GNN with attention mechanisms
- Each vertex attends to its neighbors
- Learns relational patterns independent of graph size
- Same network handles K₂₀ through K₄₂

### Implementation Considerations

1. **Action space**: 861 edges × 2 colors = 1722 possible actions per turn
2. **State representation**: Edge features (uncolored/red/blue)
3. **Terminal check**: Must verify all (42 choose 5) = 850,668 possible 5-cliques
4. **Success metric**: Even ONE successful complete coloring proves R(5,5) ≥ 43

### Why This Matters
- Finding even one valid K₄₂ coloring would be a significant mathematical breakthrough
- Current approaches use random search or SAT solvers
- AlphaZero could discover non-obvious patterns through self-play
- Even near-misses (coloring 800+ edges) provide valuable insights
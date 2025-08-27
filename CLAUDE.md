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

### JAX Version (Experimental/GPU)
```bash
# Install JAX dependencies
pip install -r requirements_jax.txt

# Run JAX pipeline with optimizations
cd jax_full_src
python run_jax_optimized.py \
    --experiment_name my_jax_exp \
    --num_iterations 10 --num_episodes 100 \
    --mcts_sims 50 --game_batch_size 32 \
    --vertices 6 --k 3 \
    --use_true_mctx  # Optional: 5x faster MCTS
```

### Common Development Tasks
```bash
# Run tests on a trained model
python src/test_clique_model.py

# Visualize game boards
python src/visualize_clique.py

# Analyze training results
python src/analyze_games.py
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
   - Experimental GPU-accelerated version
   - Pure JAX/Flax neural networks
   - Vectorized batch processing
   - Memory-optimized MCTX (allocates only num_sims+1 nodes)
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
  - `mctx_final_optimized.py`: Memory-efficient, allocates only needed nodes
  - `mctx_true_jax.py`: Pure JAX primitives for 5x speedup (optional)
- Both implement proper MCTS (SELECT → EXPAND → EVALUATE → BACKUP)

#### Training Pipeline (`pipeline_clique.py` / `run_jax_optimized.py`)
1. **Self-Play**: Generate games using current model + MCTS
2. **Training**: Update network on collected game data
3. **Evaluation**: Compare new model vs best model
4. **Model Update**: Replace best model if threshold met
5. **Iterate**: Repeat for N iterations

### Critical Performance Considerations

#### JAX vs PyTorch Trade-offs
- **JAX excels**: Large batch processing, GPU utilization, vectorized operations
- **PyTorch excels**: Tree-based algorithms, CPU efficiency, single-game performance
- **Key insight**: JAX struggles with tree algorithms due to dynamic structure vs static compilation

#### Draw-Heavy Scenarios (e.g., n=7, k=4)
- Use `--skill-variation` to create skill imbalances between players
- Use `--perspective-mode alternating` for better value learning
- Evaluation uses two metrics: decided-games-only for model updates, all-games for monitoring

### Experiment Management
- Results stored in `/experiments/<experiment_name>/`
- Models saved as `clique_net.pth.tar` (best) and `clique_net_iter{N}.pth.tar`
- Training logs in `training_log.json`
- Optional Weights & Biases integration for tracking

## Important Implementation Notes

1. **MCTS Correctness**: Ensure any MCTS implementation performs actual tree search. Some removed JAX implementations achieved "speedup" by not searching.

2. **GPU Availability**: JAX version requires GPU for claimed performance. CPU-only JAX is ~27x slower than PyTorch.

3. **Model Compatibility**: Models include `num_vertices`, `clique_size`, and `hidden_dim` for compatibility checking in interactive interface.

4. **Asymmetric Mode**: Requires special handling in neural network (dual policy heads) and training (role-specific values).

5. **Batch Sizes**: JAX benefits from larger batches (>50 games) while PyTorch works well with smaller batches.

## Testing and Validation

- **Unit Tests**: Located in `src/test_*.py`
- **Integration Test**: `src/minimal_pipeline_test.py` - Quick pipeline validation
- **Performance Benchmarks**: `test/speed_comparison*.py` - Compare implementations
- **Interactive Testing**: Web interface at `http://localhost:8080`

## Common Issues and Solutions

1. **JAX CUDA Errors**: Ensure correct CUDA/JAX version match (see requirements_jax.txt)
2. **High Draw Rate**: Use skill variation and alternating perspective mode
3. **Memory Issues**: Reduce batch size or MCTS simulations
4. **Slow Training**: Check CPU count matches `--num-cpus` parameter

## Ramsey Number R(5,5) Counterexample Search

### Objective
Find a 2-coloring of K₄₂ (complete graph with 42 vertices) that avoids all monochromatic 5-cliques, proving R(5,5) ≥ 43. Currently, we only know 43 ≤ R(5,5) ≤ 48.

### Game Design Approaches

#### Approach 1: Cooperative (Both players avoid 5-cliques)
- **Win condition**: All 861 edges colored without any monochromatic 5-clique
- **Advantage**: Direct alignment with goal, both players search for safe colorings
- **Challenge**: AlphaZero designed for adversarial games, not cooperation

#### Approach 2: Adversarial (Defender vs Attacker)
- **Defender wins**: All edges colored without monochromatic 5-cliques
- **Attacker wins**: Any monochromatic 5-clique appears
- **Advantage**: Stress-tests defensive strategies, natural fit for AlphaZero
- **Challenge**: May never provide positive training signal if defense is impossible

#### Approach 3: Progressive Difficulty with Partial Graphs
- Start with smaller graphs (K₂₀ → K₃₀ → K₄₂)
- Train on subgraph sampling of K₄₂
- Gradually expand active vertices during training

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
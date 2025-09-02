# Transfer Learning for AlphaZero Clique Models

This module enables transfer learning between different graph sizes in the AlphaZero Clique game implementation. You can take a model trained on one graph size (e.g., n=9, k=4) and use it to initialize training for a larger graph (e.g., n=13, k=4).

## Why Transfer Learning?

Training AlphaZero from scratch on large graphs (n≥13) is computationally expensive. By transferring knowledge from smaller graphs, we can:
- **Accelerate training**: Start with better-than-random initialization
- **Preserve learned patterns**: Transfer pattern recognition and strategic knowledge
- **Save compute time**: Reach good performance faster

## How It Works

The Graph Neural Network (GNN) architecture learns patterns that are partially size-independent:

### Transferable Components
- **GNN message passing layers**: Learn local patterns that work across graph sizes
- **Hidden representations**: 64-dimensional embeddings are size-agnostic
- **Value head**: Position evaluation skills transfer well
- **Edge and node embedding layers**: Can be adapted to new sizes

### Non-Transferable Components
- **Policy head output layer**: Must match number of edges (n*(n-1)/2)
- **Node count dependencies**: Need to expand for more vertices

## Usage

### Basic Transfer (n=9 → n=13)

```bash
# Transfer from your trained n=9,k=4 model to n=13,k=4
python transfer_weights.py \
    ../experiments/n9k4_pure_mctx/checkpoints/checkpoint_iter_1.pkl \
    --target_vertices 13 \
    --target_k 4 \
    --output checkpoint_n13k4_transferred.pkl
```

### Analyze a Checkpoint

```bash
# See what's inside a checkpoint
python transfer_weights.py \
    ../experiments/n9k4_pure_mctx/checkpoints/checkpoint_iter_1.pkl \
    --analyze_only
```

### Start Training with Transferred Weights

```bash
# Use the transferred checkpoint to start training
cd /workspace/alphazero_clique/jax_full_src
python run_jax_optimized.py \
    --experiment_name n13k4_transfer \
    --vertices 13 \
    --k 4 \
    --resume_from transfer_learning/checkpoint_n13k4_transferred.pkl \
    --num_iterations 20 \
    --num_episodes 100 \
    --mcts_sims 50 \
    --use_true_mctx \
    --parallel_evaluation
```

## Transfer Strategy

The script uses the following strategy for weight transfer:

1. **Direct Transfer**: Weights with matching dimensions are copied directly
2. **Adaptation**: 
   - Policy output layer: Initialized randomly for new edge count
   - Node embeddings: Expanded by padding with zeros or small random values
3. **Smart Initialization**: New parameters use Xavier/He initialization

## Supported Transfers

| Source | Target | Edge Count Change | Success Rate |
|--------|--------|------------------|--------------|
| n=6, k=3 | n=9, k=3 | 15 → 36 edges | High |
| n=9, k=4 | n=13, k=4 | 36 → 78 edges | High |
| n=9, k=4 | n=15, k=5 | 36 → 105 edges | Medium |
| n=6, k=3 | n=13, k=4 | 15 → 78 edges | Medium |

Generally, smaller jumps in graph size work better than large jumps.

## Command-Line Options

```bash
python transfer_weights.py --help

Arguments:
  source_checkpoint     Path to source model checkpoint
  --target_vertices     Number of vertices for target model (required)
  --target_k           Clique size for target model (required)
  --output             Output path for transferred checkpoint
  --hidden_dim         Hidden dimension (default: 64, should match source)
  --num_layers         Number of GNN layers (default: 3, should match source)
  --asymmetric         Use asymmetric game mode
  --analyze_only       Only analyze checkpoint without transferring
```

## What Gets Transferred?

### Example Transfer (n=9 → n=13)

```
✓ Transferred (exact match):
  - GNN layer weights (EdgeAwareGNNBlock)
  - Edge processing weights (EdgeBlock)
  - Value head layers
  - Most dense layers

🔧 Adapted (modified):
  - Policy output: 36 → 78 outputs
  - Node embeddings: 9 → 13 nodes

🎲 Randomly initialized:
  - New policy output weights
  - Additional node embeddings
```

## Tips for Best Results

1. **Similar k values**: Transfer works best when k (clique size) is the same or similar
2. **Incremental transfers**: Consider stepping through sizes (9→11→13) rather than large jumps
3. **Fine-tune learning rate**: Transferred models may benefit from lower initial learning rates
4. **More MCTS simulations**: Since the model starts stronger, use more simulations for better data

## Technical Details

The transfer process:
1. Loads source checkpoint and analyzes parameter shapes
2. Creates target model with new dimensions
3. Matches parameters by name and path
4. Transfers compatible weights directly
5. Adapts or initializes incompatible weights
6. Saves new checkpoint with transfer metadata

## Limitations

- Cannot transfer between very different architectures
- Performance gain decreases with larger size differences
- Asymmetric ↔ symmetric transfer requires careful handling
- May need adjustment of training hyperparameters

## Future Improvements

- [ ] Support for different hidden dimensions
- [ ] Smart policy head initialization based on source patterns
- [ ] Automatic hyperparameter adjustment
- [ ] Cross-architecture transfer (PyTorch ↔ JAX)
- [ ] Progressive unfreezing strategies
# Full Vectorization Implementation Plan

## Overview
Create a truly parallelized AlphaZero implementation that processes 256+ games simultaneously on GPU for 100x speedup.

## Key Principle
Every operation must handle batches of games, not single games. No Python loops over games!

---

## Phase 1: Vectorized Game Board

### Original Features to Preserve (from src/clique_board.py)
- [ ] Complete graph initialization (adjacency matrix)
- [ ] Edge state tracking (0=unselected, 1=player1, 2=player2)
- [ ] Player turn management
- [ ] Valid move generation
- [ ] Move execution
- [ ] Win detection (k-clique checking)
- [ ] Draw detection (no moves left)
- [ ] Game state serialization (get_board_state)
- [ ] Board copying functionality
- [ ] String representation for debugging

### Implementation Changes
1. **State Storage**
   - Original: Single board with 2D arrays
   - Vectorized: Shape (batch_size, n, n) for all matrices
   
2. **Move Generation**
   - Original: `get_valid_moves()` returns list of edges
   - Vectorized: Returns (batch_size, num_edges) boolean mask

3. **Win Checking**
   - Original: Checks one board for k-cliques
   - Vectorized: Checks all boards simultaneously using tensor operations

4. **Move Making**
   - Original: `make_move(edge)` for one move
   - Vectorized: `make_moves(actions)` for batch_size moves at once

### Testing Requirements
- [ ] Verify game rules are identical
- [ ] Check win detection works correctly
- [ ] Ensure move validation is accurate
- [ ] Compare game outcomes with original

---

## Phase 2: Vectorized Neural Network

### Original Features (from src/alpha_net_clique.py)
- [ ] GNN architecture with message passing
- [ ] Edge-aware convolutions
- [ ] Node feature initialization
- [ ] Policy head (outputs move probabilities)
- [ ] Value head (outputs position evaluation)
- [ ] Proper masking of invalid moves
- [ ] Model saving/loading
- [ ] Training compatibility

### Implementation Changes
1. **Forward Pass**
   - Original: Process one position
   - Vectorized: Process (batch_size, ...) positions in one call

2. **Feature Encoding**
   - Original: Single board → edge_index, edge_attr
   - Vectorized: Batch of boards → (batch_size, edge_index), (batch_size, edge_attr)

3. **Message Passing**
   - Original: Operations on single graph
   - Vectorized: Batched graph operations using vmap

### Testing Requirements
- [ ] Output shapes: (batch_size, num_actions), (batch_size, 1)
- [ ] Verify policy sums to 1 after masking
- [ ] Check value is in [-1, 1] range
- [ ] Compare outputs with original on same positions

---

## Phase 3: Vectorized MCTS

### Original Features (from src/mcts_clique.py)
- [ ] Tree node structure with visit counts, values, priors
- [ ] PUCT selection formula
- [ ] Expansion with neural network evaluation
- [ ] Backup of values through tree
- [ ] Dirichlet noise for exploration
- [ ] Temperature-based action selection
- [ ] Proper handling of terminal nodes

### Implementation Changes
1. **Tree Storage**
   - Original: Tree of node objects per game
   - Vectorized: Tensor arrays for all trees
     - visit_counts: (batch_size, max_nodes, num_actions)
     - values: (batch_size, max_nodes)
     - priors: (batch_size, max_nodes, num_actions)

2. **Selection**
   - Original: Traverse one tree following PUCT
   - Vectorized: Select best actions for all trees simultaneously

3. **Neural Network Calls**
   - Original: Evaluate one position when expanding
   - Vectorized: Batch evaluate all positions needing expansion

4. **Backup**
   - Original: Update values along one path
   - Vectorized: Update all paths in parallel

### Testing Requirements
- [ ] MCTS returns same move distribution as original
- [ ] Visit counts match expected distribution
- [ ] Proper exploration vs exploitation balance
- [ ] Terminal node handling

---

## Phase 4: Vectorized Self-Play

### Original Features (from src/self_play.py)
- [ ] Generate N games of experience
- [ ] Store (state, policy, value) tuples
- [ ] Temperature-based move selection (exploration → exploitation)
- [ ] Proper value assignment based on game outcome
- [ ] Save experiences in correct format for training
- [ ] Support for different game modes
- [ ] Multiprocessing support

### Implementation Changes
1. **Game Generation**
   - Original: Sequential games, one at a time
   - Vectorized: 256 games play simultaneously

2. **Experience Collection**
   - Original: List of experiences from each game
   - Vectorized: Batched tensors of experiences

3. **Data Format**
   - Original: List of dicts with board_state, policy, value
   - Vectorized: Need converter to match training format

### Testing Requirements
- [ ] Same number of experiences generated
- [ ] Policy targets match MCTS visit distributions
- [ ] Value targets correct based on game outcomes
- [ ] Data format compatible with training

---

## Phase 5: Complete Pipeline

### Original Features (from src/pipeline_clique.py)
- [ ] Iteration loop (self-play → train → evaluate)
- [ ] Model checkpointing and loading
- [ ] Training integration
- [ ] Model evaluation (new vs best)
- [ ] Logging and metrics
- [ ] Wandb integration
- [ ] Plot generation
- [ ] Command-line arguments
- [ ] Resume from checkpoint

### Implementation Changes
1. **Self-Play Phase**
   - Original: Launch multiple processes
   - Vectorized: Single process handles 256 games

2. **Data Handling**
   - Original: Pickle files with list of experiences
   - Vectorized: Convert batched data to training format

3. **Evaluation**
   - Original: Play games sequentially
   - Vectorized: Evaluate models with parallel games

### Testing Requirements
- [ ] Full pipeline runs end-to-end
- [ ] Training loss decreases
- [ ] Models improve over iterations
- [ ] Metrics match original pipeline

---

## Phase 6: Feature Completeness Verification

### Must Check Every Feature
1. **Game Logic**
   - [ ] Symmetric vs asymmetric game modes
   - [ ] All edge cases (draws, invalid moves)
   - [ ] Correct player switching

2. **MCTS Features**
   - [ ] Dirichlet noise application
   - [ ] Virtual loss for parallel MCTS
   - [ ] Proper c_puct scaling

3. **Training Compatibility**
   - [ ] Data format matches exactly
   - [ ] Model architecture compatible
   - [ ] Checkpoint loading/saving works

4. **Evaluation Modes**
   - [ ] Model vs model evaluation
   - [ ] Different MCTS simulation counts
   - [ ] Temperature settings

5. **Logging/Debugging**
   - [ ] Training metrics
   - [ ] Game statistics
   - [ ] Performance measurements

---

## Testing Strategy

### After Each Phase
1. **Unit Tests**
   - Test individual components
   - Compare outputs with original

2. **Integration Tests**
   - Test component interactions
   - Verify data flow

3. **Performance Tests**
   - Measure actual speedup
   - Check GPU utilization

4. **Correctness Tests**
   - Play full games
   - Compare with original outcomes
   - Statistical validation

---

## Success Criteria

1. **Functional**: Every feature from original code works
2. **Performance**: 50-100x speedup over CPU implementation
3. **Correctness**: Identical game behavior and learning
4. **Compatibility**: Can load/save models from original pipeline

---

## Implementation Order

1. Start with vectorized board (Phase 1)
2. Test thoroughly before proceeding
3. Add vectorized NN (Phase 2)
4. Implement vectorized MCTS (Phase 3)
5. Create self-play system (Phase 4)
6. Build complete pipeline (Phase 5)
7. Verify all features (Phase 6)

Each phase must be fully tested before moving to the next!
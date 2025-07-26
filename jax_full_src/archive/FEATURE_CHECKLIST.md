# Detailed Feature Checklist from Original Code

## 1. clique_board.py Features

### Core Features
- [x] `__init__(num_vertices, k, game_mode="asymmetric")`
  - Initialize complete graph adjacency matrix
  - Set up edge tracking
  - Support symmetric/asymmetric modes
  
- [x] `get_valid_moves()` → List of valid edges
  - Only unselected edges (state == 0)
  - Must return as list of tuples
  
- [x] `make_move(edge: tuple)` 
  - Update edge state to current player + 1
  - Switch players
  - Check for win/draw
  - Increment move counter
  
- [x] `check_win()` 
  - Find all k-cliques for current player
  - Uses combination search
  
- [x] `is_draw()`
  - No valid moves remaining
  - No winner
  
- [x] `get_board_state()` → dict
  - Must include: adjacency_matrix, num_vertices, player, move_count, game_state, game_mode, k
  
- [x] `copy()` → new board instance
  
- [x] `__str__()` for debugging

### Edge Cases to Handle
- Empty valid moves list
- Invalid move attempts
- Game already over checks

---

## 2. encoder_decoder_clique.py Features

### Core Functions
- [x] `prepare_state_for_network(board)` → dict with edge_index, edge_attr
  - Edge features: [unselected, player1, player2] 
  - Bidirectional edges
  - Self-loops with [0,0,0] features
  
- [x] `encode_action(board, edge)` → action index
  - Map (i,j) edge to 0-14 index
  
- [x] `decode_action(board, action_idx)` → edge tuple
  - Map 0-14 index to (i,j) edge
  - Return (-1,-1) for invalid
  
- [x] `get_valid_moves_mask(board)` → boolean array

---

## 3. alpha_net_clique.py Features

### Architecture Components
- [x] Node embedding layer (1D → hidden_dim)
- [x] Edge embedding layer (3D → hidden_dim)
- [x] GNN layers with:
  - Message passing
  - Node updates
  - Residual connections
  - Layer normalization
  
- [x] Policy head:
  - Edge features → FC layers → softmax over edges
  
- [x] Value head:
  - Global pooling → FC layers → tanh output

### Training Features
- [x] Loss functions:
  - Policy: Cross-entropy
  - Value: MSE
  - Total: policy_weight * policy_loss + value_loss
  
- [x] Optimizer: Adam with learning rate scheduling
- [x] Batch processing with PyTorch Geometric
- [x] Model checkpointing

---

## 4. mcts_clique.py Features

### MCTS Node
- [x] Store: state, move, parent, children, visits, value, prior, player

### MCTS Algorithm  
- [x] Selection: PUCT formula with c_puct parameter
- [x] Expansion: NN evaluation, add children for valid moves
- [x] Backup: Update all nodes in path
- [x] Get action probabilities with temperature

### Special Features
- [x] Dirichlet noise (alpha=0.3) at root
- [x] Virtual loss for parallel MCTS
- [x] Handle terminal nodes properly

---

## 5. self_play.py Features

### Game Generation
- [x] Play N games with MCTS
- [x] Temperature schedule: 1.0 for first 10 moves, then 0
- [x] Store experiences: (state, policy, outcome)
- [x] Assign values based on final outcome

### Multiprocessing
- [x] Split games across workers
- [x] Each worker saves separate file
- [x] Filename format: `game_{timestamp}_iter{iteration}.pkl`

---

## 6. train_clique.py Features

### Data Loading
- [x] Load pickle files from directory
- [x] Combine multiple files
- [x] Train/validation split

### Training Loop
- [x] Batch creation with PyTorch Geometric
- [x] Learning rate warmup
- [x] Learning rate reduction on plateau
- [x] Validation after each epoch
- [x] Best model checkpointing

---

## 7. pipeline_clique.py Features

### Main Loop
- [x] Iterations of self-play → train → evaluate
- [x] Model management (current, best)
- [x] Evaluation: new model vs best model
- [x] Update best if win rate > threshold

### Additional Features
- [x] Experiment directories
- [x] Logging to JSON
- [x] Command line arguments
- [x] Resume from checkpoint
- [x] Wandb integration
- [x] Plot generation
- [x] Multiple execution modes

---

## 8. Special Features Not to Miss

### Game Modes
- [x] Asymmetric: Fixed player roles
- [x] Symmetric: Players can win with either color

### Evaluation Modes
- [x] Regular evaluation with MCTS
- [x] Policy-only evaluation
- [x] Different MCTS simulation counts

### Data Format
```python
{
    'board_state': dict,  # Full board state
    'edge_index': tensor,  # Shape (2, E)
    'edge_attr': tensor,   # Shape (E, 3)
    'policy': array,       # Shape (15,)
    'value': float,        # Game outcome
    'player': int          # Who made the move
}
```

---

## Critical Implementation Notes

1. **Edge Indexing**: Must maintain consistent edge ordering
2. **Player Indexing**: Internal 0/1 vs display 1/2
3. **Move Validation**: Check game not over + edge unselected
4. **Value Assignment**: Flip based on player perspective
5. **Policy Masking**: Only valid moves get probability

This checklist will ensure our vectorized implementation has 100% feature parity!
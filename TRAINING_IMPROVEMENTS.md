# AlphaZero Clique Game Training Improvements

## Summary of Issues and Fixes

This document outlines the key issues identified in the AlphaZero implementation for the Clique Game and the changes made to improve learning.

### Identified Issues

1. **Value Loss Collapse**: The value network was rapidly converging to predict the same value for all positions (validation_value_loss around 10^-7 or 10^-8).

2. **High Policy Loss**: The policy loss remained consistently high, indicating that the policy head was not learning meaningful policies.

3. **Inconsistent Win Rates**: Win rates against best and initial models showed no clear improvement pattern, fluctuating around 50%.

4. **Graph Neural Network Issues**: The GNN architecture had potential issues in how it processed graph-based game states.

### Applied Fixes

#### Neural Network Architecture

1. **Improved Model Initialization**:
   - Added proper Xavier initialization for all layers
   - Initialized biases to zero

2. **Enhanced Architecture**:
   - Simplified the policy head for better gradient flow
   - Added batch normalization to the value head
   - Added dropout to prevent overfitting

3. **Forward Pass Improvements**:
   - Fixed handling of batch normalization for both training and inference
   - Improved edge feature processing

#### Loss Calculation

1. **Better Policy Loss**:
   - Implemented KL-divergence based loss that focuses on valid moves
   - Added proper masking of invalid moves

2. **Value Loss Improvements**:
   - Added label smoothing to prevent overconfidence
   - Switched from MSE to Huber loss for better robustness

3. **Balanced Training**:
   - Dynamic balancing of policy and value losses
   - Added L2 regularization to prevent overfitting
   - Implemented gradient clipping

#### MCTS Implementation

1. **Better Exploration**:
   - Increased exploration constant (c_puct) from 1.0 to 2.0
   - Enhanced Dirichlet noise implementation with adaptive alpha values
   - Improved noise scaling based on the number of valid actions

2. **Value Backup**:
   - Added higher weighting for terminal states to reinforce winning/losing positions
   - Improved handling of values through the tree

3. **Policy Extraction**:
   - Added temperature annealing: high temperature at the beginning for exploration, low at the end for exploitation
   - Improved policy normalization and validation

#### Training Process

1. **Early Stopping**:
   - Added patience-based early stopping to prevent overfitting
   - Implemented best model state saving

2. **Learning Rate Management**:
   - Better warmup and scheduling
   - Gradient clipping to prevent exploding gradients

3. **Training Monitoring**:
   - Enhanced logging of losses and metrics
   - Added more detailed debugging information

## Recommendations for Future Training

1. **Training Hyperparameters**:
   - Use a larger `hidden_dim` (128 or 256) for more complex game states
   - Increase `num_layers` to 3-4 for deeper representations
   - Try `initial_lr` of 0.0003 with a `lr_factor` of 0.5
   - Use `batch_size` of 64 or 128 for more stable gradients

2. **Self-Play Settings**:
   - Increase `mcts_sims` to at least 1000 for stronger play
   - Use more `self_play_games` per iteration (100+)
   - Start with a lower `eval_threshold` (0.52) to accept small improvements

3. **Architecture Experiments**:
   - Try deeper networks for larger board sizes
   - Experiment with different GNN layer types
   - Consider adding residual connections

4. **Training Curriculum**:
   - Start with smaller board sizes and transfer to larger boards
   - Begin with simpler game modes and progress to more complex ones
   - Gradually increase MCTS simulation count across iterations

## Expected Improvements

With these changes, you should observe:

1. More stable value loss (not collapsing to near-zero)
2. Decreasing policy loss over iterations
3. More consistent win rates against previous models
4. Better overall playing strength

## Monitoring Training Progress

Keep an eye on these metrics:
- Policy loss should gradually decrease
- Value loss should stabilize at a reasonable value
- Win rate against initial model should increase over time
- Look for stability in training rather than just raw numbers
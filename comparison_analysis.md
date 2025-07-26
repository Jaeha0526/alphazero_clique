# PyTorch vs JAX Implementation Comparison

## Key Differences Found

### 1. Training Loss Calculation

**PyTorch Implementation:**
```python
# Uses KL-divergence style loss with valid moves masking
valid_moves_mask = (policy_target > 1e-7)
policy_loss_terms = -policy_target * log_probs * valid_moves_mask
policy_loss_per_graph = torch.sum(policy_loss_terms, dim=1)
policy_loss = policy_loss_per_graph.mean()

# Uses Huber loss (smooth L1) with label smoothing
smoothing_factor = 0.1
smoothed_value_target = value_target * (1 - smoothing_factor)
value_loss = F.smooth_l1_loss(value_output.squeeze(), smoothed_value_target)

# Dynamic weights + L2 regularization
l2_reg = 0.0
for param in self.parameters():
    l2_reg += torch.norm(param)
l2_reg *= 1e-5
total_loss = policy_weight * policy_loss + value_weight * value_loss + l2_reg
```

**JAX Implementation:**
```python
# Same KL-divergence style loss
valid_moves_mask = batch['target_policies'] > 1e-7
log_probs = jnp.log(policies + 1e-8)
policy_loss_terms = -batch['target_policies'] * log_probs * valid_moves_mask
policy_loss_per_sample = jnp.sum(policy_loss_terms, axis=1)
policy_loss = jnp.mean(policy_loss_per_sample)

# Same Huber loss with label smoothing
smoothed_targets = batch['target_values'] * (1 - label_smoothing)
value_diff = values - smoothed_targets
huber_delta = 1.0
value_loss = jnp.where(
    jnp.abs(value_diff) <= huber_delta,
    0.5 * value_diff ** 2,
    huber_delta * (jnp.abs(value_diff) - 0.5 * huber_delta)
)
value_loss = jnp.mean(value_loss)

# Combined loss BUT missing L2 regularization!
total_loss = policy_loss + value_weight * value_loss  # Missing L2 reg
```

### 2. MCTS Implementation

**PyTorch MCTS:**
- Uses c_puct = 3.0 (higher exploration)
- Temperature annealing based on game progress:
  - First 20%: temp = 1.0
  - 20-40%: temp = 0.8
  - 40-60%: temp = 0.5
  - 60-80%: temp = 0.2
  - Last 20%: temp = 0.1
- Noise weight decreases with game progress
- Dirichlet alpha = 0.3

**JAX MCTS:**
- Uses c_puct = 1.0 (default, less exploration)
- Fixed temperature = 1.0 (no annealing)
- Fixed noise_weight = 0.25 (no decay)
- Missing temperature annealing logic

### 3. Data Format and Saving

**PyTorch:**
```python
example = {
    'board_state': {
        'edge_index': state_dict['edge_index'].numpy(),
        'edge_attr': state_dict['edge_attr'].numpy()
    },
    'policy': policies[i],
    'value': value,
    'player_role': current_player
}
```

**JAX:**
```python
experience = {
    'board_state': {
        'edge_index': np.array(edge_indices[game_idx]),
        'edge_attr': np.array(edge_features[game_idx])
    },
    'policy': np.array(action_probs[game_idx]),
    'value': value,
    'player_role': int(current_players[game_idx])
}
```

Formats match, but JAX might be missing proper value assignment logic.

### 4. Value Assignment

**PyTorch:**
- Clear logic for fixed vs alternating perspective
- Values assigned based on final game outcome
- Propagates final value to all positions in the game

**JAX:**
- Has the logic but implementation seems incomplete
- The `_get_game_value` always returns 0.0 for intermediate positions
- Final value assignment looks correct

### 5. Training Parameters

**PyTorch:**
- Learning rate warmup for 15% of steps
- Gradient clipping with max_norm=1.0
- ReduceLROnPlateau scheduler
- Weight decay = 1e-4 in optimizer

**JAX:**
- No learning rate warmup
- No gradient clipping mentioned
- Fixed learning rate (no scheduler)
- No weight decay in optimizer

### 6. Early Stopping

**PyTorch:**
- Restores best model state on early stopping
- Tracks best loss for model selection

**JAX:**
- Early stopping implemented but doesn't restore best model
- Just stops training

## Critical Issues to Fix

1. **MCTS c_puct**: JAX uses 1.0 vs PyTorch's 3.0 - this significantly affects exploration
2. **Temperature annealing**: JAX missing the temperature schedule
3. **L2 regularization**: JAX missing L2 regularization in loss
4. **Learning rate warmup**: JAX missing warmup
5. **Gradient clipping**: JAX missing gradient clipping
6. **Best model tracking**: JAX doesn't restore best model on early stopping

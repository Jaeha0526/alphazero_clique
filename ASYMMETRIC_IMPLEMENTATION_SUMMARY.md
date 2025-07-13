# Asymmetric Mode Implementation Summary

This document outlines the changes needed to implement asymmetric mode training with dual policy heads (attacker/defender) for the AlphaZero Clique Game.

## Overview

**Current State:** Single policy head trained for symmetric mode (both players try to form cliques)
**Target State:** Unified architecture supporting both modes:
- **Symmetric Mode (DEFAULT):** Single policy head for both players (existing behavior)
- **Asymmetric Mode (NEW):** Dual policy heads when explicitly enabled
  - **Attacker Policy Head:** Player 1 learns to form k-cliques
  - **Defender Policy Head:** Player 2 learns to prevent k-clique formation
- **Shared Value Head:** Single value function for position evaluation

**CRITICAL:** Symmetric mode remains the default to ensure existing functionality is preserved.

## Architecture Changes

### 1. Network Architecture (`src/alpha_net_clique.py`)

**Modified CliqueGNN Class (Backward Compatible):**
```python
class CliqueGNN(nn.Module):
    def __init__(self, num_vertices, hidden_dim=64, num_layers=2, asymmetric_mode=False):
        # Keep all existing: node_embedding, edge_embedding, GNN layers, attention
        
        self.asymmetric_mode = asymmetric_mode
        
        if asymmetric_mode:
            # Dual policy heads for asymmetric mode
            self.attacker_policy_head = nn.Sequential(...)
            self.defender_policy_head = nn.Sequential(...)
        else:
            # Single policy head for symmetric mode (EXISTING BEHAVIOR)
            self.policy_head = nn.Sequential(...)
        
        # Keep existing: single value_head
        self.value_head = nn.Sequential(...)
    
    def forward(self, edge_index, edge_attr, batch=None, player_role=None):
        # Backward compatible: if not asymmetric_mode, use existing policy_head
        # If asymmetric_mode, select appropriate policy head based on player_role
        # Return: policy_output, value_output
```

**Changes Required:**
- **Line 393:** Add `asymmetric_mode=False` parameter to `__init__()`
- **Lines 410-417:** Conditional policy head creation based on `asymmetric_mode`
- **Line 464:** Add optional `player_role` parameter to `forward()` method  
- **Lines 500-510:** Add conditional role-based policy selection logic
- **Lines 650-770:** Modify training loop to handle dual policy losses (only when asymmetric)
- **IMPORTANT:** All existing symmetric mode code paths remain unchanged

### 2. Training Data Collection (`src/MCTS_clique.py`)

**Modified Data Format:**
```python
# Current format:
example = {
    'board_state': board_state,
    'policy': policy,
    'value': value
}

# New format:
example = {
    'board_state': board_state,
    'attacker_policy': policy if player == 0 else None,
    'defender_policy': policy if player == 1 else None,
    'value': value,
    'player_role': player  # 0=attacker, 1=defender
}
```

**Changes Required:**
- **Line 95:** `expand()` method - determine player role and get appropriate policy
- **Line 230:** `search()` method - pass player role to network
- **Line 350:** `MCTS_self_play()` - track roles during data collection
- **Line 400-450:** Modify data saving format for role-specific policies

### 3. Training Pipeline (`src/pipeline_clique.py`)

**Changes Required:**
- **Line 8:** Import `AsymmetricCliqueGNN` instead of `CliqueGNN`
- **Line 213:** Instantiate `AsymmetricCliqueGNN` 
- **Line 258:** Pass `game_mode` to ensure asymmetric data collection
- **Line 277:** Modify training call for dual policy data
- **Line 291:** Load `AsymmetricCliqueGNN` for evaluation
- **Lines 320-340:** Update evaluation to specify player roles

### 4. Training Logic (`src/train_clique.py`)

**Modified Loss Computation:**
```python
# Current:
total_loss = policy_weight * policy_loss + value_weight * value_loss + l2_reg

# Asymmetric:
total_loss = policy_weight * (attacker_policy_loss + defender_policy_loss) + \
             value_weight * value_loss + l2_reg
```

**Changes Required:**
- **Dataset class:** Modify `CliqueGameData.__getitem__()` to handle dual policies
- **Loss computation:** Split policy loss into attacker + defender components
- **Training loop:** Handle cases where only one policy head has data per example

## Logging and Monitoring Changes

### 1. Training Logs

**Current Format:**
```
Policy Loss: 1.234, Value Loss: 0.567
```

**New Format:**
```
Attacker Policy Loss: 1.234, Defender Policy Loss: 0.987, Value Loss: 0.567
Combined Policy Loss: 1.111
```

**File Changes:**
- `src/alpha_net_clique.py` lines 755-766: Update print statements
- Training log JSON structure: Add separate loss fields

### 2. Training Plots (`src/pipeline_clique.py`)

**Current Plot Lines:**
- Policy Loss (red)
- Value Loss (blue)  
- Win Rate vs Initial (green)

**New Plot Lines:**
- Attacker Policy Loss (red)
- Defender Policy Loss (orange)
- Combined Policy Loss (dark red)
- Value Loss (blue)
- Win Rate vs Initial (green)

**File Changes:**
- Lines 650-720: Modify plotting logic in `plot_training_losses()`

### 3. Training Log JSON

**Current Structure:**
```json
{
  "iteration": 1,
  "validation_policy_loss": 1.234,
  "validation_value_loss": 0.567,
  "win_rate_vs_best": 0.65
}
```

**New Structure:**
```json
{
  "iteration": 1,
  "validation_attacker_policy_loss": 1.234,
  "validation_defender_policy_loss": 0.987,
  "validation_combined_policy_loss": 1.111,
  "validation_value_loss": 0.567,
  "win_rate_vs_best": 0.65
}
```

## Evaluation Changes

### 1. Model Evaluation (`src/pipeline_clique.py`)

**Role-Aware Evaluation:**
```python
# Current:
policy_output, value = model(state)

# Asymmetric:
if board.player == 0:  # Attacker
    policy_output = model.get_attacker_policy(state)
else:  # Defender
    policy_output = model.get_defender_policy(state)
```

**Changes Required:**
- **Lines 76-115:** Modify evaluation loop to pass player roles
- **Lines 160-177:** Ensure fair evaluation (both models test as attacker/defender)

### 2. Interactive Interface (`src/interactive_clique_game.py`)

**Changes Required:**
- **Model loading:** Use `AsymmetricCliqueGNN`
- **AI prediction:** Pass current player role to get appropriate policy
- **UI updates:** Show separate policy predictions for attacker vs defender

## Implementation Strategy

### Phase 1: Backward Compatible Architecture
1. **CRITICAL FIRST STEP:** Modify `CliqueGNN` with `asymmetric_mode=False` default
2. **Test existing functionality:** Ensure all symmetric mode tests pass
3. **Add dual policy heads:** Only when `asymmetric_mode=True`
4. **Conditional logic:** Ensure no impact on symmetric code paths

### Phase 2: Asymmetric Mode Implementation
1. Implement asymmetric data collection (when `game_mode="asymmetric"`)
2. Update training loop for dual policies (conditional on mode)
3. Test asymmetric mode with simple scenarios
4. Ensure mode switching works correctly

### Phase 3: Evaluation & Monitoring  
1. Update evaluation logic for role-aware play
2. Modify logging and plotting (backward compatible)
3. Update training log JSON structure
4. Test complete pipeline with both modes

### Phase 4: Integration & Polish
1. Update interactive interface
2. Documentation updates
3. Performance optimization
4. Final testing across all configurations

## Command Line Arguments

**Default Behavior (NO CHANGES NEEDED):**
```bash
# Existing symmetric mode commands work exactly as before
python src/pipeline_clique.py --mode pipeline --vertices 6 --k 3
python src/pipeline_clique.py --game-mode symmetric  # Explicit but same as default
```

**New Asymmetric Mode (EXPLICIT OPT-IN):**
```bash
--game-mode asymmetric          # Must explicitly enable asymmetric mode
--attacker-weight 1.0           # Weight for attacker policy loss (optional)
--defender-weight 1.0           # Weight for defender policy loss (optional)
```

**Existing Arguments (Unchanged):**
- `--game-mode` already exists, defaults to "symmetric"
- `--value-weight` already exists for value loss weighting

**GUARANTEE:** All existing command lines and scripts continue to work without modification.

## Testing Strategy

**Phase 1: Backward Compatibility**
1. **Regression Tests:** Ensure all existing symmetric mode functionality works unchanged
2. **Performance Tests:** Verify no performance degradation in symmetric mode

**Phase 2: Asymmetric Implementation**  
1. **Unit Tests:** Test dual policy head selection logic
2. **Integration Tests:** Full pipeline with small game configurations (n=4, k=3)
3. **Comparison Tests:** Compare symmetric vs asymmetric training convergence
4. **Evaluation Tests:** Verify fair evaluation across both roles

## Expected Benefits

1. **Specialized Strategies:** Separate heads can learn role-specific tactics
2. **Better Performance:** Asymmetric games may converge faster than symmetric
3. **Strategic Insights:** Analysis of attacker vs defender policies
4. **Extensibility:** Framework for other asymmetric games

## Potential Challenges

1. **Data Imbalance:** Ensuring equal training data for both roles
2. **Evaluation Complexity:** Fair comparison requires testing both roles
3. **Hyperparameter Tuning:** May need separate learning rates for policy heads
4. **Debugging:** More complex architecture increases debugging difficulty

## Files to Modify

### Core Implementation
- `src/alpha_net_clique.py` - New network architecture
- `src/MCTS_clique.py` - Role-aware MCTS and data collection
- `src/pipeline_clique.py` - Training pipeline updates
- `src/train_clique.py` - Training logic modifications

### Supporting Files
- `src/interactive_clique_game.py` - UI updates
- `src/encoder_decoder_clique.py` - Potential input encoding changes
- Analysis scripts - Update for dual policy analysis

### Documentation
- `README.md` - Add asymmetric mode documentation
- This file - Implementation tracking
# Training Data Flow Analysis: Why Dumb Moves Get Reinforced

## The Problem

**Training is reinforcing dumb moves instead of penalizing them!**

Let me trace through exactly what happens when a player makes a "dumb move" (choosing to lose when safe options exist).

## Step-by-Step Data Flow

### Step 1: Self-Play - Collecting Data

**File:** `run_jax_optimized.py`, lines 163-210

```python
# During MCTS search, for each move:
move_data = {
    'edge_indices': edge_indices[i],
    'edge_features': edge_features[i],
    'policy': mcts_probs[i],           # ← MCTS probabilities
    'visit_counts': visit_counts[i],
    'player': boards.current_players[i],
    'action': None  # Will be filled later
}

# Action selection (line 200):
action = np.random.choice(num_actions, p=probs)  # Sample from MCTS probs
move_data['action'] = int(action)
```

**What gets saved:**
- Board state (edge features)
- **MCTS policy** (the probabilities MCTS computed)
- **Action taken** (sampled from MCTS policy)

### Step 2: Game Ends - Assigning Values

**File:** `run_jax_optimized.py`, lines 238-250

```python
for move_data in game_data[i]:
    if winner == -1:  # Draw
        value = 0.0
    elif self.config.perspective_mode == "alternating":
        # From current player's perspective
        value = 1.0 if move_data['player'] == winner else -1.0
    else:
        # Always from Player 1's perspective
        value = 1.0 if winner == 0 else -1.0

    move_data['value'] = value  # ← Assign value to move
```

**Critical Issue Here:**

For a "dumb move" (move that immediately loses):
- Player makes move → immediately forms k-clique → loses
- **value = -1.0** (correct - this player lost)

But this value is assigned to **ALL moves in the game**, not just the dumb move!

### Step 3: Training - Learning from Data

**File:** `train_jax_fully_optimized.py`, lines 49-69

```python
# Policy loss (line 54):
policy_loss_terms = -batch['target_policies'] * log_probs * valid_moves_mask
policy_loss = jnp.mean(policy_loss_per_sample)

# Value loss (line 62):
value_diff = values - smoothed_targets
value_loss = jnp.mean(value_loss)
```

**What the network learns:**

For **each move** in the training data:
1. **Policy target:** `target_policies = mcts_probs` (what MCTS chose)
2. **Value target:** `target_values = value` (game outcome)

The network is trained to:
- **Predict the MCTS policy** that was used
- **Predict the game outcome**

## The Fatal Flaw: Example Walkthrough

### Example Game: Dumb Move at Move 11

**Iteration 15, Game 8 (from our analysis):**

```
Move 11: Player 0 makes DUMB MOVE
  Chose action 0 (MCTS prob = 1.0000)
  This move immediately forms a k-clique → Player 0 LOSES
  Safe alternatives existed: [3, 7, 12, 14]
  MCTS gave them: [0.0, 0.0, 0.0, 0.0] probability
```

**What gets saved in training data:**

```python
# Move 11 data:
{
    'edge_features': [board state at move 11],
    'policy': [0.0, 1.0, 0.0, ...],  # 100% on action 0
    'value': -1.0,  # Player 0 lost
    'action': 0
}
```

**What the network learns:**

```python
# Training step:
# 1. Policy loss: Learn to output policy = [0.0, 1.0, 0.0, ...]
#    → Network learns: "In this board state, action 0 should have prob 1.0"
#    → This is WRONG! Action 0 immediately loses!

# 2. Value loss: Learn to output value = -1.0
#    → Network learns: "This board state leads to loss"
#    → This is CORRECT!
```

**The Problem:**

The network is being told:
- ✅ "This position is losing" (correct - value = -1)
- ❌ "The correct policy is to play action 0 with 100% probability" (WRONG!)

**Why is this happening?**

The network is trained to **imitate MCTS**, not to **improve on MCTS**!

## Root Cause: Policy Target is MCTS Output

**The Issue:**

```python
# In training:
target_policies = mcts_probs  # ← This is the problem!
```

The network is learning to **reproduce** what MCTS did, including its mistakes!

### Why MCTS Makes Mistakes with Low c_puct

With c_puct=0.3:
1. MCTS tries a few moves
2. Network says "action 0 is good" (network is initially random/wrong)
3. MCTS doesn't explore much (low c_puct)
4. Action 0 gets most visits → high probability
5. Action 0 is chosen → immediate loss

Then in training:
6. Network learns "action 0 should have high probability"
7. Next iteration: Network is even MORE confident action 0 is good
8. MCTS trusts network even more → gives action 0 even higher probability
9. **Vicious cycle!**

## Comparison: What SHOULD Happen

### Correct AlphaZero Training (with sufficient exploration)

With c_puct=3.0:
1. MCTS tries a few moves including action 0
2. Network says "action 0 is good"
3. **MCTS explores other moves too** (higher c_puct)
4. Action 0 chosen → immediate loss detected
5. Action 3 explored → game continues
6. Over many simulations: action 0 gets negative value, action 3 gets better value
7. **MCTS policy gives low prob to action 0, high prob to action 3**

Then in training:
8. Network learns "action 0 should have LOW probability"
9. Network learns "action 3 should have HIGH probability"
10. **Virtuous cycle!**

## The Key Insight

**The policy target in AlphaZero is the MCTS output, not the action taken!**

This works fine IF:
- MCTS explores enough to discover bad moves
- MCTS assigns low probability to bad moves

This FAILS if:
- MCTS doesn't explore (low c_puct)
- MCTS commits to a bad move
- Policy target becomes "100% on the bad move"

## Evidence from Our Data

### Iteration 0 (Random network):
```
Game 5, Move 11: Dumb move
  Chosen action 1: MCTS prob = 0.8020
  Best safe action: MCTS prob = 0.1006
→ MCTS already preferring dumb move (network is random, but got unlucky)
```

### Iteration 15 (After training):
```
Game 8, Move 11: Dumb move
  Chosen action 0: MCTS prob = 1.0000
  Best safe action: MCTS prob = 0.0000
→ MCTS now EXTREMELY confident in dumb move (learned from earlier mistakes!)
```

## Why Value Loss Doesn't Help

You might think: "But the value is -1, won't the network learn this position is bad?"

**Yes, but:**
1. Network learns: "This position with my policy → loss"
2. Network doesn't learn: "This position with DIFFERENT policy → win/draw"
3. Because we never try different policies (low c_puct = no exploration)

The network thinks:
- "If I'm in this position, I'm doomed" (correct value)
- "So I should definitely play action 0" (wrong policy, but matches MCTS target)

## Additional Evidence: Policy Loss Trends

From training log:
```
Iteration 0:  policy_loss = 2.32
Iteration 5:  policy_loss = 2.01
Iteration 10: policy_loss = 1.95
Iteration 15: policy_loss = 1.76
Iteration 19: policy_loss = 1.83
```

**Policy loss is DECREASING!**

This means the network is getting BETTER at predicting MCTS output.
But MCTS output is WRONG (choosing dumb moves)!

So the network is learning to be confidently wrong.

## The Fix: Increase c_puct

With c_puct=3.0 or higher:

### Before (c_puct=0.3):
```
MCTS simulations:
  Action 0: 500 visits (network said good, no exploration)
  Action 3: 0 visits (not explored)
→ MCTS prob = [1.0, 0.0, ...]
→ Action 0 chosen → loss
→ Network learns: "In this state, action 0 should be 100%"
```

### After (c_puct=3.0):
```
MCTS simulations:
  Action 0: 200 visits (network says good)
  Action 3: 250 visits (exploration finds this works!)
  Action 7: 50 visits (also explored)
→ Action 0 → immediate loss → negative value
→ Action 3 → game continues → better outcome
→ MCTS prob = [0.1, 0.0, 0.6, ...] (based on outcomes)
→ Network learns: "In this state, action 3 should be high, action 0 should be low"
```

## Summary: The Training Loop

### Current (Broken) Loop with c_puct=0.3:

1. Network (wrong) → MCTS (no exploration) → Bad policy → Dumb move chosen
2. Training: Learn to reproduce bad policy
3. Network (more wrong) → MCTS (trusts network more) → Worse policy → More confident dumb move
4. **Spiral of increasing confidence in mistakes**

### Fixed Loop with c_puct=3.0:

1. Network (wrong) → MCTS (explores) → Discovers bad moves → Good policy
2. Training: Learn to reproduce good policy
3. Network (better) → MCTS (needs less exploration) → Better policy → Smarter moves
4. **Spiral of increasing confidence in correct moves**

## Conclusion

**The training is working as designed!**

The problem is:
- Training learns from MCTS policy
- With low c_puct, MCTS policy is bad
- Training learns to reproduce bad policy
- **Garbage in, garbage out**

The solution:
- Increase c_puct → Better MCTS policies → Better training targets → Better network

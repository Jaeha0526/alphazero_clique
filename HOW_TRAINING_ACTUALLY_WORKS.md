# How Training Actually Works: Why Losing Moves DON'T Get Less Preferred

## Your Question

> "If there is losing, then isn't it be less preferred for later experiments?"

**This is a great intuition, but it's NOT how AlphaZero training works!**

Let me explain the crucial misunderstanding.

## What You Might Think Training Does

You might think:
```
Training sees:
  - Action 14 was chosen → Game was lost

Training learns:
  - "Action 14 is bad, decrease its probability"
  - "Don't choose action 14 in the future"
```

**This would be correct for supervised learning or simple RL, but NOT for AlphaZero!**

## What Training Actually Does in AlphaZero

AlphaZero has **TWO separate neural network heads:**

### 1. Policy Head (What move to make)
```python
# Training target: MCTS probabilities (NOT the game outcome!)
policy_loss = -target_policy * log(predicted_policy)

# What this means:
target_policy = [0.0, 0.0, 0.0, 1.0, ...]  # ← From MCTS (100% on action 14)
predicted_policy = network output            # ← What network predicts

# Network learns: "My output should match MCTS probabilities"
```

### 2. Value Head (Who will win)
```python
# Training target: Game outcome from current player's perspective
value_loss = (predicted_value - target_value)^2

# What this means:
target_value = -1.0  # ← Current player lost the game
predicted_value = network output

# Network learns: "This board position leads to a loss"
```

## The Critical Insight: Policy and Value Are Separate!

Let me show you what happens with a dumb move:

### Example: Move 11 in Game 4, Iteration 10

**Board State:**
- Player 0's turn
- Action 14 immediately forms k-clique → Player 0 loses
- Safe alternatives: [6, 9, 12, 13]

**MCTS Output:**
```python
mcts_probabilities = [
    0.0,    # action 0
    0.0,    # action 1
    ...
    1.0,    # action 14 ← MCTS gives 100% to losing move!
    0.0     # action 15
]
```

**Action Taken:**
- Action 14 chosen → Player 0 immediately loses

**Training Data Saved:**
```python
training_example = {
    'board_state': [current edge features],
    'policy_target': [0.0, 0.0, ..., 1.0, 0.0],  # ← MCTS probs
    'value_target': -1.0,  # ← Player 0 lost
}
```

## What the Network Learns

### Policy Head Learns:
```python
Input: [board state]
Target policy: [0.0, 0.0, ..., 1.0, 0.0]  # 100% on action 14

Loss = -1.0 * log(predicted_prob_on_action_14)
     + other small terms for other actions

To minimize loss:
→ predicted_prob_on_action_14 should be HIGH (close to 1.0)
→ predicted_prob_on_other_actions should be LOW
```

**The network learns: "In this position, put 100% probability on action 14"**

### Value Head Learns:
```python
Input: [board state]
Target value: -1.0

Loss = (predicted_value - (-1.0))^2

To minimize loss:
→ predicted_value should be -1.0
```

**The network learns: "This position is losing"**

## The Problem: Policy Target is MCTS, Not Outcome!

Here's the key:
- ✅ Value target = game outcome (-1 for loss)
- ❌ Policy target = MCTS probabilities (NOT related to outcome!)

So the network learns:
- **Value head:** "This position is bad" ✓ Correct
- **Policy head:** "Action 14 should have 100% probability" ✗ Wrong!

## Why Doesn't the Value Head Fix This?

You might think: "The value head knows position is bad, won't that prevent choosing action 14?"

**No! Here's why:**

### During MCTS in the Next Iteration:

```python
# Network evaluation for the position BEFORE action 14:
policy_probs = network.policy([board_before_action_14])
# Returns: [0.0, ..., 0.9, ...] ← High prob on action 14

value = network.value([board_before_action_14])
# Returns: 0.2 ← "This position is okay-ish"

# Note: The value is for the CURRENT position, not after action 14!
```

The network has **never been trained on the position AFTER action 14**!

### What Happens:
1. MCTS starts at current position
2. Network says: "Policy: 90% action 14, Value: this position is 0.2"
3. MCTS simulates action 14 → board changes → game ends
4. **But network was never trained on "board after action 14 → immediate loss"**
5. Network doesn't know action 14 causes immediate loss!

## The Missing Link: Network Doesn't See Immediate Consequences

**What network WAS trained on:**
- Position at move 11 → Lost eventually (value = -1)
- But it doesn't know WHY it lost or WHEN it lost

**What network was NOT trained on:**
- Action 14 causes immediate loss
- Safe actions allow game to continue

The network thinks:
- "I'm in a losing position (value = -1)"
- "The correct move here is action 14 (policy = 100%)"
- **"I don't know action 14 is what CAUSES the loss!"**

## Why This Works in Normal AlphaZero

In Go/Chess with high c_puct, this works because:

### Good MCTS Targets (c_puct = 3.0):
```python
# MCTS explores many moves:
action_14: tried 50 times → immediate loss → gets Q-value = -1.0
action_12: tried 200 times → game continues → gets Q-value = 0.3
action_9: tried 150 times → game continues → gets Q-value = 0.1

# Final MCTS probabilities:
mcts_probs = [
    ...,
    0.02,  # action 9  (150 visits, but lower Q)
    0.65,  # action 12 (200 visits, highest Q)
    0.01,  # action 14 (50 visits, terrible Q)
    ...
]

# Training target: "action 12 should be 65%, action 14 should be 1%"
# Network learns: "Prefer action 12, avoid action 14"
```

### Bad MCTS Targets (c_puct = 0.3):
```python
# MCTS barely explores:
action_14: tried 500 times (network said it's good, no exploration)
action_12: tried 0 times
action_9: tried 0 times

# Final MCTS probabilities:
mcts_probs = [
    ...,
    0.0,   # action 9  (not explored)
    0.0,   # action 12 (not explored)
    1.0,   # action 14 (all visits)
    ...
]

# Training target: "action 14 should be 100%"
# Network learns: "Always choose action 14"
```

## Concrete Example from Your Data

### Iteration 0, Game 4, Move 11:

**MCTS probabilities (training target):**
```python
policy_target = {
    'action_14': 1.0000,  # Losing move
    'action_6':  0.0000,  # Safe
    'action_9':  0.0000,  # Safe
    'action_12': 0.0000,  # Safe
    'action_13': 0.0000,  # Safe
}
```

**What network learns:**
- Policy head: "Output 100% on action 14"
- Value head: "This position value = -1"

**Next iteration, same position:**
- Network now even MORE confident action 14 is correct
- MCTS trusts network MORE
- Gives action 14 even HIGHER probability (if that's possible)

## The Key Point: MCTS Does the Learning, Not the Loss Function!

**In AlphaZero:**
- Training does NOT directly learn from wins/losses
- Training learns to **imitate MCTS**
- **MCTS** learns from wins/losses (through simulations)
- If MCTS explores well → learns good moves → network imitates good moves
- If MCTS doesn't explore → never learns → network imitates bad moves

## Why Your Intuition is Correct for Other Algorithms

Your intuition would be correct for:

### Supervised Learning:
```python
# Training sees:
action_14 → label = "bad"
action_12 → label = "good"

# Learns: Decrease prob of action_14, increase prob of action_12
```

### Policy Gradient RL (REINFORCE):
```python
# Training sees:
action_14 → reward = -1
action_12 → reward = +1

# Loss: -reward * log(prob_action)
# Learns: Decrease prob of bad actions, increase prob of good actions
```

### But AlphaZero is Different:
```python
# Training sees:
action_14 → MCTS_prob = 1.0 (← This is what we learn!)
           reward = -1 (← This only affects value head)

# Policy loss: -MCTS_prob * log(predicted_prob)
# Learns: Match MCTS probabilities (regardless of reward!)
```

## Visual Summary

```
Traditional RL:
Action → Outcome → Learn from outcome directly
  ↓        ↓         ↓
  14   →  Loss  →  "Avoid action 14"

AlphaZero:
Action → Outcome → MCTS processes → Training learns from MCTS
  ↓        ↓           ↓                ↓
  14   →  Loss  →  MCTS: "14=bad"  →  "Match MCTS"
                        ↑
                   BUT: If c_puct low,
                   MCTS never realizes 14=bad!
```

## The Solution

**Make sure MCTS produces good targets:**

### With c_puct = 0.3 (Bad):
```
MCTS (no exploration) → Bad policies → Network learns bad policies
```

### With c_puct = 3.0 (Good):
```
MCTS (explores) → Discovers bad moves → Good policies → Network learns good policies
```

## Conclusion

**Your intuition is correct, but applies to the wrong component!**

- ❌ Network training does NOT learn directly from wins/losses
- ✓ **MCTS** learns from wins/losses (through exploration)
- ✓ Network learns from MCTS
- ❌ With low c_puct, MCTS doesn't explore enough to learn
- ❌ Network imitates bad MCTS → gets worse

**The fix:** Increase c_puct so MCTS actually learns from the losses!

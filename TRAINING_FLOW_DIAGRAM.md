# AlphaZero Training Flow: Visual Explanation

## Your Question: "Why doesn't losing make the move less preferred?"

Short answer: **Because the policy target comes from MCTS, not from the game outcome!**

## Traditional Reinforcement Learning (What You Expected)

```
┌─────────────────────────────────────────────────────────────┐
│ Traditional Policy Gradient (e.g., REINFORCE)               │
└─────────────────────────────────────────────────────────────┘

Step 1: Take action
    Network → Action 14 (prob=0.3)

Step 2: See outcome
    Action 14 → LOSS (reward=-1)

Step 3: Update network
    Loss = -reward * log(prob_action_14)
         = -(-1) * log(0.3)
         = positive value

    Gradient descent:
    → DECREASE prob of action 14
    → Network learns "action 14 is bad"

✓ This is what you expected!
```

## AlphaZero (What Actually Happens)

```
┌─────────────────────────────────────────────────────────────┐
│ AlphaZero: Two-Step Process                                 │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ STEP 1: MCTS (During Self-Play)                              │
│ "Search for the best move using simulations"                 │
└──────────────────────────────────────────────────────────────┘

Network says: "Action 14 looks good (score=0.8)"

MCTS with c_puct=0.3 (LOW EXPLORATION):
  Simulation 1: Try action 14 → immediate loss → Q=-1
  Simulation 2: Try action 14 again (network likes it!) → loss → Q=-1
  Simulation 3: Try action 14 again → loss → Q=-1
  ...
  Simulation 500: Try action 14 → loss → Q=-1

  Don't explore actions 6,9,12,13 (c_puct too low!)

MCTS final probabilities:
  Action 14: 500 visits → 100% probability
  Action 6:  0 visits   → 0% probability
  Action 12: 0 visits   → 0% probability

Wait, what? Why 100% on a losing move?
→ Because MCTS trusts the network MORE than the simulation results!
→ With low c_puct, exploration bonus is tiny
→ UCB formula heavily weights network prior


┌──────────────────────────────────────────────────────────────┐
│ STEP 2: Training (After Self-Play)                           │
│ "Learn to imitate MCTS"                                       │
└──────────────────────────────────────────────────────────────┘

Training example:
{
  'board_state': [...],
  'policy_target': [0, 0, 0, 1.0, ...],  ← MCTS said 100% on action 14!
  'value_target': -1.0                   ← Game was lost
}

Policy loss:
  Target: [0, 0, 0, 1.0, ...]
  Predicted: network output

  Loss = -target * log(predicted)
       = -1.0 * log(predicted[14]) + (other small terms)

  To minimize loss:
  → predicted[14] should be HIGH (close to 1.0)

  Network learns: "In this position, action 14 should have high prob"

Value loss:
  Target: -1.0
  Predicted: network output

  Loss = (predicted - (-1.0))^2

  To minimize loss:
  → predicted value should be -1.0

  Network learns: "This position is losing"

✗ Policy learned: "Prefer action 14" (WRONG!)
✓ Value learned: "Position is bad" (CORRECT!)
```

## The Vicious Cycle with Low c_puct

```
┌─────────────────────────────────────────────────────────────┐
│ Iteration 0: Random Network                                  │
└─────────────────────────────────────────────────────────────┘

Network (random): "Action 14 score = 0.6 (by chance)"
MCTS (c_puct=0.3): Doesn't explore → 100% on action 14
Training: Learn "action 14 = 100%"

     ↓

┌─────────────────────────────────────────────────────────────┐
│ Iteration 1: After Training on Random MCTS                   │
└─────────────────────────────────────────────────────────────┘

Network (trained): "Action 14 score = 0.8 (learned from iter 0)"
MCTS (c_puct=0.3): Trusts network even more → 100% on action 14
Training: Learn "action 14 = 100%" (reinforced!)

     ↓

┌─────────────────────────────────────────────────────────────┐
│ Iteration 10: Confidently Wrong                              │
└─────────────────────────────────────────────────────────────┘

Network (trained): "Action 14 score = 0.95"
MCTS (c_puct=0.3): Extremely confident → 100% on action 14
Training: Learn "action 14 = 100%" (very reinforced!)

     ↓

┌─────────────────────────────────────────────────────────────┐
│ Iteration 19: Hopeless                                       │
└─────────────────────────────────────────────────────────────┘

Network (trained): "Action 14 score = 0.999"
MCTS (c_puct=0.3): Absolutely certain → 100% on action 14
Training: Learn "action 14 = 100%" (maximally reinforced!)

Result: 90% of training targets prefer losing moves!
```

## What SHOULD Happen with High c_puct

```
┌─────────────────────────────────────────────────────────────┐
│ MCTS with c_puct=3.0 (HIGH EXPLORATION)                      │
└─────────────────────────────────────────────────────────────┘

Network says: "Action 14 looks good (score=0.8)"

MCTS with c_puct=3.0:
  Sim 1-50:   Try action 14 → loss → Q=-1
  Sim 51-150: Try action 6 (exploration!) → game continues → Q=0.2
  Sim 151-300: Try action 12 (exploration!) → game continues → Q=0.5
  Sim 301-350: Try action 9 (exploration!) → game continues → Q=0.3
  Sim 351-400: Try action 14 again → still loses → Q=-1
  Sim 401-500: More on action 12 (best Q-value) → Q=0.5

MCTS final probabilities:
  Action 14: 100 visits, Q=-1.0  → 5% probability
  Action 6:  100 visits, Q=0.2   → 15% probability
  Action 12: 200 visits, Q=0.5   → 60% probability (best!)
  Action 9:  50 visits,  Q=0.3   → 10% probability
  Action 13: 50 visits,  Q=0.2   → 10% probability

┌──────────────────────────────────────────────────────────────┐
│ Training with Good MCTS Targets                               │
└──────────────────────────────────────────────────────────────┘

Training example:
{
  'policy_target': [0, 0, 0, 0.05, 0.15, 0.60, ...],  ← Good distribution!
  'value_target': 0.3  ← From player who made good moves
}

Network learns:
  → "Action 14 should have LOW probability (5%)"
  → "Action 12 should have HIGH probability (60%)"
  → "This is a winnable position (value=0.3)"

Next iteration:
  Network now suggests action 12 is good
  MCTS confirms it
  Virtuous cycle!
```

## Key Differences

| Aspect | c_puct = 0.3 (Bad) | c_puct = 3.0 (Good) |
|--------|-------------------|---------------------|
| **MCTS explores safe moves?** | ❌ No | ✅ Yes |
| **MCTS discovers action 14 loses?** | ⚠️ Knows but ignores (trusts network) | ✅ Yes, reflects in probabilities |
| **Policy target on action 14** | 100% | 5% |
| **Policy target on best move** | 0% | 60% |
| **Network learns** | "Prefer action 14" | "Avoid action 14" |
| **Next iteration** | Worse | Better |

## Why the Value Head Doesn't Save Us

You might think: "The value head knows position is bad (-1), won't that help?"

**No, because:**

```
Value head learns: "This exact board state → loss"

But during MCTS next iteration:
1. We're in the board state BEFORE action 14
2. Value head evaluates that position (not after action 14)
3. Value head says: "Current position is neutral (0.2)"
4. MCTS doesn't know that action 14 leads to immediate loss!

The value is for the POSITION, not for specific ACTIONS from that position.
```

## Summary: The Answer to Your Question

> "If there is losing, then isn't it be less preferred for later experiments?"

**What you expected (traditional RL):**
```
Loss → Directly decrease probability of losing action
```

**What actually happens (AlphaZero):**
```
Loss → Stored in value target
MCTS probabilities → Stored in policy target
Training → Learn to match MCTS probabilities (NOT directly from loss!)
```

**The problem:**
- With low c_puct, MCTS probabilities are bad (prefer losing moves)
- Training faithfully learns these bad probabilities
- Next iteration: Even worse

**The solution:**
- Increase c_puct → MCTS explores → finds good moves
- MCTS probabilities are good (prefer safe moves)
- Training learns good probabilities
- Next iteration: Better!

## Analogy

**Bad Teacher (c_puct=0.3):**
```
Teacher: "The answer is 42"
Student: "Are you sure?"
Teacher: "I'm 100% certain!" (but teacher is wrong)
Student learns: "The answer is 42"
Next test: Student writes 42 (wrong)
Student becomes teacher: "The answer is 42, I'm even more certain!"
```

**Good Teacher (c_puct=3.0):**
```
Teacher: "Let me check multiple sources..."
Teacher: "Source 1 says 42... but that's wrong, it leads to failure"
Teacher: "Source 2 says 17... that seems better"
Teacher: "After checking, the answer is probably 17 (60% sure)"
Student learns: "The answer is probably 17"
Next test: Student writes 17 (correct!)
Student becomes teacher: "The answer is 17, I'm very certain!"
```

The quality of what you learn depends on the quality of your teacher!
In AlphaZero, MCTS is the teacher, and c_puct determines how thorough the teacher is.

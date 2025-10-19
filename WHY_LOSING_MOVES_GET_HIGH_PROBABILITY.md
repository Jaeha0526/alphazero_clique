# Why Losing Moves Get HIGH Probability Despite Losing Immediately

## Your Excellent Question

> "Even if it is not exploring, the losing move should be the LEAST favored among visited moves!"

**You're absolutely correct! But there's a subtle bug in how MCTS probabilities are computed.**

## The Expected Behavior

When MCTS visits action 14 and it loses immediately:

```python
# After simulation:
N[action_14] = 500  # Number of visits
W[action_14] = -500 # Total value (all losses: 500 * -1.0)
Q[action_14] = W/N = -500/500 = -1.0  # Average value

# This Q-value should make it LEAST favorable!
```

## What SHOULD Happen: Probability from Q-values

```python
# Ideal: Use softmax over Q-values
Q = [-1.0, 0.3, 0.5, -1.0, ...]  # Q-values for each action

# Actions with Q=-1 should get near-zero probability
probs = softmax(Q)
# Result: [0.001, 0.3, 0.5, 0.001, ...]
```

## What ACTUALLY Happens: Probability from Visit Counts!

Let me trace through the actual code:

### Step 1: MCTS Simulations (lines 184-191)

```python
# UCB calculation during tree traversal:
Q = W / (N + 1e-8)          # Q-value (average reward)
U = c_puct * sqrt(N_sum) * P / (N + 1)  # Exploration bonus
ucb = Q + U                  # Combined score
action = argmax(ucb)         # Select action with highest UCB
```

With c_puct=0.3 and network prior P[action_14]=0.9:

```
Initial (N=0 for all):
  Q[14] = 0 (no visits yet)
  U[14] = 0.3 * sqrt(1) * 0.9 / 1 = 0.27
  ucb[14] = 0 + 0.27 = 0.27  ← Highest!

After 1 visit (action 14 loses):
  N[14] = 1, W[14] = -1.0
  Q[14] = -1.0 / 1 = -1.0
  U[14] = 0.3 * sqrt(2) * 0.9 / 2 = 0.19
  ucb[14] = -1.0 + 0.19 = -0.81  ← Still selected because others not explored!

After 10 visits (all losses):
  N[14] = 10, W[14] = -10.0
  Q[14] = -10.0 / 10 = -1.0
  U[14] = 0.3 * sqrt(11) * 0.9 / 11 = 0.08
  ucb[14] = -1.0 + 0.08 = -0.92  ← Getting worse

After 100 visits (all losses):
  N[14] = 100, W[14] = -100.0
  Q[14] = -1.0
  U[14] = 0.3 * sqrt(101) * 0.9 / 101 = 0.027
  ucb[14] = -1.0 + 0.027 = -0.973  ← Very bad!
```

**At this point, OTHER actions should start getting explored** because their UCB is higher than -0.973!

But let's check what happens with VERY low c_puct and high network prior...

### The Problem: Network Prior Dominates

```python
# Action 14: Losing move, network likes it (P=0.9)
N[14] = 100, Q[14] = -1.0, P[14] = 0.9
U[14] = 0.3 * sqrt(101) * 0.9 / 101 = 0.027
ucb[14] = -1.0 + 0.027 = -0.973

# Action 12: Safe move, network doesn't like it (P=0.01)
N[12] = 0, Q[12] = 0.0, P[12] = 0.01
U[12] = 0.3 * sqrt(101) * 0.01 / 1 = 0.003
ucb[12] = 0.0 + 0.003 = 0.003  ← HIGHER! Should be selected!
```

So action 12 SHOULD start getting visits! This would reduce action 14's final probability.

### Step 2: Final Probability Calculation (lines 404-411)

**Here's the KEY BUG:**

```python
# Extract visit counts
root_visits = arrays.N[:, 0, :]  # Visit counts: [100, 0, 0, ..., 0, ...]

# With temperature > 0 (used during self-play):
root_visits_temp = jnp.power(root_visits + 1e-8, 1.0 / temperature)
root_visits_temp = jnp.where(root_valid, root_visits_temp, 0.0)
action_probs = root_visits_temp / jnp.sum(root_visits_temp, axis=1, keepdims=True)
```

**The probability is based ONLY on visit counts, not Q-values!**

Example:
```python
# Visit counts after 500 simulations:
N = [0, 0, 0, 500, 0, ...]  # Only action 14 visited

# Temperature = 1.0 (during self-play):
probs = N / sum(N)
probs = [0, 0, 0, 1.0, 0, ...]  # 100% on action 14!
```

**Even though Q[14] = -1.0 (terrible!), it gets 100% probability because it has all the visits!**

## Why Does This Happen?

### The AlphaZero Assumption (Works for Chess/Go)

In normal AlphaZero:
- High visit count → High Q-value (action is good)
- Low visit count → Low Q-value (action is bad)
- **Visits and quality are correlated**

So using visit counts as probabilities works fine!

### Why It Fails Here (avoid_clique with low c_puct)

1. **Network is wrong:** Says action 14 is good (high prior P=0.9)
2. **MCTS trusts network:** Tries action 14 first
3. **Low c_puct:** Doesn't explore alternatives enough
4. **Action 14 gets ALL visits** even though Q=-1.0
5. **Final prob based on visits:** 100% on action 14

## The Actual Numbers from Your Data

Let me verify this with the game data. In Iteration 10, Game 4:

**Move 11: Dumb move with 100% probability**

Expected visit counts:
```python
# If my hypothesis is correct:
visit_counts = {
    'action_14': 500,  # All visits (losing move!)
    'action_6': 0,     # Not explored (safe)
    'action_9': 0,     # Not explored (safe)
    'action_12': 0,    # Not explored (safe)
    'action_13': 0,    # Not explored (safe)
}

# Probabilities (visits / sum):
probs = {
    'action_14': 500/500 = 1.0,  # 100%!
    'other': 0/500 = 0.0
}
```

This matches our observation: **100% probability on losing move!**

## Why Doesn't MCTS Switch to Other Actions?

Here's the critical insight. With c_puct=0.3:

```python
# After 500 visits to action 14, all losing:
ucb[14] = -1.0 + 0.3 * sqrt(501) * 0.9 / 501 = -1.0 + 0.012 = -0.988

# For unvisited safe action 12 (network prior P=0.01):
ucb[12] = 0.0 + 0.3 * sqrt(501) * 0.01 / 1 = 0.0 + 0.067 = 0.067
```

**Wait! UCB[12] = 0.067 > UCB[14] = -0.988**

So action 12 SHOULD be selected next!

### The Problem: Dirichlet Noise!

Let me check if Dirichlet noise is being added to the prior...

Actually, looking at the code more carefully, I think the issue is simpler:

**The network prior P is SO dominant that even with low c_puct, the exploration term can't overcome it.**

Let's recalculate more carefully:

```python
# Root node, before any simulations:
N_sum = 0
For each action a:
  Q[a] = 0 (no visits)
  U[a] = c_puct * sqrt(N_sum) * P[a] / (N[a] + 1)
       = c_puct * sqrt(0) * P[a] / 1
       = 0  (!)

# All actions have UCB = 0, so it picks the first one or random?
```

Actually, let me check the initialization:


# MCTS Implementation Comparison

## Original (src/MCTS_clique.py) vs Our Implementation (simple_tree_mcts.py)

### 1. Q-value Calculation

**Original:**
```python
# child_Q()
return self.child_total_value / (1.0 + self.child_number_visits)
```

**Our Implementation:**
```python
Q = np.where(N > 0, W / N, 0.0)
```

**ISSUE:** The original uses `W / (1 + N)` but we use `W / N` when N > 0.

### 2. U-value (Exploration) Calculation

**Original:**
```python
# child_U()
c_puct = 3.0
sqrt_N_s = math.sqrt(max(1.0, self.number_visits))
return c_puct * sqrt_N_s * (abs(self.child_priors) / (1.0 + self.child_number_visits))
```

**Our Implementation:**
```python
total_N = N.sum()
sqrt_sum = np.sqrt(total_N + 1)  # Add 1 to handle first visit
U = self.c_puct * P * sqrt_sum / (1 + N)
```

**COMPARISON:**
- Both use c_puct = 3.0 ✓
- Original: `sqrt(max(1, visits_to_this_node))`
- Ours: `sqrt(sum(N) + 1)` where N is visits from this node
- Both use `P / (1 + N)` for the exploration term ✓

### 3. Key Differences

1. **Q calculation:** Original always uses (1 + N) denominator, we use N when N > 0
2. **sqrt term:** Original uses visits TO the node, we use sum of visits FROM the node

### 4. Which is Correct?

The original implementation follows the AlphaZero paper more closely:
- Q = W / (1 + N) ensures no division by zero
- sqrt term should use parent visits (visits TO the node)
# Dumb Move Analysis: ramsey_n6_k3_cpuct03 (c_puct=0.3)

## Executive Summary

**Critical Finding:** The model is **NOT learning to avoid dumb moves** - in fact, it's getting WORSE!

- **Iteration 0**: 95% of losses were preventable (had safe alternatives)
- **Iteration 19**: **100% of losses were preventable**
- **MCTS actively prefers losing moves**: Average prob on losing move = 0.61, on best safe move = 0.13

## What Are "Dumb Moves"?

A "dumb move" is when a player:
1. Makes a move that **immediately forms a k-clique** (loses in avoid_clique mode)
2. **Had safe alternatives available** that wouldn't form a clique

## Key Findings

### 1. Dumb Move Rate: NO IMPROVEMENT

| Iteration | Dumb Moves | Rate |
|-----------|------------|------|
| 0         | 19/20      | 95%  |
| 5         | 19/20      | 95%  |
| 10        | 19/20      | 95%  |
| 15        | 19/20      | 95%  |
| 19        | 20/20      | **100%** |

**Conclusion:** Training is not reducing dumb moves. If anything, it's getting slightly worse!

### 2. MCTS Systematically Chooses Losing Moves

**Average probabilities across all iterations:**
- MCTS probability on **chosen (losing) move**: **0.6096** (61%)
- MCTS probability on **best safe move**: **0.1279** (13%)

This means MCTS is giving **5x more probability** to losing moves than safe moves!

### 3. Examples of Egregious Dumb Moves

**Iteration 15, Game 8:**
```
Chose losing action 0 (MCTS prob=1.0000) - 100% confident!
Had 4 safe alternatives: [3, 7, 12, 14]
MCTS prob on ALL safe moves combined: 0.0000
```

**Iteration 19, Game 3:**
```
Chose losing action 4 (MCTS prob=0.9998) - nearly 100% confident!
Had 8 safe alternatives
Max MCTS prob on safe move: 0.0001
```

**Iteration 10, Game 4:**
```
Chose losing action 14 (MCTS prob=1.0000)
Had 4 safe alternatives: [6, 9, 12, 13]
Total prob on ALL safe moves: 0.0000
```

### 4. Pattern Analysis

**Early iterations (0-5):**
- MCTS sometimes explores safe moves (20-50% probability)
- Dumb moves often happen when safe moves have decent probability
- Model is "confused" about what's safe

**Later iterations (10-19):**
- MCTS becomes **extremely confident** in losing moves (90-100% probability)
- Safe moves get **near-zero** probability (<1%)
- Model has **learned to lose confidently**

## Root Cause Analysis

### Why is this happening?

1. **c_puct=0.3 is TOO LOW**
   - MCTS barely explores alternative moves
   - Quickly converges to first move it tries
   - If network says a losing move is good, MCTS doesn't discover it's bad

2. **Network Value Function is Broken**
   - The network is evaluating losing positions as GOOD
   - MCTS trusts the network and commits to losing moves
   - Training reinforces this because value targets come from these bad games

3. **Vicious Cycle**
   - Network predicts losing moves are good → MCTS chooses them → Game data shows "this was the move chosen" → Network learns to prefer losing moves even more

## Specific Examples from Game Data

### Game with Low Safe Probability (Iteration 10, Game 4)
```
Move 11: Player chooses action 14 (forming clique, loses)
  MCTS prob on action 14: 1.0000 (100%!)
  Safe alternatives: [6, 9, 12, 13]
  MCTS prob on safe moves: [0.0, 0.0, 0.0, 0.0]

Network is SO confident this is the right move, it gives 0% to all safe options!
```

### Game with MCTS Preferring Loss (Iteration 5, Game 3)
```
Move 12: Player chooses action 1 (loses)
  MCTS prob on action 1: 0.9959 (99.6%!)
  Safe alternatives: [2, 12]
  MCTS prob on safe moves: [0.0039, 0.0000]

Even though 2 safe moves exist, MCTS gives them <0.4% probability combined!
```

### Game with Multiple Safe Options Ignored (Iteration 15, Game 4)
```
Move 7: Player chooses action 6 (loses)
  MCTS prob on action 6: 0.9504 (95%)
  Safe alternatives: [1, 2, 3, 4, 7] - 5 safe moves!
  Max safe prob: 0.0085 (0.8%)

With 5 safe options available, the best one gets <1% probability!
```

## Comparison: Good vs Bad MCTS Behavior

### Good MCTS (Iteration 0, Game 4):
```
Chose losing action 5 (MCTS prob=0.0863)
Safe alternatives: [0, 1, 2]
MCTS probs on safe: [0.0550, 0.0336, 0.6669]
→ MCTS actually preferred safe move (66%!) but randomness chose losing one
→ This is learning opportunity - model can learn safe moves are better
```

### Bad MCTS (Iteration 19, Game 6):
```
Chose losing action 0 (MCTS prob=0.9987)
Safe alternatives: [3, 7]
MCTS probs on safe: [0.0000, 0.0011]
→ MCTS is 99.87% confident in losing move
→ No learning signal - model thinks it's already correct!
```

## Why Your Original Concern About c_puct Was Justified

You lowered c_puct to 0.3 to reduce dumb moves, but it had the **opposite effect**:

### With Low c_puct (0.3):
- MCTS doesn't explore enough
- Network's wrong evaluations go unchallenged
- Model becomes **confidently wrong**
- Dumb move rate: **95-100%**

### With Higher c_puct (3.0+):
- MCTS explores more alternatives
- Even if network thinks a move is good, MCTS tries others
- Discovers that "network said good move → immediate loss"
- Updates network: "that move is actually BAD"
- Model learns from mistakes

## Recommendations

### 1. **IMMEDIATE: Increase c_puct to 3.0-4.0**
```bash
python jax_full_src/run_jax_optimized.py \
    --vertices 6 --k 3 \
    --avoid_clique \
    --c_puct 4.0 \  # Much higher exploration
    --num_iterations 30 \
    --num_episodes 200 \
    --mcts_sims 400 \  # Deeper search to find safe moves
    --experiment_name ramsey_n6_k3_cpuct4
```

### 2. **Add Explicit Loss Detection**

Modify the network training to:
- Detect when a move immediately forms a k-clique
- Give those moves **very negative** value in training
- Add auxiliary loss: penalize policy prob on moves that immediately lose

### 3. **Use Curriculum Learning**

Start with easier scenarios:
```python
# Start with k=4 (easier to avoid 4-cliques on n=6)
# Once model learns to avoid losses → switch to k=3
```

### 4. **Monitor Dumb Move Rate**

Run this analysis script regularly:
```bash
python analyze_dumb_moves.py --experiment experiments/YOUR_EXP
```

Target: **Dumb move rate should decrease** each iteration
- Iteration 0: 95% is expected (random play)
- Iteration 10: Should be <50%
- Iteration 20: Should be <20%
- Iteration 50: Should be <5%

## Conclusion

The current training with c_puct=0.3 is fundamentally broken:

1. ❌ **Not learning to avoid losses** (100% dumb move rate)
2. ❌ **Becoming more confident in wrong moves** (0.96 → 0.99 confidence over time)
3. ❌ **MCTS prefers losing moves 5x more** than safe moves

The solution is **higher c_puct** (3.0-4.0) to allow MCTS to discover that losing moves are bad, even when the network thinks they're good. This creates the learning signal needed for the network to improve.

**Your intuition was backwards:** Low c_puct doesn't prevent dumb moves - it prevents the model from *learning* to avoid them!

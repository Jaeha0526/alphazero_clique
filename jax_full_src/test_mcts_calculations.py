#!/usr/bin/env python
"""Compare MCTS calculations between original and our implementation."""

import numpy as np
import math

print("=== MCTS Calculation Comparison ===\n")

# Test scenario
N = np.array([0, 2, 1, 0])  # Visit counts for 4 actions
W = np.array([0, 1.5, -0.5, 0])  # Total values
P = np.array([0.25, 0.25, 0.25, 0.25])  # Priors
node_visits = 4  # Total visits to this node (sum of N + 1)
c_puct = 3.0

print(f"N (visits from node): {N}")
print(f"W (total values): {W}")
print(f"P (priors): {P}")
print(f"Node visits (to node): {node_visits}")
print(f"c_puct: {c_puct}")

print("\n--- Original AlphaZero Calculation ---")
# Original uses W/(1+N) for Q
Q_orig = W / (1.0 + N)
print(f"Q = W/(1+N) = {Q_orig}")

# Original uses sqrt(visits_to_node) * P/(1+N) for U
sqrt_visits = math.sqrt(max(1.0, node_visits))
U_orig = c_puct * sqrt_visits * (P / (1.0 + N))
print(f"U = {c_puct} * sqrt({node_visits}) * P/(1+N) = {U_orig}")

UCB_orig = Q_orig + U_orig
print(f"UCB = Q + U = {UCB_orig}")
print(f"Best action: {np.argmax(UCB_orig)}")

print("\n--- Our Implementation ---")
# We now use the same formulas
Q_ours = W / (1.0 + N)
print(f"Q = W/(1+N) = {Q_ours}")

sqrt_visits_ours = np.sqrt(max(1.0, node_visits))
U_ours = c_puct * sqrt_visits_ours * (P / (1.0 + N))
print(f"U = {c_puct} * sqrt({node_visits}) * P/(1+N) = {U_ours}")

UCB_ours = Q_ours + U_ours
print(f"UCB = Q + U = {UCB_ours}")
print(f"Best action: {np.argmax(UCB_ours)}")

print("\n--- Comparison ---")
print(f"Q match: {np.allclose(Q_orig, Q_ours)}")
print(f"U match: {np.allclose(U_orig, U_ours)}")
print(f"UCB match: {np.allclose(UCB_orig, UCB_ours)}")
print(f"Best action match: {np.argmax(UCB_orig) == np.argmax(UCB_ours)}")

# Test edge case: first visit
print("\n--- Edge Case: First Visit (all N=0) ---")
N_first = np.zeros(4)
W_first = np.zeros(4)

Q_first = W_first / (1.0 + N_first)
print(f"Q = {Q_first}")

# With node_visits = 1 (first time visiting)
U_first = c_puct * 1.0 * (P / (1.0 + N_first))
print(f"U = {U_first}")

UCB_first = Q_first + U_first
print(f"UCB = {UCB_first}")
print(f"All actions have UCB = {UCB_first[0]:.4f} (uniform exploration)")
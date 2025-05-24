#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLARIFICATION: Where do the 36 edges come from?

The confusion is: 
- Training data has 15 policy TARGETS
- But GNN processes 36 edge INPUTS

These are DIFFERENT things!
"""

print("=" * 80)
print("WHERE THE 36 EDGES COME FROM vs WHERE THE 15 TARGETS COME FROM")
print("=" * 80)
print()

print("PART 1: BOARD STATE -> 36 EDGE INPUTS")
print("-" * 50)
print("The 36 edges come from the BOARD STATE, not from policy targets!")
print()

print("Example board state (6x6 matrix):")
board_example = """
    0  1  2  3  4  5
0 [ 0  1  0  2  0  0 ]   # 0=unselected, 1=player1, 2=player2
1 [ 1  0  1  0  0  0 ]
2 [ 0  1  0  0  0  1 ]
3 [ 2  0  0  0  1  0 ]
4 [ 0  0  0  1  0  0 ]
5 [ 0  0  1  0  0  0 ]
"""
print(board_example)

print("From this board state, _board_to_graph() creates:")
print()

print("1. 30 BIDIRECTIONAL EDGES (from 15 undirected edges):")
edges_list = []
for i in range(6):
    for j in range(i+1, 6):
        edges_list.append(f"({i},{j})")
        edges_list.append(f"({j},{i})")

print("   Undirected edges: " + ", ".join(edges_list[::2]))  # Every other one
print("   Forward:  " + ", ".join(edges_list[::2]))
print("   Reverse:  " + ", ".join(edges_list[1::2]))
print()

print("2. 6 SELF-LOOPS:")
self_loops = [f"({i},{i})" for i in range(6)]
print("   Self-loops: " + ", ".join(self_loops))
print()

print("3. EDGE FEATURES (from board state):")
print("   Each edge gets [unselected, player1, player2] features")
print("   Examples:")
print("   - Edge (0,1): state=1 -> [0, 1, 0]  (player1 selected)")
print("   - Edge (1,0): state=1 -> [0, 1, 0]  (same as (0,1))")
print("   - Edge (0,2): state=0 -> [1, 0, 0]  (unselected)")
print("   - Edge (0,0): self-loop -> [1, 0, 0]  (special feature)")
print()

print("TOTAL: 30 + 6 = 36 edges with features")
print("These 36 edges are the INPUT to the GNN!")
print()

print("=" * 80)
print()

print("PART 2: POLICY TARGETS (15 values)")
print("-" * 50)
print("The 15 policy targets are SEPARATE - they're the training labels!")
print()

print("Policy targets represent probabilities for 15 UNDIRECTED edges:")
policy_targets = [
    "policy[0] = 0.1   # edge (0,1)",
    "policy[1] = 0.05  # edge (0,2)", 
    "policy[2] = 0.2   # edge (0,3)",
    "policy[3] = 0.0   # edge (0,4)",
    "policy[4] = 0.15  # edge (0,5)",
    "policy[5] = 0.3   # edge (1,2)",
    "policy[6] = 0.1   # edge (1,3)",
    "policy[7] = 0.0   # edge (1,4)",
    "policy[8] = 0.05  # edge (1,5)",
    "policy[9] = 0.0   # edge (2,3)",
    "policy[10] = 0.0  # edge (2,4)",
    "policy[11] = 0.05 # edge (2,5)",
    "policy[12] = 0.0  # edge (3,4)",
    "policy[13] = 0.0  # edge (3,5)",
    "policy[14] = 0.0  # edge (4,5)"
]

for target in policy_targets:
    print(f"   {target}")
print()

print("These are the TRAINING TARGETS (what model should predict)")
print()

print("=" * 80)
print()

print("THE DATA FLOW:")
print("-" * 50)
print("1. Board state -> 36 edges with features (GNN INPUT)")
print("2. GNN processes 36 edges -> 36 learned edge features")  
print("3. Policy head: 36 edge features -> 36 edge scores")
print("4. Mapping: 36 edge scores -> 15 policy outputs")
print("5. Loss: 15 policy outputs vs 15 policy targets")
print()

print("THE PROBLEM:")
print("-" * 50)
print("- INPUT: 36 edges (all have meaningful features from board state)")
print("- OUTPUT: 15 actions (what we want to predict)")
print("- MISMATCH: 36 to 15 mapping is lossy and inconsistent!")
print()

print("ANSWER TO YOUR QUESTION:")
print("- NO, they don't pad with zeros for missing edges")
print("- YES, all 36 edges have real features from the board state")  
print("- BUT only 15 of them matter for the final policy prediction")
print("- The other 21 edges (reverse directions + self-loops) are wasted!") 
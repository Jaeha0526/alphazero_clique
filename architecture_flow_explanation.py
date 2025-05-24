# -*- coding: utf-8 -*-
"""
DETAILED FLOW ANALYSIS: 15-Action Input to 36-Edge Architecture to 15-Action Output

This explains why the training is not working properly.
"""

# ======================================
# STEP 1: DATA LOADING (15 actions)
# ======================================
print("STEP 1: TRAINING DATA")
print("- Saved policy shape: (15,) for 6-node graph")
print("- Represents 15 undirected edges: (0,1), (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)")
print("- Example: [0.1, 0.05, 0.2, 0.0, 0.15, ...]  # 15 probabilities")
print()

# ======================================
# STEP 2: GRAPH CREATION (36 edges)
# ======================================
print("STEP 2: _board_to_graph() CREATES 36 EDGES")
print("Lines 157-178 in alpha_net_clique.py:")
print("""
# For 6 nodes, creates:
# 1. 30 bidirectional edges:
for i in range(6):
    for j in range(i + 1, 6):
        edge_index.append([i, j])  # Forward direction
        edge_index.append([j, i])  # Reverse direction
        # Results in: (0->1), (1->0), (0->2), (2->0), ..., (4->5), (5->4)

# 2. 6 self-loops:
for i in range(6):
    edge_index.append([i, i])  # Self-loop
    # Results in: (0->0), (1->1), (2->2), (3->3), (4->4), (5->5)

# Total: 30 + 6 = 36 edges
""")
print()

# ======================================
# STEP 3: GNN PROCESSING (36 edges)
# ======================================
print("STEP 3: GNN LAYERS PROCESS ALL 36 EDGES")
print("Lines 434-485 in forward():")
print("""
# Each GNN layer processes ALL 36 edges:
for node_layer, edge_layer in zip(self.node_layers, self.edge_layers):
    x = node_layer(x, edge_index, edge_features)         # Uses all 36 edges
    edge_features = edge_layer(x, edge_index, edge_features)  # Updates all 36 edge features

# Policy head processes all 36 edge features:
edge_scores = self.policy_head(edge_features)  # Shape: (36, 1)
""")
print()

# ======================================
# STEP 4: MAPPING BACK TO 15 (LOSSY!)
# ======================================
print("STEP 4: MAPPING 36 EDGE SCORES TO 15 POLICY OUTPUTS")
print("Lines 521-560 in forward() - THE PROBLEMATIC PART:")
print("""
# This is where information is LOST!

# 1. Loop through all 36 edges and their scores
canonical_scores = {}
for j in range(36):  # Process all 36 edge scores
    src = edge_index[0, j] 
    dst = edge_index[1, j]
    
    if src != dst:  # Skip self-loops (loses 6 edges)
        canonical_edge = tuple(sorted((src, dst)))  # Convert (1,0) to (0,1)
        if canonical_edge not in canonical_scores:
            canonical_scores[canonical_edge] = edge_scores[j]  # Keep first occurrence only!

# 2. Map to 15-dimensional policy vector
policy = torch.zeros(15)
for edge_tuple, edge_idx in edge_map.items():
    canonical_edge = tuple(sorted(edge_tuple))
    if edge_tuple == canonical_edge:  # Only process canonical form
        score = canonical_scores.get(canonical_edge, 0.0)
        policy[edge_idx] = score  # Assign to one of 15 positions
""")
print()

# ======================================
# STEP 5: LOSS CALCULATION MISMATCH
# ======================================
print("STEP 5: LOSS CALCULATION - DIMENSION MISMATCH")
print("Lines 729-738 in train_network():")
print("""
# Training targets: 15 values
policy_target = batch_data.policy  # Shape: (batch_size x 15)

# Model output: 15 values  
policy_output = model(...)          # Shape: (batch_size x 15)

# Loss calculation:
loss = -policy_target * log(policy_output)  # Works dimensionally but...
""")
print()

# ======================================
# THE CORE PROBLEMS
# ======================================
print("*** WHY THIS ARCHITECTURE BREAKS TRAINING ***")
print("""
1. INFORMATION LOSS:
   - GNN processes 36 edges but only 15 matter for the loss
   - 21 edge features (6 self-loops + 15 reverse edges) are computed but ignored
   - Wasted computation and lost gradients

2. INCONSISTENT MAPPING:
   - Edge (0->1) and edge (1->0) have different learned features
   - But they map to the same policy index  
   - The model can't learn which direction is important

3. GRADIENT FLOW PROBLEMS:
   - Only 15 out of 36 edge scores receive gradient updates
   - 21 edges learn random/useless features
   - Self-loop features never contribute to policy loss

4. TRAINING INSTABILITY:
   - The mapping from 36 to 15 is not learnable
   - Model fights between bidirectional edge representations
   - Inconsistent target supervision

5. ARCHITECTURAL MISMATCH:
   - Data format (15 actions) != Model format (36 edges)
   - Forces complex, lossy mapping instead of clean design
""")
print()

print("*** SOLUTION ***")
print("""
Either:
A) Design model to work with 15 edges directly (cleaner)
B) Change data format to 36 edge targets (more complex)
C) Use proper edge aggregation instead of first-occurrence mapping

The current approach tries to have both and succeeds at neither!
""")

# Example of the mapping issue
print("\n*** CONCRETE EXAMPLE ***")
print("Training target: [0.1, 0.05, 0.2, 0.0, 0.15, 0.3, 0.1, 0.0, 0.05, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0]")
print("                  ^                             ^")
print("              edge (0,1)                   edge (1,5)")
print()
print("Internal GNN learns:")
print("  edge_scores[0] = 0.8   # (0->1)")  
print("  edge_scores[1] = 0.3   # (1->0)  <- Different from (0->1)!")
print("  edge_scores[15] = 0.9  # (1->5)")
print("  edge_scores[16] = 0.1  # (5->1)  <- Different from (1->5)!")
print()
print("Mapping takes ONLY FIRST occurrence:")
print("  policy[0] = edge_scores[0] = 0.8   # But target is 0.1!")
print("  policy[9] = edge_scores[15] = 0.9  # But target is 0.05!")
print()
print("Result: Model learns wrong associations and can't converge!") 
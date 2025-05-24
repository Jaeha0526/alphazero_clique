import pickle
import numpy as np

# Load datasets
data1 = pickle.load(open('experiments/improved_n4k3_h16_l1_mcts50_20250518_232024/datasets/game_20250518_232034_cpu1_game0_iter0.pkl', 'rb'))
data2 = pickle.load(open('experiments/improved_n6k3_h128_l3_mcts1000_20250518_222614/datasets/game_20250518_222640_cpu2_game0_iter0.pkl', 'rb'))

print("=== COMPREHENSIVE DATA ANALYSIS ===")
print()

# Values analysis
print("1. VALUES:")
print(f"Dataset 1 (n4k3): {[d['value'] for d in data1]}")
print(f"Dataset 2 (n6k3): {[d['value'] for d in data2]}")
print()

# Edge attributes analysis
print("2. EDGE_ATTR:")
edge_attrs1 = [d['board_state']['edge_attr'] for d in data1]
edge_attrs2 = [d['board_state']['edge_attr'] for d in data2]
all_edge_vals1 = np.concatenate([ea.flatten() for ea in edge_attrs1])
all_edge_vals2 = np.concatenate([ea.flatten() for ea in edge_attrs2])
print(f"Dataset 1 edge_attr unique: {np.unique(all_edge_vals1)}")
print(f"Dataset 2 edge_attr unique: {np.unique(all_edge_vals2)}")
print()

# Policy analysis
print("3. POLICY VALUES:")
policies1 = [d['policy'] for d in data1]
policies2 = [d['policy'] for d in data2]

print("Dataset 1 policy analysis:")
for i, p in enumerate(policies1):
    zeros = np.sum(p == 0.0)
    ones = np.sum(p == 1.0)
    print(f"  State {i}: {zeros} zeros, {ones} ones, range=[{np.min(p):.3f}, {np.max(p):.3f}]")

print()
print("Dataset 2 policy analysis:")
for i, p in enumerate(policies2):
    zeros = np.sum(p == 0.0)
    ones = np.sum(p == 1.0)
    print(f"  State {i}: {zeros} zeros, {ones} ones, range=[{np.min(p):.3f}, {np.max(p):.3f}]")

print()
print("4. SUMMARY:")
print("Components that contain both 0s and 1s:")
print("- edge_attr: YES (both datasets)")
print("- policy: Dataset 1 YES, Dataset 2 NO (only 0s, no exact 1s)")
print("- value: NO (Dataset 1 only 0s, Dataset 2 only 1s)") 
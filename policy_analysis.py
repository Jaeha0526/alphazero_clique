import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
data1 = pickle.load(open('experiments/improved_n4k3_h16_l1_mcts50_20250518_232024/datasets/game_20250518_232034_cpu1_game0_iter0.pkl', 'rb'))
data2 = pickle.load(open('experiments/improved_n6k3_h128_l3_mcts1000_20250518_222614/datasets/game_20250518_222640_cpu2_game0_iter0.pkl', 'rb'))

print("=== POLICY PATTERN ANALYSIS ===")
print()

def analyze_policy_progression(data, dataset_name):
    print(f"=== {dataset_name} ===")
    policies = [d['policy'] for d in data]
    
    print("Policy progression:")
    for i, policy in enumerate(policies):
        print(f"State {i}: {policy}")
    
    print()
    print("Action availability over time:")
    num_actions = len(policies[0])
    for action in range(num_actions):
        action_probs = [p[action] for p in policies]
        print(f"Action {action}: {action_probs}")
    
    print()
    print("Policy entropy over time:")
    entropies = []
    for i, policy in enumerate(policies):
        # Add small epsilon to avoid log(0)
        entropy = -np.sum(policy * np.log(policy + 1e-10))
        entropies.append(entropy)
        print(f"State {i}: {entropy:.3f}")
    
    print()
    print("Actions that become unavailable (go to 0):")
    for action in range(num_actions):
        states_zero = []
        for i, policy in enumerate(policies):
            if policy[action] == 0.0:
                states_zero.append(i)
        if states_zero:
            print(f"Action {action}: becomes 0 at states {states_zero}")
    
    print()
    print("Most probable action per state:")
    for i, policy in enumerate(policies):
        best_action = np.argmax(policy)
        best_prob = policy[best_action]
        print(f"State {i}: Action {best_action} ({best_prob:.3f})")
    
    print()
    return policies, entropies

policies1, entropies1 = analyze_policy_progression(data1, "DATASET 1 (n4k3)")
print("="*50)
policies2, entropies2 = analyze_policy_progression(data2, "DATASET 2 (n6k3)")

print("="*50)
print("PATTERN OBSERVATIONS:")
print()

print("1. Action Masking Pattern:")
print("   - Actions progressively become unavailable (set to 0)")
print("   - Suggests invalid moves get masked as game progresses")
print()

print("2. Entropy Trend:")
print(f"   Dataset 1: {entropies1[0]:.3f} → {entropies1[-1]:.3f}")
print(f"   Dataset 2: {entropies2[0]:.3f} → {entropies2[-1]:.3f}")
print("   - Both show decreasing uncertainty over time")
print()

print("3. Action Space Reduction:")
print(f"   Dataset 1: {len([p for p in policies1[0] if p > 0])} → {len([p for p in policies1[-1] if p > 0])} available actions")
print(f"   Dataset 2: {len([p for p in policies2[0] if p > 0])} → {len([p for p in policies2[-1] if p > 0])} available actions")
print()

print("4. Convergence Pattern:")
final_action1 = np.argmax(policies1[-1])
final_action2 = np.argmax(policies2[-1])
print(f"   Dataset 1: Converges to action {final_action1} ({policies1[-1][final_action1]:.3f})")
print(f"   Dataset 2: Converges to action {final_action2} ({policies2[-1][final_action2]:.3f})") 
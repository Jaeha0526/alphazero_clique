import pickle
import numpy as np

# Load datasets
data1 = pickle.load(open('experiments/improved_n4k3_h16_l1_mcts50_20250518_232024/datasets/game_20250518_232034_cpu1_game0_iter0.pkl', 'rb'))
data2 = pickle.load(open('experiments/improved_n6k3_h128_l3_mcts1000_20250518_222614/datasets/game_20250518_222640_cpu2_game0_iter0.pkl', 'rb'))

print("=== ACTION SEQUENCE PATTERN ANALYSIS ===")
print()

def analyze_action_sequences(data, dataset_name):
    print(f"=== {dataset_name} ===")
    policies = [d['policy'] for d in data]
    
    # Track the most probable action at each step
    action_sequence = [np.argmax(p) for p in policies]
    print(f"Action sequence (most probable): {action_sequence}")
    
    # Track when actions become permanently unavailable
    num_actions = len(policies[0])
    action_death_state = {}
    
    for action in range(num_actions):
        for state in range(len(policies)):
            if policies[state][action] == 0.0:
                if action not in action_death_state:
                    action_death_state[action] = state
                break
    
    print(f"Action elimination order:")
    for action, death_state in sorted(action_death_state.items(), key=lambda x: x[1]):
        print(f"  Action {action} eliminated at state {death_state}")
    
    # Check if there's a pattern: actions taken get eliminated
    print(f"Pattern check - Actions taken vs eliminated:")
    for i, taken_action in enumerate(action_sequence[:-1]):  # Exclude last state
        if taken_action in action_death_state:
            death_state = action_death_state[taken_action]
            next_state = i + 1
            if death_state == next_state:
                print(f"  ✓ Action {taken_action} taken at state {i}, eliminated at state {death_state}")
            else:
                print(f"  ✗ Action {taken_action} taken at state {i}, but eliminated at state {death_state}")
        else:
            print(f"  ? Action {taken_action} taken at state {i}, never eliminated")
    
    print()
    return action_sequence, action_death_state

seq1, deaths1 = analyze_action_sequences(data1, "DATASET 1 (n4k3)")
seq2, deaths2 = analyze_action_sequences(data2, "DATASET 2 (n6k3)")

print("="*60)
print("DISCOVERED PATTERNS:")
print()

print("1. ACTION SELECTION PATTERN:")
print("   - Each state seems to prefer a different action")
print("   - Actions are selected in a specific sequence")
print()

print("2. ACTION ELIMINATION PATTERN:")
print("   - Actions often become unavailable after being preferred")
print("   - Suggests 'once taken, action becomes invalid'")
print()

print("3. GAME PROGRESSION:")
print("   - Dataset 1: Progressive elimination until only action 5 remains")
print("   - Dataset 2: More complex, but similar elimination pattern")
print()

print("4. POSSIBLE INTERPRETATION:")
print("   - This could be a graph construction game")
print("   - Each action might represent adding an edge to the graph")
print("   - Once an edge is added, that action becomes invalid")
print("   - Goal might be to form a clique")
print("   - Action space shrinks as graph gets constructed") 
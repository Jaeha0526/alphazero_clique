#!/usr/bin/env python3
"""
Verify what the network is actually learning from the training data.
Check if policy targets match the dumb moves we identified.
"""
import pickle
import numpy as np
import itertools

def check_move_creates_clique(edge_states, action, player, k=3, num_vertices=6):
    """Check if a move immediately creates a k-clique."""
    new_edge_states = np.array(edge_states, copy=True)
    new_edge_states[action] = player + 1

    adj = np.zeros((num_vertices, num_vertices), dtype=int)
    edge_idx = 0
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if new_edge_states[edge_idx] == player + 1:
                adj[i, j] = 1
                adj[j, i] = 1
            edge_idx += 1

    for clique in itertools.combinations(range(num_vertices), k):
        is_clique = True
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                if adj[clique[i], clique[j]] == 0:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            return True
    return False

def analyze_training_targets(iter_num, base_path='experiments/ramsey_n6_k3_cpuct03'):
    """Analyze what the network is being trained to learn."""
    path = f'{base_path}/game_data/iteration_{iter_num}.pkl'

    with open(path, 'rb') as f:
        data = pickle.load(f)

    training_data = data.get('training_data', [])
    games_info = data.get('games_info', [])
    k = data.get('k', 3)
    num_vertices = data.get('vertices', 6)

    print(f"\n{'='*70}")
    print(f"ITERATION {iter_num} - Training Target Analysis")
    print(f"{'='*70}")

    # Analyze first 5 games
    dumb_move_examples = []

    for game_idx, game_info in enumerate(games_info[:5]):
        start_idx = game_info['start_idx']
        end_idx = game_info['end_idx']
        winner = game_info['winner']

        if winner == -1:
            continue

        game_moves = training_data[start_idx:end_idx]

        # Reconstruct game
        num_edges = num_vertices * (num_vertices - 1) // 2
        edge_states = np.zeros(num_edges, dtype=np.int32)

        for move_idx, move_data in enumerate(game_moves):
            player = int(move_data['player'])
            action = int(move_data['action'])
            policy_target = np.array(move_data['policy'])
            value_target = move_data['value']

            # Check if this is a dumb move
            creates_clique = check_move_creates_clique(edge_states, action, player, k, num_vertices)

            if creates_clique:
                # Find safe alternatives
                safe_moves = []
                for a in range(num_edges):
                    if edge_states[a] == 0:
                        if not check_move_creates_clique(edge_states, a, player, k, num_vertices):
                            safe_moves.append(a)

                if len(safe_moves) > 0:
                    # This is a dumb move!
                    dumb_move_examples.append({
                        'game': game_idx + 1,
                        'move': move_idx + 1,
                        'action': action,
                        'policy_target': policy_target,
                        'value_target': value_target,
                        'safe_moves': safe_moves,
                        'player': player
                    })
                break

            edge_states[action] = player + 1

    # Print analysis
    if dumb_move_examples:
        print(f"\nFound {len(dumb_move_examples)} dumb moves in first 5 games")
        print(f"\n{'='*70}")
        print("WHAT THE NETWORK IS LEARNING:")
        print(f"{'='*70}")

        for ex in dumb_move_examples[:3]:  # Show first 3
            print(f"\nGame {ex['game']}, Move {ex['move']}:")
            print(f"  Player {ex['player']} chose LOSING action {ex['action']}")
            print(f"  Safe alternatives: {ex['safe_moves'][:5]}")
            print(f"\n  TRAINING TARGETS:")
            print(f"    Value target: {ex['value_target']:.2f}")
            print(f"      → Network learns: 'This position leads to {ex['value_target']:+.0f} outcome'")

            print(f"\n    Policy target (what network is trained to output):")
            policy = ex['policy_target']

            # Show prob on losing action
            print(f"      Losing action {ex['action']}: {policy[ex['action']]:.4f}")

            # Show prob on safe actions
            if ex['safe_moves']:
                safe_probs = [policy[a] for a in ex['safe_moves'][:5]]
                print(f"      Safe actions {ex['safe_moves'][:5]}: {[f'{p:.4f}' for p in safe_probs]}")

                total_safe_prob = sum(policy[a] for a in ex['safe_moves'])
                print(f"      Total prob on ALL safe moves: {total_safe_prob:.4f}")

                if policy[ex['action']] > max(safe_probs):
                    print(f"\n      ⚠️  NETWORK IS BEING TRAINED TO PREFER THE LOSING MOVE!")
                    print(f"      ⚠️  Target says: {100*policy[ex['action']]:.1f}% on losing, {100*max(safe_probs):.1f}% on best safe")
                else:
                    print(f"\n      ✓ Target prefers safe moves (but action selection was unlucky)")

    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"{'='*70}")

    all_dumb_moves = 0
    targets_prefer_losing = 0

    for game_info in games_info[:20]:
        start_idx = game_info['start_idx']
        end_idx = game_info['end_idx']
        winner = game_info['winner']

        if winner == -1:
            continue

        game_moves = training_data[start_idx:end_idx]
        num_edges = num_vertices * (num_vertices - 1) // 2
        edge_states = np.zeros(num_edges, dtype=np.int32)

        for move_data in game_moves:
            player = int(move_data['player'])
            action = int(move_data['action'])
            policy = np.array(move_data['policy'])

            creates_clique = check_move_creates_clique(edge_states, action, player, k, num_vertices)

            if creates_clique:
                safe_moves = []
                for a in range(num_edges):
                    if edge_states[a] == 0:
                        if not check_move_creates_clique(edge_states, a, player, k, num_vertices):
                            safe_moves.append(a)

                if len(safe_moves) > 0:
                    all_dumb_moves += 1
                    if policy[action] > max([policy[a] for a in safe_moves]):
                        targets_prefer_losing += 1
                break

            edge_states[action] = player + 1

    print(f"  Dumb moves in first 20 games: {all_dumb_moves}")
    print(f"  Training targets prefer losing move: {targets_prefer_losing} ({100*targets_prefer_losing/max(1,all_dumb_moves):.1f}%)")
    print(f"\n  → The network is being trained to reproduce MCTS mistakes!")
    print(f"  → MCTS with c_puct={data.get('c_puct', 'N/A')} doesn't explore enough")
    print(f"  → Training reinforces the bad choices instead of correcting them")

if __name__ == "__main__":
    print("="*70)
    print("TRAINING TARGET VERIFICATION")
    print("="*70)
    print("\nThis script verifies what the network is actually being trained to learn.")
    print("We check if the policy targets (from MCTS) prefer losing moves or safe moves.\n")

    for iteration in [0, 10, 19]:
        try:
            analyze_training_targets(iteration)
        except Exception as e:
            print(f"\n⚠️  Error analyzing iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()

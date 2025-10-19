#!/usr/bin/env python3
"""
Analyze game data to find "dumb moves" - moves that immediately form a k-clique (lose)
when safe alternatives exist.
"""
import pickle
import numpy as np
import itertools
import argparse
from pathlib import Path

def check_move_creates_clique(edge_states, action, player, k=3, num_vertices=6):
    """
    Check if a move immediately creates a k-clique for the player.
    Returns True if this move causes immediate loss in avoid_clique mode.
    """
    # Make a copy and apply the move
    new_edge_states = np.array(edge_states, copy=True)
    new_edge_states[action] = player + 1  # player 0 -> edge_state 1, player 1 -> edge_state 2

    # Build adjacency matrix
    adj = np.zeros((num_vertices, num_vertices), dtype=int)
    edge_idx = 0
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if new_edge_states[edge_idx] == player + 1:
                adj[i, j] = 1
                adj[j, i] = 1
            edge_idx += 1

    # Check all k-cliques
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

def get_safe_moves(edge_states, player, k=3, num_vertices=6):
    """
    Get list of moves that don't immediately form a k-clique.
    """
    safe_moves = []
    num_edges = num_vertices * (num_vertices - 1) // 2

    for action in range(num_edges):
        # Check if move is valid (edge not yet colored)
        if edge_states[action] == 0:
            # Check if move creates clique
            if not check_move_creates_clique(edge_states, action, player, k, num_vertices):
                safe_moves.append(action)

    return safe_moves

def analyze_iteration(iter_num, base_path='experiments/ramsey_n6_k3_cpuct03'):
    """Analyze a single iteration for dumb moves."""
    path = f'{base_path}/game_data/iteration_{iter_num}.pkl'

    with open(path, 'rb') as f:
        data = pickle.load(f)

    training_data = data.get('training_data', [])
    games_info = data.get('games_info', [])
    k = data.get('k', 3)
    num_vertices = data.get('vertices', 6)

    print(f"\n{'='*70}")
    print(f"ITERATION {iter_num} - Dumb Move Analysis")
    print(f"{'='*70}")
    print(f"Graph: n={num_vertices}, k={k} (avoid_clique mode)")

    if not games_info:
        print("⚠️  No game boundary information available")
        return None

    total_games = len(games_info)
    games_with_dumb_moves = 0
    total_dumb_moves = 0
    total_moves_analyzed = 0

    immediate_loss_moves = 0  # Moves that immediately lose
    had_safe_alternative = 0   # Of those, how many had safe options

    dumb_details = []

    # Analyze each game
    for game_idx, game_info in enumerate(games_info[:20]):  # First 20 games
        start_idx = game_info['start_idx']
        end_idx = game_info['end_idx']
        winner = game_info['winner']
        num_moves = game_info['num_moves']

        if winner == -1:
            continue  # Skip draws

        game_moves = training_data[start_idx:end_idx]

        # Reconstruct the game
        num_edges = num_vertices * (num_vertices - 1) // 2
        edge_states = np.zeros(num_edges, dtype=np.int32)

        game_had_dumb = False
        losing_move = None

        for move_idx, move_data in enumerate(game_moves):
            player = int(move_data['player'])
            action = int(move_data['action'])

            total_moves_analyzed += 1

            # Check if this move creates a clique (immediate loss in avoid_clique)
            creates_clique = check_move_creates_clique(edge_states, action, player, k, num_vertices)

            if creates_clique:
                # This move loses! Check if there were safe alternatives
                safe_moves = get_safe_moves(edge_states, player, k, num_vertices)

                immediate_loss_moves += 1

                if len(safe_moves) > 0:
                    # DUMB MOVE! There were safe options but player chose to lose
                    had_safe_alternative += 1
                    game_had_dumb = True
                    total_dumb_moves += 1

                    policy = np.array(move_data['policy'])
                    losing_move = {
                        'game_idx': game_idx + 1,
                        'move_num': move_idx + 1,
                        'action': action,
                        'num_safe_alternatives': len(safe_moves),
                        'safe_actions': safe_moves[:5],  # First 5 safe moves
                        'policy_on_chosen': float(policy[action]),
                        'policy_on_safe': [float(policy[a]) for a in safe_moves[:5]],
                        'max_policy_on_safe': float(max([policy[a] for a in safe_moves])),
                        'total_safe_prob': float(sum([policy[a] for a in safe_moves]))
                    }
                    dumb_details.append(losing_move)

                # Game ends here
                break

            # Apply the move
            edge_states[action] = player + 1

        if game_had_dumb and len(dumb_details) <= 10:  # Print first 10
            loser = 1 - winner  # If winner is 0, loser is 1
            print(f"\n🎯 Game {losing_move['game_idx']}: Player {loser} DUMB LOSS at move {losing_move['move_num']}/{num_moves}")
            print(f"   Chose losing action {losing_move['action']} (MCTS prob={losing_move['policy_on_chosen']:.4f})")
            print(f"   Had {losing_move['num_safe_alternatives']} safe alternatives: {losing_move['safe_actions']}")
            print(f"   MCTS probs on safe moves: {[f'{p:.4f}' for p in losing_move['policy_on_safe']]}")
            print(f"   Max MCTS prob on safe: {losing_move['max_policy_on_safe']:.4f}")
            print(f"   Total prob on ALL safe: {losing_move['total_safe_prob']:.4f}")

            # Key insight
            if losing_move['policy_on_chosen'] > losing_move['max_policy_on_safe']:
                print(f"   ⚠️  MCTS ACTIVELY PREFERRED THE LOSING MOVE!")
            elif losing_move['total_safe_prob'] < 0.5:
                print(f"   ⚠️  MCTS put <50% probability on ALL safe moves combined!")

    # Summary
    print(f"\n{'-'*70}")
    print(f"SUMMARY for iteration {iter_num}:")
    print(f"  Games analyzed: {min(20, total_games)}")
    print(f"  Total moves: {total_moves_analyzed}")
    print(f"  Immediate loss moves: {immediate_loss_moves}")
    print(f"  Dumb moves (had safe alternative): {had_safe_alternative}")
    if immediate_loss_moves > 0:
        print(f"  Dumb move rate: {100*had_safe_alternative/immediate_loss_moves:.1f}%")

    return {
        'iteration': iter_num,
        'dumb_moves': had_safe_alternative,
        'total_losses': immediate_loss_moves,
        'dumb_pct': 100*had_safe_alternative/max(1,immediate_loss_moves),
        'dumb_details': dumb_details
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze dumb moves in game data')
    parser.add_argument('--experiment', type=str, default='experiments/ramsey_n6_k3_cpuct03',
                        help='Path to experiment directory')
    parser.add_argument('--iterations', type=int, nargs='+', default=[0, 5, 10, 15, 19],
                        help='Iterations to analyze')

    args = parser.parse_args()

    print("="*70)
    print(f"DUMB MOVE ANALYSIS: {args.experiment}")
    print("="*70)
    print("\nLooking for moves that form a k-clique (lose) when safe moves exist...")

    results = []
    for iteration in args.iterations:
        try:
            result = analyze_iteration(iteration, args.experiment)
            if result:
                results.append(result)
        except FileNotFoundError:
            print(f"\n⚠️  Iteration {iteration} data not found")
        except Exception as e:
            print(f"\n⚠️  Error analyzing iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()

    # Overall summary
    if results:
        print(f"\n\n{'='*70}")
        print("OVERALL TREND:")
        print(f"{'='*70}")
        print(f"{'Iteration':<12} {'Dumb Moves':<15} {'Rate':<10} {'Status'}")
        print("-" * 70)

        for r in results:
            status = ""
            print(f"{r['iteration']:<12} {r['dumb_moves']}/{r['total_losses']:<13} {r['dumb_pct']:>6.1f}%    {status}")

        if len(results) >= 2:
            first_pct = results[0]['dumb_pct']
            last_pct = results[-1]['dumb_pct']
            improvement = first_pct - last_pct

            print(f"\n{'='*70}")
            if improvement > 10:
                print(f"✅ SIGNIFICANT IMPROVEMENT: Dumb moves reduced by {improvement:.1f}%")
                print(f"   Training is helping the model avoid obvious blunders!")
            elif improvement > 0:
                print(f"✓ Slight improvement: Dumb moves reduced by {improvement:.1f}%")
            elif improvement < -10:
                print(f"⚠️  REGRESSION: Dumb moves increased by {abs(improvement):.1f}%")
                print(f"   Model may be overfitting or c_puct is too low!")
            else:
                print(f"➡️  NO SIGNIFICANT CHANGE: Dumb moves ~{last_pct:.1f}%")
                print(f"   Model is not learning to avoid obvious losses")

            # Additional insights
            print(f"\nKey insight from detailed analysis:")
            avg_chosen_prob = np.mean([d['policy_on_chosen'] for r in results for d in r['dumb_details']])
            avg_safe_prob = np.mean([d['max_policy_on_safe'] for r in results for d in r['dumb_details']])

            print(f"  Avg MCTS prob on chosen (losing) move: {avg_chosen_prob:.4f}")
            print(f"  Avg MCTS prob on best safe move: {avg_safe_prob:.4f}")

            if avg_chosen_prob > avg_safe_prob:
                print(f"\n⚠️  MCTS is systematically preferring losing moves!")
                print(f"   → Problem: Network evaluation or MCTS exploration")
                print(f"   → Current c_puct may be too low for proper exploration")

if __name__ == "__main__":
    main()

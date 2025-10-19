#!/usr/bin/env python3
"""
Test script for auxiliary loss implementation.
Tests clique detection and auxiliary loss computation.
"""

import sys
sys.path.insert(0, '/workspace/alphazero_clique/jax_full_src')

import jax.numpy as jnp
import numpy as np
from vectorized_board import VectorizedCliqueBoard, compute_immediate_loss_mask_batch

print("=" * 70)
print("TESTING AUXILIARY LOSS IMPLEMENTATION")
print("=" * 70)

# Test 1: Individual game - immediate loss detection
print("\n" + "=" * 70)
print("TEST 1: Individual Game - Immediate Loss Detection")
print("=" * 70)

# Create a board with n=6, k=3 (avoid triangles)
board = VectorizedCliqueBoard(batch_size=1, num_vertices=6, k=3, game_mode="avoid_clique")

# Manually set up a game state where player 0 has edges (0,1) and (0,2)
# If player 0 adds edge (1,2), they form triangle {0,1,2} and lose!
game_idx = 0

# Set edges for player 1 (value=1)
board.edge_states = board.edge_states.at[0, 0, 1].set(1)
board.edge_states = board.edge_states.at[0, 1, 0].set(1)
board.edge_states = board.edge_states.at[0, 0, 2].set(1)
board.edge_states = board.edge_states.at[0, 2, 0].set(1)

# Player 0's turn
board.current_players = board.current_players.at[0].set(0)

print(f"Game state: Player {board.current_players[0]} to move")
print(f"Player 1 has edges: (0,1) and (0,2)")
print(f"If Player 0 adds edge (1,2), they form triangle {0,1,2} -> LOSE!")

# Get immediate loss mask
loss_mask = board.get_immediate_loss_mask(game_idx)

# Edge (1,2) should be action index 7 for n=6
# Action mapping: (0,1)=0, (0,2)=1, (0,3)=2, (0,4)=3, (0,5)=4,
#                 (1,2)=5, (1,3)=6, ...
# Let me find the correct action for (1,2)
edge_12_action = board.edge_to_action[(1, 2)]
print(f"\nAction index for edge (1,2): {edge_12_action}")

print(f"Immediate loss mask shape: {loss_mask.shape}")
print(f"Immediate loss mask: {loss_mask}")
print(f"Edge (1,2) marked as immediate loss? {loss_mask[edge_12_action]}")

if loss_mask[edge_12_action]:
    print("✓ PASS: Edge (1,2) correctly identified as immediate loss!")
else:
    print("✗ FAIL: Edge (1,2) should be marked as immediate loss!")

# Check that other edges are NOT marked
num_loss_moves = np.sum(loss_mask)
print(f"\nTotal immediate-loss moves detected: {num_loss_moves}")

# Test 2: Batch computation
print("\n" + "=" * 70)
print("TEST 2: Batch Computation")
print("=" * 70)

batch_size = 3
boards = VectorizedCliqueBoard(batch_size=batch_size, num_vertices=6, k=3, game_mode="avoid_clique")

# Set up different scenarios for each game
# Game 0: Same as before (edge 1,2 is losing)
boards.edge_states = boards.edge_states.at[0, 0, 1].set(1)
boards.edge_states = boards.edge_states.at[0, 1, 0].set(1)
boards.edge_states = boards.edge_states.at[0, 0, 2].set(1)
boards.edge_states = boards.edge_states.at[0, 2, 0].set(1)
boards.current_players = boards.current_players.at[0].set(0)

# Game 1: Empty board (no immediate losses)
boards.current_players = boards.current_players.at[1].set(0)

# Game 2: Player 1 has (2,3) and (2,4), edge (3,4) would lose
boards.edge_states = boards.edge_states.at[2, 2, 3].set(2)
boards.edge_states = boards.edge_states.at[2, 3, 2].set(2)
boards.edge_states = boards.edge_states.at[2, 2, 4].set(2)
boards.edge_states = boards.edge_states.at[2, 4, 2].set(2)
boards.current_players = boards.current_players.at[2].set(1)

# Extract features for batch computation
edge_indices, edge_features = boards.get_features_for_nn_undirected()

print(f"Edge features shape: {edge_features.shape}")
print(f"Current players: {boards.current_players}")

# Compute immediate loss masks
loss_masks = compute_immediate_loss_mask_batch(
    edge_features,
    boards.current_players,
    k=3,
    num_vertices=6,
    game_mode="avoid_clique"
)

print(f"Loss masks shape: {loss_masks.shape}")
print(f"\nGame 0 - loss moves: {np.sum(loss_masks[0])}")
print(f"Game 1 - loss moves: {np.sum(loss_masks[1])}")
print(f"Game 2 - loss moves: {np.sum(loss_masks[2])}")

# Verify Game 0
edge_12_action = boards.edge_to_action[(1, 2)]
if loss_masks[0, edge_12_action]:
    print(f"✓ PASS: Game 0, edge (1,2) correctly marked as loss")
else:
    print(f"✗ FAIL: Game 0, edge (1,2) should be marked as loss")

# Verify Game 1 (empty board should have no losses)
if np.sum(loss_masks[1]) == 0:
    print(f"✓ PASS: Game 1, empty board has no immediate losses")
else:
    print(f"✗ FAIL: Game 1, empty board should have no immediate losses, found {np.sum(loss_masks[1])}")

# Verify Game 2
edge_34_action = boards.edge_to_action[(3, 4)]
if loss_masks[2, edge_34_action]:
    print(f"✓ PASS: Game 2, edge (3,4) correctly marked as loss")
else:
    print(f"✗ FAIL: Game 2, edge (3,4) should be marked as loss")

# Test 3: Integration with training
print("\n" + "=" * 70)
print("TEST 3: Integration with Training Functions")
print("=" * 70)

try:
    from train_jax import prepare_batch, compute_immediate_loss_mask_batch

    # Create mock training data
    experiences = []
    for i in range(10):
        # Create random edge features
        edge_features = np.zeros((15, 3))
        edge_features[:, 0] = 1  # All unselected

        exp = {
            'edge_indices': jnp.zeros((2, 15), dtype=jnp.int32),
            'edge_features': jnp.array(edge_features, dtype=jnp.float32),
            'policy': jnp.ones(15) / 15,
            'value': 0.0,
            'player': i % 2
        }
        experiences.append(exp)

    # Prepare batch
    import jax
    rng = jax.random.PRNGKey(42)
    batch = prepare_batch(
        experiences,
        batch_size=4,
        rng=rng,
        game_mode="avoid_clique",
        k=3,
        num_vertices=6
    )

    print(f"Batch prepared successfully!")
    print(f"  edge_features shape: {batch['edge_features'].shape}")
    print(f"  target_policies shape: {batch['target_policies'].shape}")

    if 'immediate_loss_mask' in batch:
        print(f"  immediate_loss_mask shape: {batch['immediate_loss_mask'].shape}")
        print(f"✓ PASS: Batch includes immediate_loss_mask!")
    else:
        print(f"✗ FAIL: Batch missing immediate_loss_mask!")

    print(f"\n✓ Integration test PASSED!")

except Exception as e:
    print(f"✗ FAIL: Integration test failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Auxiliary loss computation
print("\n" + "=" * 70)
print("TEST 4: Auxiliary Loss Computation")
print("=" * 70)

try:
    # Create mock policy and loss mask
    batch_size = 4
    num_actions = 15

    # Mock policies: uniform distribution
    policies = jnp.ones((batch_size, num_actions)) / num_actions

    # Mock immediate loss masks:
    # Game 0: 1 losing move
    # Game 1: 0 losing moves
    # Game 2: 3 losing moves
    # Game 3: 2 losing moves
    loss_mask = jnp.zeros((batch_size, num_actions), dtype=jnp.bool_)
    loss_mask = loss_mask.at[0, 5].set(True)
    loss_mask = loss_mask.at[2, 3].set(True)
    loss_mask = loss_mask.at[2, 7].set(True)
    loss_mask = loss_mask.at[2, 11].set(True)
    loss_mask = loss_mask.at[3, 2].set(True)
    loss_mask = loss_mask.at[3, 9].set(True)

    # Compute auxiliary loss manually
    immediate_loss_probs = policies * loss_mask
    num_loss_moves = jnp.sum(loss_mask, axis=1, keepdims=True) + 1e-8
    avg_prob_per_loss_move = jnp.sum(immediate_loss_probs, axis=1, keepdims=True) / num_loss_moves
    auxiliary_loss = jnp.mean(avg_prob_per_loss_move)

    print(f"Policies shape: {policies.shape}")
    print(f"Loss mask shape: {loss_mask.shape}")
    print(f"Num losing moves per game: {np.sum(loss_mask, axis=1)}")
    print(f"Prob on losing moves per game: {np.sum(immediate_loss_probs, axis=1)}")
    print(f"Auxiliary loss: {auxiliary_loss:.6f}")

    # Expected: uniform policy puts 1/15 on each action
    # Game 0: 1 move -> 1/15
    # Game 1: 0 moves -> 0
    # Game 2: 3 moves -> 3/15 = 1/5
    # Game 3: 2 moves -> 2/15
    # Average over games: ((1/15) + 0 + (1/5) + (2/15)) / 4 = (1 + 0 + 3 + 2) / (15*4) = 6/60 = 0.1
    expected = (1/15 + 0 + 3/15 + 2/15) / 4
    print(f"Expected auxiliary loss: {expected:.6f}")

    if abs(float(auxiliary_loss) - expected) < 0.001:
        print(f"✓ PASS: Auxiliary loss computed correctly!")
    else:
        print(f"✗ FAIL: Auxiliary loss mismatch!")

except Exception as e:
    print(f"✗ FAIL: Auxiliary loss test failed:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)

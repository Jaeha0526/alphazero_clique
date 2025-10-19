#!/usr/bin/env python
"""
JAX-based training module for AlphaZero
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
from flax.training import train_state
import numpy as np
from typing import List, Dict, Tuple
import time
from vectorized_board import compute_immediate_loss_mask_batch


class TrainState(train_state.TrainState):
    """Custom training state with additional metrics."""
    policy_loss: float
    value_loss: float


def create_train_state(model, learning_rate: float, rng):
    """Create initial training state."""
    # Initialize with dummy input
    dummy_indices = jnp.zeros((1, 2, 36), dtype=jnp.int32)
    dummy_features = jnp.zeros((1, 36, 3), dtype=jnp.float32)
    params = model.init(rng, dummy_indices, dummy_features)
    
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        policy_loss=0.0,
        value_loss=0.0
    )


def train_step(state: TrainState, batch: Dict, rng, asymmetric_mode: bool = False,
               value_weight: float = 1.0, label_smoothing: float = 0.1,
               aux_loss_weight: float = 0.0):
    """Single training step matching PyTorch implementation with auxiliary loss."""

    def loss_fn(params):
        # Forward pass with proper RNG for dropout
        if asymmetric_mode and 'player_roles' in batch:
            policies, values = state.apply_fn(
                params,
                batch['edge_indices'],
                batch['edge_features'],
                batch.get('player_roles'),
                deterministic=False,
                rngs={'dropout': rng}
            )
        else:
            policies, values = state.apply_fn(
                params,
                batch['edge_indices'],
                batch['edge_features'],
                deterministic=False,
                rngs={'dropout': rng}
            )

        # Policy loss with valid moves masking (matching PyTorch)
        # Create valid moves mask based on target policies
        valid_moves_mask = batch['target_policies'] > 1e-7

        # KL-divergence style loss focusing on valid moves
        log_probs = jnp.log(policies + 1e-8)
        policy_loss_terms = -batch['target_policies'] * log_probs * valid_moves_mask
        policy_loss_per_sample = jnp.sum(policy_loss_terms, axis=1)
        policy_loss = jnp.mean(policy_loss_per_sample)

        # Value loss with label smoothing and Huber loss (matching PyTorch)
        # Apply label smoothing: move targets slightly toward zero
        smoothed_targets = batch['target_values'] * (1 - label_smoothing)

        # Huber loss (smooth L1) - more robust to outliers than MSE
        value_diff = values - smoothed_targets
        huber_delta = 1.0  # Standard delta for Huber loss
        value_loss = jnp.where(
            jnp.abs(value_diff) <= huber_delta,
            0.5 * value_diff ** 2,
            huber_delta * (jnp.abs(value_diff) - 0.5 * huber_delta)
        )
        value_loss = jnp.mean(value_loss)

        # L2 regularization (matching PyTorch)
        l2_reg = 0.0
        for p in jax.tree_util.tree_leaves(params):
            l2_reg += jnp.sum(p ** 2)
        l2_reg *= 1e-5  # Small regularization factor

        # NEW: Auxiliary loss for immediate-loss moves (avoid_clique mode)
        auxiliary_loss = 0.0
        if aux_loss_weight > 0 and 'immediate_loss_mask' in batch:
            # immediate_loss_mask: (batch_size, num_actions) boolean
            # policies: (batch_size, num_actions)

            # Penalize high probability on immediate-loss moves
            immediate_loss_probs = policies * batch['immediate_loss_mask']

            # Average probability mass on immediate-loss moves per sample
            num_loss_moves = jnp.sum(batch['immediate_loss_mask'], axis=1, keepdims=True) + 1e-8
            avg_prob_per_loss_move = jnp.sum(immediate_loss_probs, axis=1, keepdims=True) / num_loss_moves
            auxiliary_loss = jnp.mean(avg_prob_per_loss_move)

        # Combined loss with configurable weights
        total_loss = policy_loss + value_weight * value_loss + l2_reg + aux_loss_weight * auxiliary_loss

        return total_loss, (policy_loss, value_loss, auxiliary_loss)
    
    # Compute gradients
    grad_fn = grad(loss_fn, has_aux=True)
    grads, (policy_loss, value_loss, auxiliary_loss) = grad_fn(state.params)

    # Gradient clipping (matching PyTorch max_norm=1.0)
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    grads = jax.tree_util.tree_map(
        lambda g: jnp.where(grad_norm > 1.0, g / grad_norm, g),
        grads
    )

    # Update parameters
    state = state.apply_gradients(grads=grads)
    state = state.replace(policy_loss=policy_loss, value_loss=value_loss)

    return state, auxiliary_loss


def prepare_batch(experiences: List[Dict], batch_size: int, rng,
                  game_mode: str = "symmetric", k: int = 3,
                  num_vertices: int = 6) -> Dict:
    """Prepare a training batch from experiences."""
    # Randomly sample experiences
    indices = jax.random.choice(rng, len(experiences), shape=(batch_size,))

    # Extract data
    edge_indices = []
    edge_features = []
    target_policies = []
    target_values = []
    player_roles = []
    players = []

    for idx in indices:
        exp = experiences[idx]
        # Handle different formats
        if 'edge_indices' in exp:
            # Optimized format
            edge_indices.append(exp['edge_indices'])
            edge_features.append(exp['edge_features'])
        elif 'board_state' in exp:
            # New format from improved self-play
            edge_indices.append(exp['board_state']['edge_index'])
            edge_features.append(exp['board_state']['edge_attr'])
        else:
            # Old format
            edge_indices.append(exp['edge_index'])
            edge_features.append(exp['edge_attr'])
        target_policies.append(exp['policy'])
        target_values.append(exp['value'])

        # Add player role if available
        if 'player_role' in exp:
            player_roles.append(exp['player_role'])

        # Add player for immediate loss mask computation
        if 'player' in exp:
            players.append(exp['player'])

    # Stack into arrays
    batch = {
        'edge_indices': jnp.stack(edge_indices, dtype=jnp.int32),  # Must be int32 for indexing
        'edge_features': jnp.stack(edge_features, dtype=jnp.float32),
        'target_policies': jnp.stack(target_policies, dtype=jnp.float32),
        'target_values': jnp.array(target_values, dtype=jnp.float32).reshape(-1, 1)
    }

    # Add player roles if available
    if player_roles:
        batch['player_roles'] = jnp.array(player_roles)

    # NEW: Compute immediate loss mask for avoid_clique mode
    if game_mode == "avoid_clique" and players:
        current_players_array = jnp.array(players, dtype=jnp.int32)
        immediate_loss_mask = compute_immediate_loss_mask_batch(
            batch['edge_features'],
            current_players_array,
            k,
            num_vertices,
            game_mode
        )
        batch['immediate_loss_mask'] = immediate_loss_mask

    return batch


def train_network_jax(
    model,
    experiences: List[Dict],
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    verbose: bool = True,
    initial_state: TrainState = None,
    asymmetric_mode: bool = False,
    value_weight: float = 1.0,
    label_smoothing: float = 0.1,
    warmup_fraction: float = 0.15,
    aux_loss_weight: float = 0.0,
    game_mode: str = "symmetric",
    k: int = 3,
    num_vertices: int = 6
) -> Tuple[TrainState, float, float, float]:
    """
    Train the neural network using JAX.

    Returns:
        state: Final training state
        avg_policy_loss: Average policy loss
        avg_value_loss: Average value loss
        avg_aux_loss: Average auxiliary loss
    """
    # Initialize
    rng = jax.random.PRNGKey(42)

    if initial_state is not None:
        state = initial_state
    else:
        # Handle both Flax models and our wrapper
        rng, init_rng = jax.random.split(rng)
        if hasattr(model, 'model'):
            # It's our wrapper - use the underlying Flax model
            flax_model = model.model
            params = model.params
            tx = optax.adam(learning_rate)
            state = TrainState.create(
                apply_fn=flax_model.apply,
                params=params,
                tx=tx,
                policy_loss=0.0,
                value_loss=0.0
            )
        else:
            # It's a Flax model
            state = create_train_state(model, learning_rate, init_rng)

    # Training metrics
    policy_losses = []
    value_losses = []
    auxiliary_losses = []

    # Training loop
    steps_per_epoch = max(1, len(experiences) // batch_size)
    total_steps = epochs * steps_per_epoch

    if verbose:
        print(f"Training for {epochs} epochs, {steps_per_epoch} steps per epoch")
        print(f"Total training examples: {len(experiences)}")
        if aux_loss_weight > 0:
            print(f"Auxiliary loss weight: {aux_loss_weight}")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_aux_loss = 0.0

        for step in range(steps_per_epoch):
            # Prepare batch with immediate loss mask
            rng, batch_rng = jax.random.split(rng)
            batch = prepare_batch(experiences, batch_size, batch_rng,
                                game_mode, k, num_vertices)

            # Training step
            rng, step_rng = jax.random.split(rng)
            state, aux_loss = train_step(state, batch, step_rng, asymmetric_mode,
                                        value_weight, label_smoothing, aux_loss_weight)

            epoch_policy_loss += state.policy_loss
            epoch_value_loss += state.value_loss
            epoch_aux_loss += aux_loss

        # Record epoch metrics
        avg_policy_loss = epoch_policy_loss / steps_per_epoch
        avg_value_loss = epoch_value_loss / steps_per_epoch
        avg_aux_loss = epoch_aux_loss / steps_per_epoch
        policy_losses.append(float(avg_policy_loss))
        value_losses.append(float(avg_value_loss))
        auxiliary_losses.append(float(avg_aux_loss))

        if verbose and epoch % max(1, epochs // 10) == 0:
            msg = (f"Epoch {epoch+1}/{epochs} - "
                  f"Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}")
            if aux_loss_weight > 0:
                msg += f", Aux Loss: {avg_aux_loss:.4f}"
            print(msg)

    elapsed = time.time() - start_time

    if verbose:
        print(f"Training completed in {elapsed:.1f}s")
        msg = f"Final losses - Policy: {policy_losses[-1]:.4f}, Value: {value_losses[-1]:.4f}"
        if aux_loss_weight > 0:
            msg += f", Aux: {auxiliary_losses[-1]:.4f}"
        print(msg)

    # Return final state and average losses
    return state, np.mean(policy_losses), np.mean(value_losses), np.mean(auxiliary_losses)


def save_model_jax(state: TrainState, filepath: str):
    """Save JAX model state."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump({
            'params': state.params,
            'step': state.step,
            'policy_loss': state.policy_loss,
            'value_loss': state.value_loss
        }, f)


def load_model_jax(model, filepath: str, learning_rate: float = 0.001):
    """Load JAX model state."""
    import pickle
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Recreate state
    tx = optax.adam(learning_rate)
    state = TrainState.create(
        apply_fn=model.apply,
        params=checkpoint['params'],
        tx=tx,
        policy_loss=checkpoint.get('policy_loss', 0.0),
        value_loss=checkpoint.get('value_loss', 0.0)
    )
    
    return state
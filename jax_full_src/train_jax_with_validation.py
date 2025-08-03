#!/usr/bin/env python
"""JAX training with validation split and early stopping, matching PyTorch implementation"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
from flax.training import train_state
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import time
from functools import partial
import copy


class TrainState(train_state.TrainState):
    """Custom training state with additional metrics."""
    policy_loss: float
    value_loss: float


# JIT-compiled train step with static arguments
@partial(jit, static_argnames=['asymmetric_mode', 'value_weight', 'label_smoothing'])
def train_step_optimized(state: TrainState, batch: Dict, rng, asymmetric_mode: bool = False,
                        value_weight: float = 1.0, label_smoothing: float = 0.1):
    """JIT-compiled training step for maximum performance."""
    
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
        
        # Policy loss with valid moves masking
        valid_moves_mask = batch['target_policies'] > 1e-7
        
        # KL-divergence style loss
        log_probs = jnp.log(policies + 1e-8)
        policy_loss_terms = -batch['target_policies'] * log_probs * valid_moves_mask
        policy_loss_per_sample = jnp.sum(policy_loss_terms, axis=1)
        policy_loss = jnp.mean(policy_loss_per_sample)
        
        # Value loss with label smoothing and Huber loss
        smoothed_targets = batch['target_values'] * (1 - label_smoothing)
        
        # Huber loss
        value_diff = values - smoothed_targets
        huber_delta = 1.0
        value_loss = jnp.where(
            jnp.abs(value_diff) <= huber_delta,
            0.5 * value_diff ** 2,
            huber_delta * (jnp.abs(value_diff) - 0.5 * huber_delta)
        )
        value_loss = jnp.mean(value_loss)
        
        # L2 regularization
        l2_reg = 0.0
        for p in jax.tree_util.tree_leaves(params):
            l2_reg += jnp.sum(p ** 2)
        l2_reg *= 1e-5
        
        # Combined loss
        total_loss = policy_loss + value_weight * value_loss + l2_reg
        
        return total_loss, (policy_loss, value_loss)
    
    # Compute gradients
    grad_fn = grad(loss_fn, has_aux=True)
    grads, (policy_loss, value_loss) = grad_fn(state.params)
    
    # Gradient clipping
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    grads = jax.tree_util.tree_map(
        lambda g: jnp.where(grad_norm > 1.0, g / grad_norm, g),
        grads
    )
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    state = state.replace(policy_loss=policy_loss, value_loss=value_loss)
    
    return state


@partial(jit, static_argnames=['asymmetric_mode'])
def compute_validation_metrics(state: TrainState, batch: Dict, asymmetric_mode: bool = False):
    """Compute validation metrics without updating parameters."""
    
    # Forward pass in eval mode (deterministic=True)
    if asymmetric_mode and 'player_roles' in batch:
        policies, values = state.apply_fn(
            state.params,
            batch['edge_indices'],
            batch['edge_features'],
            batch.get('player_roles'),
            deterministic=True
        )
    else:
        policies, values = state.apply_fn(
            state.params,
            batch['edge_indices'],
            batch['edge_features'],
            deterministic=True
        )
    
    # Policy loss
    valid_moves_mask = batch['target_policies'] > 1e-7
    log_probs = jnp.log(policies + 1e-8)
    policy_loss_terms = -batch['target_policies'] * log_probs * valid_moves_mask
    policy_loss = jnp.mean(jnp.sum(policy_loss_terms, axis=1))
    
    # Value loss (MSE, no label smoothing for validation)
    value_loss = jnp.mean((values - batch['target_values']) ** 2)
    
    # For asymmetric mode, compute per-role losses
    if asymmetric_mode and 'player_roles' in batch:
        attacker_mask = batch['player_roles'] == 0
        defender_mask = batch['player_roles'] == 1
        
        # Use JAX-compatible conditional computation
        attacker_count = jnp.sum(attacker_mask)
        defender_count = jnp.sum(defender_mask)
        
        attacker_policy_loss = jnp.where(
            attacker_count > 0,
            jnp.sum(policy_loss_terms * attacker_mask[:, None]) / attacker_count,
            0.0
        )
        defender_policy_loss = jnp.where(
            defender_count > 0,
            jnp.sum(policy_loss_terms * defender_mask[:, None]) / defender_count,
            0.0
        )
        
        return policy_loss, value_loss, attacker_policy_loss, defender_policy_loss
    
    return policy_loss, value_loss, None, None


def prepare_batch_vectorized(experiences_array: Dict[str, jnp.ndarray], 
                            indices: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Vectorized batch preparation using JAX operations."""
    batch = {
        'edge_indices': experiences_array['edge_indices'][indices],
        'edge_features': experiences_array['edge_features'][indices],
        'target_policies': experiences_array['policies'][indices],
        'target_values': experiences_array['values'][indices]
    }
    
    if 'player_roles' in experiences_array:
        batch['player_roles'] = experiences_array['player_roles'][indices]
    
    return batch


def preprocess_experiences(experiences: List[Dict]) -> Dict[str, jnp.ndarray]:
    """Convert list of experiences to stacked JAX arrays for fast indexing."""
    print("Preprocessing experiences into JAX arrays...")
    
    # Pre-allocate arrays
    num_exp = len(experiences)
    first_exp = experiences[0]
    
    # Get shapes
    edge_indices_shape = first_exp['edge_indices'].shape
    edge_features_shape = first_exp['edge_features'].shape
    policy_shape = first_exp['policy'].shape
    
    # Allocate numpy arrays first (faster than lists)
    edge_indices_arr = np.zeros((num_exp,) + edge_indices_shape, dtype=np.int32)
    edge_features_arr = np.zeros((num_exp,) + edge_features_shape, dtype=np.float32)
    policies_arr = np.zeros((num_exp,) + policy_shape, dtype=np.float32)
    values_arr = np.zeros((num_exp, 1), dtype=np.float32)
    
    has_roles = 'player_role' in first_exp and first_exp['player_role'] is not None
    if has_roles:
        player_roles_arr = np.zeros(num_exp, dtype=np.int32)
    
    # Fill arrays
    for i, exp in enumerate(experiences):
        edge_indices_arr[i] = exp['edge_indices']
        edge_features_arr[i] = exp['edge_features']
        policies_arr[i] = exp['policy']
        values_arr[i, 0] = exp['value']
        if has_roles and exp.get('player_role') is not None:
            player_roles_arr[i] = exp['player_role']
    
    # Convert to JAX arrays
    result = {
        'edge_indices': jnp.array(edge_indices_arr),
        'edge_features': jnp.array(edge_features_arr),
        'policies': jnp.array(policies_arr),
        'values': jnp.array(values_arr)
    }
    
    if has_roles:
        result['player_roles'] = jnp.array(player_roles_arr)
    
    return result


def train_network_jax_with_validation(
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
    validation_split: float = 0.1,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.001
) -> Tuple[TrainState, float, float, Dict]:
    """
    Training with validation split and early stopping, matching PyTorch implementation.
    
    Returns:
        - Final train state
        - Best validation policy loss
        - Best validation value loss
        - Training history dict
    """
    
    # Split data into train and validation
    num_examples = len(experiences)
    train_size = int((1 - validation_split) * num_examples)
    
    # Shuffle before splitting
    np.random.shuffle(experiences)
    train_experiences = experiences[:train_size]
    val_experiences = experiences[train_size:]
    
    print(f"\nData split: {len(train_experiences)} training, {len(val_experiences)} validation")
    
    # Preprocess both sets
    start_preprocess = time.time()
    train_array = preprocess_experiences(train_experiences)
    val_array = preprocess_experiences(val_experiences)
    preprocess_time = time.time() - start_preprocess
    
    if verbose:
        print(f"Preprocessing completed in {preprocess_time:.2f}s")
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    
    if initial_state is not None:
        state = initial_state
    else:
        # Handle both Flax models and our wrapper
        rng, init_rng = jax.random.split(rng)
        if hasattr(model, 'model'):
            # It's our wrapper
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
            from train_jax import create_train_state
            state = create_train_state(model, learning_rate, init_rng)
    
    # Training history
    history = {
        'train_policy_loss': [],
        'train_value_loss': [],
        'val_policy_loss': [],
        'val_value_loss': [],
        'val_attacker_loss': [],
        'val_defender_loss': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_state_params = None
    patience_counter = 0
    
    # Training loop
    steps_per_epoch = max(1, len(train_experiences) // batch_size)
    val_batch_size = min(256, len(val_experiences))  # Larger batch for validation
    
    if verbose:
        print(f"\nTraining with validation:")
        print(f"  Epochs: {epochs}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Batch size: {batch_size}")
        print(f"  Validation split: {validation_split:.1%}")
        print(f"  Early stopping patience: {early_stopping_patience}")
    
    start_time = time.time()
    
    # Create index arrays
    train_indices = jnp.arange(len(train_experiences))
    val_indices = jnp.arange(len(val_experiences))
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        epoch_train_policy_loss = 0.0
        epoch_train_value_loss = 0.0
        
        # Shuffle training indices
        rng, shuffle_rng = jax.random.split(rng)
        shuffled_train_indices = jax.random.permutation(shuffle_rng, train_indices)
        
        for step in range(steps_per_epoch):
            # Get batch indices
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, len(train_experiences))
            batch_indices = shuffled_train_indices[start_idx:end_idx]
            
            # Prepare batch
            batch = prepare_batch_vectorized(train_array, batch_indices)
            
            # Training step
            rng, step_rng = jax.random.split(rng)
            state = train_step_optimized(
                state, batch, step_rng, 
                asymmetric_mode=asymmetric_mode,
                value_weight=value_weight,
                label_smoothing=label_smoothing
            )
            
            epoch_train_policy_loss += state.policy_loss
            epoch_train_value_loss += state.value_loss
        
        # Average training losses
        avg_train_policy_loss = float(epoch_train_policy_loss / steps_per_epoch)
        avg_train_value_loss = float(epoch_train_value_loss / steps_per_epoch)
        history['train_policy_loss'].append(avg_train_policy_loss)
        history['train_value_loss'].append(avg_train_value_loss)
        
        # Validation phase
        val_policy_losses = []
        val_value_losses = []
        val_attacker_losses = []
        val_defender_losses = []
        
        for i in range(0, len(val_experiences), val_batch_size):
            end_idx = min(i + val_batch_size, len(val_experiences))
            batch_indices = val_indices[i:end_idx]
            val_batch = prepare_batch_vectorized(val_array, batch_indices)
            
            # Compute validation metrics
            policy_loss, value_loss, attacker_loss, defender_loss = compute_validation_metrics(
                state, val_batch, asymmetric_mode=asymmetric_mode
            )
            
            val_policy_losses.append(float(policy_loss))
            val_value_losses.append(float(value_loss))
            
            if attacker_loss is not None:
                val_attacker_losses.append(float(attacker_loss))
            if defender_loss is not None:
                val_defender_losses.append(float(defender_loss))
        
        # Average validation losses
        avg_val_policy_loss = np.mean(val_policy_losses)
        avg_val_value_loss = np.mean(val_value_losses)
        history['val_policy_loss'].append(avg_val_policy_loss)
        history['val_value_loss'].append(avg_val_value_loss)
        
        if val_attacker_losses:
            history['val_attacker_loss'].append(np.mean(val_attacker_losses))
        if val_defender_losses:
            history['val_defender_loss'].append(np.mean(val_defender_losses))
        
        # Combined validation loss for early stopping
        current_val_loss = avg_val_policy_loss + value_weight * avg_val_value_loss
        
        # Early stopping check
        if current_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = current_val_loss
            best_state_params = jax.tree.map(lambda x: x.copy(), state.params)
            patience_counter = 0
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: P={avg_train_policy_loss:.4f} V={avg_train_value_loss:.4f} | "
                      f"Val Loss: P={avg_val_policy_loss:.4f} V={avg_val_value_loss:.4f} | "
                      f"*BEST* (patience reset)")
        else:
            patience_counter += 1
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: P={avg_train_policy_loss:.4f} V={avg_train_value_loss:.4f} | "
                      f"Val Loss: P={avg_val_policy_loss:.4f} V={avg_val_value_loss:.4f} | "
                      f"Patience: {patience_counter}/{early_stopping_patience}")
        
        # Print asymmetric metrics if available
        if asymmetric_mode and val_attacker_losses and verbose:
            print(f"  Asymmetric Val - Attacker: {np.mean(val_attacker_losses):.4f}, "
                  f"Defender: {np.mean(val_defender_losses):.4f}")
        
        # Check for early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Restoring best model from epoch {epoch+1-patience_counter}")
            if best_state_params is not None:
                state = state.replace(params=best_state_params)
            break
        
        epoch_time = time.time() - epoch_start
    
    elapsed = time.time() - start_time
    
    # Final validation metrics with best model
    if best_state_params is not None:
        state = state.replace(params=best_state_params)
    
    # Compute final validation metrics
    final_val_policy_losses = []
    final_val_value_losses = []
    
    for i in range(0, len(val_experiences), val_batch_size):
        end_idx = min(i + val_batch_size, len(val_experiences))
        batch_indices = val_indices[i:end_idx]
        val_batch = prepare_batch_vectorized(val_array, batch_indices)
        
        policy_loss, value_loss, _, _ = compute_validation_metrics(
            state, val_batch, asymmetric_mode=asymmetric_mode
        )
        
        final_val_policy_losses.append(float(policy_loss))
        final_val_value_losses.append(float(value_loss))
    
    final_policy_loss = np.mean(final_val_policy_losses)
    final_value_loss = np.mean(final_val_value_losses)
    
    if verbose:
        print(f"\nTraining completed in {elapsed:.1f}s")
        print(f"Average time per epoch: {elapsed/len(history['train_policy_loss']):.2f}s")
        print(f"Best validation loss achieved: {best_val_loss:.4f}")
        print(f"Final validation losses - Policy: {final_policy_loss:.4f}, Value: {final_value_loss:.4f}")
    
    return state, final_policy_loss, final_value_loss, history


# Re-export save/load functions from original module
from train_jax import save_model_jax, load_model_jax

__all__ = ['train_network_jax_with_validation', 'save_model_jax', 'load_model_jax']
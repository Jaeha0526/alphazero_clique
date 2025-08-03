#!/usr/bin/env python
"""Fully optimized JAX training with all performance improvements"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
from flax.training import train_state
import numpy as np
from typing import List, Dict, Tuple, Any
import time
from functools import partial


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


def prepare_batch_vectorized(experiences_array: Dict[str, jnp.ndarray], 
                            indices: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Vectorized batch preparation using JAX operations."""
    # Simply gather from pre-stacked arrays
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
    
    has_roles = 'player_role' in first_exp
    if has_roles:
        player_roles_arr = np.zeros(num_exp, dtype=np.int32)
    
    # Fill arrays
    for i, exp in enumerate(experiences):
        edge_indices_arr[i] = exp['edge_indices']
        edge_features_arr[i] = exp['edge_features']
        policies_arr[i] = exp['policy']
        values_arr[i, 0] = exp['value']
        if has_roles:
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


def train_network_jax_optimized(
    model,
    experiences: List[Dict],
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    verbose: bool = True,
    initial_state: TrainState = None,
    asymmetric_mode: bool = False,
    value_weight: float = 1.0,
    label_smoothing: float = 0.1
) -> Tuple[TrainState, float, float]:
    """
    Fully optimized training with:
    - JIT-compiled train step
    - Vectorized batch preparation
    - Pre-processed experience arrays
    """
    
    # Preprocess experiences once
    start_preprocess = time.time()
    experiences_array = preprocess_experiences(experiences)
    preprocess_time = time.time() - start_preprocess
    
    if verbose:
        print(f"Preprocessing completed in {preprocess_time:.2f}s")
        print(f"Total training examples: {len(experiences)}")
    
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
    
    # Training metrics
    policy_losses = []
    value_losses = []
    
    # Training loop
    steps_per_epoch = max(1, len(experiences) // batch_size)
    total_steps = epochs * steps_per_epoch
    
    if verbose:
        print(f"\nOptimized Training:")
        print(f"  Epochs: {epochs}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Batch size: {batch_size}")
        print(f"  Using JIT-compiled train step")
        print(f"  Using vectorized batch preparation")
    
    start_time = time.time()
    
    # Create index array for all experiences
    all_indices = jnp.arange(len(experiences))
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        
        # Shuffle indices once per epoch
        rng, shuffle_rng = jax.random.split(rng)
        shuffled_indices = jax.random.permutation(shuffle_rng, all_indices)
        
        for step in range(steps_per_epoch):
            # Get batch indices
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, len(experiences))
            batch_indices = shuffled_indices[start_idx:end_idx]
            
            # Prepare batch (vectorized)
            batch = prepare_batch_vectorized(experiences_array, batch_indices)
            
            # Training step (JIT-compiled)
            rng, step_rng = jax.random.split(rng)
            state = train_step_optimized(
                state, batch, step_rng, 
                asymmetric_mode=asymmetric_mode,
                value_weight=value_weight,
                label_smoothing=label_smoothing
            )
            
            epoch_policy_loss += state.policy_loss
            epoch_value_loss += state.value_loss
        
        # Record epoch metrics
        avg_policy_loss = epoch_policy_loss / steps_per_epoch
        avg_value_loss = epoch_value_loss / steps_per_epoch
        policy_losses.append(float(avg_policy_loss))
        value_losses.append(float(avg_value_loss))
        
        epoch_time = time.time() - epoch_start
        
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nTraining completed in {elapsed:.1f}s")
        print(f"Average time per epoch: {elapsed/epochs:.2f}s")
        print(f"Average time per step: {elapsed/(epochs*steps_per_epoch)*1000:.1f}ms")
        print(f"Throughput: {len(experiences)*epochs/elapsed:.0f} samples/sec")
        print(f"Final losses - Policy: {policy_losses[-1]:.4f}, Value: {value_losses[-1]:.4f}")
    
    return state, np.mean(policy_losses), np.mean(value_losses)


# Re-export save/load functions from original module
from train_jax import save_model_jax, load_model_jax

__all__ = ['train_network_jax_optimized', 'save_model_jax', 'load_model_jax']
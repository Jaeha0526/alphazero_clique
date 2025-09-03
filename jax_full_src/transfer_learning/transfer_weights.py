#!/usr/bin/env python
"""
Transfer learning utilities for AlphaZero Clique models.
Enables transferring learned weights from one graph size to another (e.g., n=9 to n=13).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from functools import partial

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vectorized_nn import ImprovedBatchedNeuralNetwork


def analyze_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Analyze a checkpoint file to understand its structure and dimensions.
    """
    print(f"\n📊 Analyzing checkpoint: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    params = checkpoint['params']
    metadata = checkpoint.get('model_config', {})
    
    print(f"  Model configuration:")
    print(f"    - num_vertices: {metadata.get('num_vertices', 'unknown')}")
    print(f"    - hidden_dim: {metadata.get('hidden_dim', 'unknown')}")
    print(f"    - num_layers: {metadata.get('num_gnn_layers', 'unknown')}")
    print(f"    - asymmetric: {metadata.get('asymmetric_mode', False)}")
    
    # Analyze parameter shapes
    print(f"\n  Parameter shapes:")
    for key, value in jax.tree_util.tree_flatten_with_path(params)[0]:
        shape = value.shape if hasattr(value, 'shape') else 'scalar'
        param_name = '/'.join(str(k) for k in key)
        print(f"    {param_name}: {shape}")
    
    return checkpoint


def transfer_compatible_weights(
    source_params: Dict[str, Any],
    target_model: ImprovedBatchedNeuralNetwork,
    source_vertices: int,
    target_vertices: int,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Transfer compatible weights from source to target model.
    
    Strategy:
    1. GNN layers (EdgeAwareGNNBlock, EdgeBlock) - fully transferable
    2. Value head - fully transferable
    3. Policy head - transfer all except final layer
    4. Node embeddings - expand if needed
    """
    if verbose:
        print(f"\n🔄 Transferring weights from n={source_vertices} to n={target_vertices}")
    
    # Initialize target with random weights
    dummy_edge_index = jnp.zeros((1, 2, target_vertices * (target_vertices - 1) // 2), dtype=jnp.int32)
    dummy_edge_features = jnp.zeros((1, target_vertices * (target_vertices - 1) // 2, 3))
    
    rng = jax.random.PRNGKey(42)
    target_params_full = target_model.model.init(
        rng, 
        dummy_edge_index, 
        dummy_edge_features,
        deterministic=True
    )
    
    # Match source structure
    if 'batch_stats' in source_params:
        target_params = target_params_full  # Keep full structure
    else:
        target_params = target_params_full['params']  # Just params
    
    # Flatten parameters for easier manipulation
    source_flat, source_tree = jax.tree_util.tree_flatten_with_path(source_params)
    target_flat, target_tree = jax.tree_util.tree_flatten_with_path(target_params)
    
    # Create mapping of parameter paths to values
    source_dict = {'/'.join(str(k) for k in key): value for key, value in source_flat}
    target_dict = {'/'.join(str(k) for k in key): value for key, value in target_flat}
    
    transferred = 0
    skipped = 0
    adapted = 0
    
    # Transfer weights
    new_params = {}
    for target_key, target_value in target_dict.items():
        if target_key in source_dict:
            source_value = source_dict[target_key]
            
            # Check if shapes match
            if source_value.shape == target_value.shape:
                # Direct transfer
                new_params[target_key] = source_value
                transferred += 1
                if verbose and transferred <= 10:  # Show first 10 transfers
                    print(f"  ✓ Transferred: {target_key} {source_value.shape}")
            
            elif 'policy' in target_key.lower() and 'dense_2' in target_key.lower():
                # Policy head final layer - different output size
                # Initialize randomly (or could do smart initialization)
                new_params[target_key] = target_value
                adapted += 1
                if verbose:
                    print(f"  🔧 Adapted (policy output): {target_key} {source_value.shape} → {target_value.shape}")
            
            elif len(source_value.shape) == len(target_value.shape):
                # Try to adapt if dimensions are compatible
                if source_value.shape[0] < target_value.shape[0]:
                    # Expand by padding/repeating
                    padding = [(0, target_value.shape[i] - source_value.shape[i]) 
                              for i in range(len(source_value.shape))]
                    padded = jnp.pad(source_value, padding, mode='constant')
                    new_params[target_key] = padded
                    adapted += 1
                    if verbose:
                        print(f"  🔧 Adapted (expanded): {target_key} {source_value.shape} → {target_value.shape}")
                else:
                    # Use target's random initialization
                    new_params[target_key] = target_value
                    skipped += 1
            else:
                # Incompatible - use target's initialization
                new_params[target_key] = target_value
                skipped += 1
        else:
            # Parameter doesn't exist in source - use target's initialization
            new_params[target_key] = target_value
            skipped += 1
    
    if verbose:
        print(f"\n  Summary:")
        print(f"    - Transferred: {transferred} parameters")
        print(f"    - Adapted: {adapted} parameters")
        print(f"    - Initialized: {skipped} parameters")
    
    # Reconstruct parameter tree
    flat_list = [(key.split('/'), value) for key, value in new_params.items()]
    
    # Convert string keys back to proper structure
    def reconstruct_tree(flat_list):
        tree = {}
        for keys, value in flat_list:
            current = tree
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        return tree
    
    return reconstruct_tree(flat_list)


def create_transfer_checkpoint(
    source_checkpoint_path: str,
    target_vertices: int,
    target_k: int,
    output_path: Optional[str] = None,
    hidden_dim: int = 64,
    num_layers: int = 3,
    asymmetric: bool = False
) -> str:
    """
    Create a new checkpoint with transferred weights for a different graph size.
    
    Args:
        source_checkpoint_path: Path to source model checkpoint
        target_vertices: Number of vertices for target model
        target_k: Clique size for target model
        output_path: Where to save the new checkpoint
        hidden_dim: Hidden dimension (should match source)
        num_layers: Number of GNN layers (should match source)
        asymmetric: Whether to use asymmetric mode
    
    Returns:
        Path to the created checkpoint
    """
    print(f"\n🚀 Creating transfer checkpoint")
    print(f"  Source: {source_checkpoint_path}")
    print(f"  Target: n={target_vertices}, k={target_k}")
    
    # Load source checkpoint
    with open(source_checkpoint_path, 'rb') as f:
        source_checkpoint = pickle.load(f)
    
    # Handle both old and new checkpoint formats
    if 'params' in source_checkpoint and isinstance(source_checkpoint['params'], dict):
        # Check if it has the nested structure with 'params' key inside
        if 'params' in source_checkpoint['params']:
            source_params = source_checkpoint['params']  # Full structure including batch_stats
        else:
            source_params = {'params': source_checkpoint['params']}  # Wrap in expected structure
    else:
        source_params = source_checkpoint['params']
    
    source_config = source_checkpoint.get('model_config', {})
    source_vertices = source_config.get('num_vertices', 9)  # Default to 9 if not specified
    
    # Create target model
    target_model = ImprovedBatchedNeuralNetwork(
        num_vertices=target_vertices,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        asymmetric_mode=asymmetric
    )
    
    # Transfer weights
    transferred_params = transfer_compatible_weights(
        source_params,
        target_model,
        source_vertices,
        target_vertices,
        verbose=True
    )
    
    # Create new checkpoint
    new_checkpoint = {
        'params': transferred_params,
        'iteration': 0,  # Reset iteration count
        'source_checkpoint': source_checkpoint_path,
        'transfer_info': {
            'source_vertices': source_vertices,
            'target_vertices': target_vertices,
            'source_k': source_config.get('k', 4),
            'target_k': target_k,
            'transfer_method': 'weight_transfer'
        },
        'model_config': {
            'num_vertices': target_vertices,
            'hidden_dim': hidden_dim,
            'num_gnn_layers': num_layers,
            'asymmetric_mode': asymmetric,
        }
    }
    
    # Save checkpoint
    if output_path is None:
        source_name = Path(source_checkpoint_path).stem
        output_path = f"checkpoint_n{target_vertices}k{target_k}_from_{source_name}.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(new_checkpoint, f)
    
    print(f"\n✅ Transfer checkpoint saved to: {output_path}")
    print(f"  You can now use this for training with --resume_from {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Transfer AlphaZero Clique model weights between graph sizes')
    
    parser.add_argument('source_checkpoint', type=str,
                        help='Path to source model checkpoint (e.g., n9k4 model)')
    parser.add_argument('--target_vertices', type=int, required=True,
                        help='Number of vertices for target model')
    parser.add_argument('--target_k', type=int, required=True,
                        help='Clique size for target model')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for transferred checkpoint')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension (should match source model)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers (should match source model)')
    parser.add_argument('--asymmetric', action='store_true',
                        help='Use asymmetric game mode')
    parser.add_argument('--analyze_only', action='store_true',
                        help='Only analyze checkpoint without transferring')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_checkpoint(args.source_checkpoint)
    else:
        create_transfer_checkpoint(
            args.source_checkpoint,
            args.target_vertices,
            args.target_k,
            args.output,
            args.hidden_dim,
            args.num_layers,
            args.asymmetric
        )


if __name__ == "__main__":
    main()
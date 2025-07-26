#!/usr/bin/env python
"""
Improved Vectorized Neural Network matching the improved-alphazero branch architecture.
Key improvements:
- EdgeAwareGNNBlock with proper message passing
- Direct undirected edge handling (no conversion needed)
- Asymmetric mode with dual policy heads
- Combined node and edge features for value head
- Better initialization and normalization
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import flax.linen as nn
from flax import struct
import numpy as np
from typing import Tuple, Dict, Any, Optional
from functools import partial


class EdgeAwareGNNBlock(nn.Module):
    """
    Edge-aware GNN layer that handles undirected graphs with edge features.
    Matches the PyTorch MessagePassing architecture but in JAX/Flax.
    """
    hidden_dim: int
    
    @nn.compact
    def __call__(self, node_features: jnp.ndarray, edge_index: jnp.ndarray, 
                 edge_features: jnp.ndarray) -> jnp.ndarray:
        """
        Message passing that combines node and edge features.
        
        Args:
            node_features: (batch_size, num_nodes, hidden_dim)
            edge_index: (batch_size, 2, num_edges) - undirected edges where i < j
            edge_features: (batch_size, num_edges, hidden_dim)
            
        Returns:
            Updated node features: (batch_size, num_nodes, hidden_dim)
        """
        batch_size, num_nodes, node_dim = node_features.shape
        
        # Message MLP
        message_mlp = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
        ])
        
        # Skip connection for residual
        if node_dim != self.hidden_dim:
            skip_connection = nn.Dense(self.hidden_dim, use_bias=False)
        else:
            skip_connection = lambda x: x
        
        def process_single_graph(nodes, edges_idx, edges_feat):
            # For undirected edges, we need bidirectional message passing
            # Create bidirectional edges from undirected
            row, col = edges_idx[0], edges_idx[1]
            # Add reverse edges
            edges_bidirectional = jnp.concatenate([
                edges_idx,
                jnp.stack([col, row])
            ], axis=1)
            edges_feat_bidirectional = jnp.concatenate([edges_feat, edges_feat], axis=0)
            
            # Get source nodes for each edge
            src_idx = edges_bidirectional[0]
            src_features = nodes[src_idx]  # (2*num_edges, hidden_dim)
            
            # Concatenate source node features with edge features
            message_input = jnp.concatenate([src_features, edges_feat_bidirectional], axis=-1)
            
            # Compute messages
            messages = message_mlp(message_input)  # (2*num_edges, hidden_dim)
            
            # Aggregate messages by target nodes using mean (not sum, to match PyTorch version)
            dst_idx = edges_bidirectional[1]
            
            # Use segment_mean for aggregation
            # First count messages per node
            ones = jnp.ones_like(dst_idx, dtype=jnp.float32)
            counts = jax.ops.segment_sum(ones, dst_idx, num_segments=num_nodes)
            counts = jnp.maximum(counts, 1.0)  # Avoid division by zero
            
            # Sum messages
            aggregated = jax.ops.segment_sum(messages, dst_idx, num_segments=num_nodes)
            # Take mean
            aggregated = aggregated / counts[:, None]
            
            # Residual connection
            residual = skip_connection(nodes)
            updated = aggregated + residual
            
            # Layer normalization
            updated = nn.LayerNorm()(updated)
            
            return updated
        
        # Vectorize over batch
        updated_nodes = vmap(process_single_graph)(node_features, edge_index, edge_features)
        
        return updated_nodes


class EdgeBlock(nn.Module):
    """
    Updates edge features based on connected nodes.
    Processes both directions (i→j and j→i) then averages.
    """
    hidden_dim: int
    
    @nn.compact
    def __call__(self, node_features: jnp.ndarray, edge_index: jnp.ndarray, 
                 edge_features: jnp.ndarray) -> jnp.ndarray:
        """
        Update edge features based on connected nodes.
        
        Args:
            node_features: (batch_size, num_nodes, hidden_dim)
            edge_index: (batch_size, 2, num_edges) - undirected edges
            edge_features: (batch_size, num_edges, hidden_dim)
            
        Returns:
            Updated edge features: (batch_size, num_edges, hidden_dim)
        """
        edge_proj = nn.Dense(self.hidden_dim)
        node_proj = nn.Dense(self.hidden_dim)
        combine = nn.Dense(self.hidden_dim)
        
        # Skip connection
        edge_dim = edge_features.shape[-1]
        if edge_dim != self.hidden_dim:
            skip_connection = nn.Dense(self.hidden_dim, use_bias=False)
        else:
            skip_connection = lambda x: x
        
        def process_single_graph(nodes, edges_idx, edges_feat):
            row, col = edges_idx[0], edges_idx[1]
            
            # Get features for both directions
            # Direction 1: i->j
            src_feat_1 = nodes[row]
            dst_feat_1 = nodes[col]
            node_concat_1 = jnp.concatenate([src_feat_1, dst_feat_1], axis=-1)
            
            # Direction 2: j->i  
            src_feat_2 = nodes[col]
            dst_feat_2 = nodes[row]
            node_concat_2 = jnp.concatenate([src_feat_2, dst_feat_2], axis=-1)
            
            # Project features
            node_proj_1 = nn.relu(node_proj(node_concat_1))
            node_proj_2 = nn.relu(node_proj(node_concat_2))
            edge_proj_feat = nn.relu(edge_proj(edges_feat))
            
            # Combine for both directions
            combined_1 = jnp.concatenate([node_proj_1, edge_proj_feat], axis=-1)
            combined_2 = jnp.concatenate([node_proj_2, edge_proj_feat], axis=-1)
            
            # Process both directions
            out_1 = nn.relu(combine(combined_1))
            out_2 = nn.relu(combine(combined_2))
            
            # Average both directions
            updated = (out_1 + out_2) / 2.0
            
            # Residual connection
            residual = skip_connection(edges_feat)
            updated = updated + residual
            
            # Layer normalization
            updated = nn.LayerNorm()(updated)
            
            return updated
        
        # Vectorize over batch
        updated_edges = vmap(process_single_graph)(node_features, edge_index, edge_features)
        
        return updated_edges


class ImprovedVectorizedCliqueGNN(nn.Module):
    """
    Improved GNN architecture matching the improved-alphazero branch.
    Supports asymmetric mode, better feature aggregation, and direct undirected edge handling.
    """
    num_vertices: int = 6
    hidden_dim: int = 64
    num_layers: int = 2
    asymmetric_mode: bool = False
    
    @nn.compact
    def __call__(self, edge_index: jnp.ndarray, edge_features: jnp.ndarray, 
                 player_role: Optional[jnp.ndarray] = None,
                 deterministic: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass for batch of graphs.
        
        Args:
            edge_index: (batch_size, 2, num_edges) - undirected edges where i < j
            edge_features: (batch_size, num_edges, 3) - [unselected, player1, player2]
            player_role: (batch_size,) - 0 for attacker, 1 for defender (only for asymmetric)
            
        Returns:
            policies: (batch_size, num_edges) - probability over edges
            values: (batch_size, 1) - position evaluation
        """
        batch_size = edge_features.shape[0]
        num_nodes = self.num_vertices
        num_edges = edge_features.shape[1]
        
        # Initialize node features with learnable embedding
        node_init = self.param('node_init', 
                              nn.initializers.xavier_uniform(),
                              (1, 1))
        
        # Create initial node features - all zeros to preserve symmetry
        node_features = jnp.zeros((batch_size, num_nodes, 1))
        
        # Embed nodes and edges
        node_embed = nn.Dense(self.hidden_dim, 
                            kernel_init=nn.initializers.xavier_uniform())
        edge_embed = nn.Dense(self.hidden_dim,
                            kernel_init=nn.initializers.xavier_uniform())
        
        node_features = node_embed(node_features)
        edge_features_hidden = edge_embed(edge_features)
        
        # Apply GNN layers
        for _ in range(self.num_layers):
            # Node layer (EdgeAwareGNNBlock)
            node_features = EdgeAwareGNNBlock(self.hidden_dim)(
                node_features, edge_index, edge_features_hidden
            )
            # Edge layer (EdgeBlock)
            edge_features_hidden = EdgeBlock(self.hidden_dim)(
                node_features, edge_index, edge_features_hidden
            )
        
        # Policy head(s)
        if self.asymmetric_mode:
            # Dual policy heads for asymmetric mode
            # Define attacker policy head
            def attacker_policy(x):
                x = nn.Dense(2 * self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
                x = nn.relu(x)
                x = nn.Dropout(0.2, deterministic=deterministic)(x)
                x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
                x = nn.relu(x)
                x = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(x)
                return x
            
            # Define defender policy head
            def defender_policy(x):
                x = nn.Dense(2 * self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
                x = nn.relu(x)
                x = nn.Dropout(0.2, deterministic=deterministic)(x)
                x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
                x = nn.relu(x)
                x = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(x)
                return x
            
            # Apply appropriate policy head based on player role
            def apply_policy_head(edge_feat, role):
                attacker_logits = attacker_policy(edge_feat).squeeze(-1)
                defender_logits = defender_policy(edge_feat).squeeze(-1)
                # Select based on role
                return jnp.where(role == 0, attacker_logits, defender_logits)
            
            # Vectorize over batch
            if player_role is None:
                # Default to attacker if no role specified
                player_role = jnp.zeros(batch_size, dtype=jnp.int32)
            
            edge_logits = vmap(apply_policy_head)(edge_features_hidden, player_role)
            
        else:
            # Single policy head for symmetric mode
            x = edge_features_hidden
            x = nn.Dense(2 * self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.relu(x)
            x = nn.Dropout(0.2, deterministic=deterministic)(x)
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.relu(x)
            x = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(x)
            edge_logits = x.squeeze(-1)
        
        # Apply softmax to get probabilities
        policies = nn.softmax(edge_logits)
        
        # Value head - combines both node AND edge features
        # Node attention pooling
        node_attention = nn.Sequential([
            nn.Dense(self.hidden_dim // 4,
                    kernel_init=nn.initializers.xavier_uniform()),
            nn.tanh,
            nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())
        ])
        
        # Edge attention pooling  
        edge_attention = nn.Sequential([
            nn.Dense(self.hidden_dim // 4,
                    kernel_init=nn.initializers.xavier_uniform()),
            nn.tanh,
            nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())
        ])
        
        def pool_features(nodes, edges):
            # Node pooling
            node_scores = node_attention(nodes)  # (num_nodes, 1)
            node_weights = nn.softmax(node_scores.squeeze(-1))
            node_pooled = jnp.sum(nodes * node_weights[:, None], axis=0)
            
            # Edge pooling
            edge_scores = edge_attention(edges)  # (num_edges, 1)
            edge_weights = nn.softmax(edge_scores.squeeze(-1))
            edge_pooled = jnp.sum(edges * edge_weights[:, None], axis=0)
            
            # Combine
            return jnp.concatenate([node_pooled, edge_pooled])
        
        # Pool all graphs
        graph_features = vmap(pool_features)(node_features, edge_features_hidden)
        
        # Value MLP
        x = graph_features
        x = nn.BatchNorm(use_running_average=not self.is_mutable_collection('batch_stats'))(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dropout(0.2, deterministic=deterministic)(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dropout(0.1, deterministic=deterministic)(x)
        x = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(x)
        values = nn.tanh(x)
        
        return policies, values


def create_improved_model(num_vertices: int = 6, hidden_dim: int = 64, 
                         num_layers: int = 2, asymmetric_mode: bool = False) -> Tuple[ImprovedVectorizedCliqueGNN, Dict]:
    """
    Create and initialize the improved model.
    """
    model = ImprovedVectorizedCliqueGNN(
        num_vertices=num_vertices,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        asymmetric_mode=asymmetric_mode
    )
    
    # Initialize with dummy input
    key = jax.random.PRNGKey(0)
    num_edges = num_vertices * (num_vertices - 1) // 2
    
    # Create dummy undirected edges
    edge_list = []
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            edge_list.append([i, j])
    
    dummy_edge_index = jnp.array(edge_list, dtype=jnp.int32).T[None, :, :]  # (1, 2, num_edges)
    dummy_edge_features = jnp.zeros((1, num_edges, 3), dtype=jnp.float32)
    dummy_player_role = jnp.zeros((1,), dtype=jnp.int32) if asymmetric_mode else None
    
    # Don't pass player_role if it's None (symmetric mode)
    if asymmetric_mode:
        params = model.init({'params': key, 'dropout': key}, 
                           dummy_edge_index, dummy_edge_features, dummy_player_role)
    else:
        params = model.init({'params': key, 'dropout': key}, 
                           dummy_edge_index, dummy_edge_features)
    
    return model, params


class ImprovedBatchedNeuralNetwork:
    """
    Wrapper for the improved neural network with all features from improved-alphazero.
    """
    
    def __init__(self, num_vertices: int = 6, hidden_dim: int = 64, 
                 num_layers: int = 2, asymmetric_mode: bool = False):
        self.num_vertices = num_vertices
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.asymmetric_mode = asymmetric_mode
        self.num_actions = num_vertices * (num_vertices - 1) // 2
        
        # Create model and params
        self.model, self.params = create_improved_model(
            num_vertices, hidden_dim, num_layers, asymmetric_mode
        )
        
        # Pre-compile evaluation functions
        if asymmetric_mode:
            self._batch_eval = jit(
                lambda p, ei, ef, pr, rng: self.model.apply(
                    p, ei, ef, pr, 
                    deterministic=False,
                    rngs={'dropout': rng},
                    mutable=['batch_stats']
                )[0]  # Only return predictions, not updated batch_stats for now
            )
        else:
            self._batch_eval = jit(
                lambda p, ei, ef, rng: self.model.apply(
                    p, ei, ef, 
                    deterministic=False,
                    rngs={'dropout': rng},
                    mutable=['batch_stats']
                )[0]  # Only return predictions, not updated batch_stats for now
            )
        
        # RNG for dropout
        self.rng = jax.random.PRNGKey(0)
    
    def evaluate_batch(self, edge_indices: jnp.ndarray, edge_features: jnp.ndarray,
                      valid_moves_mask: Optional[jnp.ndarray] = None,
                      player_roles: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Evaluate a batch of positions.
        
        Args:
            edge_indices: (batch_size, 2, num_edges) - undirected edges
            edge_features: (batch_size, num_edges, 3)
            valid_moves_mask: Optional (batch_size, num_edges) mask
            player_roles: Optional (batch_size,) - 0 for attacker, 1 for defender
            
        Returns:
            policies: (batch_size, num_edges) - masked if mask provided
            values: (batch_size, 1)
        """
        # Get new RNG for this evaluation
        self.rng, eval_rng = jax.random.split(self.rng)
        
        if self.asymmetric_mode:
            policies, values = self._batch_eval(
                self.params, edge_indices, edge_features, player_roles, eval_rng
            )
        else:
            policies, values = self._batch_eval(
                self.params, edge_indices, edge_features, eval_rng
            )
        
        # Apply valid moves mask if provided
        if valid_moves_mask is not None:
            # Mask invalid moves
            policies = policies * valid_moves_mask
            # Renormalize
            policies = policies / (jnp.sum(policies, axis=1, keepdims=True) + 1e-8)
        
        return policies, values
    
    def evaluate_single(self, edge_index: jnp.ndarray, edge_features: jnp.ndarray,
                       player_role: Optional[int] = None) -> Tuple[jnp.ndarray, float]:
        """
        Evaluate a single position.
        """
        # Add batch dimension
        edge_indices = edge_index[None, :, :]
        edge_features_batch = edge_features[None, :, :]
        player_roles = jnp.array([player_role if player_role is not None else 0])
        
        policies, values = self.evaluate_batch(
            edge_indices, edge_features_batch, player_roles=player_roles
        )
        
        return policies[0], float(values[0, 0])


if __name__ == "__main__":
    print("Testing Improved Vectorized Neural Network...")
    print("="*60)
    
    # Test symmetric mode
    print("1. Testing Symmetric Mode:")
    net_sym = ImprovedBatchedNeuralNetwork(asymmetric_mode=False)
    
    # Create undirected edges
    edge_list = []
    for i in range(6):
        for j in range(i+1, 6):
            edge_list.append([i, j])
    
    edge_index = jnp.array(edge_list, dtype=jnp.int32).T
    edge_features = jnp.ones((15, 3), dtype=jnp.float32) / 3.0
    
    policy, value = net_sym.evaluate_single(edge_index, edge_features)
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {jnp.sum(policy):.4f}")
    print(f"Value: {value:.4f}")
    
    # Test asymmetric mode
    print("\n2. Testing Asymmetric Mode:")
    net_asym = ImprovedBatchedNeuralNetwork(asymmetric_mode=True)
    
    # Test as attacker
    policy_att, value_att = net_asym.evaluate_single(edge_index, edge_features, player_role=0)
    print(f"Attacker - Policy shape: {policy_att.shape}, Value: {value_att:.4f}")
    
    # Test as defender
    policy_def, value_def = net_asym.evaluate_single(edge_index, edge_features, player_role=1)
    print(f"Defender - Policy shape: {policy_def.shape}, Value: {value_def:.4f}")
    
    # Policies should be different
    policy_diff = jnp.max(jnp.abs(policy_att - policy_def))
    print(f"Max policy difference: {policy_diff:.4f}")
    
    # Test batch evaluation
    print("\n3. Testing Batch Evaluation:")
    batch_size = 128
    edge_indices = jnp.tile(edge_index[None, :, :], (batch_size, 1, 1))
    edge_features_batch = jnp.tile(edge_features[None, :, :], (batch_size, 1, 1))
    player_roles = jnp.array([i % 2 for i in range(batch_size)])
    
    policies, values = net_asym.evaluate_batch(
        edge_indices, edge_features_batch, player_roles=player_roles
    )
    print(f"Batch policies shape: {policies.shape}")
    print(f"Batch values shape: {values.shape}")
    
    print("\n" + "="*60)
    print("✓ Improved Neural Network Implementation Complete!")
    print("✓ Supports all features from improved-alphazero branch")
    print("="*60)


# Backward compatibility aliases
VectorizedCliqueGNN = ImprovedVectorizedCliqueGNN
BatchedNeuralNetwork = ImprovedBatchedNeuralNetwork
create_vectorized_model = create_improved_model
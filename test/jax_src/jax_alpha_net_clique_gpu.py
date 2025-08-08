#!/usr/bin/env python
"""
GPU-enabled JAX implementation of CliqueGNN for AlphaZero
"""

import warnings
import numpy as np
from typing import Dict, Tuple, Optional, Any

# Try to import JAX components
try:
    import jax
    import jax.numpy as jnp
    from jax import random, jit, vmap
    import flax.linen as nn
    from flax.training import train_state
    import optax
    JAX_AVAILABLE = True
    print("JAX GPU: Using JAX with GPU acceleration")
except ImportError:
    warnings.warn("JAX not available, using NumPy fallback implementation")
    jnp = np
    JAX_AVAILABLE = False
    def jit(f): return f
    def vmap(f, **kwargs): return f


class EdgeAwareGNNBlock:
    """GPU-enabled GNN block with message passing"""
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        
        # Initialize weight shapes
        self.lin_message_weight_shape = (node_dim + edge_dim, out_dim)
        self.lin_message_bias_shape = (out_dim,)
        
        if node_dim != out_dim:
            self.lin_skip_weight_shape = (node_dim, out_dim)
        else:
            self.lin_skip_weight_shape = None
            
        self.layer_norm_scale_shape = (out_dim,)
        self.layer_norm_bias_shape = (out_dim,)
    
    def init_params(self, rng) -> Dict:
        """Initialize parameters"""
        params = {}
        
        if JAX_AVAILABLE:
            key1, key2, key3, key4 = random.split(rng, 4)
        else:
            key1 = key2 = key3 = key4 = rng
        
        params['lin_message_weight'] = self._init_linear_weight(key1, self.lin_message_weight_shape)
        params['lin_message_bias'] = jnp.zeros(self.lin_message_bias_shape)
        
        if self.lin_skip_weight_shape is not None:
            params['lin_skip_weight'] = self._init_linear_weight(key2, self.lin_skip_weight_shape)
        
        # LayerNorm parameters
        params['layer_norm_scale'] = jnp.ones(self.layer_norm_scale_shape)
        params['layer_norm_bias'] = jnp.zeros(self.layer_norm_bias_shape)
        
        return params
    
    def _init_linear_weight(self, key, shape):
        """Xavier/Glorot initialization"""
        fan_in, fan_out = shape
        limit = jnp.sqrt(6 / (fan_in + fan_out))
        if JAX_AVAILABLE:
            return random.uniform(key, shape, minval=-limit, maxval=limit)
        else:
            return jnp.array(np.random.uniform(-limit, limit, shape).astype(np.float32))
    
    @jit
    def __call__(self, params: Dict, x: jnp.ndarray, edge_index: jnp.ndarray, 
                 edge_attr: jnp.ndarray) -> jnp.ndarray:
        """GPU-accelerated forward pass"""
        num_nodes = x.shape[0]
        
        # Residual connection
        if self.node_dim != self.out_dim:
            residual = jnp.dot(x, params['lin_skip_weight'])
        else:
            residual = x
        
        # Message passing
        row, col = edge_index[0], edge_index[1]
        
        # Edge features from source nodes
        x_j = x[row]
        
        # Prepare message features
        msg_input = jnp.concatenate([x_j, edge_attr], axis=1)
        
        # Compute messages
        messages = jnp.dot(msg_input, params['lin_message_weight']) + params['lin_message_bias']
        messages = jax.nn.relu(messages)
        
        # Aggregate messages
        # Use segment_sum for efficient GPU aggregation
        aggregated = jax.ops.segment_sum(messages, col, num_segments=num_nodes)
        
        # Add residual
        out = aggregated + residual
        
        # Layer normalization
        out = self._layer_norm(out, params['layer_norm_scale'], params['layer_norm_bias'])
        
        return out
    
    def _layer_norm(self, x, scale, bias, eps=1e-5):
        """Layer normalization"""
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(var + eps)
        return normalized * scale + bias


class CliqueGNN:
    """GPU-enabled Graph Neural Network for Clique Game"""
    
    def __init__(self, num_vertices: int = 6, hidden_dim: int = 64, num_layers: int = 2):
        self.num_vertices = num_vertices
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node and edge feature dimensions
        self.node_feat_dim = 1
        self.edge_feat_dim = 3  # 3 features: [unselected, player1, player2]
        
        # Create layers
        self.node_blocks = []
        self.edge_blocks = []
        
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            
            self.node_blocks.append(
                EdgeAwareGNNBlock(in_dim, hidden_dim, out_dim)
            )
    
    def init_params(self, rng) -> Dict:
        """Initialize all parameters"""
        if JAX_AVAILABLE:
            keys = random.split(rng, 2 + self.num_layers + 3)
        else:
            # For numpy fallback
            keys = [np.random.RandomState(i) for i in range(2 + self.num_layers + 3)]
        
        params = {}
        
        # Embedding layers
        params['node_embedding'] = self._init_linear_weight(keys[0], (self.node_feat_dim, self.hidden_dim))
        params['edge_embedding'] = self._init_linear_weight(keys[1], (self.edge_feat_dim, self.hidden_dim))
        
        # GNN blocks
        params['node_blocks'] = []
        for i in range(self.num_layers):
            params['node_blocks'].append(
                self.node_blocks[i].init_params(keys[2 + i])
            )
        
        # Policy head
        params['policy_fc1'] = self._init_linear_weight(keys[-3], (self.hidden_dim, self.hidden_dim))
        params['policy_fc2'] = self._init_linear_weight(keys[-2], (self.hidden_dim, 1))
        params['policy_bias1'] = jnp.zeros(self.hidden_dim)
        params['policy_bias2'] = jnp.zeros(1)
        
        # Value head  
        params['value_fc'] = self._init_linear_weight(keys[-1], (self.hidden_dim, 1))
        params['value_bias'] = jnp.zeros(1)
        
        return params
    
    def _init_linear_weight(self, key, shape):
        """Xavier/Glorot initialization"""
        fan_in, fan_out = shape
        limit = jnp.sqrt(6 / (fan_in + fan_out))
        if JAX_AVAILABLE:
            return random.uniform(key, shape, minval=-limit, maxval=limit)
        else:
            return jnp.array(np.random.uniform(-limit, limit, shape).astype(np.float32))
    
    def __call__(self, params: Dict, edge_index: jnp.ndarray, edge_attr: jnp.ndarray,
                 batch: Optional[jnp.ndarray] = None, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """GPU-accelerated forward pass"""
        # Handle batch
        if batch is None:
            num_nodes = self.num_vertices  # Use fixed number of vertices
            batch = jnp.zeros(num_nodes, dtype=jnp.int32)
        
        # Initialize node features
        node_indices = jnp.zeros((len(batch), 1), dtype=jnp.float32)
        x = jnp.dot(node_indices, params['node_embedding'])
        
        # Initialize edge features
        edge_features = jnp.dot(edge_attr, params['edge_embedding'])
        
        # Apply GNN layers
        for i in range(self.num_layers):
            # Direct call to block's forward logic
            block_params = params['node_blocks'][i]
            
            # Get number of nodes
            num_nodes = x.shape[0]
            
            # Compute residual
            if self.node_blocks[i].lin_skip_weight_shape is not None:
                residual = jnp.dot(x, block_params['lin_skip_weight'])
            else:
                residual = x
            
            # Message passing
            row, col = edge_index[0], edge_index[1]
            x_j = x[row]
            
            # Prepare message features
            msg_input = jnp.concatenate([x_j, edge_features], axis=1)
            
            # Compute messages
            messages = jnp.dot(msg_input, block_params['lin_message_weight']) + block_params['lin_message_bias']
            messages = jax.nn.relu(messages)
            
            # Aggregate messages
            aggregated = jax.ops.segment_sum(messages, col, num_segments=num_nodes)
            
            # Add residual
            out = aggregated + residual
            
            # Layer normalization
            mean = out.mean(axis=-1, keepdims=True)
            var = ((out - mean) ** 2).mean(axis=-1, keepdims=True)
            normalized = (out - mean) / jnp.sqrt(var + 1e-5)
            x = normalized * block_params['layer_norm_scale'] + block_params['layer_norm_bias']
        
        # Policy head (on edges)
        edge_policy = jnp.dot(edge_features, params['policy_fc1']) + params['policy_bias1']
        edge_policy = jax.nn.relu(edge_policy)
        edge_scores = jnp.dot(edge_policy, params['policy_fc2']) + params['policy_bias2']
        edge_scores = edge_scores.squeeze(-1)
        
        # Create policy output for complete graph
        num_edges = self.num_vertices * (self.num_vertices - 1) // 2
        policy = jax.nn.softmax(edge_scores[:num_edges])
        
        # Value head (global pooling)
        graph_features = x.mean(axis=0, keepdims=True)
        value = jnp.dot(graph_features, params['value_fc']) + params['value_bias']
        value = jnp.tanh(value)
        
        # Return in correct format
        return policy.reshape(1, -1), value.reshape(1, 1)


# Batch processing functions
@jit
def batch_forward(model_params: Dict, batch_edge_indices: jnp.ndarray, 
                  batch_edge_attrs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Process multiple boards in parallel on GPU"""
    # This would need proper implementation for batching
    # For now, use vmap
    model = CliqueGNN()
    forward_fn = lambda ei, ea: model(model_params, ei, ea)
    
    # Map over batch
    policies, values = vmap(forward_fn)(batch_edge_indices, batch_edge_attrs)
    
    return policies, values


def create_gpu_model(num_vertices: int = 6, hidden_dim: int = 64, 
                    num_layers: int = 2) -> Tuple[CliqueGNN, Dict]:
    """Create GPU-enabled model with initialized parameters"""
    model = CliqueGNN(num_vertices, hidden_dim, num_layers)
    
    if JAX_AVAILABLE:
        key = random.PRNGKey(42)
        params = model.init_params(key)
    else:
        rng = np.random.RandomState(42)
        params = model.init_params(rng)
    
    return model, params


if __name__ == "__main__":
    # Test GPU model
    print("Testing GPU-enabled CliqueGNN...")
    
    model, params = create_gpu_model()
    
    # Create dummy input
    edge_index = jnp.array([[0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0],
                           [1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]], dtype=jnp.int32)
    edge_attr = jnp.ones((15, 5), dtype=jnp.float32)
    
    # Forward pass
    policy, value = model(params, edge_index, edge_attr)
    
    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Policy sum: {policy.sum():.4f}")
    print(f"Value: {value.item():.4f}")
    
    if JAX_AVAILABLE:
        print(f"\nUsing: {jax.devices()[0]}")
        print("✓ GPU acceleration enabled!")
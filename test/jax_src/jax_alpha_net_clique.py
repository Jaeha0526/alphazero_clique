#!/usr/bin/env python
"""
JAX/Flax implementation of CliqueGNN.
Maintains exact same architecture as PyTorch version for compatibility.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import warnings

# Try to import JAX components, fall back to numpy if not available
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import flax.linen as nn
    from flax.training import train_state
    import optax
    JAX_AVAILABLE = True
except ImportError:
    warnings.warn("JAX not available, using NumPy fallback implementation")
    jnp = np
    JAX_AVAILABLE = False


# Define the architecture using NumPy for now (will be converted to JAX when available)
class EdgeAwareGNNBlock:
    """JAX version of EdgeAwareGNNBlock with message passing"""
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
    
    def init_params(self, rng):
        """Initialize parameters"""
        params = {}
        
        # Message linear layer
        key1, key2, key3, key4 = random.split(rng, 4) if JAX_AVAILABLE else [rng] * 4
        params['lin_message_weight'] = self._init_linear_weight(key1, self.lin_message_weight_shape)
        params['lin_message_bias'] = np.zeros(self.lin_message_bias_shape)
        
        # Skip connection
        if self.lin_skip_weight_shape:
            params['lin_skip_weight'] = self._init_linear_weight(key2, self.lin_skip_weight_shape)
        
        # Layer norm
        params['layer_norm_scale'] = np.ones(self.layer_norm_scale_shape)
        params['layer_norm_bias'] = np.zeros(self.layer_norm_bias_shape)
        
        return params
    
    def _init_linear_weight(self, key, shape):
        """Xavier/Glorot initialization"""
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        if JAX_AVAILABLE:
            return random.uniform(key, shape, minval=-limit, maxval=limit)
        else:
            return np.random.uniform(-limit, limit, shape).astype(np.float32)
    
    def __call__(self, params, x, edge_index, edge_attr):
        """Forward pass"""
        # Message passing
        messages = self._message_passing(params, x, edge_index, edge_attr)
        
        # Residual connection
        if 'lin_skip_weight' in params:
            residual = np.dot(x, params['lin_skip_weight'])
        else:
            residual = x
            
        updated_nodes = messages + residual
        
        # Layer normalization
        updated_nodes = self._layer_norm(updated_nodes, 
                                       params['layer_norm_scale'], 
                                       params['layer_norm_bias'])
        
        return updated_nodes
    
    def _message_passing(self, params, x, edge_index, edge_attr):
        """Compute messages and aggregate"""
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        # Get source node features for each edge
        x_j = x[src_idx]  # Shape: (num_edges, node_dim)
        
        # Concatenate with edge features
        msg_input = np.concatenate([x_j, edge_attr], axis=1)
        
        # Apply linear transformation and ReLU
        messages = np.dot(msg_input, params['lin_message_weight']) + params['lin_message_bias']
        messages = np.maximum(0, messages)  # ReLU
        
        # Aggregate messages by destination node (sum aggregation)
        num_nodes = x.shape[0]
        aggregated = np.zeros((num_nodes, self.out_dim))
        
        # Aggregate using scatter add
        for i, dst in enumerate(dst_idx):
            aggregated[dst] += messages[i]
            
        return aggregated
    
    def _layer_norm(self, x, scale, bias, eps=1e-5):
        """Layer normalization"""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)
        return scale * normalized + bias


class EdgeBlock:
    """JAX version of EdgeBlock"""
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        
        # Initialize weight shapes
        self.edge_proj_weight_shape = (edge_dim, out_dim)
        self.edge_proj_bias_shape = (out_dim,)
        self.node_proj_weight_shape = (2 * node_dim, out_dim)
        self.node_proj_bias_shape = (out_dim,)
        self.combine_weight_shape = (2 * out_dim, out_dim)
        self.combine_bias_shape = (out_dim,)
        
        if edge_dim != out_dim:
            self.lin_skip_weight_shape = (edge_dim, out_dim)
        else:
            self.lin_skip_weight_shape = None
            
        self.layer_norm_scale_shape = (out_dim,)
        self.layer_norm_bias_shape = (out_dim,)
    
    def init_params(self, rng):
        """Initialize parameters"""
        params = {}
        
        keys = random.split(rng, 6) if JAX_AVAILABLE else [rng] * 6
        
        # Projection layers
        params['edge_proj_weight'] = self._init_linear_weight(keys[0], self.edge_proj_weight_shape)
        params['edge_proj_bias'] = np.zeros(self.edge_proj_bias_shape)
        params['node_proj_weight'] = self._init_linear_weight(keys[1], self.node_proj_weight_shape)
        params['node_proj_bias'] = np.zeros(self.node_proj_bias_shape)
        params['combine_weight'] = self._init_linear_weight(keys[2], self.combine_weight_shape)
        params['combine_bias'] = np.zeros(self.combine_bias_shape)
        
        # Skip connection
        if self.lin_skip_weight_shape:
            params['lin_skip_weight'] = self._init_linear_weight(keys[3], self.lin_skip_weight_shape)
        
        # Layer norm
        params['layer_norm_scale'] = np.ones(self.layer_norm_scale_shape)
        params['layer_norm_bias'] = np.zeros(self.layer_norm_bias_shape)
        
        return params
    
    def _init_linear_weight(self, key, shape):
        """Xavier/Glorot initialization"""
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        if JAX_AVAILABLE:
            return random.uniform(key, shape, minval=-limit, maxval=limit)
        else:
            return np.random.uniform(-limit, limit, shape).astype(np.float32)
    
    def __call__(self, params, x, edge_index, edge_attr):
        """Forward pass"""
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        # Gather node features for each edge
        src_features = x[src_idx]
        dst_features = x[dst_idx]
        
        # Concatenate source and destination features
        node_features = np.concatenate([src_features, dst_features], axis=1)
        
        # Project features
        projected_node = np.dot(node_features, params['node_proj_weight']) + params['node_proj_bias']
        projected_node = np.maximum(0, projected_node)  # ReLU
        
        projected_edge = np.dot(edge_attr, params['edge_proj_weight']) + params['edge_proj_bias']
        projected_edge = np.maximum(0, projected_edge)  # ReLU
        
        # Combine
        combined = np.concatenate([projected_node, projected_edge], axis=1)
        combined_out = np.dot(combined, params['combine_weight']) + params['combine_bias']
        combined_activated = np.maximum(0, combined_out)  # ReLU
        
        # Residual connection
        if 'lin_skip_weight' in params:
            residual = np.dot(edge_attr, params['lin_skip_weight'])
        else:
            residual = edge_attr
            
        out = combined_activated + residual
        
        # Layer normalization
        out = self._layer_norm(out, params['layer_norm_scale'], params['layer_norm_bias'])
        
        return out
    
    def _layer_norm(self, x, scale, bias, eps=1e-5):
        """Layer normalization"""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)
        return scale * normalized + bias


class EnhancedPolicyHead:
    """JAX version of EnhancedPolicyHead"""
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Multi-head attention parameters
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        
        # Weight shapes
        self.attn_qkv_weight_shape = (hidden_dim, 3 * hidden_dim)
        self.attn_out_weight_shape = (hidden_dim, hidden_dim)
        
        # Policy network shapes
        self.policy_fc1_weight_shape = (hidden_dim, hidden_dim * 2)
        self.policy_fc1_bias_shape = (hidden_dim * 2,)
        self.policy_fc2_weight_shape = (hidden_dim * 2, hidden_dim * 2)
        self.policy_fc2_bias_shape = (hidden_dim * 2,)
        self.policy_fc3_weight_shape = (hidden_dim * 2, hidden_dim)
        self.policy_fc3_bias_shape = (hidden_dim,)
        
        # Residual and output shapes
        self.residual_weight_shape = (hidden_dim, hidden_dim)
        self.residual_bias_shape = (hidden_dim,)
        self.output_weight_shape = (hidden_dim, 1)
        self.output_bias_shape = (1,)
        
        # Layer norm shapes
        self.ln1_scale_shape = (hidden_dim,)
        self.ln1_bias_shape = (hidden_dim,)
        self.ln2_scale_shape = (hidden_dim,)
        self.ln2_bias_shape = (hidden_dim,)
        self.ln3_scale_shape = (hidden_dim,)
        self.ln3_bias_shape = (hidden_dim,)
    
    def init_params(self, rng):
        """Initialize parameters"""
        params = {}
        
        keys = random.split(rng, 15) if JAX_AVAILABLE else [rng] * 15
        idx = 0
        
        # Attention
        params['attn_qkv_weight'] = self._init_linear_weight(keys[idx], self.attn_qkv_weight_shape); idx += 1
        params['attn_out_weight'] = self._init_linear_weight(keys[idx], self.attn_out_weight_shape); idx += 1
        
        # Policy network
        params['policy_fc1_weight'] = self._init_linear_weight(keys[idx], self.policy_fc1_weight_shape); idx += 1
        params['policy_fc1_bias'] = np.zeros(self.policy_fc1_bias_shape)
        params['policy_fc2_weight'] = self._init_linear_weight(keys[idx], self.policy_fc2_weight_shape); idx += 1
        params['policy_fc2_bias'] = np.zeros(self.policy_fc2_bias_shape)
        params['policy_fc3_weight'] = self._init_linear_weight(keys[idx], self.policy_fc3_weight_shape); idx += 1
        params['policy_fc3_bias'] = np.zeros(self.policy_fc3_bias_shape)
        
        # Residual and output
        params['residual_weight'] = self._init_linear_weight(keys[idx], self.residual_weight_shape); idx += 1
        params['residual_bias'] = np.zeros(self.residual_bias_shape)
        params['output_weight'] = self._init_linear_weight(keys[idx], self.output_weight_shape); idx += 1
        params['output_bias'] = np.zeros(self.output_bias_shape)
        
        # Layer norms
        params['ln1_scale'] = np.ones(self.ln1_scale_shape)
        params['ln1_bias'] = np.zeros(self.ln1_bias_shape)
        params['ln2_scale'] = np.ones(self.ln2_scale_shape)
        params['ln2_bias'] = np.zeros(self.ln2_bias_shape)
        params['ln3_scale'] = np.ones(self.ln3_scale_shape)
        params['ln3_bias'] = np.zeros(self.ln3_bias_shape)
        
        return params
    
    def _init_linear_weight(self, key, shape):
        """Xavier/Glorot initialization"""
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        if JAX_AVAILABLE:
            return random.uniform(key, shape, minval=-limit, maxval=limit)
        else:
            return np.random.uniform(-limit, limit, shape).astype(np.float32)
    
    def __call__(self, params, x, training=False):
        """Forward pass"""
        # Multi-head attention
        attn_output = self._multi_head_attention(params, x)
        
        # Policy network with layer norm
        x_norm = self._layer_norm(attn_output, params['ln1_scale'], params['ln1_bias'])
        
        # FC layers with GELU activation
        h = np.dot(x_norm, params['policy_fc1_weight']) + params['policy_fc1_bias']
        h = self._gelu(h)
        if training and self.dropout_rate > 0:
            h = self._dropout(h, self.dropout_rate)
            
        h = np.dot(h, params['policy_fc2_weight']) + params['policy_fc2_bias']
        h = self._gelu(h)
        if training and self.dropout_rate > 0:
            h = self._dropout(h, self.dropout_rate)
            
        policy_features = np.dot(h, params['policy_fc3_weight']) + params['policy_fc3_bias']
        
        # Residual connection
        x_norm2 = self._layer_norm(x, params['ln2_scale'], params['ln2_bias'])
        residual_features = np.dot(x_norm2, params['residual_weight']) + params['residual_bias']
        
        combined = policy_features + residual_features
        
        # Final output
        combined_norm = self._layer_norm(combined, params['ln3_scale'], params['ln3_bias'])
        output = np.dot(combined_norm, params['output_weight']) + params['output_bias']
        
        return output
    
    def _multi_head_attention(self, params, x):
        """Simple multi-head self-attention"""
        batch_size = x.shape[0]
        
        # QKV projection
        qkv = np.dot(x, params['attn_qkv_weight'])
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim)
        qkv = np.transpose(qkv, (1, 2, 0, 3))  # (3, num_heads, batch, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = np.matmul(q, np.transpose(k, (0, 2, 1))) / np.sqrt(self.head_dim)
        attn_weights = self._softmax(scores, axis=-1)
        
        # Apply attention to values
        attn_output = np.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = np.transpose(attn_output, (1, 0, 2))  # (batch, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, self.hidden_dim)
        attn_output = np.dot(attn_output, params['attn_out_weight'])
        
        return attn_output
    
    def _gelu(self, x):
        """GELU activation"""
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    
    def _softmax(self, x, axis=-1):
        """Stable softmax"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _dropout(self, x, rate):
        """Dropout (only during training)"""
        # Simple dropout implementation
        keep_prob = 1 - rate
        mask = np.random.binomial(1, keep_prob, x.shape) / keep_prob
        return x * mask
    
    def _layer_norm(self, x, scale, bias, eps=1e-5):
        """Layer normalization"""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)
        return scale * normalized + bias


class CliqueGNN:
    """JAX version of CliqueGNN"""
    def __init__(self, num_vertices: int = 6, hidden_dim: int = 64, num_layers: int = 2):
        self.num_vertices = num_vertices
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initialize layers
        self.node_embedding_shape = (1, hidden_dim)
        self.edge_embedding_shape = (3, hidden_dim)
        
        # GNN blocks
        self.node_blocks = []
        self.edge_blocks = []
        for _ in range(num_layers):
            self.node_blocks.append(EdgeAwareGNNBlock(hidden_dim, hidden_dim, hidden_dim))
            self.edge_blocks.append(EdgeBlock(hidden_dim, hidden_dim, hidden_dim))
        
        # Policy and value heads
        self.policy_head = EnhancedPolicyHead(hidden_dim)
        
        # Value head shapes
        self.value_fc1_shape = (hidden_dim, hidden_dim // 2)
        self.value_fc1_bias_shape = (hidden_dim // 2,)
        self.value_fc2_shape = (hidden_dim // 2, 1)
        self.value_fc2_bias_shape = (1,)
    
    def init_params(self, rng):
        """Initialize all parameters"""
        params = {}
        
        if JAX_AVAILABLE:
            keys = random.split(rng, 2 + 2 * self.num_layers + 2)
        else:
            keys = [rng] * (2 + 2 * self.num_layers + 2)
        key_idx = 0
        
        # Embeddings
        params['node_embedding'] = self._init_linear_weight(keys[key_idx], self.node_embedding_shape)
        key_idx += 1
        params['edge_embedding'] = self._init_linear_weight(keys[key_idx], self.edge_embedding_shape)
        key_idx += 1
        
        # GNN blocks
        params['node_blocks'] = []
        params['edge_blocks'] = []
        for i in range(self.num_layers):
            params['node_blocks'].append(self.node_blocks[i].init_params(keys[key_idx]))
            key_idx += 1
            params['edge_blocks'].append(self.edge_blocks[i].init_params(keys[key_idx]))
            key_idx += 1
        
        # Policy head
        params['policy_head'] = self.policy_head.init_params(keys[key_idx])
        key_idx += 1
        
        # Value head
        params['value_fc1_weight'] = self._init_linear_weight(keys[key_idx], self.value_fc1_shape)
        params['value_fc1_bias'] = np.zeros(self.value_fc1_bias_shape)
        params['value_fc2_weight'] = self._init_linear_weight(keys[key_idx], self.value_fc2_shape)
        params['value_fc2_bias'] = np.zeros(self.value_fc2_bias_shape)
        
        return params
    
    def _init_linear_weight(self, key, shape):
        """Xavier/Glorot initialization"""
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        if JAX_AVAILABLE:
            return random.uniform(key, shape, minval=-limit, maxval=limit)
        else:
            return np.random.uniform(-limit, limit, shape).astype(np.float32)
    
    def __call__(self, params, edge_index, edge_attr, batch=None, training=False):
        """Forward pass"""
        # Handle batch
        if batch is None:
            num_nodes = int(edge_index.max()) + 1
            batch = np.zeros(num_nodes, dtype=np.int32)
        
        # Initialize node features
        node_indices = np.zeros((len(batch), 1), dtype=np.float32)
        x = np.dot(node_indices, params['node_embedding'])
        
        # Initialize edge features
        edge_features = np.dot(edge_attr, params['edge_embedding'])
        
        # Apply GNN layers
        for i in range(self.num_layers):
            x_new = self.node_blocks[i](params['node_blocks'][i], x, edge_index, edge_features)
            edge_features = self.edge_blocks[i](params['edge_blocks'][i], x, edge_index, edge_features)
            x = x_new
        
        # Policy head (on edges)
        edge_scores = self.policy_head(params['policy_head'], edge_features, training)
        
        # Process policy output to match PyTorch format
        num_graphs = len(np.unique(batch))
        num_nodes_per_graph = self.num_vertices
        num_edges_per_graph = num_nodes_per_graph * (num_nodes_per_graph - 1) // 2
        
        # Create policy output tensor
        policy_outputs = []
        
        for g in range(num_graphs):
            # Create edge-to-index mapping for complete graph
            edge_to_idx = {}
            idx = 0
            for i in range(num_nodes_per_graph):
                for j in range(i+1, num_nodes_per_graph):
                    edge_to_idx[(i, j)] = idx
                    edge_to_idx[(j, i)] = idx
                    idx += 1
            
            # Initialize policy vector for this graph
            graph_policy = np.zeros(num_edges_per_graph)
            
            # Fill in edge scores
            edge_idx = 0
            for i in range(len(edge_index[0])):
                src, dst = int(edge_index[0][i]), int(edge_index[1][i])
                # Map global node indices to local
                if g == 0:  # Single graph case
                    local_src, local_dst = src, dst
                else:
                    # Multi-graph: need to adjust indices
                    local_src = src % num_nodes_per_graph
                    local_dst = dst % num_nodes_per_graph
                
                if local_src < local_dst:
                    canonical_edge = (local_src, local_dst)
                else:
                    canonical_edge = (local_dst, local_src)
                
                if canonical_edge in edge_to_idx:
                    policy_idx = edge_to_idx[canonical_edge]
                    if policy_idx < num_edges_per_graph:
                        graph_policy[policy_idx] = edge_scores[edge_idx].item()
                
                edge_idx += 1
            
            policy_outputs.append(graph_policy)
        
        # Format output to match PyTorch
        if num_graphs == 1:
            policy_output = np.array(policy_outputs[0]).reshape(1, num_edges_per_graph)
        else:
            policy_output = np.array(policy_outputs)
        
        # Value head (on aggregated node features)
        # Aggregate node features by graph
        graph_features = []
        for g in range(num_graphs):
            mask = (batch == g)
            graph_nodes = x[mask]
            graph_feat = graph_nodes.mean(axis=0)
            graph_features.append(graph_feat)
        
        if num_graphs == 1:
            graph_features = graph_features[0].reshape(1, -1)
        else:
            graph_features = np.array(graph_features)
        
        # Value network
        value = np.dot(graph_features, params['value_fc1_weight']) + params['value_fc1_bias']
        value = np.maximum(0, value)  # ReLU
        value = np.dot(value, params['value_fc2_weight']) + params['value_fc2_bias']
        value = np.tanh(value)  # Output between -1 and 1
        
        # Format to match PyTorch output shape
        value_output = value.reshape(num_graphs, 1, 1)
        
        return policy_output, value_output


# Helper function to convert PyTorch state dict to JAX params
def convert_pytorch_to_jax(pytorch_state_dict, jax_model):
    """Convert PyTorch state dict to JAX parameters"""
    # This will be implemented when we have both models
    # For now, return empty dict
    return {}


# Helper function to prepare graph data
def prepare_graph_data(board):
    """Convert board state to graph format for GNN"""
    from src.encoder_decoder_clique import prepare_state_for_network
    
    # Use the existing encoder
    state_dict = prepare_state_for_network(board)
    
    # Convert to numpy arrays
    edge_index = state_dict['edge_index'].numpy()
    edge_attr = state_dict['edge_attr'].numpy()
    
    return edge_index, edge_attr
#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
from typing import Tuple, List, Dict, Optional
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import MessagePassing # Import MessagePassing
from torch_geometric.data import Data, Batch, DataLoader as PyGDataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import argparse

DEBUG = False

class CliqueGameData(Dataset):
    def __init__(self, examples):
        # Filter out any clearly invalid examples
        self.examples = []
        for example in examples:
            try:
                if isinstance(example, dict):
                    board_state = example['board_state']
                    policy = example['policy']
                    value = example['value']
                else:
                    print(f"Skipping example: wrong format")
                    continue
                
                # Basic validation checks
                if isinstance(policy, np.ndarray) and isinstance(value, (int, float, np.ndarray)):
                    self.examples.append((board_state, policy, value))
                else:
                    print(f"Skipping example: invalid types - policy: {type(policy)}, value: {type(value)}")
            except Exception as e:
                print(f"Error initializing example: {e}")
                continue
        
        print(f"Kept {len(self.examples)}/{len(examples)} examples after validation")
        # print(f"[DEBUG] Examples: {self.examples}")
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        try:
            board_state, policy, value = self.examples[idx]
            
            # Convert board state to graph format
            edge_index, edge_attr = self._board_to_graph(board_state)
            
            # Verify edge_index integrity
            if edge_index.numel() == 0 or edge_index.min() < 0:
                raise ValueError(f"Invalid edge_index: min={edge_index.min().item() if edge_index.numel() > 0 else 'empty'}")
            
            # Determine number of nodes
            num_nodes = edge_index.max().item() + 1 if edge_index.numel() > 0 else 0
            
            if num_nodes == 0:
                raise ValueError("No nodes in graph")
            
            # # Debug information
            # print(f"Graph {idx}:")
            # print(f"  Number of nodes: {num_nodes}")
            # print(f"  Edge index shape: {edge_index.shape}")
            # print(f"  Edge index min/max: {edge_index.min().item()}/{edge_index.max().item()}")
            # print(f"  Edge attr shape: {edge_attr.shape}")
            
            # Calculate expected policy size (number of edges in complete graph)
            expected_policy_size = num_nodes * (num_nodes - 1) // 2
            
            # Ensure policy is the right shape and on CPU
            if isinstance(policy, np.ndarray):
                policy = torch.tensor(policy, dtype=torch.float)
                
                # Check if policy size matches expected size
                if policy.shape[0] != expected_policy_size:
                    print(f"Warning: Policy shape mismatch in example {idx}. Expected: {expected_policy_size}, Got: {policy.shape[0]}")
                    # Resize policy if needed
                    if policy.shape[0] < expected_policy_size:
                        # Pad with zeros
                        padding = torch.zeros(expected_policy_size - policy.shape[0])
                        policy = torch.cat([policy, padding])
                    else:
                        # Truncate
                        policy = policy[:expected_policy_size]
                    
                    # Renormalize
                    if policy.sum() > 0:
                        policy = policy / policy.sum()
            
            # Ensure value is a single float on CPU
            if isinstance(value, np.ndarray):
                value = torch.tensor(value.item() if value.size == 1 else value[0], dtype=torch.float)
            else:
                value = torch.tensor(value, dtype=torch.float)
            
            # Create PyG Data object with explicit num_nodes
            return Data(
                edge_index=edge_index,
                edge_attr=edge_attr,
                policy=policy,
                value=value,
                num_nodes=num_nodes
            )
        except Exception as e:
            print(f"Error getting item {idx}: {e}")
            # Return a simple default graph as fallback
            num_nodes = 2
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
            edge_attr = torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.float)
            policy = torch.zeros(1, dtype=torch.float)
            value = torch.tensor(0.0, dtype=torch.float)
            
            return Data(
                edge_index=edge_index,
                edge_attr=edge_attr,
                policy=policy,
                value=value,
                num_nodes=num_nodes
            )
    
    def _board_to_graph(self, board_state):
        try:
            # Check if board_state is already in the correct format
            if isinstance(board_state, dict) and 'edge_index' in board_state and 'edge_attr' in board_state:
                return torch.tensor(board_state['edge_index'], dtype=torch.long), \
                       torch.tensor(board_state['edge_attr'], dtype=torch.float)
            
            # If not, try to convert from old format
            if isinstance(board_state, dict):
                if 'edge_states' in board_state and 'num_vertices' in board_state:
                    edge_states = board_state['edge_states']
                    num_vertices = board_state['num_vertices']
                else:
                    raise KeyError("Required keys not found in board_state")
            else:
                edge_states = board_state
                num_vertices = edge_states.shape[0]
            
            # Ensure edge_states is a numpy array
            if not isinstance(edge_states, np.ndarray):
                edge_states = np.array(edge_states)
            
            # Validate num_vertices
            if num_vertices <= 0:
                raise ValueError(f"Invalid num_vertices: {num_vertices}")
            
            # Create edge indices for all possible edges
            edge_index = []
            edge_attr = []
            
            # Add edges between all pairs of vertices
            for i in range(num_vertices):
                for j in range(i + 1, num_vertices):
                    # Bounds check on edge_states
                    if i < edge_states.shape[0] and j < edge_states.shape[1]:
                        edge_index.append([i, j])
                        edge_index.append([j, i])  # Add reverse edge
                        
                        # Create one-hot encoding for edge state (safely)
                        state = int(edge_states[i, j])
                        if state < 0 or state > 2:
                            state = 0  # Default to unselected if invalid
                        
                        edge_features = [0, 0, 0]  # [unselected, player1, player2]
                        edge_features[state] = 1
                        edge_attr.append(edge_features)
                        edge_attr.append(edge_features)  # Same features for reverse edge
            
            # Add self-loops
            for i in range(num_vertices):
                edge_index.append([i, i])
                edge_attr.append([1, 0, 0])  # Special feature for self-loops
            
            # Check if we have any edges
            if not edge_index:
                raise ValueError("No edges created in _board_to_graph")
            
            return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), \
                   torch.tensor(edge_attr, dtype=torch.float)
                   
        except Exception as e:
            print(f"Error in _board_to_graph: {e}")
            print(f"Board state type: {type(board_state)}")
            if isinstance(board_state, dict):
                print(f"Board state keys: {list(board_state.keys())}")
            
            # Return a minimal valid graph as a fallback
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
            edge_attr = torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.float)
            return edge_index, edge_attr

# New GNN Block incorporating edge features using MessagePassing
class EdgeAwareGNNBlock(MessagePassing):
    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__(aggr='add') # Use 'add' aggregation
        # Linear layer to process concatenated node and edge features for message creation
        self.lin_message = nn.Linear(node_dim + edge_dim, out_dim)
        # Layer normalization applied AFTER residual addition
        self.layer_norm = nn.LayerNorm(out_dim)
        # Optional: Add a final linear transformation within the update step if needed
        # self.lin_update = nn.Linear(out_dim, out_dim)

        # Linear layer for the residual connection if dimensions differ
        if node_dim != out_dim:
            self.lin_skip = nn.Linear(node_dim, out_dim, bias=False) # Bias often False in skip connections
        else:
            self.lin_skip = nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        # x: [N, node_dim], Node features
        # edge_index: [2, E], Graph connectivity
        # edge_attr: [E, edge_dim], Edge features

        # Start the message passing process.
        # The `message` method now includes ReLU.
        aggregated_messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Add residual connection (transform original x if needed)
        residual = self.lin_skip(x)
        updated_nodes = aggregated_messages + residual

        # Apply layer normalization AFTER the residual connection
        updated_nodes = self.layer_norm(updated_nodes)

        # No final activation here, non-linearity is within message/update
        return updated_nodes

    def message(self, x_j, edge_attr):
        # x_j: [E, node_dim] - features of source nodes j for each edge (j, i)
        # edge_attr: [E, edge_dim] - features of the edge (j, i)

        # Concatenate source node features and edge features.
        msg = torch.cat([x_j, edge_attr], dim=1)
        # Transform the concatenated features to create the message.
        msg = self.lin_message(msg)
        # Apply non-linearity (ReLU) within the message function
        return F.relu(msg)

    # Optional: If more complex update logic is needed beyond aggregation
    # def update(self, aggr_out, x):
    #     # aggr_out contains the sum of F.relu(self.lin_message(msg))
    #     # Add residual connection
    #     residual = self.lin_skip(x)
    #     updated_nodes = aggr_out + residual
    #     # Apply layer normalization
    #     return self.layer_norm(updated_nodes)

class EdgeBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim

        # Linear layers to process node and edge features
        self.edge_proj = nn.Linear(edge_dim, out_dim)
        self.node_proj = nn.Linear(2 * node_dim, out_dim)
        # Combine layer - potentially add activation after this
        self.combine = nn.Linear(2 * out_dim, out_dim)
        # Layer normalization applied AFTER residual addition
        self.layer_norm = nn.LayerNorm(out_dim)

        # Linear layer for the residual connection if dimensions differ
        if edge_dim != out_dim:
            self.lin_skip = nn.Linear(edge_dim, out_dim, bias=False)
        else:
            self.lin_skip = nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        # Get source and target node features
        src, dst = edge_index[0], edge_index[1]

        # Gather node features for each edge
        src_features = x[src]  # [num_edges, node_dim]
        dst_features = x[dst]  # [num_edges, node_dim]

        # Concatenate source and destination features
        node_features = torch.cat([src_features, dst_features], dim=1)  # [num_edges, 2*node_dim]

        # Project node and edge features
        # Apply activation after projection?
        projected_node_features = F.relu(self.node_proj(node_features))  # [num_edges, out_dim]
        projected_edge_features = F.relu(self.edge_proj(edge_attr))  # [num_edges, out_dim]

        # Combine node and edge information
        combined = torch.cat([projected_node_features, projected_edge_features], dim=1)  # [num_edges, 2*out_dim]
        combined_out = self.combine(combined)  # [num_edges, out_dim]
        # Apply activation after combine?
        combined_activated = F.relu(combined_out)

        # Add residual connection (transform original edge_attr if needed)
        residual = self.lin_skip(edge_attr)
        out = combined_activated + residual

        # Apply layer normalization AFTER residual connection
        out = self.layer_norm(out)

        # No final activation here
        return out

class EnhancedPolicyHead(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        
        # First use multi-head attention to focus on relevant edges
        self.edge_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout_rate
        )
        
        # Main policy network with residual connections
        self.policy_network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.GELU(),  # GELU often works better than ReLU for policy networks
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        
        # Residual connection
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final prediction layer
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # Apply self-attention to focus on important edges
        attn_output, _ = self.edge_attention(x, x, x)
        
        # Apply main policy network
        policy_features = self.policy_network(attn_output)
        
        # Add residual connection
        residual_features = self.residual(x)
        combined = policy_features + residual_features
        
        # Final output
        return self.output(combined)


class CliqueGNN(nn.Module):
    def __init__(self, num_vertices=6, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input embedding layers with initialization
        self.node_embedding = nn.Linear(1, hidden_dim)  # Assuming node features are just indices initially
        self.edge_embedding = nn.Linear(3, hidden_dim)  # 3 features for edge state [unselected, p1, p2]
        
        # Initialize embeddings with Xavier (helps with training convergence)
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.xavier_uniform_(self.edge_embedding.weight)
        
        # Dynamically create GNN layers
        self.node_layers = nn.ModuleList()
        self.edge_layers = nn.ModuleList()
        for _ in range(num_layers):
            # Replace GNNBlock with EdgeAwareGNNBlock
            self.node_layers.append(EdgeAwareGNNBlock(hidden_dim, hidden_dim, hidden_dim))
            self.edge_layers.append(EdgeBlock(hidden_dim, hidden_dim, hidden_dim))
        
        # Simpler policy head with properly initialized weights
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout to prevent overfitting
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize policy head weights
        for m in self.policy_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Global attention pooling for nodes
        self.node_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Global attention pooling for edges
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Enhanced value head that combines both node and edge features
        # Input size is 2*hidden_dim (concatenated node and edge global features)
        self.value_head = nn.Sequential(
            nn.BatchNorm1d(2*hidden_dim),  # Batch norm for combined features
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Add residual block
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Final value prediction
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Initialize value head weights
        for m in self.value_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, edge_index, edge_attr, batch=None, debug=False):
        # If no batch assignment is provided, assume single graph
        if batch is None:
            batch = torch.zeros(edge_index.max().item() + 1, dtype=torch.long, device=edge_index.device)
        
        # Get number of nodes per graph
        num_nodes_per_graph = torch.bincount(batch)
        num_graphs = len(num_nodes_per_graph)
        
        # Initialize node features
        # node_indices = torch.arange(len(batch), device=edge_index.device).float().unsqueeze(1)
        node_indices = torch.zeros(len(batch), device=edge_index.device).float().unsqueeze(1)
        x = self.node_embedding(node_indices)
        if debug:
            print(f"[DEBUG] Node indices: {node_indices}")
            print(f"[DEBUG] Node features: {x}")
            print(f"[DEBUG] Edge index: {edge_index}")
            print(f"[DEBUG] Edge index shape: {edge_index.shape}")
            print(f"[DEBUG] Edge attributes: {edge_attr}")
            print(f"[DEBUG] Edge attributes shape: {edge_attr.shape}")
            print(f"[DEBUG] Batch: {batch}")
        
        # Initialize edge features
        edge_features = self.edge_embedding(edge_attr)
        if debug:
            print(f"[DEBUG] Edge features: {edge_features}")
            print(f"[DEBUG] Edge features shape: {edge_features.shape}")
            pass
        
        # Apply GNN layers
        for i, (node_layer, edge_layer) in enumerate(zip(self.node_layers, self.edge_layers)):
            # Pass edge_features to the node_layer (EdgeAwareGNNBlock)
            x_new = node_layer(x, edge_index, edge_features) 
            edge_features = edge_layer(x, edge_index, edge_features)
            x = x_new
            if debug:
                print(f"[DEBUG] Edge features after layer {i}: {edge_features}")
                print(f"[DEBUG] Node features after layer {i}: {x}")
        
        # --- Debug Start --- 
        # Print sample edge features before policy head
        # if edge_features.numel() > 0:
        #      print(f"DEBUG: Edge features into policy_head (sample): {edge_features[0]}")
        # else:
        #      print("DEBUG: Edge features tensor is empty!")
        # --- Debug End --- 
        
        # Generate edge scores
        edge_scores = self.policy_head(edge_features)
        if debug:
            print(f"[DEBUG] Edge scores: {edge_scores}")
            print(f"[DEBUG] Edge scores shape: {edge_scores.shape}")
        
        # --- Debug Start --- 
        # Print sample edge scores after policy head
        # if edge_scores.numel() > 0:
        #      print(f"DEBUG: Edge scores from policy_head (sample): {edge_scores[0]}")
        # else:
        #      print("DEBUG: Edge scores tensor is empty!")
        # --- Debug End ---
        
        # Process each graph in the batch separately
        policies = []
        values = []
        
        # Calculate cumulative sum of nodes for indexing
        node_cumsum = torch.cat([torch.tensor([0], device=edge_index.device), torch.cumsum(num_nodes_per_graph, dim=0)])
        
        for i in range(num_graphs):
            # Get node indices for this graph
            start_idx = node_cumsum[i]
            end_idx = node_cumsum[i + 1]
            num_nodes = num_nodes_per_graph[i]
            
            # Get edges for this graph
            graph_mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
            graph_edge_index = edge_index[:, graph_mask]
            graph_edge_scores = edge_scores[graph_mask]
            
            # Calculate number of edges in complete graph for this size
            num_edges = num_nodes * (num_nodes - 1) // 2
            
            # Create edge mapping for this graph
            edge_map = {}
            idx = 0
            for j in range(num_nodes):
                for k in range(j+1, num_nodes):
                    edge_map[(j, k)] = idx
                    edge_map[(k, j)] = idx
                    idx += 1
            
            # --- Reworked Mapping Logic ---
            # Create a temporary dict to hold scores for canonical edges
            canonical_scores = {}
            # population_count = 0 # Debug Counter
            for j in range(graph_edge_index.shape[1]): # Loop through edges found for this graph
                # Explicitly cast to int after .item()
                src_item = graph_edge_index[0, j].item() - start_idx
                dst_item = graph_edge_index[1, j].item() - start_idx
                src = int(src_item)
                dst = int(dst_item)
                
                if src != dst: # Ignore self-loops for policy assignment
                     canonical_edge = tuple(sorted((src, dst))) # Key is now definitely (int, int)
                     # Store the score associated with this edge (take first occurrence if duplicated)
                     if canonical_edge not in canonical_scores:
                          score_value = graph_edge_scores[j].item() # Get the score as a float
                          canonical_scores[canonical_edge] = score_value # Use .item()
                          # if score_value != 0.0: # Debug Print
                          #      population_count += 1
                          #      print(f"DEBUG: Populating canonical_scores[{canonical_edge}] = {score_value}") # ADDED

            # ADDED Debug Print: Show the dictionary contents
            # print(f"DEBUG: Populated canonical_scores dict ({population_count} non-zero entries): {canonical_scores}")
    
            # Now, fill the policy vector using the canonical scores
            policy = torch.zeros(num_edges, device=edge_index.device)
            edge_map_items = edge_map.items()
    
            # assigned_count = 0 # Debug counter
            for edge_tuple, edge_idx in edge_map_items:
                 # edge_map contains both (j,k) and (k,j) mapping to same index
                 # We only need to assign once per index, using the canonical tuple
                 canonical_edge = tuple(sorted(edge_tuple))
                 if edge_tuple == canonical_edge: # Process only for the canonical key e.g. (0, 1) not (1, 0)
                     score = canonical_scores.get(canonical_edge, 0.0) # Get score, default to 0.0 if missing
                     policy[edge_idx] = score
                     # if score != 0.0: # Debug print
                     #      assigned_count += 1
                     #      # print(f"DEBUG: Assigning score {score} to index {edge_idx} for edge {canonical_edge}")

            # print(f"DEBUG: Assigned {assigned_count} non-zero scores to policy vector") # Debug print
            # --- End Reworked Mapping Logic ---
            
            # Print policy vector before softmax
            # print(f"DEBUG: Graph {i} Policy before softmax: {policy}")
            # --- Debug End --- 
            
            # Normalize policy
            policy = F.softmax(policy, dim=0)
            policies.append(policy)
            
            # Extract nodes and edges for this graph
            graph_nodes = x[start_idx:end_idx]  # [num_nodes, hidden_dim]
            
            # Extract edges for this graph 
            graph_mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
            graph_edge_features = edge_features[graph_mask]  # [num_edges, hidden_dim]
            
            # Apply attention-based pooling for nodes
            if len(graph_nodes) > 0:  # Ensure we have nodes
                # Calculate attention scores for nodes
                node_attn_scores = self.node_attention(graph_nodes)  # [num_nodes, 1]
                node_attn_weights = F.softmax(node_attn_scores, dim=0)  # [num_nodes, 1]
                # Weighted sum of node features
                node_features_pooled = torch.sum(graph_nodes * node_attn_weights, dim=0)  # [hidden_dim]
            else:
                # Fallback if no nodes (shouldn't happen)
                node_features_pooled = torch.zeros(self.hidden_dim, device=edge_index.device)
            
            # Apply attention-based pooling for edges
            if len(graph_edge_features) > 0:  # Ensure we have edges
                # Calculate attention scores for edges
                edge_attn_scores = self.edge_attention(graph_edge_features)  # [num_edges, 1]
                edge_attn_weights = F.softmax(edge_attn_scores, dim=0)  # [num_edges, 1]
                # Weighted sum of edge features
                edge_features_pooled = torch.sum(graph_edge_features * edge_attn_weights, dim=0)  # [hidden_dim]
            else:
                # Fallback if no edges (shouldn't happen)
                edge_features_pooled = torch.zeros(self.hidden_dim, device=edge_index.device)
            
            # Concatenate node and edge pooled features
            combined_features = torch.cat([node_features_pooled, edge_features_pooled], dim=0)  # [2*hidden_dim]
            
            # Store the combined features for batch processing
            if self.training:
                # During training, we'll collect all value inputs and process them as a batch
                values.append(combined_features)  # [2*hidden_dim]
            else:
                # During inference, handle batch norm for single instance
                value = self.value_head(combined_features.unsqueeze(0))  # Shape [1, 1]
                values.append(value)
        
        # Stack policies
        policies = torch.stack(policies)  # [num_graphs, num_edges]
        
        # Handle values based on training/inference mode
        if self.training:
            # In training mode, we process all value inputs together through the value head
            values_tensor = torch.stack(values)  # [num_graphs, 2*hidden_dim]
            values = self.value_head(values_tensor)  # [num_graphs, 1]
        else:
            # In inference mode, we've already processed each value input separately
            values = torch.stack(values)  # [num_graphs, 1]
        
        return policies, values

    def train_network(self, train_examples: List, start_iter: int, num_iterations: int, 
              save_interval: int = 10, device: torch.device = None,
              args: argparse.Namespace = None, # Pass args object
             ) -> float:
        """
        Train the network on the given examples.
        Expects LR parameters within the args object.
        
        Returns:
            avg_epoch_loss: Average loss for this epoch/training run
        """
        if device is None:
            device = torch.device("cpu")
        
        print(f"Training on device: {device}")
        self.to(device)
        self.train()
        
        # Initialize optimizer with learning rate scheduler
        # Get parameters from args object (with defaults if args is None)
        initial_lr = getattr(args, 'initial_lr', 0.0001)
        lr_factor = getattr(args, 'lr_factor', 0.95)
        lr_patience = getattr(args, 'lr_patience', 7)
        lr_threshold = getattr(args, 'lr_threshold', 1e-5)
        min_lr = getattr(args, 'min_lr', 1e-7)

        print(f"Initializing Adam with LR: {initial_lr}") # Log the LR used
        optimizer = optim.Adam(self.parameters(), lr=initial_lr, weight_decay=1e-4)
        print(f"Initializing ReduceLROnPlateau with factor={lr_factor}, patience={lr_patience}, threshold={lr_threshold}, min_lr={min_lr}") # Log scheduler params
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
            threshold=lr_threshold,
            threshold_mode='rel'
        )
        
        # Create dataset and dataloader
        train_dataset = CliqueGameData(train_examples)
        # Use batch_size from args
        batch_size = getattr(args, 'batch_size', 16) # Default if args is None
        print(f"Using batch size: {batch_size}")
        train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                   num_workers=0, pin_memory=True) # Use 0 workers if issues arise

        # --- LR Warmup Setup ---
        warmup_fraction = 0.15
        warmup_steps = int(warmup_fraction * num_iterations)
        print(f"LR Warmup enabled for the first {warmup_steps} steps.")
        # --- End Warmup Setup ---
        
        # Training loop (now uses num_training_steps)
        print(f"Starting training loop for {num_iterations} steps...")
        step = 0
        done = False
        epoch_loss = 0.0
        num_batches = 0
        
        # Performance tracking
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        for batch_data in train_loader:
            if step >= num_iterations:
                break
            
            self.train() # Ensure model is in training mode
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            # --- LR Warmup Logic ---
            if step < warmup_steps and warmup_steps > 0:
                # Calculate linearly increasing LR
                current_lr = initial_lr * (step + 1) / warmup_steps
                # Apply LR to optimizer
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            # --- End LR Warmup Logic ---
            
            # Forward pass
            try:
                if DEBUG:
                    print(f"[DEBUG] Batch data: {batch_data}")
                    print(f"[DEBUG] Batch data edge attr: {batch_data.edge_attr[-25:]}")
                    print(f"[DEBUG] Batch data edge index: {batch_data.edge_index[-25:]}")
                policy_output, value_output = self(batch_data.edge_index, batch_data.edge_attr, batch=batch_data.batch, debug=DEBUG)
            except Exception as e:
                print(f"Error during forward pass at step {step}: {e}")
                # Handle potential batch-related errors (e.g., size mismatch)
                continue # Skip this batch
            
            # Add small epsilon to policy output for stability
            policy_output_stable = policy_output + 1e-8 # Use stable version for log
            
            # Calculate losses
            # Reshape policy target to match output shape [batch_size, num_edges]
            policy_target = batch_data.policy
            num_graphs = batch_data.num_graphs # Get number of graphs in batch
            num_edges = policy_output.shape[1] # Get num_edges from output tensor
            
            if policy_target.numel() == num_graphs * num_edges:
                 policy_target = policy_target.view(num_graphs, num_edges)
            else:
                # Add error handling or logging if reshape is not possible
                print(f"ERROR: Cannot reshape policy_target (size {policy_target.numel()}) to ({num_graphs}, {num_edges})")
                continue # Skip this batch
            
            # Improved policy loss calculation
            # 1. Create a valid moves mask
            valid_moves_mask = (policy_target > 1e-7)
            
            # 2. Create a KL-divergence based loss that focuses on valid moves
            # Get raw logits from the policy network
            log_probs = torch.log(policy_output_stable) # Use original output + epsilon
            
            # Calculate KL divergence loss only on valid moves
            # We multiply by policy_target to weight more important moves higher
            policy_loss_terms = -policy_target * log_probs * valid_moves_mask
            
            # Sum over the edge dimension for each graph
            policy_loss_per_graph = torch.sum(policy_loss_terms, dim=1)
            
            # Average over the batch
            policy_loss = policy_loss_per_graph.mean()
            
            # Value loss with label smoothing to prevent the model from being too confident
            value_target = batch_data.value
            # Apply label smoothing: move targets slightly toward zero
            smoothing_factor = 0.1
            smoothed_value_target = value_target * (1 - smoothing_factor)
            
            # Huber loss is more robust to outliers than MSE
            value_loss = F.smooth_l1_loss(value_output.squeeze(), smoothed_value_target)
            
            # Dynamically balance policy and value losses
            # This prevents one loss from dominating the other
            policy_weight = 1.0
            value_weight = getattr(args, 'value_weight', 1.0)  # Default to 1.0 if not specified
            
            # Add L2 regularization to prevent overfitting
            l2_reg = 0.0
            for param in self.parameters():
                l2_reg += torch.norm(param)
            l2_reg *= 1e-5  # Small regularization factor
            
            # Combined loss
            total_loss = policy_weight * policy_loss + value_weight * value_loss + l2_reg
            
            # Track losses
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # Backward pass and optimize
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Print progress
            if step % 20 == 0:
                print(f"Step: {step}/{num_iterations}, Policy Loss: {policy_loss.item():.6f}, Value Loss: {value_loss.item():.6f}")
            step += 1

        # End of epoch logic
        avg_epoch_loss = epoch_loss / max(1, num_batches)
        avg_policy_loss = policy_loss_sum / max(1, num_batches)
        avg_value_loss = value_loss_sum / max(1, num_batches)
        
        print(f"Training completed. Steps: {step}, Avg Loss: {avg_epoch_loss:.6f}")
        print(f"  Policy Loss: {avg_policy_loss:.6f}, Value Loss: {avg_value_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.7f}")
        
        # Step the scheduler based on epoch loss
        scheduler.step(avg_epoch_loss)
        
        return avg_epoch_loss

class CliqueAlphaLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_value, value, y_policy, policy):
        # MSE loss for value
        value_error = (value - y_value) ** 2
        
        # Cross-entropy loss for policy (handling very small probabilities)
        policy_error = torch.sum((-policy * 
                              (1e-8 + y_policy.float()).float().log()), 1)
        
        # Ensure dimensions match before adding
        total_error = (value_error.float() + policy_error).mean()
        return total_error

def create_encoder_decoder_for_clique(num_vertices):
    """
    Create functions to encode/decode moves for the Clique Game.
    
    For the Clique Game, moves are edges between vertices.
    
    Args:
        num_vertices: Number of vertices in the graph
        
    Returns:
        encode_move: Function to encode a move (edge) to an index
        decode_move: Function to decode an index to a move (edge)
    """
    # Create a mapping from edge to index
    edge_to_idx = {}
    idx_to_edge = {}
    
    idx = 0
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            edge_to_idx[(i, j)] = idx
            idx_to_edge[idx] = (i, j)
            idx += 1
    
    def encode_move(edge):
        """Convert an edge (i,j) to a flat index"""
        i, j = min(edge), max(edge)  # Ensure i < j
        return edge_to_idx.get((i, j), -1)
    
    def decode_move(idx):
        """Convert a flat index to an edge (i,j)"""
        return idx_to_edge.get(idx, (-1, -1))
    
    return encode_move, decode_move

# Example usage
if __name__ == "__main__":
    from clique_board import CliqueBoard
    
    # Create a model
    model = CliqueGNN(num_vertices=6)
    print(model)
    
    # Create a sample board
    board = CliqueBoard(6, 3)  # 6 vertices, need 3-clique to win
    
    # Make some moves
    board.make_move((0, 1))  # Player 1
    board.make_move((1, 2))  # Player 2
    board.make_move((2, 3))  # Player 1
    
    # Get board state
    board_state = board.get_board_state()
    
    # Convert to graph representation
    dataset = CliqueGameData([[board_state, np.ones(15)/15, 0.5]])
    edge_index, edge_attr, policy, value = dataset[0]
    
    # Forward pass
    policy_pred, value_pred = model(edge_index, edge_attr)
    
    print(f"Policy shape: {policy_pred.shape}")
    print(f"Value: {value_pred.item()}") 
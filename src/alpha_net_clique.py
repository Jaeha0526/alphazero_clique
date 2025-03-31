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
from torch_geometric.data import Data, Batch, DataLoader as PyGDataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import argparse

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

class GNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = pyg_nn.GCNConv(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x

class EdgeBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        
        # Linear layers to process node and edge features
        self.edge_proj = nn.Linear(edge_dim, out_dim)
        self.node_proj = nn.Linear(2 * node_dim, out_dim)
        self.combine = nn.Linear(2 * out_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
    
    def forward(self, x, edge_index, edge_attr):
        # Get source and target node features
        src, dst = edge_index[0], edge_index[1]
        
        # Gather node features for each edge
        src_features = x[src]  # [num_edges, node_dim]
        dst_features = x[dst]  # [num_edges, node_dim]
        
        # Concatenate source and destination features
        node_features = torch.cat([src_features, dst_features], dim=1)  # [num_edges, 2*node_dim]
        
        # Project node and edge features
        node_features = self.node_proj(node_features)  # [num_edges, out_dim]
        edge_features = self.edge_proj(edge_attr)  # [num_edges, out_dim]
        
        # Combine node and edge information
        combined = torch.cat([node_features, edge_features], dim=1)  # [num_edges, 2*out_dim]
        out = self.combine(combined)  # [num_edges, out_dim]
        
        # Apply normalization and activation
        out = self.batch_norm(out)
        out = F.relu(out)
        
        return out

class CliqueGNN(nn.Module):
    def __init__(self, num_vertices=6, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input embedding layers
        self.node_embedding = nn.Linear(1, hidden_dim)  # Assuming node features are just indices initially
        self.edge_embedding = nn.Linear(3, hidden_dim)  # 3 features for edge state [unselected, p1, p2]
        
        # Dynamically create GNN layers
        self.node_layers = nn.ModuleList()
        self.edge_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.node_layers.append(GNNBlock(hidden_dim, hidden_dim))
            self.edge_layers.append(EdgeBlock(hidden_dim, hidden_dim, hidden_dim))
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 1) # Outputs score for each edge
        )
        
        # Value head
        # Input to value head is mean of node features from the last layer
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh() # Output between -1 and 1
        )
    
    def forward(self, edge_index, edge_attr, batch=None):
        # If no batch assignment is provided, assume single graph
        if batch is None:
            batch = torch.zeros(edge_index.max().item() + 1, dtype=torch.long, device=edge_index.device)
        
        # Get number of nodes per graph
        num_nodes_per_graph = torch.bincount(batch)
        num_graphs = len(num_nodes_per_graph)
        
        # Initialize node features
        node_indices = torch.arange(len(batch), device=edge_index.device).float().unsqueeze(1)
        x = self.node_embedding(node_indices)
        
        # Initialize edge features
        edge_features = self.edge_embedding(edge_attr)
        
        # Apply GNN layers
        for node_layer, edge_layer in zip(self.node_layers, self.edge_layers):
            x_new = node_layer(x, edge_index)
            edge_features = edge_layer(x, edge_index, edge_features)
            x = x_new
        
        # --- Debug Start --- 
        # Print sample edge features before policy head
        # if edge_features.numel() > 0:
        #      print(f"DEBUG: Edge features into policy_head (sample): {edge_features[0]}")
        # else:
        #      print("DEBUG: Edge features tensor is empty!")
        # --- Debug End --- 
        
        # Generate edge scores
        edge_scores = self.policy_head(edge_features)
        
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
            
            # Calculate value for this graph
            graph_nodes = x[start_idx:end_idx]
            value_input = torch.mean(graph_nodes, dim=0, keepdim=True)
            value = self.value_head(value_input)
            values.append(value)
        
        # Stack policies and values instead of concatenating
        policies = torch.stack(policies)  # [num_graphs, num_edges]
        values = torch.stack(values)  # [num_graphs, 1]
        
        return policies, values

    def train_network(self, train_examples: List, start_iter: int, num_iterations: int, 
              save_interval: int = 10, device: torch.device = None,
              args: argparse.Namespace = None # Pass args object
             ) -> None:
        """
        Train the network on the given examples.
        Expects LR parameters within the args object.
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
        optimizer = optim.Adam(self.parameters(), lr=initial_lr)
        print(f"Initializing ReduceLROnPlateau with factor={lr_factor}, patience={lr_patience}, threshold={lr_threshold}, min_lr={min_lr}") # Log scheduler params
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=lr_factor,
            patience=lr_patience,
            verbose=True,
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

        # Training loop (now uses num_training_steps)
        print(f"Starting training loop for {num_iterations} steps...")
        step = 0
        done = False
        epoch = 0 # Track epochs for scheduler
        
        while not done:
            epoch_loss = 0.0
            num_batches = 0
            for batch_data in train_loader:
                if step >= num_iterations:
                    done = True
                    break
                
                self.train() # Ensure model is in training mode
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                try:
                    policy_output, value_output = self(batch_data.edge_index, batch_data.edge_attr, batch=batch_data.batch)
                except Exception as e:
                    print(f"Error during forward pass at step {step}: {e}")
                    # Handle potential batch-related errors (e.g., size mismatch)
                    # Option 1: Skip batch (might lose data)
                    # continue 
                    # Option 2: Try to recover (difficult)
                    # Option 3: Raise error (stops training)
                    raise e
                
                # Add small epsilon to policy output for stability
                policy_output = policy_output + 1e-8
                
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
                
                # Policy loss (Cross-Entropy)
                policy_loss = -torch.sum(policy_target * torch.log(policy_output), dim=1).mean()
                
                # Value loss (MSE)
                value_target = batch_data.value
                value_loss = F.mse_loss(value_output.squeeze(), value_target)
                
                total_loss = policy_loss + value_loss
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # Backward pass and optimize
                total_loss.backward()
                optimizer.step()
                
                # Print progress
                if step % 50 == 0:
                    print(f"Step: {step}/{num_iterations}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
                step += 1

            # End of epoch logic
            if not done:
                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
                print(f"End of Epoch {epoch}. Avg Loss: {avg_epoch_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.7f}")
                # Step the scheduler based on epoch loss (or validation loss if validation is done per epoch)
                scheduler.step(avg_epoch_loss)
                epoch += 1
                
        print(f"Training finished after {step} steps.")

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
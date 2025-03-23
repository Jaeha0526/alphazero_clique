#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
from typing import Tuple, List, Dict, Optional
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse

class CliqueGameData(Dataset):
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        board_state, policy, value = self.examples[idx]
        
        # Convert board state to graph format
        edge_index, edge_attr = self._board_to_graph(board_state)
        
        return Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            policy=torch.tensor(policy, dtype=torch.float),
            value=torch.tensor(value, dtype=torch.float)
        )
    
    def _board_to_graph(self, board_state):
        try:
            # Check what format the board state is in
            if isinstance(board_state, dict):
                # Try the standard format first
                if 'edge_states' in board_state and 'num_vertices' in board_state:
                    edge_states = board_state['edge_states']
                    num_vertices = board_state['num_vertices']
                # For backwards compatibility or different formats
                elif 'states' in board_state:
                    edge_states = board_state['states']
                    num_vertices = edge_states.shape[0]
                else:
                    # Print available keys for debugging
                    print(f"Board state keys: {list(board_state.keys())}")
                    raise KeyError("Required keys not found in board_state")
            else:
                # If board_state is not a dict but the raw edge states array
                edge_states = board_state
                num_vertices = edge_states.shape[0]
            
            # Create edge indices for all possible edges
            edge_index = []
            edge_attr = []
            
            # Add edges between all pairs of vertices
            for i in range(num_vertices):
                for j in range(i + 1, num_vertices):
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # Add reverse edge
                    
                    # Create one-hot encoding for edge state
                    state = edge_states[i, j]  # Note: edge_states is a numpy array
                    edge_features = [0, 0, 0]  # [unselected, player1, player2]
                    edge_features[state] = 1
                    edge_attr.append(edge_features)
                    edge_attr.append(edge_features)  # Same features for reverse edge
            
            # Add self-loops
            for i in range(num_vertices):
                edge_index.append([i, i])
                edge_attr.append([1, 0, 0])  # Special feature for self-loops
            
            return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), \
                   torch.tensor(edge_attr, dtype=torch.float)
                   
        except Exception as e:
            print(f"Error in _board_to_graph: {e}")
            print(f"Board state type: {type(board_state)}")
            if isinstance(board_state, dict):
                print(f"Board state keys: {list(board_state.keys())}")
            raise

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
    def __init__(self, num_vertices=6, hidden_dim=64):
        super().__init__()
        self.num_vertices = num_vertices
        self.num_edges = num_vertices * (num_vertices - 1) // 2
        self.hidden_dim = hidden_dim
        
        # Initial embeddings
        self.node_embedding = nn.Embedding(num_vertices, hidden_dim)
        self.edge_embedding = nn.Linear(3, hidden_dim)  # 3 features for edge state
        
        # Node update layers
        self.node_layers = nn.ModuleList([
            GNNBlock(hidden_dim, hidden_dim),
            GNNBlock(hidden_dim, hidden_dim)
        ])
        
        # Edge update layers
        self.edge_layers = nn.ModuleList([
            EdgeBlock(hidden_dim, hidden_dim, hidden_dim),
            EdgeBlock(hidden_dim, hidden_dim, hidden_dim)
        ])
        
        # Policy head
        self.policy_head = nn.Linear(hidden_dim, 1)
        
        # Value head
        self.value_pool = pyg_nn.global_mean_pool
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
    
    def forward(self, edge_index, edge_attr):
        # Get number of nodes
        num_nodes = edge_index.max().item() + 1
        
        # Initialize node features
        node_indices = torch.arange(num_nodes, device=edge_index.device)
        x = self.node_embedding(node_indices)  # [num_nodes, hidden_dim]
        
        # Initialize edge features
        edge_features = self.edge_embedding(edge_attr)  # [num_edges, hidden_dim]
        
        # Apply GNN layers
        for node_layer, edge_layer in zip(self.node_layers, self.edge_layers):
            # Update node representations
            x_new = node_layer(x, edge_index)
            
            # Update edge representations using the original node features
            edge_features = edge_layer(x, edge_index, edge_features)
            
            # Update node features
            x = x_new
        
        # Generate policy output
        edge_scores = self.policy_head(edge_features)  # [num_edges, 1]
        
        # Map edge scores to policy
        policy = torch.zeros(self.num_edges, device=edge_index.device)
        edge_map = {}
        idx = 0
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                edge_map[(i, j)] = idx
                edge_map[(j, i)] = idx  # Same index for both directions
                idx += 1
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src < dst:  # Only consider upper triangular
                edge_idx = edge_map.get((src, dst), -1)
                if edge_idx >= 0:
                    policy[edge_idx] = edge_scores[i]
        
        # Normalize policy
        policy = F.softmax(policy, dim=0)
        
        # Generate value output
        batch = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        value = self.value_head(self.value_pool(x, batch))
        
        return policy, value

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

def train(net, dataset, epoch_start=0, epoch_stop=50, cpu=0):
    """Train the network on the given dataset"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.train()
    
    # Create data loader
    train_set = CliqueGameData(dataset)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    
    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epoch_start, epoch_stop):
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            policy_output, value_output = net(batch)
            
            # Calculate losses
            policy_loss = policy_criterion(policy_output, batch.policy)
            value_loss = value_criterion(value_output.squeeze(), batch.value)
            loss = policy_loss + value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            policy_loss += policy_loss.item()
            value_loss += value_loss.item()
        
        # Print epoch statistics
        num_batches = len(train_loader)
        print(f"Epoch {epoch}: Total Loss = {total_loss/num_batches:.4f}, "
              f"Policy Loss = {policy_loss/num_batches:.4f}, "
              f"Value Loss = {value_loss/num_batches:.4f}")

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
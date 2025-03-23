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
    def __init__(self, dataset: List):
        self.X_edge_index = []  # Edge indices (graph structure)
        self.X_edge_attr = []   # Edge attributes (player markings)
        self.y_p = []           # Policy outputs
        self.y_v = []           # Value outputs
        
        # Process each item in the dataset
        for item in dataset:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                board_state, policy, value = item
                # Convert board_state to graph representation
                edge_index, edge_attr = self._board_to_graph(board_state)
                self.X_edge_index.append(edge_index)
                self.X_edge_attr.append(edge_attr)
                self.y_p.append(policy)
                # Make sure value is a scalar
                if isinstance(value, (list, tuple, np.ndarray)):
                    value = value[0] if len(value) > 0 else 0.0
                self.y_v.append(float(value))
            else:
                print(f"Unexpected item structure: {item}")
                continue
        
        # Convert policies and values to tensors
        self.y_p = torch.stack([torch.from_numpy(np.array(p)).float() for p in self.y_p])
        self.y_v = torch.tensor(self.y_v, dtype=torch.float32).view(-1, 1)
        
        print(f"Final shapes - edge_index: {len(self.X_edge_index)}, "
              f"y_p: {self.y_p.shape}, y_v: {self.y_v.shape}")
    
    def _board_to_graph(self, board_state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a board state to a graph representation with edge features.
        
        Args:
            board_state: Dictionary with adjacency_matrix and edge_states
            
        Returns:
            edge_index: Tensor of shape [2, E] containing the indices of the edges
            edge_attr: Tensor of shape [E, F] containing the features of the edges
        """
        # Convert edge_states to features (one-hot encoding)
        # 0: unselected, 1: player1, 2: player2
        num_vertices = board_state['adjacency_matrix'].shape[0]
        edge_states = board_state['edge_states']
        
        # Get edge indices (sparse representation)
        edge_indices = []
        edge_features = []
        
        for i in range(num_vertices):
            for j in range(i+1, num_vertices):  # Only upper triangular
                if board_state['adjacency_matrix'][i, j] == 1:  # if edge exists
                    # Add both directions for undirected graph
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])
                    
                    # One-hot encode the edge state
                    state = edge_states[i, j]
                    if state == 0:  # unselected
                        feat = [1, 0, 0]
                    elif state == 1:  # player 1
                        feat = [0, 1, 0]
                    else:  # player 2
                        feat = [0, 0, 1]
                    
                    edge_features.append(feat)
                    edge_features.append(feat)  # Same feature for both directions
        
        # Add self-loops for each vertex with special feature
        for i in range(num_vertices):
            edge_indices.append([i, i])
            edge_features.append([0, 0, 0])  # Special feature for self-loops
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()  # Shape [2, E]
        edge_attr = torch.tensor(edge_features, dtype=torch.float)     # Shape [E, 3]
        
        return edge_index, edge_attr
    
    def __len__(self) -> int:
        return len(self.X_edge_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Return graph data, policy, and value
        return self.X_edge_index[idx], self.X_edge_attr[idx], self.y_p[idx], self.y_v[idx]
    
    def collate_fn(self, batch):
        """Custom collate function for batching graphs of different sizes"""
        edge_indices, edge_attrs, policies, values = zip(*batch)
        
        # Create a batch of graphs
        # This is where we'd use PyTorch Geometric's Batch class
        # For now, we'll just return the separate components
        return edge_indices, edge_attrs, torch.stack(policies), torch.stack(values)

class GNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = pyg_nn.GCNConv(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x

class EdgeGNNBlock(nn.Module):
    def __init__(self, in_edge_channels, out_edge_channels, node_channels):
        super().__init__()
        self.edge_conv = pyg_nn.EdgeConv(
            nn.Sequential(
                nn.Linear(2 * node_channels + in_edge_channels, out_edge_channels),
                nn.ReLU(),
                nn.Linear(out_edge_channels, out_edge_channels)
            ),
            aggr='add'
        )
        self.batch_norm = nn.BatchNorm1d(out_edge_channels)
        
    def forward(self, x, edge_index, edge_attr):
        # Process edge features
        edge_attr = self.edge_conv(x, edge_index, edge_attr)
        edge_attr = self.batch_norm(edge_attr)
        edge_attr = F.relu(edge_attr)
        return edge_attr

class CliqueGNN(nn.Module):
    def __init__(self, num_vertices=6):
        super().__init__()
        # Number of possible edges in a complete graph
        self.num_edges = num_vertices * (num_vertices - 1) // 2
        
        # Initial node embeddings
        self.node_embedding = nn.Embedding(num_vertices, 32)
        
        # Edge feature processing
        self.edge_embedding = nn.Linear(3, 32)  # 3 features for edge state
        
        # Graph layers
        self.gnn_layers = nn.ModuleList([
            GNNBlock(32, 64),
            GNNBlock(64, 128),
            GNNBlock(128, 128)
        ])
        
        # Edge processing layers
        self.edge_layers = nn.ModuleList([
            EdgeGNNBlock(32, 64, 32),
            EdgeGNNBlock(64, 128, 64),
            EdgeGNNBlock(128, 128, 128)
        ])
        
        # Policy head (outputs probability for each edge)
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Value head (outputs game state evaluation)
        self.value_node_pool = pyg_nn.global_mean_pool
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
    def forward(self, edge_index, edge_attr):
        # Initialize node features
        batch_size = 1  # For single graph inference
        num_vertices = edge_index.max().item() + 1
        
        # Create node indices - ensure they're on the same device as edge_index
        node_indices = torch.arange(num_vertices, device=edge_index.device)
        
        # Get node embeddings
        x = self.node_embedding(node_indices)
        
        # Process edge features
        edge_features = self.edge_embedding(edge_attr)
        
        # Apply GNN layers
        for i, (gnn_layer, edge_layer) in enumerate(zip(self.gnn_layers, self.edge_layers)):
            # Update node representations
            x = gnn_layer(x, edge_index)
            
            # Update edge representations
            edge_features = edge_layer(x, edge_index, edge_features)
        
        # Generate policy output (one value per edge)
        # We need to map edge_index to the original edges of the complete graph
        edge_scores = self.policy_head(edge_features)
        
        # Extract only the upper triangular part (original edges)
        policy = torch.zeros(self.num_edges, device=edge_index.device)
        
        # Create a mapping from edge_index to the flattened adjacency matrix
        edge_map = {}
        idx = 0
        for i in range(num_vertices):
            for j in range(i+1, num_vertices):
                edge_map[(i, j)] = idx
                edge_map[(j, i)] = idx  # Same index for both directions
                idx += 1
        
        # Fill policy vector
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src < dst:  # Only consider upper triangular
                edge_idx = edge_map.get((src, dst), -1)
                if edge_idx >= 0:
                    policy[edge_idx] = edge_scores[i]
        
        # Normalize policy to get a probability distribution
        policy = F.softmax(policy, dim=0)
        
        # Generate value output
        batch = torch.zeros(num_vertices, dtype=torch.long, device=edge_index.device)
        value = self.value_head(self.value_node_pool(x, batch))
        
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

def train(net, dataset, epoch_start=0, epoch_stop=20, cpu=0):
    # Set device
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpu_id = cpu % num_gpus
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Process {cpu} using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print(f"Process {cpu} using CPU")
    
    torch.manual_seed(cpu)
    
    net.to(device)
    net.train()
    criterion = CliqueAlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                             milestones=[50, 100, 150], 
                                             gamma=0.2)
    
    train_set = CliqueGameData(dataset)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, 
                            num_workers=2, collate_fn=train_set.collate_fn)
    
    losses_per_epoch = []
    for epoch in range(epoch_start, epoch_stop):
        total_loss = 0.0
        batch_count = 0
        
        for i, (edge_indices, edge_attrs, policies, values) in enumerate(train_loader):
            batch_count += 1
            
            # Process each graph in the batch
            batch_loss = 0.0
            for edge_index, edge_attr, policy, value in zip(edge_indices, edge_attrs, policies, values):
                # Move data to device
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                policy = policy.to(device)
                value = value.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                policy_pred, value_pred = net(edge_index, edge_attr)
                
                # Compute loss
                loss = criterion(value_pred, value, policy_pred, policy)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                optimizer.step()
                batch_loss += loss.item()
            
            # Average loss for the batch
            batch_loss /= len(edge_indices)
            total_loss += batch_loss
            
            if i % 5 == 4:
                print(f'Process ID: {os.getpid()} [Epoch: {epoch + 1}, '
                      f'Batch: {i + 1}/{len(train_loader)}] '
                      f'loss per batch: {batch_loss:.3f}')
        
        # Step the scheduler after each epoch
        scheduler.step()
        
        # Average loss for the epoch
        epoch_loss = total_loss / batch_count
        losses_per_epoch.append(epoch_loss)
        print(f'Epoch {epoch + 1} completed. Average loss: {epoch_loss:.3f}')
        
        # Early stopping
        if len(losses_per_epoch) > 20:
            recent_avg = sum(losses_per_epoch[-3:]) / 3
            earlier_avg = sum(losses_per_epoch[-10:-7]) / 3
            if abs(recent_avg - earlier_avg) <= 0.005:
                print("Early stopping triggered")
                break

    # Plot training loss
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(losses_per_epoch)+1), losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss vs Epoch")
    plt.savefig(os.path.join("./model_data/", 
                           f"Clique_GNN_Loss_{datetime.datetime.today().strftime('%Y-%m-%d')}.png"))
    print('Finished Training')

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
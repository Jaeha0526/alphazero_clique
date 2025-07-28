"""
Efficient board that stores only edge states, not full adjacency matrix.
This reduces memory and copy overhead while keeping the same interface.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple


class EfficientCliqueBoard:
    """
    Board that stores edges as a flat array instead of adjacency matrix.
    Preserves the same interface as VectorizedCliqueBoard.
    """
    
    def __init__(self, batch_size: int, num_vertices: int, k: int, game_mode: str = "symmetric"):
        self.batch_size = batch_size
        self.num_vertices = num_vertices
        self.k = k
        self.game_mode = game_mode
        
        # Store edges as flat array instead of n x n matrix
        self.num_edges = num_vertices * (num_vertices - 1) // 2
        self.edge_states = jnp.zeros((batch_size, self.num_edges), dtype=jnp.int32)
        
        # Game state
        self.current_players = jnp.ones(batch_size, dtype=jnp.int32)
        self.game_states = jnp.zeros(batch_size, dtype=jnp.int32)
        self.winners = jnp.zeros(batch_size, dtype=jnp.int32)
        self.move_counts = jnp.zeros(batch_size, dtype=jnp.int32)
        
        # Precompute edge to vertex mapping
        self._edge_to_vertices = self._compute_edge_mapping()
    
    def _compute_edge_mapping(self):
        """Compute mapping from edge index to vertex pairs."""
        mapping = []
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                mapping.append((i, j))
        return jnp.array(mapping)
    
    def copy(self):
        """Efficient copy - only copies small arrays."""
        new_board = EfficientCliqueBoard(
            self.batch_size, self.num_vertices, self.k, self.game_mode
        )
        new_board.edge_states = self.edge_states.copy()
        new_board.current_players = self.current_players.copy()
        new_board.game_states = self.game_states.copy()
        new_board.winners = self.winners.copy()
        new_board.move_counts = self.move_counts.copy()
        return new_board
    
    def get_single_board(self, idx: int):
        """Get a single board from batch."""
        board = EfficientCliqueBoard(1, self.num_vertices, self.k, self.game_mode)
        board.edge_states = self.edge_states[idx:idx+1].copy()
        board.current_players = self.current_players[idx:idx+1].copy()
        board.game_states = self.game_states[idx:idx+1].copy()
        board.winners = self.winners[idx:idx+1].copy()
        board.move_counts = self.move_counts[idx:idx+1].copy()
        return board
    
    def make_moves(self, actions: jnp.ndarray):
        """Make moves - same interface as original."""
        active_mask = self.game_states == 0
        batch_indices = jnp.arange(self.batch_size)
        
        # Update edge states
        self.edge_states = self.edge_states.at[batch_indices, actions].set(
            jnp.where(active_mask, self.current_players, self.edge_states[batch_indices, actions])
        )
        
        # Update move counts
        self.move_counts = jnp.where(active_mask, self.move_counts + 1, self.move_counts)
        
        # Check for cliques
        self._check_cliques_for_moves(actions)
        
        # Switch players
        if self.game_mode == "alternating":
            self.current_players = jnp.where(
                active_mask & (self.game_states == 0),
                3 - self.current_players,
                self.current_players
            )
    
    def get_valid_moves_mask(self):
        """Get valid moves - edges that haven't been taken."""
        return self.edge_states == 0
    
    def get_features_for_nn_undirected(self):
        """Get features for neural network."""
        # Convert edge states to adjacency matrix format for compatibility
        adjacency = jnp.zeros((self.batch_size, self.num_vertices, self.num_vertices))
        
        for idx, (i, j) in enumerate(self._edge_to_vertices):
            adjacency = adjacency.at[:, i, j].set(self.edge_states[:, idx])
            adjacency = adjacency.at[:, j, i].set(self.edge_states[:, idx])
        
        # Create edge indices and features
        edge_indices = []
        edge_features = []
        
        for b in range(self.batch_size):
            edges = []
            features = []
            
            # Add edges for adjacency
            for idx, (i, j) in enumerate(self._edge_to_vertices):
                if self.edge_states[b, idx] != 0:
                    edges.extend([[i, j], [j, i]])
                    player = self.edge_states[b, idx]
                    features.extend([[player], [player]])
            
            # Add self-loops
            for i in range(self.num_vertices):
                edges.append([i, i])
                features.append([0])
            
            edge_indices.append(jnp.array(edges).T if edges else jnp.zeros((2, 0)))
            edge_features.append(jnp.array(features))
        
        return edge_indices, edge_features
    
    def _check_cliques_for_moves(self, actions):
        """Check if moves created k-cliques."""
        # Simplified - would need actual clique detection
        # For now just check if game should end
        all_edges_taken = jnp.all(self.edge_states != 0, axis=1)
        self.game_states = jnp.where(all_edges_taken, 3, self.game_states)
        self.winners = jnp.where(all_edges_taken, 0, self.winners)
    
    @property
    def adjacency_matrices(self):
        """Property for compatibility - converts to adjacency matrix on demand."""
        adjacency = jnp.zeros((self.batch_size, self.num_vertices, self.num_vertices))
        for idx, (i, j) in enumerate(self._edge_to_vertices):
            adjacency = adjacency.at[:, i, j].set(self.edge_states[:, idx])
            adjacency = adjacency.at[:, j, i].set(self.edge_states[:, idx])
        return adjacency
"""
Efficient board that stores only edge states with proper clique detection.
Maintains exact same interface and game logic as VectorizedCliqueBoard.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List
from itertools import combinations


class EfficientCliqueBoard:
    """
    Board that stores edges as a flat array instead of adjacency matrix.
    Reduces memory and copy overhead while keeping exact same game logic.
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
        edge_to_vertices = []
        edge_map = {}
        idx = 0
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                edge_to_vertices.append((i, j))
                edge_map[(i, j)] = idx
                edge_map[(j, i)] = idx  # Symmetric
                idx += 1
        self._edge_to_vertices = np.array(edge_to_vertices)
        self._edge_map = edge_map
        
        # Precompute all k-cliques for efficient checking
        self._all_k_cliques = list(combinations(range(num_vertices), k))
    
    def get_single_board(self, idx: int):
        """Get a single board from batch - exact same interface."""
        board = EfficientCliqueBoard(1, self.num_vertices, self.k, self.game_mode)
        board.edge_states = self.edge_states[idx:idx+1].copy()
        board.current_players = self.current_players[idx:idx+1].copy()
        board.game_states = self.game_states[idx:idx+1].copy()
        board.winners = self.winners[idx:idx+1].copy()
        board.move_counts = self.move_counts[idx:idx+1].copy()
        return board
    
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
    
    def make_moves(self, actions: jnp.ndarray):
        """Make moves - exact same logic as original."""
        active_mask = self.game_states == 0
        batch_indices = jnp.arange(self.batch_size)
        
        # Update edge states
        self.edge_states = self.edge_states.at[batch_indices, actions].set(
            jnp.where(active_mask, self.current_players, self.edge_states[batch_indices, actions])
        )
        
        # Update move counts
        self.move_counts = jnp.where(active_mask, self.move_counts + 1, self.move_counts)
        
        # Check for cliques after each move - exact same logic as original
        for batch_idx in range(self.batch_size):
            if active_mask[batch_idx]:
                self._check_win_condition(batch_idx)
        
        # Switch players if game is still ongoing
        if self.game_mode == "alternating":
            self.current_players = jnp.where(
                active_mask & (self.game_states == 0),
                3 - self.current_players,
                self.current_players
            )
    
    def _check_win_condition(self, batch_idx: int):
        """Check win condition - exact same logic as original CliqueBoard."""
        if self.game_mode == "asymmetric":
            # Player 1 wins by forming k-clique
            # Player 2 wins by preventing it (all edges taken)
            
            # Check if Player 1 has formed a k-clique
            for vertices in self._all_k_cliques:
                if self._is_clique(batch_idx, vertices, 1):
                    self.game_states = self.game_states.at[batch_idx].set(1)
                    self.winners = self.winners.at[batch_idx].set(1)
                    return
            
            # Check if all edges are taken (Player 2 wins)
            if jnp.all(self.edge_states[batch_idx] != 0):
                self.game_states = self.game_states.at[batch_idx].set(2)
                self.winners = self.winners.at[batch_idx].set(2)
                
        else:  # symmetric mode
            # Both players can win by forming k-clique
            current_player = int(self.current_players[batch_idx])
            
            # Check if current player has formed a k-clique
            for vertices in self._all_k_cliques:
                if self._is_clique(batch_idx, vertices, current_player):
                    self.game_states = self.game_states.at[batch_idx].set(current_player)
                    self.winners = self.winners.at[batch_idx].set(current_player)
                    return
            
            # Check for draw (all edges taken)
            if jnp.all(self.edge_states[batch_idx] != 0):
                self.game_states = self.game_states.at[batch_idx].set(3)
                self.winners = self.winners.at[batch_idx].set(0)
    
    def _is_clique(self, batch_idx: int, vertices: tuple, player: int) -> bool:
        """Check if vertices form a clique with player's edges."""
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                v1, v2 = vertices[i], vertices[j]
                edge_idx = self._edge_map[(v1, v2)]
                if self.edge_states[batch_idx, edge_idx] != player:
                    return False
        return True
    
    def get_valid_moves_mask(self):
        """Get valid moves - edges that haven't been taken."""
        return self.edge_states == 0
    
    def get_features_for_nn_undirected(self):
        """Get features for neural network - maintains compatibility."""
        batch_edge_indices = []
        batch_edge_features = []
        
        for game_idx in range(self.batch_size):
            edge_indices = []
            edge_features = []
            
            # Add undirected edges (i < j)
            for idx, (i, j) in enumerate(self._edge_to_vertices):
                edge_indices.append([i, j])
                
                # Get edge state
                state = self.edge_states[game_idx, idx]
                if state == 0:
                    feat = [1, 0, 0]  # Unselected
                elif state == 1:
                    feat = [0, 1, 0]  # Player 1
                else:
                    feat = [0, 0, 1]  # Player 2
                
                edge_features.append(feat)
            
            # Add self-loops
            for i in range(self.num_vertices):
                edge_indices.append([i, i])
                edge_features.append([0, 0, 0])
            
            batch_edge_indices.append(jnp.array(edge_indices, dtype=jnp.int32).T)
            batch_edge_features.append(jnp.array(edge_features, dtype=jnp.float32))
        
        return batch_edge_indices, batch_edge_features
    
    @property
    def adjacency_matrices(self):
        """Property for compatibility - converts to adjacency matrix on demand."""
        adjacency = jnp.zeros((self.batch_size, self.num_vertices, self.num_vertices))
        
        for batch_idx in range(self.batch_size):
            for edge_idx, (i, j) in enumerate(self._edge_to_vertices):
                state = self.edge_states[batch_idx, edge_idx]
                adjacency = adjacency.at[batch_idx, i, j].set(state)
                adjacency = adjacency.at[batch_idx, j, i].set(state)
        
        return adjacency
    
    @adjacency_matrices.setter
    def adjacency_matrices(self, value):
        """Set adjacency matrices by converting to edge format."""
        for batch_idx in range(self.batch_size):
            for edge_idx, (i, j) in enumerate(self._edge_to_vertices):
                self.edge_states = self.edge_states.at[batch_idx, edge_idx].set(
                    value[batch_idx, i, j]
                )
"""
Improved board representation that avoids expensive copying.
Uses edge lists instead of adjacency matrices.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple


class ImprovedCliqueBoard:
    """
    Optimized board representation using edge lists instead of adjacency matrices.
    This avoids the O(n^2) copying overhead.
    """
    
    def __init__(self, batch_size: int, num_vertices: int, k: int, game_mode: str = "symmetric"):
        self.batch_size = batch_size
        self.num_vertices = num_vertices
        self.k = k
        self.game_mode = game_mode
        self.num_edges = num_vertices * (num_vertices - 1) // 2
        
        # Instead of adjacency matrix, use edge list representation
        # Each edge is represented by its index in the flattened upper triangle
        self.edge_states = jnp.zeros((batch_size, self.num_edges), dtype=jnp.int32)
        
        # Game state tracking
        self.current_players = jnp.ones(batch_size, dtype=jnp.int32)
        self.game_states = jnp.zeros(batch_size, dtype=jnp.int32)
        self.winners = jnp.zeros(batch_size, dtype=jnp.int32)
        self.move_counts = jnp.zeros(batch_size, dtype=jnp.int32)
        
        # Precompute edge to vertex mapping
        edge_to_vertices = []
        idx = 0
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                edge_to_vertices.append((i, j))
                idx += 1
        self.edge_to_vertices = jnp.array(edge_to_vertices)
    
    def copy_single_board(self, board_idx: int) -> 'ImprovedCliqueBoard':
        """Efficient copy of a single board state."""
        new_board = ImprovedCliqueBoard(1, self.num_vertices, self.k, self.game_mode)
        new_board.edge_states = self.edge_states[board_idx:board_idx+1].copy()
        new_board.current_players = self.current_players[board_idx:board_idx+1].copy()
        new_board.game_states = self.game_states[board_idx:board_idx+1].copy()
        new_board.winners = self.winners[board_idx:board_idx+1].copy()
        new_board.move_counts = self.move_counts[board_idx:board_idx+1].copy()
        return new_board
    
    def make_moves(self, actions: jnp.ndarray) -> None:
        """Make moves on the boards."""
        # Only update boards that are still active
        active_mask = self.game_states == 0
        
        # Update edge states
        batch_indices = jnp.arange(self.batch_size)
        self.edge_states = self.edge_states.at[batch_indices, actions].set(
            jnp.where(active_mask, self.current_players, self.edge_states[batch_indices, actions])
        )
        
        # Update move counts
        self.move_counts = jnp.where(active_mask, self.move_counts + 1, self.move_counts)
        
        # Check for cliques (simplified - would need proper implementation)
        self._check_cliques()
        
        # Switch players
        if self.game_mode == "alternating":
            self.current_players = jnp.where(
                active_mask & (self.game_states == 0),
                3 - self.current_players,
                self.current_players
            )
    
    def get_valid_moves_mask(self) -> jnp.ndarray:
        """Get valid moves as a boolean mask."""
        # Valid moves are edges that haven't been claimed yet
        return self.edge_states == 0
    
    def _check_cliques(self):
        """Check for k-cliques (simplified placeholder)."""
        # This would need proper clique detection logic
        # For now, just check if all edges are taken
        all_edges_taken = jnp.all(self.edge_states != 0, axis=1)
        self.game_states = jnp.where(all_edges_taken, 3, self.game_states)  # Draw
        self.winners = jnp.where(all_edges_taken, 0, self.winners)
    
    def get_features_for_nn(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get features for neural network in edge list format."""
        # Return edge states directly - much more efficient
        return self.edge_to_vertices, self.edge_states
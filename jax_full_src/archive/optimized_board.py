"""
Optimized board with vectorized feature extraction
"""

import jax
import jax.numpy as jnp
from typing import Tuple

class OptimizedCliqueBoard:
    """Board with fully vectorized operations"""
    
    def __init__(self, batch_size: int, num_vertices: int = 6, k: int = 3, game_mode: str = "symmetric"):
        self.batch_size = batch_size
        self.num_vertices = num_vertices
        self.k = k
        self.game_mode = game_mode
        self.num_edges = num_vertices * (num_vertices - 1) // 2
        
        # Edge states: 0=unplayed, 1=player1, 2=player2
        self.edge_states = jnp.zeros((batch_size, num_vertices, num_vertices), dtype=jnp.int32)
        self.current_players = jnp.zeros(batch_size, dtype=jnp.int32)
        self.game_states = jnp.zeros(batch_size, dtype=jnp.int32)
        self.winners = jnp.zeros(batch_size, dtype=jnp.int32)
        
        # Pre-compute edge indices for fast feature extraction
        self._precompute_edge_indices()
    
    def _precompute_edge_indices(self):
        """Pre-compute edge indices for vectorized feature extraction"""
        # Create arrays of i,j indices for all edges where i < j
        i_indices = []
        j_indices = []
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                i_indices.append(i)
                j_indices.append(j)
        
        self.edge_i = jnp.array(i_indices)
        self.edge_j = jnp.array(j_indices)
        
        # Create edge index array (constant for all batches)
        self.edge_indices_const = jnp.stack([self.edge_i, self.edge_j], axis=0)  # [2, num_edges]
    
    def get_features_for_nn_undirected_vectorized(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Fully vectorized feature extraction - no Python loops!
        
        Returns:
            edge_indices: [batch_size, 2, num_edges]
            edge_features: [batch_size, num_edges, 3]
        """
        # Broadcast edge indices to all batches
        edge_indices = jnp.broadcast_to(
            self.edge_indices_const[None, :, :], 
            (self.batch_size, 2, self.num_edges)
        )
        
        # Extract edge states using advanced indexing
        # edge_states shape: [batch_size, num_vertices, num_vertices]
        # We want states at positions [batch, i, j] for all edges
        batch_indices = jnp.arange(self.batch_size)[:, None]  # [batch_size, 1]
        edge_states_flat = self.edge_states[batch_indices, self.edge_i, self.edge_j]  # [batch_size, num_edges]
        
        # Convert states to one-hot features
        # state 0 -> [1, 0, 0]
        # state 1 -> [0, 1, 0]  
        # state 2 -> [0, 0, 1]
        edge_features = jnp.zeros((self.batch_size, self.num_edges, 3))
        edge_features = edge_features.at[:, :, 0].set(edge_states_flat == 0)
        edge_features = edge_features.at[:, :, 1].set(edge_states_flat == 1)
        edge_features = edge_features.at[:, :, 2].set(edge_states_flat == 2)
        
        return edge_indices, edge_features.astype(jnp.float32)
    
    def get_valid_moves_mask(self) -> jnp.ndarray:
        """Get mask of valid moves (unplayed edges)"""
        batch_indices = jnp.arange(self.batch_size)[:, None]
        edge_states_flat = self.edge_states[batch_indices, self.edge_i, self.edge_j]
        return edge_states_flat == 0
    
    def make_moves(self, actions: jnp.ndarray):
        """Make moves on the board"""
        # Convert actions to i,j coordinates
        i_coords = self.edge_i[actions]
        j_coords = self.edge_j[actions]
        
        # Update edge states
        batch_indices = jnp.arange(self.batch_size)
        player_values = self.current_players + 1
        
        # Update both directions (symmetric)
        self.edge_states = self.edge_states.at[batch_indices, i_coords, j_coords].set(player_values)
        self.edge_states = self.edge_states.at[batch_indices, j_coords, i_coords].set(player_values)
        
        # Switch players
        self.current_players = 1 - self.current_players
    
    # Copy other necessary methods from VectorizedCliqueBoard
    @property
    def adjacency_matrices(self):
        """For compatibility"""
        return self.edge_states


def test_performance():
    """Compare optimized vs original feature extraction"""
    import time
    import sys
    sys.path.append('.')
    from vectorized_board import VectorizedCliqueBoard
    
    print("Comparing Feature Extraction Performance")
    print("="*40)
    
    batch_sizes = [1, 8, 32, 128]
    
    for batch_size in batch_sizes:
        # Original
        orig_board = VectorizedCliqueBoard(batch_size, 6, 3, "symmetric")
        
        # Warm up
        _ = orig_board.get_features_for_nn_undirected()
        
        # Time original
        start = time.time()
        for _ in range(10):
            _, _ = orig_board.get_features_for_nn_undirected()
        orig_time = (time.time() - start) / 10
        
        # Optimized
        opt_board = OptimizedCliqueBoard(batch_size, 6, 3, "symmetric")
        
        # Warm up
        _ = opt_board.get_features_for_nn_undirected_vectorized()
        
        # Time optimized
        start = time.time()
        for _ in range(100):
            _, _ = opt_board.get_features_for_nn_undirected_vectorized()
        opt_time = (time.time() - start) / 100
        
        speedup = orig_time / opt_time
        print(f"Batch {batch_size:3}: Original {orig_time*1000:6.1f}ms, Optimized {opt_time*1000:6.2f}ms, Speedup {speedup:6.1f}x")
    
    # Verify correctness
    print("\nVerifying correctness...")
    orig_board = VectorizedCliqueBoard(4, 6, 3, "symmetric")
    opt_board = OptimizedCliqueBoard(4, 6, 3, "symmetric")
    
    orig_idx, orig_feat = orig_board.get_features_for_nn_undirected()
    opt_idx, opt_feat = opt_board.get_features_for_nn_undirected_vectorized()
    
    print(f"Indices match: {jnp.allclose(orig_idx, opt_idx)}")
    print(f"Features match: {jnp.allclose(orig_feat, opt_feat)}")

if __name__ == "__main__":
    test_performance()
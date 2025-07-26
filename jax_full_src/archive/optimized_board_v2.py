#!/usr/bin/env python
"""
Optimized Vectorized Clique Board v2 - Compatible with NN
Uses directed edge representation to match neural network
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
from typing import Tuple, List, Dict, Optional
from functools import partial


class OptimizedVectorizedBoard:
    """
    Fully optimized board that matches neural network's edge representation.
    Uses directed edges (i,j) for all i,j pairs where i≠j.
    """
    
    def __init__(self, batch_size: int, num_vertices: int = 6, k: int = 3,
                 game_mode: str = "asymmetric"):
        self.batch_size = batch_size
        self.num_vertices = num_vertices
        self.k = k
        self.game_mode = game_mode
        
        # For neural network compatibility, we use ALL vertex pairs (directed)
        self.num_edges_directed = num_vertices * num_vertices  # 36 for 6 vertices
        self.num_edges_undirected = num_vertices * (num_vertices - 1) // 2  # 15 for 6 vertices
        self.num_actions = self.num_edges_undirected  # Actions are still undirected
        
        # Create mappings between undirected and directed representations
        self._create_edge_mappings()
        
        # Initialize game states
        self.reset_all()
        
        # Precompute clique information
        self._precompute_cliques()
        
        # Create JIT-compiled functions
        self._create_jit_functions()
    
    def _create_edge_mappings(self):
        """Create mappings between undirected edges and directed representation."""
        # Undirected edges (action space)
        self.edge_list = []
        self.action_to_edge = {}
        self.edge_to_action = {}
        
        idx = 0
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                edge = (i, j)
                self.edge_list.append(edge)
                self.action_to_edge[idx] = edge
                self.edge_to_action[edge] = idx
                idx += 1
        
        # Directed edge indices for NN (all i,j pairs where i≠j)
        self.directed_edges = []
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if i != j:
                    self.directed_edges.append((i, j))
        
        # Mapping from directed index to undirected action
        self.directed_to_action = {}
        for idx, (i, j) in enumerate(self.directed_edges):
            if i < j:
                self.directed_to_action[idx] = self.edge_to_action[(i, j)]
            else:
                self.directed_to_action[idx] = self.edge_to_action[(j, i)]
    
    def _precompute_cliques(self):
        """Precompute all possible k-cliques."""
        from itertools import combinations
        vertices = list(range(self.num_vertices))
        all_cliques = list(combinations(vertices, self.k))
        
        # Create clique edge masks for undirected edges
        clique_masks = []
        for clique in all_cliques:
            mask = np.zeros(self.num_edges_undirected, dtype=np.bool_)
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    edge_idx = self.edge_to_action.get((clique[i], clique[j]))
                    if edge_idx is not None:
                        mask[edge_idx] = True
            clique_masks.append(mask)
        
        self.clique_masks = jnp.array(clique_masks)
        self.num_cliques = len(all_cliques)
    
    def _create_jit_functions(self):
        """Create JIT-compiled versions of key functions."""
        
        @jit
        def get_valid_moves_mask_jit(edge_states: jnp.ndarray, game_states: jnp.ndarray) -> jnp.ndarray:
            """JIT-compiled valid moves computation."""
            edge_unselected = (edge_states == 0)
            game_ongoing = (game_states == 0)[:, None]
            return edge_unselected & game_ongoing
        
        @jit
        def check_cliques_jit(edge_states: jnp.ndarray, clique_masks: jnp.ndarray,
                            current_players: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Check if any player has formed a k-clique."""
            p1_edges = (edge_states == 1)
            p1_has_clique = jnp.all(p1_edges[:, None, :] >= clique_masks[None, :, :], axis=2)
            p1_wins = jnp.any(p1_has_clique, axis=1)
            
            p2_edges = (edge_states == 2)
            p2_has_clique = jnp.all(p2_edges[:, None, :] >= clique_masks[None, :, :], axis=2)
            p2_wins = jnp.any(p2_has_clique, axis=1)
            
            return p1_wins, p2_wins
        
        @jit
        def make_moves_jit(edge_states: jnp.ndarray, game_states: jnp.ndarray,
                          current_players: jnp.ndarray, actions: jnp.ndarray,
                          clique_masks: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Make moves and update game states."""
            batch_size = edge_states.shape[0]
            batch_indices = jnp.arange(batch_size)
            
            new_edge_states = edge_states.at[batch_indices, actions].set(
                jnp.where(game_states == 0, current_players + 1, edge_states[batch_indices, actions])
            )
            
            p1_wins, p2_wins = check_cliques_jit(new_edge_states, clique_masks, current_players)
            
            new_game_states = jnp.where(
                p1_wins, 1,
                jnp.where(p2_wins, 2, game_states)
            )
            
            new_players = jnp.where(
                game_states == 0,
                1 - current_players,
                current_players
            )
            
            return new_edge_states, new_game_states, new_players
        
        self.get_valid_moves_mask_jit = get_valid_moves_mask_jit
        self.check_cliques_jit = check_cliques_jit
        self.make_moves_jit = make_moves_jit
    
    def reset_all(self):
        """Reset all games to initial state."""
        # Edge states for undirected edges only
        self.edge_states = jnp.zeros((self.batch_size, self.num_edges_undirected), dtype=jnp.int32)
        self.game_states = jnp.zeros(self.batch_size, dtype=jnp.int32)
        self.current_players = jnp.zeros(self.batch_size, dtype=jnp.int32)
    
    def get_valid_moves_mask(self) -> jnp.ndarray:
        """Get valid moves mask (for undirected actions)."""
        return self.get_valid_moves_mask_jit(self.edge_states, self.game_states)
    
    def make_moves(self, actions: jnp.ndarray):
        """Make moves using JIT-compiled function."""
        self.edge_states, self.game_states, self.current_players = self.make_moves_jit(
            self.edge_states, self.game_states, self.current_players,
            actions, self.clique_masks
        )
    
    def get_features_for_nn(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get features for neural network in directed edge format.
        
        Returns:
            edge_indices: (batch_size, 2, num_vertices^2)
            edge_features: (batch_size, num_vertices^2, 3)
        """
        batch_size = self.batch_size
        num_vertices = self.num_vertices
        
        # Create directed edge indices
        edge_indices = []
        for i in range(num_vertices):
            for j in range(num_vertices):
                edge_indices.append([i, j])
        edge_indices = jnp.array(edge_indices).T  # Shape: (2, 36)
        edge_indices = jnp.broadcast_to(edge_indices[None, :, :], (batch_size, 2, 36))
        
        # Create features for directed edges
        # For each directed edge (i,j), get features from undirected edge
        edge_features = jnp.zeros((batch_size, 36, 3))
        
        for dir_idx, (i, j) in enumerate(self.directed_edges):
            if i == j:
                # Self-loop: always zeros
                continue
            
            # Get undirected edge state
            if i < j:
                undirected_idx = self.edge_to_action[(i, j)]
            else:
                undirected_idx = self.edge_to_action[(j, i)]
            
            # Set features: [is_unselected, is_player1, is_player2]
            edge_state = self.edge_states[:, undirected_idx]
            is_unselected = (edge_state == 0).astype(jnp.float32)
            is_player1 = (edge_state == 1).astype(jnp.float32)
            is_player2 = (edge_state == 2).astype(jnp.float32)
            
            edge_features = edge_features.at[:, dir_idx, 0].set(is_unselected)
            edge_features = edge_features.at[:, dir_idx, 1].set(is_player1)
            edge_features = edge_features.at[:, dir_idx, 2].set(is_player2)
        
        return edge_indices, edge_features
    
    def get_board_states(self) -> List[Dict]:
        """Get board states for compatibility with training."""
        states = []
        for i in range(self.batch_size):
            state = {
                'edges': {},
                'valid_edges': [],
                'turn': int(self.current_players[i]),
                'done': int(self.game_states[i]) != 0,
                'winner': int(self.game_states[i]) if self.game_states[i] != 0 else None
            }
            
            for idx, (v1, v2) in enumerate(self.edge_list):
                edge_state = int(self.edge_states[i, idx])
                if edge_state == 0:
                    state['valid_edges'].append((v1, v2))
                else:
                    state['edges'][(v1, v2)] = edge_state - 1
            
            states.append(state)
        
        return states


def test_compatibility():
    """Test compatibility with neural network."""
    print("Testing Board-NN Compatibility")
    print("="*60)
    
    from vectorized_nn import BatchedNeuralNetwork
    
    batch_size = 16
    board = OptimizedVectorizedBoard(batch_size)
    nn = BatchedNeuralNetwork()
    
    # Get features
    edge_indices, edge_features = board.get_features_for_nn()
    valid_mask = board.get_valid_moves_mask()
    
    print(f"Board:")
    print(f"  Undirected edges (actions): {board.num_edges_undirected}")
    print(f"  Directed edges (NN input): {board.num_edges_directed}")
    print(f"  Edge indices shape: {edge_indices.shape}")
    print(f"  Edge features shape: {edge_features.shape}")
    print(f"  Valid mask shape: {valid_mask.shape}")
    
    # Test NN evaluation
    policies, values = nn.evaluate_batch(edge_indices, edge_features, valid_mask)
    print(f"\nNN Output:")
    print(f"  Policies shape: {policies.shape}")
    print(f"  Values shape: {values.shape}")
    
    # Test move making
    actions = jnp.zeros(batch_size, dtype=jnp.int32)
    board.make_moves(actions)
    print(f"\n✓ All shapes compatible!")
    
    # Benchmark
    import time
    print("\nPerformance:")
    
    start = time.time()
    for _ in range(100):
        edge_indices, edge_features = board.get_features_for_nn()
        edge_features.block_until_ready()
    elapsed = time.time() - start
    print(f"  Get features (100x): {elapsed:.3f}s ({1000*elapsed/100:.2f}ms per call)")
    
    start = time.time()
    for _ in range(100):
        valid_mask = board.get_valid_moves_mask()
        valid_mask.block_until_ready()
    elapsed = time.time() - start
    print(f"  Valid moves (100x): {elapsed:.3f}s ({1000*elapsed/100:.2f}ms per call)")


if __name__ == "__main__":
    test_compatibility()
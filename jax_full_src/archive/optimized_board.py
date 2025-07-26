#!/usr/bin/env python
"""
Optimized Vectorized Clique Board with JIT compilation
All operations are truly vectorized with no Python loops
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
from typing import Tuple, List, Dict, Optional
from functools import partial


class OptimizedVectorizedBoard:
    """
    Fully optimized board that processes batch_size games in parallel.
    All operations are JIT-compiled for maximum GPU performance.
    """
    
    def __init__(self, batch_size: int, num_vertices: int = 6, k: int = 3,
                 game_mode: str = "asymmetric"):
        self.batch_size = batch_size
        self.num_vertices = num_vertices
        self.k = k
        self.game_mode = game_mode
        
        # Precompute edge information
        self.num_edges = num_vertices * (num_vertices - 1) // 2
        self.num_actions = self.num_edges  # For compatibility
        
        # Create edge list and mappings using JAX operations
        edge_indices = []
        idx = 0
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                edge_indices.append((i, j))
                idx += 1
        
        self.edge_array = jnp.array(edge_indices)  # Shape: (num_edges, 2)
        
        # Create reverse mapping: (i, j) -> action index
        self.edge_to_action = {}
        for idx, (i, j) in enumerate(edge_indices):
            self.edge_to_action[(i, j)] = idx
        
        # Initialize game states
        self.reset_all()
        
        # Precompute clique information
        self._precompute_cliques()
        
        # Create JIT-compiled functions
        self._create_jit_functions()
    
    def _precompute_cliques(self):
        """Precompute all possible k-cliques."""
        from itertools import combinations
        vertices = list(range(self.num_vertices))
        all_cliques = list(combinations(vertices, self.k))
        
        # Create clique edge masks
        # For each clique, which edges are part of it?
        clique_masks = []
        for clique in all_cliques:
            mask = np.zeros(self.num_edges, dtype=np.bool_)
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    edge_idx = self.edge_to_action.get((clique[i], clique[j]))
                    if edge_idx is not None:
                        mask[edge_idx] = True
            clique_masks.append(mask)
        
        self.clique_masks = jnp.array(clique_masks)  # Shape: (num_cliques, num_edges)
        self.num_cliques = len(all_cliques)
    
    def _create_jit_functions(self):
        """Create JIT-compiled versions of key functions."""
        
        @jit
        def get_valid_moves_mask_jit(edge_states: jnp.ndarray, game_states: jnp.ndarray) -> jnp.ndarray:
            """JIT-compiled valid moves computation."""
            # edge_states shape: (batch_size, num_edges)
            # Valid if edge is unselected (0) and game is ongoing (0)
            edge_unselected = (edge_states == 0)
            game_ongoing = (game_states == 0)[:, None]  # Broadcast to (batch_size, 1)
            return edge_unselected & game_ongoing
        
        @jit
        def check_cliques_jit(edge_states: jnp.ndarray, clique_masks: jnp.ndarray,
                            current_players: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Check if any player has formed a k-clique."""
            # edge_states: (batch_size, num_edges)
            # clique_masks: (num_cliques, num_edges)
            
            # For player 1 (edges with state 1)
            p1_edges = (edge_states == 1)  # (batch_size, num_edges)
            # Check each clique: does player 1 have all edges?
            # Broadcast and compare
            p1_has_clique = jnp.all(p1_edges[:, None, :] >= clique_masks[None, :, :], axis=2)
            # Any clique formed by player 1?
            p1_wins = jnp.any(p1_has_clique, axis=1)
            
            # For player 2 (edges with state 2)
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
            
            # Update edge states
            # Create mask for which edges to update
            batch_indices = jnp.arange(batch_size)
            new_edge_states = edge_states.at[batch_indices, actions].set(
                jnp.where(game_states == 0, current_players + 1, edge_states[batch_indices, actions])
            )
            
            # Check for wins
            p1_wins, p2_wins = check_cliques_jit(new_edge_states, clique_masks, current_players)
            
            # Update game states
            new_game_states = jnp.where(
                p1_wins, 1,  # Player 1 wins
                jnp.where(p2_wins, 2,  # Player 2 wins
                         game_states)  # Keep current state
            )
            
            # Switch players (only if game ongoing)
            new_players = jnp.where(
                game_states == 0,
                1 - current_players,
                current_players
            )
            
            return new_edge_states, new_game_states, new_players
        
        # Store JIT functions
        self.get_valid_moves_mask_jit = get_valid_moves_mask_jit
        self.check_cliques_jit = check_cliques_jit
        self.make_moves_jit = make_moves_jit
    
    def reset_all(self):
        """Reset all games to initial state."""
        # Edge states: 0=unselected, 1=player1, 2=player2
        self.edge_states = jnp.zeros((self.batch_size, self.num_edges), dtype=jnp.int32)
        
        # Game states: 0=ongoing, 1=player1_win, 2=player2_win
        self.game_states = jnp.zeros(self.batch_size, dtype=jnp.int32)
        
        # Current player: 0=player1, 1=player2
        self.current_players = jnp.zeros(self.batch_size, dtype=jnp.int32)
    
    def get_valid_moves_mask(self) -> jnp.ndarray:
        """Get valid moves mask using JIT-compiled function."""
        return self.get_valid_moves_mask_jit(self.edge_states, self.game_states)
    
    def make_moves(self, actions: jnp.ndarray):
        """Make moves using JIT-compiled function."""
        self.edge_states, self.game_states, self.current_players = self.make_moves_jit(
            self.edge_states, self.game_states, self.current_players,
            actions, self.clique_masks
        )
    
    def get_features_for_nn(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get features for neural network - fully vectorized.
        
        Returns:
            edge_indices: (batch_size, 2, num_edges)
            edge_features: (batch_size, num_edges, 3)
        """
        # Edge indices are the same for all games
        edge_indices = jnp.broadcast_to(
            self.edge_array.T[None, :, :],  # (1, 2, num_edges)
            (self.batch_size, 2, self.num_edges)
        )
        
        # Edge features: [is_unselected, is_player1, is_player2]
        is_unselected = (self.edge_states == 0).astype(jnp.float32)
        is_player1 = (self.edge_states == 1).astype(jnp.float32)
        is_player2 = (self.edge_states == 2).astype(jnp.float32)
        
        edge_features = jnp.stack([is_unselected, is_player1, is_player2], axis=2)
        
        return edge_indices, edge_features
    
    def get_board_states(self) -> List[Dict]:
        """Get board states for compatibility with training."""
        # This requires Python loop but is only called at end of games
        states = []
        for i in range(self.batch_size):
            state = {
                'edges': {},
                'valid_edges': [],
                'turn': int(self.current_players[i]),
                'done': int(self.game_states[i]) != 0,
                'winner': int(self.game_states[i]) if self.game_states[i] != 0 else None
            }
            
            # Fill edge information
            for idx, (v1, v2) in enumerate(self.edge_array):
                edge_state = int(self.edge_states[i, idx])
                if edge_state == 0:
                    state['valid_edges'].append((int(v1), int(v2)))
                else:
                    state['edges'][(int(v1), int(v2))] = edge_state - 1  # Convert to 0/1
            
            states.append(state)
        
        return states


def benchmark_optimized_board():
    """Benchmark the optimized board implementation."""
    print("Optimized Board Performance Test")
    print("="*60)
    
    batch_sizes = [16, 64, 256]
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        board = OptimizedVectorizedBoard(batch_size)
        
        # Warmup
        _ = board.get_valid_moves_mask()
        _ = board.get_features_for_nn()
        
        # Time operations
        import time
        
        # 1. Valid moves
        start = time.time()
        for _ in range(100):
            mask = board.get_valid_moves_mask()
            mask.block_until_ready()
        elapsed = time.time() - start
        print(f"  Valid moves (100x): {elapsed:.3f}s ({1000*elapsed/100:.2f}ms per call)")
        
        # 2. Features
        start = time.time()
        for _ in range(100):
            indices, features = board.get_features_for_nn()
            features.block_until_ready()
        elapsed = time.time() - start
        print(f"  NN features (100x): {elapsed:.3f}s ({1000*elapsed/100:.2f}ms per call)")
        
        # 3. Make moves
        actions = jnp.zeros(batch_size, dtype=jnp.int32)
        start = time.time()
        for _ in range(100):
            board.make_moves(actions)
            board.game_states.block_until_ready()
        elapsed = time.time() - start
        print(f"  Make moves (100x): {elapsed:.3f}s ({1000*elapsed/100:.2f}ms per call)")
        
        # Total for one game step
        total_ms = 3 * 1000 * elapsed / 100
        print(f"  Total per step: ~{total_ms:.1f}ms")
        print(f"  Steps per second: ~{1000/total_ms:.0f}")


if __name__ == "__main__":
    benchmark_optimized_board()
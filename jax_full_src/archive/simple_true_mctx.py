"""
Simplified True MCTX Implementation
Demonstrates the core concepts of MCTX without full complexity
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial
from typing import Tuple, NamedTuple

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class SimpleTrueMCTX:
    """
    Simplified MCTX implementation that shows the key concepts:
    1. Pre-allocated arrays (no dynamic allocation)
    2. Vectorized operations (no Python loops)
    3. Full JIT compilation
    """
    
    def __init__(self, batch_size: int, num_actions: int = 15, 
                 max_nodes: int = 500, c_puct: float = 3.0):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.max_nodes = max_nodes
        self.c_puct = c_puct
        
        # Pre-compile key functions
        self._select_and_expand_batch = jax.jit(self._select_and_expand_batch_impl)
        
    def _init_arrays(self):
        """Initialize pre-allocated arrays"""
        return {
            'N': jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            'W': jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            'P': jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            'children': jnp.full((self.batch_size, self.max_nodes, self.num_actions), -1, dtype=jnp.int32),
            'expanded': jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.bool_),
            'node_count': jnp.ones(self.batch_size, dtype=jnp.int32),
            'edge_states': jnp.zeros((self.batch_size, self.max_nodes, self.num_actions), dtype=jnp.int32),
            'current_players': jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32),
        }
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_ucb_vectorized(self, N, W, P, parent_visits, valid_mask):
        """Compute UCB scores for all games and actions at once"""
        Q = W / (N + 1e-8)
        U = self.c_puct * jnp.sqrt(parent_visits)[:, None] * P / (N + 1)
        ucb = Q + U
        ucb = jnp.where(valid_mask, ucb, -jnp.inf)
        return ucb
    
    def _select_and_expand_batch_impl(self, arrays, sim_idx):
        """
        Vectorized selection and expansion for all games.
        This is the KEY difference from previous implementations.
        """
        batch_indices = jnp.arange(self.batch_size)
        
        # Start from root
        current_nodes = jnp.zeros(self.batch_size, dtype=jnp.int32)
        
        # Vectorized selection loop using scan
        def select_step(carry, _):
            nodes = carry
            
            # Get statistics for current nodes
            N = arrays['N'][batch_indices, nodes]
            W = arrays['W'][batch_indices, nodes]
            P = arrays['P'][batch_indices, nodes]
            
            # Valid actions
            edge_states = arrays['edge_states'][batch_indices, nodes]
            valid_mask = edge_states == 0
            
            # Compute UCB for all games at once
            parent_visits = jnp.sum(N, axis=1)
            ucb = self._compute_ucb_vectorized(N, W, P, parent_visits, valid_mask)
            
            # Select best actions
            actions = jnp.argmax(ucb, axis=1)
            
            # Get children
            children = arrays['children'][batch_indices, nodes, actions]
            
            # If no child, stay at current node (will expand later)
            next_nodes = jnp.where(children >= 0, children, nodes)
            
            return next_nodes, (nodes, actions, children < 0)
        
        # Run selection for a fixed depth
        final_nodes, (path_nodes, path_actions, need_expand) = jax.lax.scan(
            select_step, current_nodes, None, length=10
        )
        
        # Find first expansion needed in path
        first_expand = jnp.argmax(need_expand, axis=0)
        expand_nodes = path_nodes[first_expand, batch_indices]
        expand_actions = path_actions[first_expand, batch_indices]
        
        # Vectorized expansion
        new_indices = arrays['node_count']
        can_expand = (new_indices < self.max_nodes) & need_expand[first_expand, batch_indices]
        
        # Update children
        arrays['children'] = arrays['children'].at[batch_indices, expand_nodes, expand_actions].set(
            jnp.where(can_expand, new_indices, -1)
        )
        
        # Update node count
        arrays['node_count'] = jnp.where(can_expand, new_indices + 1, arrays['node_count'])
        
        # Update edge states for new nodes
        parent_edges = arrays['edge_states'][batch_indices, expand_nodes]
        new_edges = parent_edges.at[batch_indices, expand_actions].set(1)
        arrays['edge_states'] = arrays['edge_states'].at[batch_indices, new_indices].set(
            jnp.where(can_expand[:, None], new_edges, 0)
        )
        
        return arrays, (new_indices, can_expand)
    
    def search(self, boards: VectorizedCliqueBoard,
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               temperature: float = 1.0) -> jnp.ndarray:
        """Run MCTS search"""
        print(f"Starting Simplified True MCTX with {num_simulations} simulations")
        start_time = time.time()
        
        # Initialize arrays
        arrays = self._init_arrays()
        
        # Set up root
        edge_states = jnp.zeros((self.batch_size, self.num_actions), dtype=jnp.int32)
        edge_idx = 0
        for i in range(6):
            for j in range(i + 1, 6):
                edge_played = boards.edge_states[:, i, j]
                edge_states = edge_states.at[:, edge_idx].set(edge_played)
                edge_idx += 1
        
        arrays['edge_states'] = arrays['edge_states'].at[:, 0].set(edge_states)
        arrays['current_players'] = arrays['current_players'].at[:, 0].set(boards.current_players)
        arrays['expanded'] = arrays['expanded'].at[:, 0].set(True)
        
        # Evaluate root
        root_policies, _ = neural_network.evaluate_batch(
            *boards.get_features_for_nn_undirected(),
            boards.get_valid_moves_mask()
        )
        arrays['P'] = arrays['P'].at[:, 0].set(root_policies)
        
        # Main loop - KEY: No Python loop over games!
        for sim in range(num_simulations):
            # 1. Select and expand for ALL games at once
            arrays, (leaf_indices, leaf_mask) = self._select_and_expand_batch(arrays, sim)
            
            # 2. Evaluate leaves (simplified - uniform policy)
            # In real implementation, would batch evaluate all leaves
            uniform_policy = jnp.ones(self.num_actions) / self.num_actions
            batch_indices = jnp.arange(self.batch_size)
            arrays['P'] = arrays['P'].at[batch_indices, leaf_indices].set(
                jnp.where(leaf_mask[:, None], uniform_policy[None, :], 0)
            )
            arrays['expanded'] = arrays['expanded'].at[batch_indices, leaf_indices].set(
                leaf_mask | arrays['expanded'][batch_indices, leaf_indices]
            )
            
            # 3. Simplified backup (just update root for demo)
            # Real implementation would trace full path
            arrays['N'] = arrays['N'].at[:, 0].add(1)
            arrays['W'] = arrays['W'].at[:, 0].add(0.5)  # Dummy value
        
        # Extract action probabilities
        root_visits = arrays['N'][:, 0]
        
        if temperature == 0:
            action_probs = (root_visits == jnp.max(root_visits, axis=1, keepdims=True)).astype(jnp.float32)
        else:
            root_visits_temp = jnp.power(root_visits + 1e-8, 1.0 / temperature)
            action_probs = root_visits_temp / jnp.sum(root_visits_temp, axis=1, keepdims=True)
        
        elapsed = time.time() - start_time
        print(f"Search complete in {elapsed:.3f}s ({elapsed/self.batch_size*1000:.1f}ms per game)")
        
        return action_probs


if __name__ == "__main__":
    print("Testing Simplified True MCTX...")
    
    # Create test instance
    mcts = SimpleTrueMCTX(batch_size=8, num_actions=15, max_nodes=500)
    
    # Create test boards and NN
    boards = VectorizedCliqueBoard(8, 6, 3, "symmetric")
    nn = ImprovedBatchedNeuralNetwork(6, 128, 4)
    
    # Run search
    action_probs = mcts.search(boards, nn, num_simulations=20, temperature=1.0)
    
    print(f"\nAction probabilities shape: {action_probs.shape}")
    print(f"Probabilities sum to 1: {jnp.allclose(jnp.sum(action_probs, axis=1), 1.0)}")
    print("\nKey insight: NO Python loops over games - everything is vectorized!")
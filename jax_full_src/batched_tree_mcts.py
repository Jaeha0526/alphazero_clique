"""
Batched tree MCTS that processes multiple games in a single tree structure.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from functools import partial


class BatchedMCTSTree(NamedTuple):
    """Batched tree structure for multiple games."""
    # All arrays have shape (batch_size, max_nodes, ...)
    N: jnp.ndarray  # (batch, max_nodes, num_actions) - visit counts
    W: jnp.ndarray  # (batch, max_nodes, num_actions) - total values  
    P: jnp.ndarray  # (batch, max_nodes, num_actions) - priors
    children: jnp.ndarray  # (batch, max_nodes, num_actions) - child indices
    node_visits: jnp.ndarray  # (batch, max_nodes) - visits per node
    expanded: jnp.ndarray  # (batch, max_nodes) - is expanded
    current_nodes: jnp.ndarray  # (batch,) - current node being processed
    edge_states: jnp.ndarray  # (batch, max_nodes, num_edges) - board states
    node_count: jnp.ndarray  # (batch,) - number of nodes per game


class BatchedTreeMCTS:
    """MCTS that processes a batch of games simultaneously."""
    
    def __init__(self, batch_size: int, num_actions: int, num_edges: int,
                 max_nodes: int = 200, c_puct: float = 3.0):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.num_edges = num_edges
        self.max_nodes = max_nodes
        self.c_puct = c_puct
        
        # Vectorized operations
        self.select_actions_vmap = jax.vmap(self._select_action, in_axes=(0, 0, 0, 0, 0))
        self.expand_nodes_vmap = jax.vmap(self._expand_single_node, in_axes=(0, 0, 0))
        self.backup_vmap = jax.vmap(self._backup_single, in_axes=(0, 0, 0, 0))
    
    def create_empty_tree(self, initial_boards: jnp.ndarray) -> BatchedMCTSTree:
        """Create empty tree for batch of games."""
        return BatchedMCTSTree(
            N=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            W=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            P=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            children=jnp.full((self.batch_size, self.max_nodes, self.num_actions), -1),
            node_visits=jnp.zeros((self.batch_size, self.max_nodes)),
            expanded=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.bool_),
            current_nodes=jnp.zeros(self.batch_size, dtype=jnp.int32),
            edge_states=jnp.zeros((self.batch_size, self.max_nodes, self.num_edges)),
            node_count=jnp.ones(self.batch_size, dtype=jnp.int32)
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _select_action(self, N: jnp.ndarray, W: jnp.ndarray, P: jnp.ndarray,
                      node_visits: float, valid_mask: jnp.ndarray) -> int:
        """Select best action using UCB for a single node."""
        Q = W / (1.0 + N)
        U = self.c_puct * jnp.sqrt(node_visits) * P / (1.0 + N)
        ucb = Q + U
        ucb = jnp.where(valid_mask, ucb, -jnp.inf)
        return jnp.argmax(ucb)
    
    @partial(jax.jit, static_argnums=(0,))
    def batched_select_phase(self, tree: BatchedMCTSTree) -> jnp.ndarray:
        """Parallel selection phase for all games."""
        # Get data for current nodes
        batch_idx = jnp.arange(self.batch_size)
        current_N = tree.N[batch_idx, tree.current_nodes]
        current_W = tree.W[batch_idx, tree.current_nodes]
        current_P = tree.P[batch_idx, tree.current_nodes]
        current_visits = tree.node_visits[batch_idx, tree.current_nodes]
        current_edges = tree.edge_states[batch_idx, tree.current_nodes]
        
        # Valid moves are unclaimed edges
        valid_masks = current_edges == 0
        
        # Select actions for all games in parallel
        actions = self.select_actions_vmap(
            current_N, current_W, current_P, current_visits, valid_masks
        )
        
        return actions
    
    @partial(jax.jit, static_argnums=(0,))
    def batched_expand_phase(self, tree: BatchedMCTSTree, 
                           priors: jnp.ndarray, new_edges: jnp.ndarray) -> BatchedMCTSTree:
        """Expand nodes for all games in parallel."""
        batch_idx = jnp.arange(self.batch_size)
        
        # Create new nodes
        new_node_ids = tree.node_count
        
        # Update tree arrays
        tree = tree._replace(
            # Set priors for current nodes
            P=tree.P.at[batch_idx, tree.current_nodes].set(priors),
            expanded=tree.expanded.at[batch_idx, tree.current_nodes].set(True),
            
            # Add new child nodes
            edge_states=tree.edge_states.at[batch_idx, new_node_ids].set(new_edges),
            node_count=tree.node_count + 1,
            
            # Update current nodes to the newly created children
            current_nodes=new_node_ids
        )
        
        return tree
    
    @partial(jax.jit, static_argnums=(0,))
    def batched_backup_phase(self, tree: BatchedMCTSTree, values: jnp.ndarray) -> BatchedMCTSTree:
        """Backup values for all games in parallel."""
        # This is simplified - would need proper path tracking
        batch_idx = jnp.arange(self.batch_size)
        
        # Update visits and values
        tree = tree._replace(
            node_visits=tree.node_visits.at[batch_idx, tree.current_nodes].add(1.0),
            # Would need to track parent nodes and actions taken
        )
        
        # Reset to root for next simulation
        tree = tree._replace(current_nodes=jnp.zeros(self.batch_size, dtype=jnp.int32))
        
        return tree
    
    def search(self, initial_boards: jnp.ndarray, neural_net, num_simulations: int) -> jnp.ndarray:
        """Run batched MCTS search."""
        tree = self.create_empty_tree(initial_boards)
        
        for _ in range(num_simulations):
            # Selection phase - all games select simultaneously
            actions = self.batched_select_phase(tree)
            
            # Neural network evaluation - batch all leaf states
            leaf_states = tree.edge_states[jnp.arange(self.batch_size), tree.current_nodes]
            priors, values = neural_net.evaluate_batch(leaf_states)
            
            # Expansion phase - create children for all games
            # Apply actions to get new board states
            new_edges = leaf_states.at[jnp.arange(self.batch_size), actions].set(1)
            tree = self.batched_expand_phase(tree, priors, new_edges)
            
            # Backup phase - propagate values
            tree = self.batched_backup_phase(tree, values)
        
        # Return action probabilities from root nodes
        root_visits = tree.N[:, 0, :]
        return root_visits / jnp.sum(root_visits, axis=1, keepdims=True)
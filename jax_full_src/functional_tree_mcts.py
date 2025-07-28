"""
Functional tree MCTS using JAX arrays instead of Python dictionaries.
This allows for better JIT compilation and parallelization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Tuple
from functools import partial


class MCTSTree(NamedTuple):
    """Tree structure using JAX arrays."""
    N: jnp.ndarray  # (max_nodes, num_actions) - visit counts
    W: jnp.ndarray  # (max_nodes, num_actions) - total values
    P: jnp.ndarray  # (max_nodes, num_actions) - priors
    children: jnp.ndarray  # (max_nodes, num_actions) - child node indices (-1 = no child)
    parents: jnp.ndarray  # (max_nodes,) - parent node indices
    node_visits: jnp.ndarray  # (max_nodes,) - visits to each node
    expanded: jnp.ndarray  # (max_nodes,) - boolean, is node expanded
    board_states: jnp.ndarray  # (max_nodes, num_edges) - edge states
    game_states: jnp.ndarray  # (max_nodes,) - game state (0=ongoing, 1=p1_win, etc)
    active_players: jnp.ndarray  # (max_nodes,) - current player


class FunctionalTreeMCTS:
    """MCTS using functional updates on JAX arrays."""
    
    def __init__(self, num_actions: int, max_nodes: int = 1000, c_puct: float = 3.0):
        self.num_actions = num_actions
        self.max_nodes = max_nodes
        self.c_puct = c_puct
        
        # JIT compile the core functions
        self.select_leaf_jit = jax.jit(self._select_leaf)
        self.expand_node_jit = jax.jit(self._expand_node)
        self.backup_jit = jax.jit(self._backup)
        self.get_action_probs_jit = jax.jit(self._get_action_probs)
    
    def create_tree(self, initial_board_state: jnp.ndarray, initial_player: int) -> MCTSTree:
        """Create empty tree with root node."""
        return MCTSTree(
            N=jnp.zeros((self.max_nodes, self.num_actions)),
            W=jnp.zeros((self.max_nodes, self.num_actions)),
            P=jnp.zeros((self.max_nodes, self.num_actions)),
            children=jnp.full((self.max_nodes, self.num_actions), -1, dtype=jnp.int32),
            parents=jnp.full(self.max_nodes, -1, dtype=jnp.int32),
            node_visits=jnp.zeros(self.max_nodes),
            expanded=jnp.zeros(self.max_nodes, dtype=jnp.bool_),
            board_states=jnp.zeros((self.max_nodes, initial_board_state.shape[0])),
            game_states=jnp.zeros(self.max_nodes, dtype=jnp.int32),
            active_players=jnp.full(self.max_nodes, initial_player, dtype=jnp.int32)
        ).at[0].set({
            'board_states': initial_board_state,
            'active_players': initial_player
        })
    
    @partial(jax.jit, static_argnums=(0,))
    def _calculate_ucb(self, N: jnp.ndarray, W: jnp.ndarray, P: jnp.ndarray, 
                      parent_visits: float, valid_mask: jnp.ndarray) -> jnp.ndarray:
        """Calculate UCB scores for actions."""
        Q = W / (1.0 + N)
        U = self.c_puct * jnp.sqrt(parent_visits) * P / (1.0 + N)
        ucb = Q + U
        return jnp.where(valid_mask, ucb, -jnp.inf)
    
    def _select_leaf(self, tree: MCTSTree, valid_moves_fn) -> Tuple[int, jnp.ndarray]:
        """Select a leaf node by following UCB."""
        node_id = 0
        path = [0]
        
        def cond_fn(state):
            node_id, path, tree = state
            is_expanded = tree.expanded[node_id]
            is_terminal = tree.game_states[node_id] != 0
            return is_expanded & ~is_terminal
        
        def body_fn(state):
            node_id, path, tree = state
            
            # Get valid moves for current board state
            board_state = tree.board_states[node_id]
            valid_mask = board_state == 0  # Unclaimed edges
            
            # Calculate UCB
            ucb = self._calculate_ucb(
                tree.N[node_id],
                tree.W[node_id], 
                tree.P[node_id],
                tree.node_visits[node_id],
                valid_mask
            )
            
            # Select best action
            action = jnp.argmax(ucb)
            
            # Get or create child
            child_id = tree.children[node_id, action]
            
            # If no child exists, this is our leaf
            child_exists = child_id >= 0
            node_id = jnp.where(child_exists, child_id, node_id)
            
            return node_id, path + [node_id], tree
        
        # Use lax.while_loop for efficiency
        final_node, path, _ = jax.lax.while_loop(cond_fn, body_fn, (node_id, path, tree))
        
        return final_node, jnp.array(path)
    
    def _expand_node(self, tree: MCTSTree, node_id: int, priors: jnp.ndarray) -> MCTSTree:
        """Expand a node by adding priors."""
        return tree._replace(
            P=tree.P.at[node_id].set(priors),
            expanded=tree.expanded.at[node_id].set(True)
        )
    
    def _backup(self, tree: MCTSTree, path: jnp.ndarray, value: float) -> MCTSTree:
        """Backup value through the tree."""
        def update_node(tree, node_data):
            node_id, parent_id, action = node_data
            
            # Update node visits
            tree = tree._replace(
                node_visits=tree.node_visits.at[node_id].add(1.0)
            )
            
            # Update parent's action statistics if not root
            def update_parent(tree):
                return tree._replace(
                    N=tree.N.at[parent_id, action].add(1.0),
                    W=tree.W.at[parent_id, action].add(value)
                )
            
            tree = jax.lax.cond(
                parent_id >= 0,
                update_parent,
                lambda t: t,
                tree
            )
            
            return tree
        
        # Process path from leaf to root
        for i in range(len(path) - 1, -1, -1):
            node_id = path[i]
            parent_id = tree.parents[node_id] if i > 0 else -1
            action = 0  # Would need to track actions taken
            
            tree = update_node(tree, (node_id, parent_id, action))
            
            # Flip value for alternating games
            value = -value
        
        return tree
    
    def search(self, board_state: jnp.ndarray, neural_net, num_simulations: int) -> jnp.ndarray:
        """Run MCTS search and return action probabilities."""
        tree = self.create_tree(board_state, initial_player=1)
        
        for _ in range(num_simulations):
            # Select leaf
            leaf_id, path = self.select_leaf_jit(tree, lambda x: x == 0)
            
            # Neural network evaluation
            leaf_state = tree.board_states[leaf_id]
            priors, value = neural_net.evaluate(leaf_state)
            
            # Expand node
            tree = self.expand_node_jit(tree, leaf_id, priors)
            
            # Backup
            tree = self.backup_jit(tree, path, value)
        
        # Get action probabilities from root
        return self.get_action_probs_jit(tree, 0)
    
    def _get_action_probs(self, tree: MCTSTree, node_id: int) -> jnp.ndarray:
        """Get action probabilities from visit counts."""
        visits = tree.N[node_id]
        total_visits = jnp.sum(visits)
        return visits / jnp.maximum(total_visits, 1.0)
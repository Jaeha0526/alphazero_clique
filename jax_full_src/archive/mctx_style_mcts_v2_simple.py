"""
MCTX-style MCTS V2 - Simplified version focusing on core optimizations
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial
from typing import Tuple, NamedTuple

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class MCTSArrays(NamedTuple):
    """Container for all MCTS arrays"""
    N: jnp.ndarray  # Visit counts [batch, num_nodes, num_actions]
    W: jnp.ndarray  # Total values [batch, num_nodes, num_actions]
    P: jnp.ndarray  # Prior probabilities [batch, num_nodes, num_actions]
    children: jnp.ndarray  # Child indices [batch, num_nodes, num_actions]
    parents: jnp.ndarray   # Parent indices [batch, num_nodes]
    expanded: jnp.ndarray  # Whether node is expanded [batch, num_nodes]
    node_visits: jnp.ndarray  # Visits to each node [batch, num_nodes]
    node_count: jnp.ndarray   # Number of nodes used [batch]
    edge_states: jnp.ndarray     # Edge states [batch, num_nodes, num_edges]
    current_players: jnp.ndarray  # Current player [batch, num_nodes]
    game_over: jnp.ndarray       # Game over flag [batch, num_nodes]
    winners: jnp.ndarray         # Winner (-1, 0, 1) [batch, num_nodes]


class SelectionOutput(NamedTuple):
    """Output from selection phase"""
    leaf_game_indices: jnp.ndarray  # Which games need expansion [num_leaves]
    leaf_node_indices: jnp.ndarray  # Which nodes are leaves [num_leaves]
    leaf_parent_indices: jnp.ndarray  # Parent of each leaf [num_leaves]
    leaf_actions: jnp.ndarray  # Actions to reach leaves [num_leaves]
    paths: jnp.ndarray  # Full paths for backup [batch, max_depth, 2]


class MCTXStyleMCTSV2Simple:
    """Simplified V2 focusing on key optimizations"""
    
    def __init__(self, 
                 batch_size: int,
                 num_actions: int = 15,
                 max_nodes: int = 800,
                 c_puct: float = 3.0,
                 max_depth: int = 50):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.max_nodes = max_nodes
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.num_edges = num_actions
        
        # Don't JIT compile yet - fix logic first
        self._init_arrays = self._init_arrays_impl
        self._select_and_expand = self._select_and_expand_impl
        self._backup_paths = self._backup_paths_impl
        
    def _init_arrays_impl(self) -> MCTSArrays:
        """Initialize all arrays"""
        return MCTSArrays(
            N=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            W=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            P=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            children=jnp.full((self.batch_size, self.max_nodes, self.num_actions), -1, dtype=jnp.int32),
            parents=jnp.full((self.batch_size, self.max_nodes), -1, dtype=jnp.int32),
            expanded=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.bool_),
            node_visits=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32),
            node_count=jnp.ones(self.batch_size, dtype=jnp.int32),
            edge_states=jnp.zeros((self.batch_size, self.max_nodes, self.num_edges), dtype=jnp.int32),
            current_players=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32),
            game_over=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.bool_),
            winners=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32),
        )
    
    def _calculate_ucb_vectorized(self, N: jnp.ndarray, W: jnp.ndarray, P: jnp.ndarray, 
                                 node_visits: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
        """Calculate UCB scores for a batch of nodes"""
        Q = W / (1.0 + N)
        sqrt_visits = jnp.sqrt(jnp.maximum(1.0, node_visits[:, None]))
        U = self.c_puct * sqrt_visits * (P / (1.0 + N))
        ucb = Q + U
        ucb = jnp.where(valid_mask, ucb, -jnp.inf)
        return ucb
    
    def _select_and_expand_impl(self, arrays: MCTSArrays) -> Tuple[MCTSArrays, SelectionOutput]:
        """Combined selection and expansion for efficiency"""
        
        # For simplicity, use a vectorized approach without while_loop first
        # This still gives us the key optimizations
        
        # Start from root for all games
        current_nodes = jnp.zeros(self.batch_size, dtype=jnp.int32)
        paths = jnp.full((self.batch_size, self.max_depth, 2), -1, dtype=jnp.int32)
        
        # Track leaves found
        leaf_games = []
        leaf_nodes = []
        leaf_parents = []
        leaf_actions = []
        
        # Simple depth-limited traversal
        for depth in range(min(self.max_depth, 10)):  # Limit depth for now
            print(f"    Depth {depth}, current_nodes: {current_nodes}")
            # Get statistics for current nodes
            batch_indices = jnp.arange(self.batch_size)
            
            # Check which games are still active
            is_expanded = arrays.expanded[batch_indices, current_nodes]
            is_game_over = arrays.game_over[batch_indices, current_nodes]
            active_mask = is_expanded & (~is_game_over)
            
            if not jnp.any(active_mask):
                break
            
            # Calculate UCB for active games
            N = arrays.N[batch_indices, current_nodes]
            W = arrays.W[batch_indices, current_nodes]
            P = arrays.P[batch_indices, current_nodes]
            node_visits = arrays.node_visits[batch_indices, current_nodes]
            edge_states = arrays.edge_states[batch_indices, current_nodes]
            valid_mask = (edge_states == 0)
            
            ucb = self._calculate_ucb_vectorized(N, W, P, node_visits, valid_mask)
            
            # Select actions
            actions = jnp.argmax(ucb, axis=1)
            
            # Record in paths
            paths = paths.at[:, depth, 0].set(current_nodes)
            paths = paths.at[:, depth, 1].set(actions)
            
            # Update node visits
            arrays = arrays._replace(
                node_visits=arrays.node_visits.at[batch_indices, current_nodes].add(
                    active_mask.astype(jnp.int32)
                )
            )
            
            # Get children
            children = arrays.children[batch_indices, current_nodes, actions]
            
            # Find games that need expansion
            need_expand = active_mask & (children == -1)
            
            # Collect leaves
            for game_idx in range(self.batch_size):
                if need_expand[game_idx]:
                    leaf_games.append(game_idx)
                    leaf_nodes.append(arrays.node_count[game_idx])
                    leaf_parents.append(current_nodes[game_idx])
                    leaf_actions.append(actions[game_idx])
                    
                    # Expand node
                    new_node_idx = arrays.node_count[game_idx]
                    if new_node_idx < self.max_nodes:
                        # Update tree structure
                        arrays = arrays._replace(
                            children=arrays.children.at[game_idx, current_nodes[game_idx], actions[game_idx]].set(new_node_idx),
                            parents=arrays.parents.at[game_idx, new_node_idx].set(current_nodes[game_idx]),
                            node_count=arrays.node_count.at[game_idx].add(1)
                        )
                        
                        # Copy board state
                        parent_edges = arrays.edge_states[game_idx, current_nodes[game_idx]]
                        new_edges = parent_edges.at[actions[game_idx]].set(1)
                        arrays = arrays._replace(
                            edge_states=arrays.edge_states.at[game_idx, new_node_idx].set(new_edges),
                            current_players=arrays.current_players.at[game_idx, new_node_idx].set(
                                1 - arrays.current_players[game_idx, current_nodes[game_idx]]
                            )
                        )
            
            # Move to children for next iteration
            current_nodes = jnp.where(
                active_mask & (children >= 0),
                children,
                current_nodes
            )
        
        # Create output
        if leaf_games:
            selection_output = SelectionOutput(
                leaf_game_indices=jnp.array(leaf_games),
                leaf_node_indices=jnp.array(leaf_nodes),
                leaf_parent_indices=jnp.array(leaf_parents),
                leaf_actions=jnp.array(leaf_actions),
                paths=paths
            )
        else:
            # No leaves found
            selection_output = SelectionOutput(
                leaf_game_indices=jnp.array([], dtype=jnp.int32),
                leaf_node_indices=jnp.array([], dtype=jnp.int32),
                leaf_parent_indices=jnp.array([], dtype=jnp.int32),
                leaf_actions=jnp.array([], dtype=jnp.int32),
                paths=paths
            )
        
        return arrays, selection_output
    
    def _backup_paths_impl(self, arrays: MCTSArrays, paths: jnp.ndarray, values: jnp.ndarray) -> MCTSArrays:
        """Backup values along paths"""
        
        # Process each game's path
        for game_idx in range(self.batch_size):
            value = values[game_idx]
            
            # Walk backwards through path
            for depth in range(self.max_depth):
                node_idx = paths[game_idx, depth, 0]
                action = paths[game_idx, depth, 1]
                
                if node_idx >= 0 and action >= 0:
                    # Update statistics
                    arrays = arrays._replace(
                        N=arrays.N.at[game_idx, node_idx, action].add(1.0),
                        W=arrays.W.at[game_idx, node_idx, action].add(value)
                    )
                    
                    # Flip value
                    value = -value
                else:
                    break
        
        return arrays
    
    def search(self, boards: VectorizedCliqueBoard, neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int, temperature: float = 1.0) -> jnp.ndarray:
        """Run MCTS search"""
        print(f"Starting MCTX-style MCTS V2 Simple with {num_simulations} simulations")
        start_time = time.time()
        
        # Initialize arrays
        arrays = self._init_arrays()
        
        # Setup root boards
        edge_states = jnp.zeros((self.batch_size, self.num_edges), dtype=jnp.int32)
        edge_idx = 0
        for i in range(6):
            for j in range(i + 1, 6):
                edge_states = edge_states.at[:, edge_idx].set(boards.edge_states[:, i, j])
                edge_idx += 1
        
        arrays = arrays._replace(
            edge_states=arrays.edge_states.at[:, 0, :].set(edge_states),
            current_players=arrays.current_players.at[:, 0].set(boards.current_players),
            game_over=arrays.game_over.at[:, 0].set(boards.game_states != 0),
            winners=arrays.winners.at[:, 0].set(boards.winners)
        )
        
        # Evaluate root
        root_policies, root_values = neural_network.evaluate_batch(
            *boards.get_features_for_nn_undirected(),
            boards.get_valid_moves_mask()
        )
        
        arrays = arrays._replace(
            P=arrays.P.at[:, 0, :].set(root_policies),
            expanded=arrays.expanded.at[:, 0].set(True)
        )
        
        # Main loop
        for sim in range(num_simulations):
            print(f"  Simulation {sim + 1}/{num_simulations}")
            # Selection and expansion
            arrays, selection_output = self._select_and_expand(arrays)
            
            # Evaluate leaves
            if len(selection_output.leaf_game_indices) > 0:
                # Create temporary boards for evaluation
                temp_boards = VectorizedCliqueBoard(
                    batch_size=len(selection_output.leaf_game_indices),
                    num_vertices=6,
                    k=3,
                    game_mode="symmetric"
                )
                
                # Set board states
                for i, (game_idx, node_idx) in enumerate(zip(
                    selection_output.leaf_game_indices, 
                    selection_output.leaf_node_indices
                )):
                    edges = arrays.edge_states[game_idx, node_idx]
                    edge_idx = 0
                    for v1 in range(6):
                        for v2 in range(v1 + 1, 6):
                            if edges[edge_idx] == 1:
                                temp_boards.adjacency_matrices = temp_boards.adjacency_matrices.at[i, v1, v2].set(1)
                                temp_boards.adjacency_matrices = temp_boards.adjacency_matrices.at[i, v2, v1].set(1)
                            edge_idx += 1
                    temp_boards.current_players = temp_boards.current_players.at[i].set(
                        arrays.current_players[game_idx, node_idx]
                    )
                
                # Evaluate
                features = temp_boards.get_features_for_nn_undirected()
                valid_masks = temp_boards.get_valid_moves_mask()
                policies, values = neural_network.evaluate_batch(*features, valid_masks)
                
                # Update arrays
                for i, (game_idx, node_idx) in enumerate(zip(
                    selection_output.leaf_game_indices, 
                    selection_output.leaf_node_indices
                )):
                    arrays = arrays._replace(
                        P=arrays.P.at[game_idx, node_idx].set(policies[i]),
                        expanded=arrays.expanded.at[game_idx, node_idx].set(True)
                    )
                
                # Create values for all games
                all_values = jnp.zeros(self.batch_size)
                for i in range(len(selection_output.leaf_game_indices)):
                    game_idx = selection_output.leaf_game_indices[i]
                    all_values = all_values.at[game_idx].set(values[i, 0] if values.ndim > 1 else values[i])
            else:
                all_values = jnp.zeros(self.batch_size)
            
            # Backup
            arrays = self._backup_paths(arrays, selection_output.paths, all_values)
        
        # Extract action probabilities
        root_visits = arrays.N[:, 0, :]
        root_valid_mask = arrays.edge_states[:, 0, :] == 0
        
        if temperature == 0:
            masked_visits = jnp.where(root_valid_mask, root_visits, -jnp.inf)
            action_probs = (masked_visits == jnp.max(masked_visits, axis=1, keepdims=True)).astype(jnp.float32)
            action_probs = jnp.where(root_valid_mask, action_probs, 0.0)
            sum_probs = jnp.sum(action_probs, axis=1, keepdims=True)
            action_probs = jnp.where(sum_probs > 0, action_probs / sum_probs, 
                                   root_valid_mask.astype(jnp.float32) / jnp.sum(root_valid_mask, axis=1, keepdims=True))
        else:
            root_visits_temp = jnp.power(root_visits + 1e-8, 1.0 / temperature)
            root_visits_temp = jnp.where(root_valid_mask, root_visits_temp, 0.0)
            sum_visits = jnp.sum(root_visits_temp, axis=1, keepdims=True)
            action_probs = jnp.where(sum_visits > 0, root_visits_temp / sum_visits, 
                                   root_valid_mask.astype(jnp.float32) / jnp.sum(root_valid_mask, axis=1, keepdims=True))
        
        elapsed = time.time() - start_time
        print(f"MCTX-style MCTS V2 Simple complete in {elapsed:.3f}s ({elapsed/self.batch_size*1000:.1f}ms per game)")
        
        return action_probs
"""
MCTX-style MCTS implementation Phase 2: With jax.lax.while_loop optimization.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial
from typing import Tuple, NamedTuple, Optional

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class MCTSArrays(NamedTuple):
    """Container for all MCTS arrays to make JAX transformations easier."""
    # Tree statistics
    N: jnp.ndarray  # Visit counts [batch, num_nodes, num_actions]
    W: jnp.ndarray  # Total values [batch, num_nodes, num_actions]
    P: jnp.ndarray  # Prior probabilities [batch, num_nodes, num_actions]
    
    # Tree structure
    children: jnp.ndarray  # Child indices [batch, num_nodes, num_actions]
    parents: jnp.ndarray   # Parent indices [batch, num_nodes]
    
    # Node state
    expanded: jnp.ndarray  # Whether node is expanded [batch, num_nodes]
    node_visits: jnp.ndarray  # Visits to each node [batch, num_nodes]
    node_count: jnp.ndarray   # Number of nodes used [batch]
    
    # Board state
    edge_states: jnp.ndarray     # Edge states [batch, num_nodes, num_edges]
    current_players: jnp.ndarray  # Current player [batch, num_nodes]
    game_over: jnp.ndarray       # Game over flag [batch, num_nodes]
    winners: jnp.ndarray         # Winner (-1, 0, 1) [batch, num_nodes]


class SelectionState(NamedTuple):
    """State for tree traversal during selection"""
    node_idx: jnp.ndarray       # Current node index
    path_nodes: jnp.ndarray     # Nodes visited [max_depth]
    path_actions: jnp.ndarray   # Actions taken [max_depth]  
    depth: jnp.ndarray          # Current depth
    found_leaf: jnp.ndarray     # Whether we found a leaf


class MCTXStyleMCTSV2:
    """
    MCTS using MCTX-style pre-allocated arrays with jax.lax.while_loop.
    
    Phase 2 improvements:
    1. jax.lax.while_loop for selection
    2. Vectorized selection across batch
    3. Batched leaf evaluation
    4. JIT compilation of entire search
    """
    
    def __init__(self, 
                 batch_size: int,
                 num_actions: int = 15,
                 max_nodes: int = 800,
                 c_puct: float = 3.0,
                 max_depth: int = 50):
        """
        Initialize MCTS with pre-allocated arrays.
        
        Args:
            batch_size: Number of parallel games
            num_actions: Number of possible actions (edges in clique game)
            max_nodes: Maximum nodes per tree (should be >= num_simulations)
            c_puct: Exploration constant
            max_depth: Maximum tree depth for selection
        """
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.max_nodes = max_nodes
        self.c_puct = c_puct
        self.max_depth = max_depth
        
        # For clique game
        self.num_edges = num_actions  # 15 for 6-vertex graph
        
        # Pre-compile key functions
        self._init_arrays_jit = jax.jit(self._init_arrays)
        self._select_batch_jit = jax.jit(self._select_batch)
        self._expand_batch_jit = jax.jit(self._expand_batch)
        self._backup_batch_jit = jax.jit(self._backup_batch)
        
    def _init_arrays(self) -> MCTSArrays:
        """Initialize all arrays. JIT-compiled."""
        return MCTSArrays(
            # Tree statistics - initialized to zero
            N=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            W=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            P=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            
            # Tree structure - children/parents initialized to -1 (no connection)
            children=jnp.full((self.batch_size, self.max_nodes, self.num_actions), -1, dtype=jnp.int32),
            parents=jnp.full((self.batch_size, self.max_nodes), -1, dtype=jnp.int32),
            
            # Node state
            expanded=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.bool_),
            node_visits=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32),
            node_count=jnp.ones(self.batch_size, dtype=jnp.int32),  # Start with 1 (root)
            
            # Board state
            edge_states=jnp.zeros((self.batch_size, self.max_nodes, self.num_edges), dtype=jnp.int32),
            current_players=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32),
            game_over=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.bool_),
            winners=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32),
        )
    
    def _select_path_single(self, arrays: MCTSArrays, game_idx: int) -> SelectionState:
        """Select a path through the tree for a single game using while_loop"""
        
        def cond_fun(state: SelectionState) -> bool:
            """Continue while node is expanded and we haven't found a leaf"""
            is_expanded = arrays.expanded[game_idx, state.node_idx]
            is_game_over = arrays.game_over[game_idx, state.node_idx]
            under_max_depth = state.depth < self.max_depth
            return is_expanded & (~state.found_leaf) & (~is_game_over) & under_max_depth
        
        def body_fun(state: SelectionState) -> SelectionState:
            """Select action and traverse tree"""
            node_idx = state.node_idx
            
            # Get node statistics
            N = arrays.N[game_idx, node_idx]
            W = arrays.W[game_idx, node_idx]
            P = arrays.P[game_idx, node_idx]
            node_visits = arrays.node_visits[game_idx, node_idx]
            
            # Get valid moves mask
            edge_states = arrays.edge_states[game_idx, node_idx]
            valid_mask = edge_states == 0
            
            # Calculate UCB
            Q = W / (1.0 + N)
            sqrt_visits = jnp.sqrt(jnp.maximum(1.0, node_visits))
            U = self.c_puct * sqrt_visits * (P / (1.0 + N))
            ucb = Q + U
            
            # Mask invalid actions
            ucb = jnp.where(valid_mask, ucb, -jnp.inf)
            
            # Check if any valid moves
            has_valid_moves = jnp.any(valid_mask)
            
            # Select action (or -1 if no valid moves)
            action = jnp.where(
                has_valid_moves,
                jnp.argmax(ucb),
                -1
            )
            
            # Get child index
            child_idx = jnp.where(
                action >= 0,
                arrays.children[game_idx, node_idx, action],
                -1
            )
            
            # Check if we need to expand (child doesn't exist)
            need_expand = (child_idx == -1) & (action >= 0)
            
            # Update path
            new_path_nodes = state.path_nodes.at[state.depth].set(node_idx)
            new_path_actions = state.path_actions.at[state.depth].set(action)
            
            # Next node is either child or current (if expansion needed or no moves)
            next_node = jnp.where(
                need_expand | ~has_valid_moves,
                node_idx,
                child_idx
            )
            
            return SelectionState(
                node_idx=next_node,
                path_nodes=new_path_nodes,
                path_actions=new_path_actions,
                depth=state.depth + 1,
                found_leaf=need_expand | ~has_valid_moves
            )
        
        # Initialize state
        init_state = SelectionState(
            node_idx=0,  # Start at root
            path_nodes=jnp.full((self.max_depth,), -1, dtype=jnp.int32),
            path_actions=jnp.full((self.max_depth,), -1, dtype=jnp.int32),
            depth=0,
            found_leaf=False
        )
        
        # Run while loop
        final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
        
        return final_state
    
    def _select_batch(self, arrays: MCTSArrays) -> Tuple[SelectionState, MCTSArrays]:
        """Select paths for all games in parallel"""
        # Vectorize selection over batch
        select_all = jax.vmap(
            self._select_path_single,
            in_axes=(None, 0)  # arrays shared, game_idx varies
        )
        
        game_indices = jnp.arange(self.batch_size)
        selection_states = select_all(arrays, game_indices)
        
        return selection_states, arrays
    
    def _expand_batch(self, arrays: MCTSArrays, selection_states: SelectionState) -> MCTSArrays:
        """Expand nodes for all games that need it"""
        
        def expand_single(arrays: MCTSArrays, game_idx: int, state: SelectionState) -> MCTSArrays:
            """Expand a single node if needed"""
            # Get the leaf node and action
            leaf_idx = state.node_idx
            leaf_depth = jnp.maximum(0, state.depth - 1)
            parent_idx = state.path_nodes[leaf_depth]
            action = state.path_actions[leaf_depth]
            
            # Check if we need to expand (found leaf with valid action)
            need_expand = state.found_leaf & (action >= 0) & (~arrays.game_over[game_idx, parent_idx])
            
            # Get next node index
            new_node_idx = arrays.node_count[game_idx]
            
            # Only update if we need to expand and have space
            can_expand = need_expand & (new_node_idx < self.max_nodes)
            
            # Update tree structure
            arrays = arrays._replace(
                children=arrays.children.at[game_idx, parent_idx, action].set(
                    jnp.where(can_expand, new_node_idx, arrays.children[game_idx, parent_idx, action])
                ),
                parents=arrays.parents.at[game_idx, new_node_idx].set(
                    jnp.where(can_expand, parent_idx, arrays.parents[game_idx, new_node_idx])
                ),
                node_count=arrays.node_count.at[game_idx].set(
                    jnp.where(can_expand, new_node_idx + 1, arrays.node_count[game_idx])
                )
            )
            
            # Copy and update board state
            parent_edges = arrays.edge_states[game_idx, parent_idx]
            new_edges = parent_edges.at[action].set(
                jnp.where(can_expand, 1, parent_edges[action])
            )
            
            arrays = arrays._replace(
                edge_states=arrays.edge_states.at[game_idx, new_node_idx].set(
                    jnp.where(can_expand, new_edges, arrays.edge_states[game_idx, new_node_idx])
                ),
                current_players=arrays.current_players.at[game_idx, new_node_idx].set(
                    jnp.where(can_expand, 1 - arrays.current_players[game_idx, parent_idx], 
                             arrays.current_players[game_idx, new_node_idx])
                )
            )
            
            return arrays
        
        # Process all games
        for game_idx in range(self.batch_size):
            state = jax.tree_map(lambda x: x[game_idx], selection_states)
            arrays = expand_single(arrays, game_idx, state)
        
        return arrays
    
    def _evaluate_leaves_batch(self, arrays: MCTSArrays, selection_states: SelectionState,
                              neural_network: ImprovedBatchedNeuralNetwork) -> Tuple[jnp.ndarray, MCTSArrays]:
        """Evaluate all leaf nodes in a single batch"""
        
        # Collect leaves to evaluate
        leaf_nodes = []
        leaf_game_indices = []
        
        for game_idx in range(self.batch_size):
            state = jax.tree_map(lambda x: x[game_idx], selection_states)
            
            # Get the actual leaf node
            if state.found_leaf and state.path_actions[state.depth - 1] >= 0:
                # Node was expanded, evaluate the new child
                parent_idx = state.path_nodes[state.depth - 1]
                action = state.path_actions[state.depth - 1]
                child_idx = arrays.children[game_idx, parent_idx, action]
                
                if child_idx >= 0 and not arrays.expanded[game_idx, child_idx]:
                    leaf_nodes.append(child_idx)
                    leaf_game_indices.append(game_idx)
            elif not arrays.expanded[game_idx, state.node_idx] and not arrays.game_over[game_idx, state.node_idx]:
                # Reached unexpanded node
                leaf_nodes.append(state.node_idx)
                leaf_game_indices.append(game_idx)
        
        if not leaf_nodes:
            return jnp.zeros(self.batch_size), arrays
        
        # Convert to arrays
        leaf_nodes = jnp.array(leaf_nodes)
        leaf_game_indices = jnp.array(leaf_game_indices)
        
        # Extract features for neural network
        leaf_edges = arrays.edge_states[leaf_game_indices, leaf_nodes]
        leaf_players = arrays.current_players[leaf_game_indices, leaf_nodes]
        
        # Convert to board features
        # For simplicity, create temporary boards
        temp_boards = VectorizedCliqueBoard(
            batch_size=len(leaf_nodes),
            num_vertices=6,
            k=3,
            game_mode="symmetric"
        )
        
        # Set board states
        for i, edges in enumerate(leaf_edges):
            edge_idx = 0
            for v1 in range(6):
                for v2 in range(v1 + 1, 6):
                    if edges[edge_idx] == 1:
                        temp_boards.adjacency_matrices = temp_boards.adjacency_matrices.at[i, v1, v2].set(1)
                        temp_boards.adjacency_matrices = temp_boards.adjacency_matrices.at[i, v2, v1].set(1)
                    edge_idx += 1
            temp_boards.current_players = temp_boards.current_players.at[i].set(leaf_players[i])
        
        # Get features and evaluate
        features = temp_boards.get_features_for_nn_undirected()
        valid_masks = leaf_edges == 0
        
        policies, values = neural_network.evaluate_batch(*features, valid_masks)
        
        # Update arrays with evaluations
        for i, (game_idx, node_idx) in enumerate(zip(leaf_game_indices, leaf_nodes)):
            arrays = arrays._replace(
                P=arrays.P.at[game_idx, node_idx].set(policies[i]),
                expanded=arrays.expanded.at[game_idx, node_idx].set(True)
            )
        
        # Create value array for all games
        all_values = jnp.zeros(self.batch_size)
        for i, game_idx in enumerate(leaf_game_indices):
            all_values = all_values.at[game_idx].set(values[i])
        
        return all_values, arrays
    
    def _backup_batch(self, arrays: MCTSArrays, selection_states: SelectionState, 
                     values: jnp.ndarray) -> MCTSArrays:
        """Backup values along all paths"""
        
        def backup_single(arrays: MCTSArrays, game_idx: int, state: SelectionState, value: float) -> MCTSArrays:
            """Backup value along a single path"""
            
            # Process path backwards
            for depth in range(state.depth):
                node_idx = state.path_nodes[depth]
                action = state.path_actions[depth]
                
                # Check if valid step in path
                is_valid = (node_idx >= 0) & (action >= 0)
                
                if is_valid:
                    # Update statistics
                    arrays = arrays._replace(
                        N=arrays.N.at[game_idx, node_idx, action].add(1.0),
                        W=arrays.W.at[game_idx, node_idx, action].add(value)
                    )
                    
                    # Flip value for opponent
                    value = -value
            
            return arrays
        
        # Process all games
        for game_idx in range(self.batch_size):
            state = jax.tree_map(lambda x: x[game_idx], selection_states)
            arrays = backup_single(arrays, game_idx, state, values[game_idx])
        
        return arrays
    
    def _initialize_root_boards(self, arrays: MCTSArrays, 
                               boards: VectorizedCliqueBoard) -> MCTSArrays:
        """Initialize root node board states from input boards."""
        edge_states = jnp.zeros((self.batch_size, self.num_edges), dtype=jnp.int32)
        
        # Convert adjacency matrix to edge list
        edge_idx = 0
        edge_list = []
        for i in range(6):  # 6 vertices
            for j in range(i + 1, 6):
                edge_list.append((i, j))
        
        # Get edge states from board
        for idx, (i, j) in enumerate(edge_list):
            edge_played = boards.edge_states[:, i, j]  # 1 if played, 0 if not
            edge_states = edge_states.at[:, idx].set(edge_played)
        
        # Update root node (index 0) board states
        arrays = arrays._replace(
            edge_states=arrays.edge_states.at[:, 0, :].set(edge_states),
            current_players=arrays.current_players.at[:, 0].set(boards.current_players),
            game_over=arrays.game_over.at[:, 0].set(boards.game_states != 0),
            winners=arrays.winners.at[:, 0].set(boards.winners)
        )
        
        return arrays
    
    def _search_iteration(self, arrays: MCTSArrays, 
                         neural_network: ImprovedBatchedNeuralNetwork) -> MCTSArrays:
        """Run one full MCTS iteration (selection, expansion, evaluation, backup)"""
        
        # Selection
        selection_states, arrays = self._select_batch(arrays)
        
        # Expansion
        arrays = self._expand_batch(arrays, selection_states)
        
        # Evaluation
        values, arrays = self._evaluate_leaves_batch(arrays, selection_states, neural_network)
        
        # Backup
        arrays = self._backup_batch(arrays, selection_states, values)
        
        return arrays
    
    def search(self,
               boards: VectorizedCliqueBoard,
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               temperature: float = 1.0) -> jnp.ndarray:
        """
        Run MCTS search using pre-allocated arrays with while_loop optimization.
        
        Args:
            boards: Current board states
            neural_network: Neural network for evaluation
            num_simulations: Number of MCTS simulations
            temperature: Temperature for action selection
            
        Returns:
            Action probabilities [batch_size, num_actions]
        """
        print(f"Starting MCTX-style MCTS V2 with {num_simulations} simulations")
        start_time = time.time()
        
        # Initialize arrays
        arrays = self._init_arrays_jit()
        
        # Set up root board states
        arrays = self._initialize_root_boards(arrays, boards)
        
        # Evaluate root positions
        root_policies, root_values = neural_network.evaluate_batch(
            *boards.get_features_for_nn_undirected(),
            boards.get_valid_moves_mask()
        )
        
        # Set root priors and mark as expanded
        arrays = arrays._replace(
            P=arrays.P.at[:, 0, :].set(root_policies),
            expanded=arrays.expanded.at[:, 0].set(True)
        )
        
        # Main MCTS loop
        for sim in range(num_simulations):
            arrays = self._search_iteration(arrays, neural_network)
        
        # Extract action probabilities from root visit counts
        root_visits = arrays.N[:, 0, :]  # [batch_size, num_actions]
        
        # Get valid moves mask for root
        root_valid_mask = arrays.edge_states[:, 0, :] == 0
        
        if temperature == 0:
            # Deterministic: choose most visited
            masked_visits = jnp.where(root_valid_mask, root_visits, -jnp.inf)
            action_probs = (masked_visits == jnp.max(masked_visits, axis=1, keepdims=True)).astype(jnp.float32)
            action_probs = jnp.where(root_valid_mask, action_probs, 0.0)
            # Normalize
            sum_probs = jnp.sum(action_probs, axis=1, keepdims=True)
            action_probs = jnp.where(sum_probs > 0, action_probs / sum_probs, 
                                   root_valid_mask.astype(jnp.float32) / jnp.sum(root_valid_mask, axis=1, keepdims=True))
        else:
            # Apply temperature
            root_visits_temp = jnp.power(root_visits + 1e-8, 1.0 / temperature)
            root_visits_temp = jnp.where(root_valid_mask, root_visits_temp, 0.0)
            sum_visits = jnp.sum(root_visits_temp, axis=1, keepdims=True)
            action_probs = jnp.where(sum_visits > 0, root_visits_temp / sum_visits, 
                                   root_valid_mask.astype(jnp.float32) / jnp.sum(root_valid_mask, axis=1, keepdims=True))
        
        elapsed = time.time() - start_time
        print(f"MCTX-style MCTS V2 complete in {elapsed:.3f}s ({elapsed/self.batch_size*1000:.1f}ms per game)")
        
        return action_probs


# Test basic functionality
if __name__ == "__main__":
    print("Testing MCTX-style MCTS V2...")
    
    # Create instance
    mcts = MCTXStyleMCTSV2(
        batch_size=4,
        num_actions=15,
        max_nodes=100,
        c_puct=3.0
    )
    
    # Create test boards
    boards = VectorizedCliqueBoard(
        batch_size=4,
        num_vertices=6,
        k=3,
        game_mode="symmetric"
    )
    
    # Create neural network
    nn = ImprovedBatchedNeuralNetwork(
        num_vertices=6,
        hidden_dim=128,
        num_layers=4
    )
    
    # Test search
    print("\nRunning test search...")
    action_probs = mcts.search(boards, nn, num_simulations=10, temperature=1.0)
    
    print(f"\nAction probabilities shape: {action_probs.shape}")
    print(f"Probabilities sum to 1: {jnp.allclose(jnp.sum(action_probs, axis=1), 1.0)}")
    print("\nPhase 2 implementation complete!")
"""
MCTX-style MCTS implementation with pre-allocated arrays.
Based on DeepMind's approach for efficient JAX tree search.
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


class MCTXStyleMCTS:
    """
    MCTS using MCTX-style pre-allocated arrays for efficiency.
    
    Key differences from SimpleTreeMCTS:
    1. All memory pre-allocated - no dynamic allocation
    2. Trees represented as arrays, not dicts
    3. Board states stored as arrays, not objects
    4. Designed for JIT compilation
    """
    
    def __init__(self, 
                 batch_size: int,
                 num_actions: int = 15,
                 max_nodes: int = 800,
                 c_puct: float = 3.0):
        """
        Initialize MCTS with pre-allocated arrays.
        
        Args:
            batch_size: Number of parallel games
            num_actions: Number of possible actions (edges in clique game)
            max_nodes: Maximum nodes per tree (should be >= num_simulations)
            c_puct: Exploration constant
        """
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.max_nodes = max_nodes
        self.c_puct = c_puct
        
        # For clique game
        self.num_edges = num_actions  # 15 for 6-vertex graph
        
        # Pre-compile key functions
        self._init_arrays_jit = jax.jit(self._init_arrays)
        self._calculate_ucb_jit = jax.jit(self._calculate_ucb)
        self._update_stats_jit = jax.jit(self._update_stats)
        self._make_move_on_edges_jit = jax.jit(self._make_move_on_edges)
        
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
    
    def _calculate_ucb(self, N: jnp.ndarray, W: jnp.ndarray, P: jnp.ndarray,
                      node_visits: float, valid_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate UCB scores for action selection.
        
        Args:
            N: Visit counts for actions [num_actions]
            W: Total values for actions [num_actions]
            P: Prior probabilities [num_actions]
            node_visits: Total visits to this node
            valid_mask: Valid actions mask [num_actions]
            
        Returns:
            UCB scores [num_actions]
        """
        # Q values with pseudocount
        Q = W / (1.0 + N)
        
        # Exploration term
        sqrt_visits = jnp.sqrt(jnp.maximum(1.0, node_visits))
        U = self.c_puct * sqrt_visits * (P / (1.0 + N))
        
        # Combine
        ucb = Q + U
        
        # Mask invalid actions
        ucb = jnp.where(valid_mask, ucb, -jnp.inf)
        
        return ucb
    
    def _update_stats(self, N: jnp.ndarray, W: jnp.ndarray, 
                     action: int, value: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update N and W statistics. JIT-compiled."""
        N_new = N.at[action].add(1.0)
        W_new = W.at[action].add(value)
        return N_new, W_new
    
    def _make_move_on_edges(self, edge_states: jnp.ndarray, action: int) -> jnp.ndarray:
        """Make a move by setting edge state. JIT-compiled."""
        return edge_states.at[action].set(1)
    
    def _get_valid_moves_mask(self, edge_states: jnp.ndarray) -> jnp.ndarray:
        """Get valid moves (unplayed edges). JIT-compiled."""
        return edge_states == 0
    
    def _initialize_root_boards(self, arrays: MCTSArrays, 
                               boards: VectorizedCliqueBoard) -> MCTSArrays:
        """
        Initialize root node board states from input boards.
        
        Args:
            arrays: MCTS arrays
            boards: Input board states
            
        Returns:
            Updated arrays with root board states
        """
        # For VectorizedCliqueBoard, the edge_states is actually stored in adjacency matrix form
        # We need to flatten it to our edge list representation
        # For clique game with 6 vertices, we have 15 edges
        # Edge mapping: (0,1)=0, (0,2)=1, ..., (4,5)=14
        
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
    
    def search(self,
               boards: VectorizedCliqueBoard,
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               temperature: float = 1.0) -> jnp.ndarray:
        """
        Run MCTS search using pre-allocated arrays.
        
        Args:
            boards: Current board states
            neural_network: Neural network for evaluation
            num_simulations: Number of MCTS simulations
            temperature: Temperature for action selection
            
        Returns:
            Action probabilities [batch_size, num_actions]
        """
        print(f"Starting MCTX-style MCTS with {num_simulations} simulations")
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
            # For now, implement a simple version without full JIT
            # This will be optimized in later phases
            
            for batch_idx in range(self.batch_size):
                # Skip finished games
                if arrays.game_over[batch_idx, 0]:
                    continue
                
                # Selection phase
                node_idx = 0
                path = []
                
                # Traverse tree
                while node_idx >= 0 and arrays.expanded[batch_idx, node_idx]:
                    # Increment node visits
                    arrays = arrays._replace(
                        node_visits=arrays.node_visits.at[batch_idx, node_idx].add(1)
                    )
                    
                    # Get valid moves
                    edge_states = arrays.edge_states[batch_idx, node_idx]
                    valid_mask = self._get_valid_moves_mask(edge_states)
                    
                    # Calculate UCB
                    ucb_scores = self._calculate_ucb_jit(
                        arrays.N[batch_idx, node_idx],
                        arrays.W[batch_idx, node_idx],
                        arrays.P[batch_idx, node_idx],
                        float(arrays.node_visits[batch_idx, node_idx]),
                        valid_mask
                    )
                    
                    # Select action
                    if jnp.all(ucb_scores == -jnp.inf):
                        break  # No valid moves
                    
                    action = jnp.argmax(ucb_scores)
                    
                    # Check if child exists
                    child_idx = arrays.children[batch_idx, node_idx, action]
                    
                    if child_idx < 0:  # Need to expand
                        # Check if we have space
                        if arrays.node_count[batch_idx] >= self.max_nodes:
                            break
                        
                        # Create new node
                        child_idx = arrays.node_count[batch_idx]
                        
                        # Update tree structure
                        arrays = arrays._replace(
                            children=arrays.children.at[batch_idx, node_idx, action].set(child_idx),
                            parents=arrays.parents.at[batch_idx, child_idx].set(node_idx),
                            node_count=arrays.node_count.at[batch_idx].add(1)
                        )
                        
                        # Copy and update board state
                        new_edge_states = self._make_move_on_edges_jit(edge_states, action)
                        new_player = 1 - arrays.current_players[batch_idx, node_idx]
                        
                        arrays = arrays._replace(
                            edge_states=arrays.edge_states.at[batch_idx, child_idx].set(new_edge_states),
                            current_players=arrays.current_players.at[batch_idx, child_idx].set(new_player)
                        )
                        
                        path.append((node_idx, action))
                        node_idx = child_idx
                        break
                    else:
                        path.append((node_idx, action))
                        node_idx = child_idx
                
                # Expansion and evaluation would go here
                # For now, using placeholder values
                if node_idx >= 0 and not arrays.expanded[batch_idx, node_idx]:
                    # Mark as expanded
                    arrays = arrays._replace(
                        expanded=arrays.expanded.at[batch_idx, node_idx].set(True)
                    )
                    
                    # Placeholder policy and value
                    # In full implementation, would batch evaluate all leaves
                    policy = jnp.ones(self.num_actions) / self.num_actions
                    value = 0.0
                    
                    arrays = arrays._replace(
                        P=arrays.P.at[batch_idx, node_idx].set(policy)
                    )
                    
                    # Backup
                    for parent_idx, action in reversed(path):
                        N_new, W_new = self._update_stats_jit(
                            arrays.N[batch_idx, parent_idx],
                            arrays.W[batch_idx, parent_idx],
                            action,
                            value
                        )
                        arrays = arrays._replace(
                            N=arrays.N.at[batch_idx, parent_idx].set(N_new),
                            W=arrays.W.at[batch_idx, parent_idx].set(W_new)
                        )
                        value = -value  # Flip for opponent
        
        # Extract action probabilities from root visit counts
        root_visits = arrays.N[:, 0, :]  # [batch_size, num_actions]
        
        # Get valid moves mask for root
        root_valid_mask = self._get_valid_moves_mask(arrays.edge_states[:, 0, :])
        
        if temperature == 0:
            # Deterministic: choose most visited
            masked_visits = jnp.where(root_valid_mask, root_visits, -jnp.inf)
            action_probs = (masked_visits == jnp.max(masked_visits, axis=1, keepdims=True)).astype(jnp.float32)
            action_probs = jnp.where(root_valid_mask, action_probs, 0.0)
            # Normalize
            sum_probs = jnp.sum(action_probs, axis=1, keepdims=True)
            action_probs = jnp.where(sum_probs > 0, action_probs / sum_probs, root_valid_mask.astype(jnp.float32) / jnp.sum(root_valid_mask, axis=1, keepdims=True))
        else:
            # Apply temperature
            root_visits_temp = jnp.power(root_visits + 1e-8, 1.0 / temperature)  # Add small epsilon
            root_visits_temp = jnp.where(root_valid_mask, root_visits_temp, 0.0)
            sum_visits = jnp.sum(root_visits_temp, axis=1, keepdims=True)
            action_probs = jnp.where(sum_visits > 0, root_visits_temp / sum_visits, root_valid_mask.astype(jnp.float32) / jnp.sum(root_valid_mask, axis=1, keepdims=True))
        
        elapsed = time.time() - start_time
        print(f"MCTX-style MCTS complete in {elapsed:.3f}s")
        
        # Print memory usage
        total_memory = sum(arr.nbytes for arr in arrays)
        print(f"Total memory used: {total_memory / 1024 / 1024:.1f} MB")
        
        return action_probs


# Test basic functionality
if __name__ == "__main__":
    print("Testing MCTX-style MCTS basic setup...")
    
    # Create instance
    mcts = MCTXStyleMCTS(
        batch_size=2,
        num_actions=15,
        max_nodes=100,
        c_puct=3.0
    )
    
    # Test array initialization
    arrays = mcts._init_arrays_jit()
    print(f"Arrays initialized successfully")
    print(f"N shape: {arrays.N.shape}")
    print(f"children shape: {arrays.children.shape}")
    print(f"edge_states shape: {arrays.edge_states.shape}")
    
    # Test JIT compilation
    print("\nTesting JIT compilation...")
    
    # Time non-JIT vs JIT
    N = jnp.zeros(15)
    W = jnp.zeros(15)
    P = jnp.ones(15) / 15
    valid_mask = jnp.ones(15, dtype=jnp.bool_)
    
    # First call (includes compilation)
    start = time.time()
    ucb1 = mcts._calculate_ucb_jit(N, W, P, 1.0, valid_mask)
    jit_first = time.time() - start
    
    # Second call (already compiled)
    start = time.time()
    ucb2 = mcts._calculate_ucb_jit(N, W, P, 1.0, valid_mask)
    jit_second = time.time() - start
    
    print(f"First JIT call: {jit_first*1000:.2f}ms (includes compilation)")
    print(f"Second JIT call: {jit_second*1000:.2f}ms")
    print(f"Speedup: {jit_first/jit_second:.1f}x")
    
    print("\nPhase 1 complete! Pre-allocated arrays working correctly.")
"""
MCTX Phase 2 Final Implementation
Focuses on core optimizations that actually matter.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import Tuple, NamedTuple

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class MCTSArrays(NamedTuple):
    """Pre-allocated arrays for MCTS"""
    N: jnp.ndarray  # Visit counts [batch, num_nodes, num_actions]
    W: jnp.ndarray  # Total values [batch, num_nodes, num_actions]
    P: jnp.ndarray  # Prior probabilities [batch, num_nodes, num_actions]
    children: jnp.ndarray  # Child indices [batch, num_nodes, num_actions]
    expanded: jnp.ndarray  # Whether node is expanded [batch, num_nodes]
    node_count: jnp.ndarray   # Number of nodes used [batch]
    edge_states: jnp.ndarray     # Edge states [batch, num_nodes, num_edges]
    current_players: jnp.ndarray  # Current player [batch, num_nodes]


class MCTXPhase2Final:
    """
    Final Phase 2 implementation with key optimizations:
    1. Pre-allocated arrays (no dynamic allocation)
    2. Vectorized operations where possible
    3. Single batched neural network evaluation per iteration
    4. Efficient tree traversal
    """
    
    def __init__(self, batch_size: int, num_actions: int = 15, 
                 max_nodes: int = 500, c_puct: float = 3.0):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.max_nodes = max_nodes
        self.c_puct = c_puct
        
    def _init_arrays(self) -> MCTSArrays:
        """Initialize all arrays"""
        return MCTSArrays(
            N=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            W=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            P=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            children=jnp.full((self.batch_size, self.max_nodes, self.num_actions), -1, dtype=jnp.int32),
            expanded=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.bool_),
            node_count=jnp.ones(self.batch_size, dtype=jnp.int32),
            edge_states=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions), dtype=jnp.int32),
            current_players=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32),
        )
    
    def _select_and_expand_batch(self, arrays: MCTSArrays) -> Tuple[MCTSArrays, list, list]:
        """
        Select paths and expand nodes for all games.
        Returns updated arrays and lists of (game_idx, node_idx, path) for evaluation.
        """
        leaves_to_eval = []
        backup_paths = []
        
        for game_idx in range(self.batch_size):
            node_idx = 0
            path = []
            
            # Traverse until we find unexpanded node
            while arrays.expanded[game_idx, node_idx]:
                # Get statistics
                N = arrays.N[game_idx, node_idx]
                W = arrays.W[game_idx, node_idx]
                P = arrays.P[game_idx, node_idx]
                
                # Valid moves
                edge_states = arrays.edge_states[game_idx, node_idx]
                valid_mask = edge_states == 0
                
                if not np.any(valid_mask):
                    break  # No valid moves
                
                # Calculate UCB
                N_sum = np.sum(N) + 1
                Q = W / (N + 1e-8)
                U = self.c_puct * np.sqrt(N_sum) * P / (N + 1)
                ucb = Q + U
                ucb = np.where(valid_mask, ucb, -np.inf)
                
                # Select action
                action = int(np.argmax(ucb))
                path.append((node_idx, action))
                
                # Check if child exists
                child_idx = arrays.children[game_idx, node_idx, action]
                
                if child_idx == -1:
                    # Need to expand
                    new_idx = int(arrays.node_count[game_idx])
                    if new_idx < self.max_nodes:
                        # Create child
                        arrays = arrays._replace(
                            children=arrays.children.at[game_idx, node_idx, action].set(new_idx),
                            node_count=arrays.node_count.at[game_idx].add(1)
                        )
                        
                        # Copy board state
                        new_edges = arrays.edge_states[game_idx, node_idx].at[action].set(1)
                        arrays = arrays._replace(
                            edge_states=arrays.edge_states.at[game_idx, new_idx].set(new_edges),
                            current_players=arrays.current_players.at[game_idx, new_idx].set(
                                1 - arrays.current_players[game_idx, node_idx]
                            )
                        )
                        
                        leaves_to_eval.append((game_idx, new_idx))
                        backup_paths.append((game_idx, path))
                    break
                else:
                    node_idx = child_idx
            
            # If we exited normally (found unexpanded node)
            if not arrays.expanded[game_idx, node_idx] and len(path) > 0:
                leaves_to_eval.append((game_idx, node_idx))
                backup_paths.append((game_idx, path[:-1]))  # Don't include last action
        
        return arrays, leaves_to_eval, backup_paths
    
    def _evaluate_and_backup(self, arrays: MCTSArrays, leaves: list, paths: list,
                           neural_network: ImprovedBatchedNeuralNetwork) -> MCTSArrays:
        """Evaluate leaves and backup values"""
        
        if not leaves:
            return arrays
        
        # Create batch for neural network
        batch_size = len(leaves)
        temp_boards = VectorizedCliqueBoard(batch_size, 6, 3, "symmetric")
        
        # Set up board states
        for i, (game_idx, node_idx) in enumerate(leaves):
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
        
        # Update nodes with evaluations
        for i, (game_idx, node_idx) in enumerate(leaves):
            arrays = arrays._replace(
                P=arrays.P.at[game_idx, node_idx].set(policies[i]),
                expanded=arrays.expanded.at[game_idx, node_idx].set(True)
            )
        
        # Backup values
        for i, (game_idx, path) in enumerate(paths):
            if i < len(values):
                value = float(values[i, 0]) if values.ndim > 1 else float(values[i])
                # Walk backwards through path
                for node_idx, action in reversed(path):
                    arrays = arrays._replace(
                        N=arrays.N.at[game_idx, node_idx, action].add(1),
                        W=arrays.W.at[game_idx, node_idx, action].add(value)
                    )
                    value = -value  # Flip for opponent
        
        return arrays
    
    def search(self, boards: VectorizedCliqueBoard, neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int, temperature: float = 1.0) -> jnp.ndarray:
        """Run MCTS search"""
        print(f"Starting MCTX Phase 2 Final with {num_simulations} simulations")
        start_time = time.time()
        
        # Initialize
        arrays = self._init_arrays()
        
        # Setup root
        edge_states = jnp.zeros((self.batch_size, self.num_actions), dtype=jnp.int32)
        edge_idx = 0
        for i in range(6):
            for j in range(i + 1, 6):
                edge_states = edge_states.at[:, edge_idx].set(boards.edge_states[:, i, j])
                edge_idx += 1
        
        arrays = arrays._replace(
            edge_states=arrays.edge_states.at[:, 0, :].set(edge_states),
            current_players=arrays.current_players.at[:, 0].set(boards.current_players)
        )
        
        # Evaluate root
        root_policies, _ = neural_network.evaluate_batch(
            *boards.get_features_for_nn_undirected(),
            boards.get_valid_moves_mask()
        )
        
        arrays = arrays._replace(
            P=arrays.P.at[:, 0, :].set(root_policies),
            expanded=arrays.expanded.at[:, 0].set(True)
        )
        
        # Main loop
        for sim in range(num_simulations):
            # Select and expand
            arrays, leaves, paths = self._select_and_expand_batch(arrays)
            
            # Evaluate and backup
            arrays = self._evaluate_and_backup(arrays, leaves, paths, neural_network)
        
        # Extract action probabilities
        root_visits = arrays.N[:, 0, :]
        root_valid = edge_states == 0
        
        if temperature == 0:
            # Deterministic
            masked_visits = jnp.where(root_valid, root_visits, -jnp.inf)
            action_probs = (masked_visits == jnp.max(masked_visits, axis=1, keepdims=True)).astype(jnp.float32)
        else:
            # Stochastic
            root_visits_temp = jnp.power(root_visits + 1e-8, 1.0 / temperature)
            root_visits_temp = jnp.where(root_valid, root_visits_temp, 0.0)
            action_probs = root_visits_temp / jnp.sum(root_visits_temp, axis=1, keepdims=True)
        
        elapsed = time.time() - start_time
        print(f"Phase 2 complete in {elapsed:.3f}s ({elapsed/self.batch_size*1000:.1f}ms per game)")
        
        return action_probs


# Test
if __name__ == "__main__":
    print("Testing MCTX Phase 2 Final...")
    
    batch_size = 4
    boards = VectorizedCliqueBoard(batch_size, 6, 3, "symmetric")
    nn = ImprovedBatchedNeuralNetwork(6, 128, 4)
    mcts = MCTXPhase2Final(batch_size)
    
    probs = mcts.search(boards, nn, 10, 1.0)
    print(f"Action probs shape: {probs.shape}")
    print(f"Sum: {jnp.sum(probs, axis=1)}")
#!/usr/bin/env python
"""
JAX implementation of MCTS for Clique Game.
Designed for massive parallelization on GPU with batched operations.
"""

import numpy as np
from typing import NamedTuple, Tuple, Dict, List, Optional
import warnings
from dataclasses import dataclass

# Import JAX components if available
try:
    import jax
    import jax.numpy as jnp
    from jax import vmap, jit
    import jax.random as jrandom
    JAX_AVAILABLE = True
except ImportError:
    warnings.warn("JAX not available, using NumPy fallback")
    jnp = np
    JAX_AVAILABLE = False
    # Mock functions for numpy fallback
    def vmap(func, in_axes=0, out_axes=0):
        return func
    def jit(func):
        return func

try:
    from jax_src.jax_clique_board import JAXCliqueBoard, CliqueBoardState
except ImportError:
    from jax_src.jax_clique_board_numpy import JAXCliqueBoard, CliqueBoardState
from jax_src.jax_alpha_net_clique import CliqueGNN
import src.encoder_decoder_clique as ed


class BatchedMCTSState(NamedTuple):
    """State for batched MCTS trees"""
    # Tree structure - all arrays have shape (batch_size, max_nodes, ...)
    node_boards: np.ndarray  # Board states for each node
    node_visits: np.ndarray  # Visit counts N(s)
    node_is_expanded: np.ndarray  # Whether node is expanded
    node_parent: np.ndarray  # Parent node indices (-1 for root)
    
    # Edge statistics - shape (batch_size, max_nodes, max_actions)
    edge_visits: np.ndarray  # N(s,a) - visit counts for edges
    edge_total_value: np.ndarray  # W(s,a) - total value for edges
    edge_priors: np.ndarray  # P(s,a) - prior probabilities
    edge_valid_mask: np.ndarray  # Valid actions mask
    
    # Tree metadata
    batch_size: int
    max_nodes: int
    max_actions: int
    num_vertices: int
    k: int
    
    # Current tree size for each batch
    tree_sizes: np.ndarray  # Shape (batch_size,)


class VectorizedMCTS:
    """Vectorized MCTS implementation for batch processing on GPU"""
    
    def __init__(self, num_vertices: int, k: int, max_nodes: int = 10000, 
                 c_puct: float = 1.0, game_mode: str = "symmetric"):
        self.num_vertices = num_vertices
        self.k = k
        self.max_nodes = max_nodes
        self.max_actions = num_vertices * (num_vertices - 1) // 2
        self.c_puct = c_puct
        self.game_mode = game_mode
        
    def initialize_batch(self, board_states: List[CliqueBoardState], batch_size: int) -> BatchedMCTSState:
        """Initialize a batch of MCTS trees"""
        # Initialize arrays
        node_boards = np.zeros((batch_size, self.max_nodes, 
                               self.num_vertices, self.num_vertices), dtype=np.int32)
        node_visits = np.zeros((batch_size, self.max_nodes), dtype=np.float32)
        node_is_expanded = np.zeros((batch_size, self.max_nodes), dtype=bool)
        node_parent = np.full((batch_size, self.max_nodes), -1, dtype=np.int32)
        
        edge_visits = np.zeros((batch_size, self.max_nodes, self.max_actions), dtype=np.float32)
        edge_total_value = np.zeros((batch_size, self.max_nodes, self.max_actions), dtype=np.float32)
        edge_priors = np.zeros((batch_size, self.max_nodes, self.max_actions), dtype=np.float32)
        edge_valid_mask = np.zeros((batch_size, self.max_nodes, self.max_actions), dtype=bool)
        
        tree_sizes = np.ones(batch_size, dtype=np.int32)  # Start with root node
        
        # Initialize root nodes
        for b in range(batch_size):
            if b < len(board_states):
                node_boards[b, 0] = board_states[b].edge_states
            
        return BatchedMCTSState(
            node_boards=node_boards,
            node_visits=node_visits,
            node_is_expanded=node_is_expanded,
            node_parent=node_parent,
            edge_visits=edge_visits,
            edge_total_value=edge_total_value,
            edge_priors=edge_priors,
            edge_valid_mask=edge_valid_mask,
            batch_size=batch_size,
            max_nodes=self.max_nodes,
            max_actions=self.max_actions,
            num_vertices=self.num_vertices,
            k=self.k,
            tree_sizes=tree_sizes
        )
    
    def batch_ucb_scores(self, state: BatchedMCTSState, node_indices: np.ndarray) -> np.ndarray:
        """
        Calculate UCB scores for all children of given nodes in batch.
        
        Args:
            state: Current MCTS state
            node_indices: Shape (batch_size,) - node index for each tree
            
        Returns:
            ucb_scores: Shape (batch_size, max_actions)
        """
        batch_indices = np.arange(state.batch_size)
        
        # Get statistics for specified nodes
        node_visit_counts = state.node_visits[batch_indices, node_indices]  # (batch_size,)
        edge_visit_counts = state.edge_visits[batch_indices, node_indices]  # (batch_size, max_actions)
        edge_values = state.edge_total_value[batch_indices, node_indices]  # (batch_size, max_actions)
        edge_priors = state.edge_priors[batch_indices, node_indices]  # (batch_size, max_actions)
        valid_mask = state.edge_valid_mask[batch_indices, node_indices]  # (batch_size, max_actions)
        
        # Calculate Q values
        q_values = np.where(
            edge_visit_counts > 0,
            edge_values / edge_visit_counts,
            0.0
        )
        
        # Calculate exploration bonus
        sqrt_parent_visits = np.sqrt(np.maximum(1.0, node_visit_counts))[:, None]
        u_values = self.c_puct * edge_priors * sqrt_parent_visits / (1.0 + edge_visit_counts)
        
        # Combine Q + U
        ucb_scores = q_values + u_values
        
        # Mask invalid actions with -inf
        ucb_scores = np.where(valid_mask, ucb_scores, -np.inf)
        
        return ucb_scores
    
    def batch_select_actions(self, state: BatchedMCTSState, node_indices: np.ndarray) -> np.ndarray:
        """
        Select best actions for given nodes based on UCB.
        
        Args:
            state: Current MCTS state  
            node_indices: Shape (batch_size,) - node index for each tree
            
        Returns:
            actions: Shape (batch_size,) - best action for each node
        """
        ucb_scores = self.batch_ucb_scores(state, node_indices)
        return np.argmax(ucb_scores, axis=1)
    
    def batch_traverse_to_leaf(self, state: BatchedMCTSState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Traverse all trees from root to leaf nodes in parallel.
        
        Returns:
            leaf_indices: Shape (batch_size,) - leaf node index for each tree
            paths: Shape (batch_size, max_depth, 2) - (node_idx, action) pairs
        """
        batch_size = state.batch_size
        current_nodes = np.zeros(batch_size, dtype=np.int32)  # Start at root
        
        max_depth = 50  # Maximum depth to traverse
        paths = np.full((batch_size, max_depth, 2), -1, dtype=np.int32)
        
        for depth in range(max_depth):
            # Check which nodes are expanded
            batch_indices = np.arange(batch_size)
            is_expanded = state.node_is_expanded[batch_indices, current_nodes]
            
            # For expanded nodes, select best action
            actions = self.batch_select_actions(state, current_nodes)
            
            # Store path
            paths[:, depth, 0] = current_nodes
            paths[:, depth, 1] = actions
            
            # Move to child nodes (only for expanded nodes)
            # In actual implementation, we need to track child node indices
            # For now, this is simplified
            
            # Break if all nodes are leaves
            if not np.any(is_expanded):
                break
                
        return current_nodes, paths
    
    def batch_expand_nodes(self, state: BatchedMCTSState, node_indices: np.ndarray, 
                          priors: np.ndarray) -> BatchedMCTSState:
        """
        Expand multiple nodes in parallel.
        
        Args:
            state: Current MCTS state
            node_indices: Shape (batch_size,) - nodes to expand
            priors: Shape (batch_size, max_actions) - prior probabilities from network
            
        Returns:
            Updated state
        """
        batch_indices = np.arange(state.batch_size)
        
        # Mark nodes as expanded
        new_is_expanded = state.node_is_expanded.copy()
        new_is_expanded[batch_indices, node_indices] = True
        
        # Set priors
        new_edge_priors = state.edge_priors.copy()
        new_edge_priors[batch_indices, node_indices] = priors
        
        # Calculate valid moves for each node
        new_valid_mask = state.edge_valid_mask.copy()
        
        # For each tree in batch, determine valid moves
        for b in range(state.batch_size):
            node_idx = node_indices[b]
            board_state = state.node_boards[b, node_idx]
            
            # Create temporary board to get valid moves
            board = JAXCliqueBoard(self.num_vertices, self.k, self.game_mode)
            board.state = board.state._replace(edge_states=board_state)
            
            valid_moves = board.get_valid_moves()
            for move in valid_moves:
                action_idx = ed.encode_action(board, move)
                if 0 <= action_idx < self.max_actions:
                    new_valid_mask[b, node_idx, action_idx] = True
        
        # Return updated state
        return state._replace(
            node_is_expanded=new_is_expanded,
            edge_priors=new_edge_priors,
            edge_valid_mask=new_valid_mask
        )
    
    def batch_backup(self, state: BatchedMCTSState, paths: np.ndarray, 
                    values: np.ndarray) -> BatchedMCTSState:
        """
        Backup values through the trees in parallel.
        
        Args:
            state: Current MCTS state
            paths: Shape (batch_size, max_depth, 2) - paths from root to leaf
            values: Shape (batch_size,) - values to backup
            
        Returns:
            Updated state
        """
        new_node_visits = state.node_visits.copy()
        new_edge_visits = state.edge_visits.copy() 
        new_edge_values = state.edge_total_value.copy()
        
        batch_size = state.batch_size
        
        # Process each tree
        for b in range(batch_size):
            value = values[b]
            
            # Walk backwards through path
            for depth in range(paths.shape[1]):
                node_idx = paths[b, depth, 0]
                action = paths[b, depth, 1]
                
                if node_idx < 0:  # End of path
                    break
                
                # Update node visit count
                new_node_visits[b, node_idx] += 1
                
                # Update edge statistics
                if action >= 0:
                    new_edge_visits[b, node_idx, action] += 1
                    new_edge_values[b, node_idx, action] += value
                    
                # Flip value for opponent
                value = -value
        
        return state._replace(
            node_visits=new_node_visits,
            edge_visits=new_edge_visits,
            edge_total_value=new_edge_values
        )
    
    def batch_get_policy(self, state: BatchedMCTSState, temperature: float = 1.0) -> np.ndarray:
        """
        Get policy vectors for all trees based on visit counts.
        
        Args:
            state: Current MCTS state
            temperature: Temperature for controlling exploration
            
        Returns:
            policies: Shape (batch_size, max_actions)
        """
        # Get root node visit counts
        root_edge_visits = state.edge_visits[:, 0, :]  # (batch_size, max_actions)
        root_valid_mask = state.edge_valid_mask[:, 0, :]
        
        # Apply temperature
        if temperature == 0:
            # Deterministic: choose most visited
            policies = np.zeros_like(root_edge_visits)
            best_actions = np.argmax(root_edge_visits, axis=1)
            batch_indices = np.arange(state.batch_size)
            policies[batch_indices, best_actions] = 1.0
        else:
            # Apply temperature and normalize
            scaled_visits = np.power(root_edge_visits, 1.0 / temperature)
            scaled_visits = np.where(root_valid_mask, scaled_visits, 0.0)
            
            # Normalize
            visit_sums = np.sum(scaled_visits, axis=1, keepdims=True)
            policies = np.where(
                visit_sums > 0,
                scaled_visits / visit_sums,
                np.where(root_valid_mask, 1.0 / np.sum(root_valid_mask, axis=1, keepdims=True), 0.0)
            )
        
        return policies
    
    def run_simulations(self, boards: List[JAXCliqueBoard], model_params: Dict, 
                       num_simulations: int, add_noise: bool = True) -> Tuple[np.ndarray, BatchedMCTSState]:
        """
        Run MCTS simulations for a batch of board positions.
        
        Args:
            boards: List of board positions to analyze
            model_params: Neural network parameters
            num_simulations: Number of simulations per position
            add_noise: Whether to add Dirichlet noise at root
            
        Returns:
            policies: Shape (batch_size, max_actions) - MCTS policies
            final_state: Final MCTS tree state
        """
        batch_size = len(boards)
        
        # Initialize batch state
        board_states = [board.state for board in boards]
        state = self.initialize_batch(board_states, batch_size)
        
        # Create neural network model
        model = CliqueGNN(self.num_vertices)
        
        # Run simulations
        for sim in range(num_simulations):
            # 1. Traverse to leaf nodes
            leaf_indices, paths = self.batch_traverse_to_leaf(state)
            
            # 2. Evaluate leaf nodes with neural network
            # Prepare batch of board states for network
            leaf_boards = []
            for b in range(batch_size):
                board_state = state.node_boards[b, leaf_indices[b]]
                board = JAXCliqueBoard(self.num_vertices, self.k, self.game_mode)
                board.state = board.state._replace(edge_states=board_state)
                leaf_boards.append(board)
            
            # Get network predictions
            policies, values = self.batch_evaluate_positions(leaf_boards, model, model_params)
            
            # 3. Expand leaf nodes
            state = self.batch_expand_nodes(state, leaf_indices, policies)
            
            # 4. Backup values
            state = self.batch_backup(state, paths, values)
        
        # Extract final policies
        final_policies = self.batch_get_policy(state, temperature=1.0)
        
        return final_policies, state
    
    def batch_evaluate_positions(self, boards: List[JAXCliqueBoard], 
                                model: CliqueGNN, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate multiple board positions with neural network.
        
        Args:
            boards: List of boards to evaluate
            model: Neural network model
            params: Model parameters
            
        Returns:
            policies: Shape (batch_size, max_actions)
            values: Shape (batch_size,)
        """
        batch_size = len(boards)
        all_policies = np.zeros((batch_size, self.max_actions))
        all_values = np.zeros(batch_size)
        
        # Process each board (in real JAX this would be vmapped)
        for i, board in enumerate(boards):
            # Prepare input
            state_dict = ed.prepare_state_for_network(board)
            edge_index = state_dict['edge_index'].numpy()
            edge_attr = state_dict['edge_attr'].numpy()
            
            # Network forward pass
            policy, value = model(params, edge_index, edge_attr)
            
            # Extract and store results
            all_policies[i] = policy.flatten()[:self.max_actions]
            all_values[i] = float(value.flatten()[0])
        
        return all_policies, all_values


# Simplified single-tree MCTS for compatibility testing
class SimpleMCTS:
    """Simple MCTS implementation matching original interface"""
    
    def __init__(self, game: JAXCliqueBoard, num_simulations: int, 
                 net_model: CliqueGNN, net_params: Dict, noise_weight: float = 0.25):
        self.game = game
        self.num_simulations = num_simulations
        self.net_model = net_model
        self.net_params = net_params
        self.noise_weight = noise_weight
        
        # Tree statistics
        self.visit_counts = {}  # State hash -> visit count
        self.value_sums = {}    # State hash -> value sum
        self.priors = {}        # State hash -> prior probabilities
        self.children = {}      # State hash -> list of (action, next_state_hash)
        
    def get_state_hash(self, board: JAXCliqueBoard) -> str:
        """Get hash of board state for tree lookup"""
        return str(board.state.edge_states.tobytes())
    
    def search(self) -> Tuple[int, Dict]:
        """Run MCTS search and return best action"""
        root_hash = self.get_state_hash(self.game)
        
        # Run simulations
        for _ in range(self.num_simulations):
            board_copy = self.game.copy()
            self._simulate(board_copy)
        
        # Get action probabilities
        action_probs = self._get_action_probs(root_hash)
        
        # Select best action
        best_action = int(np.argmax(action_probs))
        
        # Return action and tree stats
        stats = {
            'visits': self.visit_counts.get(root_hash, 0),
            'policy': action_probs
        }
        
        return best_action, stats
    
    def _simulate(self, board: JAXCliqueBoard) -> float:
        """Run one simulation from current position"""
        path = []
        
        # Selection phase - traverse tree
        while True:
            state_hash = self.get_state_hash(board)
            
            if state_hash not in self.children:
                # Leaf node - expand and evaluate
                value = self._expand_and_evaluate(board, state_hash)
                break
            
            # Select action using UCB
            action = self._select_action(board, state_hash)
            path.append((state_hash, action))
            
            # Make move
            edge = ed.decode_action(board, action)
            board.make_move(edge)
            
            # Check terminal
            if board.game_state != 0:
                value = self._get_terminal_value(board)
                break
        
        # Backup phase
        for state_hash, action in reversed(path):
            self.visit_counts[state_hash] = self.visit_counts.get(state_hash, 0) + 1
            self.value_sums[state_hash] = self.value_sums.get(state_hash, 0) + value
            value = -value  # Flip for opponent
        
        return value
    
    def _expand_and_evaluate(self, board: JAXCliqueBoard, state_hash: str) -> float:
        """Expand node and evaluate with neural network"""
        # Get network prediction
        state_dict = ed.prepare_state_for_network(board)
        edge_index = state_dict['edge_index'].numpy()
        edge_attr = state_dict['edge_attr'].numpy()
        
        policy, value = self.net_model(self.net_params, edge_index, edge_attr)
        
        # Store prior probabilities
        self.priors[state_hash] = policy.flatten()
        
        # Initialize node
        self.visit_counts[state_hash] = 1
        self.value_sums[state_hash] = float(value.flatten()[0])
        self.children[state_hash] = []
        
        # Add valid moves as children
        valid_moves = board.get_valid_moves()
        for move in valid_moves:
            action = ed.encode_action(board, move)
            self.children[state_hash].append(action)
        
        return float(value.flatten()[0])
    
    def _select_action(self, board: JAXCliqueBoard, state_hash: str) -> int:
        """Select action using UCB formula"""
        valid_actions = self.children[state_hash]
        
        if not valid_actions:
            return 0
        
        # Calculate UCB scores
        parent_visits = self.visit_counts[state_hash]
        sqrt_parent = np.sqrt(parent_visits)
        
        best_score = -np.inf
        best_action = valid_actions[0]
        
        for action in valid_actions:
            # Make move to get child state
            board_copy = board.copy()
            edge = ed.decode_action(board_copy, action)
            board_copy.make_move(edge)
            child_hash = self.get_state_hash(board_copy)
            
            # Calculate UCB
            prior = self.priors[state_hash][action]
            if child_hash in self.visit_counts:
                child_visits = self.visit_counts[child_hash]
                child_value = self.value_sums[child_hash] / child_visits
                ucb = child_value + prior * sqrt_parent / (1 + child_visits)
            else:
                ucb = prior * sqrt_parent
            
            if ucb > best_score:
                best_score = ucb
                best_action = action
        
        return best_action
    
    def _get_action_probs(self, state_hash: str) -> np.ndarray:
        """Get action probabilities based on visit counts"""
        probs = np.zeros(self.game.num_vertices * (self.game.num_vertices - 1) // 2)
        
        if state_hash not in self.children:
            return probs
        
        valid_actions = self.children[state_hash]
        total_visits = 0
        
        # Count visits for each action
        for action in valid_actions:
            # Make move to get child state
            board_copy = self.game.copy()
            edge = ed.decode_action(board_copy, action)
            board_copy.make_move(edge)
            child_hash = self.get_state_hash(board_copy)
            
            if child_hash in self.visit_counts:
                visits = self.visit_counts[child_hash]
                probs[action] = visits
                total_visits += visits
        
        # Normalize
        if total_visits > 0:
            probs /= total_visits
        else:
            # Uniform over valid actions
            for action in valid_actions:
                probs[action] = 1.0 / len(valid_actions)
        
        return probs
    
    def _get_terminal_value(self, board: JAXCliqueBoard) -> float:
        """Get value for terminal position"""
        if board.game_state == 1:  # Player 1 wins
            return 1.0 if board.player == 1 else -1.0
        elif board.game_state == 2:  # Player 2 wins
            return -1.0 if board.player == 1 else 1.0
        else:  # Draw
            return 0.0


# Compatibility function matching original interface
def UCT_search(game_state: JAXCliqueBoard, num_simulations: int, 
               net: CliqueGNN, noise_weight: float = 0.25) -> Tuple[int, SimpleMCTS]:
    """
    Run UCT search matching original interface.
    
    Returns:
        best_move: Index of best move
        mcts: MCTS object with search tree
    """
    # Initialize neural network parameters if needed
    if isinstance(net, CliqueGNN):
        # Assume net has parameters attached or create new ones
        import numpy as np
        rng = np.random.RandomState(42)
        net_params = net.init_params(rng)
    else:
        # PyTorch model - won't be used in JAX version
        raise ValueError("JAX MCTS requires JAX neural network")
    
    # Create MCTS instance
    mcts = SimpleMCTS(game_state, num_simulations, net, net_params, noise_weight)
    
    # Run search
    best_move, _ = mcts.search()
    
    return best_move, mcts
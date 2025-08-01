"""
Proper Tree-based MCTS Implementation for JAX AlphaZero.

This implementation actually builds and searches a tree, unlike the previous 
vectorized_mcts_jit.py which only reweighted the initial policy.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import copy

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    board_state: Any  # Board state at this node
    move: Optional[int] = None  # Move that led to this node
    parent: Optional['MCTSNode'] = None
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)
    
    # Statistics
    visit_count: float = 0.0
    value_sum: float = 0.0
    prior: float = 0.0
    
    # Neural network outputs stored at expansion
    child_priors: Optional[np.ndarray] = None
    is_expanded: bool = False
    is_terminal: bool = False
    terminal_value: Optional[float] = None
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (not expanded)."""
        return not self.is_expanded
    
    def q_value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visits: float, c_puct: float = 3.0) -> float:
        """Calculate UCB score for selection."""
        if self.visit_count == 0:
            q = 0.0
        else:
            q = self.value_sum / self.visit_count
        
        # Exploration term
        u = c_puct * self.prior * np.sqrt(parent_visits) / (1 + self.visit_count)
        
        return q + u


class TreeBasedMCTS:
    """Proper tree-based MCTS implementation that actually searches."""
    
    def __init__(self, 
                 num_actions: int = 15,
                 c_puct: float = 3.0,
                 noise_weight: float = 0.25,
                 perspective_mode: str = "alternating"):
        self.num_actions = num_actions
        self.c_puct = c_puct
        self.noise_weight = noise_weight
        self.perspective_mode = perspective_mode
        
    def search(self, 
               root_board: VectorizedCliqueBoard,
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               game_idx: int = 0,
               temperature: float = 1.0) -> np.ndarray:
        """
        Run MCTS search from the given position.
        
        Args:
            root_board: Current board state (vectorized, but we use game_idx)
            neural_network: Neural network for evaluation
            num_simulations: Number of simulations to run
            game_idx: Which game in the batch to search
            temperature: Temperature for final move selection
            
        Returns:
            Action probabilities based on visit counts
        """
        # Create a single-game board for this search
        single_board = self._extract_single_board(root_board, game_idx)
        
        # Create root node
        root = MCTSNode(board_state=single_board)
        
        # Add noise to root if configured
        if self.noise_weight > 0:
            noise = np.random.dirichlet([0.3] * self.num_actions)
        else:
            noise = None
        
        # Run simulations
        for sim in range(num_simulations):
            node = root
            path = [node]
            
            # 1. Selection - traverse tree to leaf using UCB
            while not node.is_leaf():
                # Select best child according to UCB
                best_action = None
                best_score = -float('inf')
                
                for action, child in node.children.items():
                    score = child.ucb_score(node.visit_count, self.c_puct)
                    if score > best_score:
                        best_score = score
                        best_action = action
                
                if best_action is None:
                    break
                    
                node = node.children[best_action]
                path.append(node)
            
            # 2. Expansion - if not terminal, expand the leaf
            if not node.is_terminal and node.visit_count > 0:
                self._expand_node(node, neural_network, noise if node is root else None)
                
                # Select a child to evaluate
                if node.children:
                    # Select child with highest prior
                    best_action = max(node.children.keys(), 
                                    key=lambda a: node.children[a].prior)
                    node = node.children[best_action]
                    path.append(node)
            
            # 3. Evaluation - get value from neural network or terminal state
            if node.is_terminal:
                value = node.terminal_value
            else:
                value = self._evaluate_node(node, neural_network)
            
            # 4. Backup - propagate value up the tree
            self._backup(path, value)
        
        # Convert visit counts to action probabilities
        visits = np.zeros(self.num_actions)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        
        # Apply temperature
        if temperature == 0:
            # Deterministic - pick best
            probs = np.zeros(self.num_actions)
            if visits.sum() > 0:
                probs[np.argmax(visits)] = 1.0
        else:
            # Apply temperature
            visits_temp = np.power(visits + 1e-8, 1.0 / temperature)
            probs = visits_temp / visits_temp.sum()
        
        # Ensure probabilities sum to 1.0
        probs = probs / probs.sum()
        
        return probs
    
    def _extract_single_board(self, batch_board: VectorizedCliqueBoard, idx: int) -> VectorizedCliqueBoard:
        """Extract a single board from a batch."""
        # Create a new single-game board
        single = VectorizedCliqueBoard(1, batch_board.num_vertices, batch_board.k, batch_board.game_mode)
        
        # Copy the state
        single.adjacency_matrices = batch_board.adjacency_matrices[idx:idx+1]
        single.edge_states = batch_board.edge_states[idx:idx+1]
        single.current_players = batch_board.current_players[idx:idx+1]
        single.move_counts = batch_board.move_counts[idx:idx+1]
        single.game_states = batch_board.game_states[idx:idx+1]
        single.winners = batch_board.winners[idx:idx+1]
        
        return single
    
    def _expand_node(self, node: MCTSNode, neural_network: ImprovedBatchedNeuralNetwork, 
                     noise: Optional[np.ndarray] = None):
        """Expand a leaf node by evaluating it and creating children."""
        if node.is_expanded:
            return
        
        # Get board features
        board = node.board_state
        edge_indices, edge_features = board.get_features_for_nn_undirected()
        valid_mask = board.get_valid_moves_mask()
        
        # Get neural network evaluation
        player_roles = board.current_players
        policies, values = neural_network.evaluate_batch(
            edge_indices, edge_features, valid_mask, player_roles
        )
        
        policy = policies[0]  # Single game
        value = values[0, 0]
        
        # Apply perspective adjustment if needed
        if self.perspective_mode == "fixed":
            # Fixed perspective: value is from Player 1's view
            if board.current_players[0] == 1:  # Player 2's turn
                value = -value
        
        # Add noise to policy if this is root
        if noise is not None and self.noise_weight > 0:
            policy = (1 - self.noise_weight) * policy + self.noise_weight * noise
            # Renormalize
            valid_sum = (policy * valid_mask[0]).sum()
            if valid_sum > 0:
                policy = policy * valid_mask[0] / valid_sum
        
        # Store child priors
        node.child_priors = policy
        
        # Create children for valid moves
        valid_actions = np.where(valid_mask[0])[0]
        for action in valid_actions:
            # Create child board state
            child_board = self._copy_and_move(board, int(action))
            
            # Check if terminal
            is_terminal = child_board.game_states[0] != 0
            terminal_value = None
            if is_terminal:
                terminal_value = self._get_terminal_value(child_board)
            
            # Create child node
            child = MCTSNode(
                board_state=child_board,
                move=int(action),
                parent=node,
                prior=float(policy[action]),
                is_terminal=is_terminal,
                terminal_value=terminal_value
            )
            
            node.children[int(action)] = child
        
        node.is_expanded = True
    
    def _evaluate_node(self, node: MCTSNode, neural_network: ImprovedBatchedNeuralNetwork) -> float:
        """Evaluate a leaf node using the neural network."""
        if node.is_terminal:
            return node.terminal_value
        
        # Get board features
        board = node.board_state
        edge_indices, edge_features = board.get_features_for_nn_undirected()
        valid_mask = board.get_valid_moves_mask()
        player_roles = board.current_players
        
        # Get neural network evaluation
        _, values = neural_network.evaluate_batch(
            edge_indices, edge_features, valid_mask, player_roles
        )
        
        value = float(values[0, 0])
        
        # Apply perspective adjustment if needed
        if self.perspective_mode == "fixed":
            # Fixed perspective: value is from Player 1's view
            if board.current_players[0] == 1:  # Player 2's turn
                value = -value
        
        return value
    
    def _backup(self, path: List[MCTSNode], value: float):
        """Backup value through the path."""
        current_value = value
        
        # Walk backwards through path
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += current_value
            
            # Flip value for alternating perspective
            if self.perspective_mode == "alternating":
                current_value = -current_value
            # For fixed perspective, value stays the same
    
    def _copy_and_move(self, board: VectorizedCliqueBoard, action: int) -> VectorizedCliqueBoard:
        """Copy board and make a move."""
        # Create a copy
        new_board = VectorizedCliqueBoard(1, board.num_vertices, board.k, board.game_mode)
        
        # Copy state
        new_board.adjacency_matrices = board.adjacency_matrices.copy()
        new_board.edge_states = board.edge_states.copy()
        new_board.current_players = board.current_players.copy()
        new_board.move_counts = board.move_counts.copy()
        new_board.game_states = board.game_states.copy()
        new_board.winners = board.winners.copy()
        
        # Make move
        new_board.make_moves(jnp.array([action]))
        
        return new_board
    
    def _get_terminal_value(self, board: VectorizedCliqueBoard) -> float:
        """Get terminal value based on perspective mode."""
        game_state = int(board.game_states[0])
        current_player = int(board.current_players[0])
        
        if self.perspective_mode == "fixed":
            # Fixed perspective: always from Player 1's view
            if game_state == 1:  # Player 1 wins
                return 1.0
            elif game_state == 2:  # Player 2 wins
                return -1.0
            else:  # Draw
                return 0.0
        else:  # alternating
            # From current player's perspective
            if game_state == 1:  # Player 1 wins
                return 1.0 if current_player == 0 else -1.0
            elif game_state == 2:  # Player 2 wins
                return -1.0 if current_player == 0 else 1.0
            else:  # Draw
                return 0.0


class ParallelTreeBasedMCTS:
    """Wrapper to run tree-based MCTS in parallel for multiple games."""
    
    def __init__(self, batch_size: int, **mcts_kwargs):
        self.batch_size = batch_size
        self.mcts_kwargs = mcts_kwargs
    
    def search(self, boards: VectorizedCliqueBoard, 
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: jnp.ndarray,
               temperature: float = 1.0) -> jnp.ndarray:
        """
        Run MCTS for multiple games in parallel.
        
        This is not JIT-compiled but runs independent tree searches.
        """
        action_probs = np.zeros((self.batch_size, 15))
        
        # Run independent MCTS for each game
        for game_idx in range(self.batch_size):
            if boards.game_states[game_idx] == 0:  # Game still active
                mcts = TreeBasedMCTS(**self.mcts_kwargs)
                
                # Run search for this game
                probs = mcts.search(
                    boards, 
                    neural_network,
                    int(num_simulations[game_idx]),
                    game_idx,
                    temperature
                )
                
                action_probs[game_idx] = probs
        
        return jnp.array(action_probs)
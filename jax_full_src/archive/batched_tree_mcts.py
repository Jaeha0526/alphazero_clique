"""
Batched Tree MCTS - A faster approach that evaluates multiple positions at once.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import jax.numpy as jnp

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from tree_based_mcts import MCTSNode


class BatchedTreeMCTS:
    """
    Tree-based MCTS that batches neural network evaluations.
    
    Instead of evaluating positions one at a time, collect multiple
    positions and evaluate them together.
    """
    
    def __init__(self, 
                 num_actions: int = 15,
                 c_puct: float = 3.0,
                 batch_size: int = 8,
                 noise_weight: float = 0.25,
                 perspective_mode: str = "alternating"):
        self.num_actions = num_actions
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.noise_weight = noise_weight
        self.perspective_mode = perspective_mode
    
    def search(self, 
               root_board: VectorizedCliqueBoard,
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               game_idx: int = 0,
               temperature: float = 1.0) -> np.ndarray:
        """
        Run MCTS with batched evaluation.
        """
        # Extract single board
        single_board = self._extract_single_board(root_board, game_idx)
        
        # Create root
        root = MCTSNode(board_state=single_board)
        
        # Add noise if needed
        if self.noise_weight > 0:
            noise = np.random.dirichlet([0.3] * self.num_actions)
        else:
            noise = None
        
        # Run simulations in batches
        num_batches = (num_simulations + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, num_simulations)
            current_batch_size = batch_end - batch_start
            
            # Collect leaf nodes for this batch
            paths = []
            leaf_nodes = []
            
            for _ in range(current_batch_size):
                node = root
                path = [node]
                
                # Selection - traverse to leaf
                while not node.is_leaf():
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
                
                paths.append(path)
                leaf_nodes.append(node)
            
            # Batch evaluate all leaf nodes
            positions_to_eval = []
            eval_indices = []
            
            for i, leaf in enumerate(leaf_nodes):
                if not leaf.is_terminal and leaf.visit_count > 0 and not leaf.is_expanded:
                    positions_to_eval.append(leaf.board_state)
                    eval_indices.append(i)
            
            if positions_to_eval:
                # Create batched board
                batch_boards = self._create_batch_boards(positions_to_eval)
                
                # Get features
                edge_indices, edge_features = batch_boards.get_features_for_nn_undirected()
                valid_mask = batch_boards.get_valid_moves_mask()
                player_roles = batch_boards.current_players
                
                # Batch evaluate
                policies, values = neural_network.evaluate_batch(
                    edge_indices, edge_features, valid_mask, player_roles
                )
                
                # Expand nodes with results
                for idx, eval_idx in enumerate(eval_indices):
                    leaf = leaf_nodes[eval_idx]
                    policy = policies[idx]
                    value = float(values[idx, 0])
                    
                    # Apply perspective adjustment
                    if self.perspective_mode == "fixed":
                        if leaf.board_state.current_players[0] == 1:
                            value = -value
                    
                    # Add noise to root
                    if leaf is root and noise is not None and self.noise_weight > 0:
                        policy = (1 - self.noise_weight) * policy + self.noise_weight * noise
                        valid_sum = (policy * valid_mask[idx]).sum()
                        if valid_sum > 0:
                            policy = policy * valid_mask[idx] / valid_sum
                    
                    # Expand leaf
                    self._expand_node_with_policy(leaf, policy)
                    
                    # If we expanded, select a child for backup
                    if leaf.children:
                        best_child_action = max(leaf.children.keys(), 
                                              key=lambda a: leaf.children[a].prior)
                        child = leaf.children[best_child_action]
                        paths[eval_idx].append(child)
                    
                    # Backup value
                    self._backup(paths[eval_idx], value)
            
            # Handle terminal nodes and already expanded nodes
            for i, leaf in enumerate(leaf_nodes):
                if i not in eval_indices:
                    if leaf.is_terminal:
                        self._backup(paths[i], leaf.terminal_value)
                    elif leaf.visit_count == 0:
                        # First visit, just evaluate
                        value = self._evaluate_single_node(leaf, neural_network)
                        self._backup(paths[i], value)
        
        # Convert visit counts to probabilities
        visits = np.zeros(self.num_actions)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        
        # Apply temperature
        if temperature == 0:
            probs = np.zeros(self.num_actions)
            if visits.sum() > 0:
                probs[np.argmax(visits)] = 1.0
        else:
            visits_temp = np.power(visits + 1e-8, 1.0 / temperature)
            probs = visits_temp / visits_temp.sum()
        
        return probs / probs.sum()
    
    def _extract_single_board(self, batch_board: VectorizedCliqueBoard, idx: int) -> VectorizedCliqueBoard:
        """Extract a single board from batch."""
        single = VectorizedCliqueBoard(1, batch_board.num_vertices, batch_board.k, batch_board.game_mode)
        single.adjacency_matrices = batch_board.adjacency_matrices[idx:idx+1]
        single.edge_states = batch_board.edge_states[idx:idx+1]
        single.current_players = batch_board.current_players[idx:idx+1]
        single.move_counts = batch_board.move_counts[idx:idx+1]
        single.game_states = batch_board.game_states[idx:idx+1]
        single.winners = batch_board.winners[idx:idx+1]
        return single
    
    def _create_batch_boards(self, boards: List[VectorizedCliqueBoard]) -> VectorizedCliqueBoard:
        """Combine multiple single boards into a batch."""
        batch_size = len(boards)
        if batch_size == 0:
            return None
            
        first = boards[0]
        batch = VectorizedCliqueBoard(batch_size, first.num_vertices, first.k, first.game_mode)
        
        # Stack all states
        batch.adjacency_matrices = jnp.concatenate([b.adjacency_matrices for b in boards], axis=0)
        batch.edge_states = jnp.concatenate([b.edge_states for b in boards], axis=0)
        batch.current_players = jnp.concatenate([b.current_players for b in boards], axis=0)
        batch.move_counts = jnp.concatenate([b.move_counts for b in boards], axis=0)
        batch.game_states = jnp.concatenate([b.game_states for b in boards], axis=0)
        batch.winners = jnp.concatenate([b.winners for b in boards], axis=0)
        
        return batch
    
    def _expand_node_with_policy(self, node: MCTSNode, policy: np.ndarray):
        """Expand node with given policy."""
        if node.is_expanded:
            return
            
        node.child_priors = policy
        board = node.board_state
        valid_mask = board.get_valid_moves_mask()[0]
        valid_actions = np.where(valid_mask)[0]
        
        for action in valid_actions:
            # Create child board
            child_board = self._copy_and_move(board, int(action))
            
            # Check terminal
            is_terminal = child_board.game_states[0] != 0
            terminal_value = None
            if is_terminal:
                terminal_value = self._get_terminal_value(child_board)
            
            # Create child
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
    
    def _evaluate_single_node(self, node: MCTSNode, neural_network: ImprovedBatchedNeuralNetwork) -> float:
        """Evaluate a single node (fallback for non-batched case)."""
        board = node.board_state
        edge_indices, edge_features = board.get_features_for_nn_undirected()
        valid_mask = board.get_valid_moves_mask()
        player_roles = board.current_players
        
        _, values = neural_network.evaluate_batch(
            edge_indices, edge_features, valid_mask, player_roles
        )
        
        value = float(values[0, 0])
        
        if self.perspective_mode == "fixed":
            if board.current_players[0] == 1:
                value = -value
                
        return value
    
    def _backup(self, path: List[MCTSNode], value: float):
        """Backup value through path."""
        current_value = value
        
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += current_value
            
            if self.perspective_mode == "alternating":
                current_value = -current_value
    
    def _copy_and_move(self, board: VectorizedCliqueBoard, action: int) -> VectorizedCliqueBoard:
        """Copy board and make move."""
        new_board = VectorizedCliqueBoard(1, board.num_vertices, board.k, board.game_mode)
        new_board.adjacency_matrices = board.adjacency_matrices.copy()
        new_board.edge_states = board.edge_states.copy()
        new_board.current_players = board.current_players.copy()
        new_board.move_counts = board.move_counts.copy()
        new_board.game_states = board.game_states.copy()
        new_board.winners = board.winners.copy()
        new_board.make_moves(jnp.array([action]))
        return new_board
    
    def _get_terminal_value(self, board: VectorizedCliqueBoard) -> float:
        """Get terminal value."""
        game_state = int(board.game_states[0])
        current_player = int(board.current_players[0])
        
        if self.perspective_mode == "fixed":
            if game_state == 1:
                return 1.0
            elif game_state == 2:
                return -1.0
            else:
                return 0.0
        else:
            if game_state == 1:
                return 1.0 if current_player == 0 else -1.0
            elif game_state == 2:
                return -1.0 if current_player == 0 else 1.0
            else:
                return 0.0
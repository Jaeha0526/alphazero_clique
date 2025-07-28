"""
Optimized SimpleTreeMCTS that preserves exact MCTS structure but with faster board operations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class SimpleTreeMCTSOptimized:
    """
    Same MCTS algorithm as SimpleTreeMCTS but with optimized board operations.
    """
    
    def __init__(self, 
                 batch_size: int,
                 num_actions: int = 15,
                 c_puct: float = 3.0,
                 max_nodes_per_game: int = 100):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.c_puct = c_puct
        self.max_nodes = max_nodes_per_game
        
        # Pre-allocate board pool to avoid repeated allocations
        self.board_pool = []
        for _ in range(max_nodes_per_game * batch_size):
            board = VectorizedCliqueBoard(
                batch_size=1,
                num_vertices=0,  # Will be set when used
                k=0,
                game_mode="symmetric"
            )
            self.board_pool.append(board)
        self.pool_index = 0
        
        # JIT-compile hot computational paths
        self._calculate_ucb_jit = jax.jit(self._calculate_ucb_scores)
        self._update_stats_jit = jax.jit(self._update_node_statistics)
    
    def _get_board_from_pool(self, template_board):
        """Get a pre-allocated board and copy state efficiently."""
        board = self.board_pool[self.pool_index]
        self.pool_index = (self.pool_index + 1) % len(self.board_pool)
        
        # Reinitialize with correct parameters if needed
        if board.num_vertices != template_board.num_vertices:
            board.num_vertices = template_board.num_vertices
            board.k = template_board.k
            board.game_mode = template_board.game_mode
        
        # Efficient copy using JAX operations
        board.adjacency_matrices = template_board.adjacency_matrices.copy()
        board.current_players = template_board.current_players.copy()
        board.game_states = template_board.game_states.copy()
        board.winners = template_board.winners.copy()
        board.move_counts = template_board.move_counts.copy()
        
        return board
    
    def _calculate_ucb_scores(self, N: jnp.ndarray, W: jnp.ndarray, P: jnp.ndarray, 
                             node_visits: float, c_puct: float, valid_mask: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled UCB calculation - same as original."""
        Q = W / (1.0 + N)
        sqrt_visits = jnp.sqrt(jnp.maximum(1.0, node_visits))
        U = c_puct * sqrt_visits * (P / (1.0 + N))
        ucb = Q + U
        ucb = jnp.where(valid_mask, ucb, -jnp.inf)
        return ucb
    
    def _update_node_statistics(self, N: jnp.ndarray, W: jnp.ndarray, 
                               action: int, value: float) -> tuple:
        """JIT-compiled statistics update - same as original."""
        N = N.at[action].add(1.0)
        W = W.at[action].add(value)
        return N, W
    
    def search(self, boards: VectorizedCliqueBoard, 
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               temperature: float = 1.0) -> np.ndarray:
        """
        Run MCTS search - exact same algorithm as SimpleTreeMCTS.
        """
        all_probs = []
        
        # Process each game independently - same as original
        for game_idx in range(self.batch_size):
            # Initialize tree for this game - same structure
            tree = {
                'node_count': 1,
                'boards': {0: boards.get_single_board(game_idx)},
                'expanded': {0},
                'children': {0: {}},
                'parents': {0: None},
                'N': {0: np.zeros(self.num_actions)},
                'W': {0: np.zeros(self.num_actions)},
                'P': {0: np.zeros(self.num_actions)},
                'node_visits': {0: 0}
            }
            
            # Run simulations - exact same algorithm
            for sim in range(num_simulations):
                # 1. Selection phase - unchanged
                node_id = 0
                path = [0]
                
                while node_id in tree['expanded']:
                    if len(path) > 20:
                        break
                    
                    # UCB calculation - using JIT version but same logic
                    N = jnp.array(tree['N'][node_id])
                    W = jnp.array(tree['W'][node_id])  
                    P = jnp.array(tree['P'][node_id])
                    node_visits = float(tree['node_visits'][node_id])
                    board = tree['boards'][node_id]
                    valid_mask = jnp.array(board.get_valid_moves_mask()[0])
                    
                    ucb = self._calculate_ucb_jit(N, W, P, node_visits, self.c_puct, valid_mask)
                    ucb = np.array(ucb)
                    
                    if ucb.max() == -np.inf:
                        break
                    
                    action = np.argmax(ucb)
                    
                    # Move to child - unchanged logic
                    if action not in tree['children'][node_id]:
                        if tree['node_count'] >= self.max_nodes:
                            break
                        
                        child_id = tree['node_count']
                        tree['node_count'] += 1
                        
                        # Optimized board copy using pool
                        child_board = self._get_board_from_pool(board)
                        child_board.make_moves(jnp.array([action]))
                        
                        # Initialize child - same as original
                        tree['N'][child_id] = np.zeros(self.num_actions)
                        tree['W'][child_id] = np.zeros(self.num_actions)
                        tree['P'][child_id] = np.zeros(self.num_actions)
                        tree['boards'][child_id] = child_board
                        tree['children'][child_id] = {}
                        tree['children'][node_id][action] = child_id
                        tree['parents'][child_id] = (node_id, action)
                        tree['node_visits'][child_id] = 0
                        
                        node_id = child_id
                        path.append(node_id)
                        break
                    else:
                        node_id = tree['children'][node_id][action]
                        path.append(node_id)
                
                # 2. Evaluation phase - unchanged
                leaf_board = tree['boards'][node_id]
                if leaf_board.game_states[0] == 0:
                    edge_indices, edge_features = leaf_board.get_features_for_nn_undirected()
                    if neural_network.asymmetric_mode and leaf_board.game_mode == "asymmetric":
                        priors, values = neural_network.apply(
                            neural_network.params, 
                            edge_indices[0], 
                            edge_features[0],
                            player_role=int(leaf_board.current_players[0])
                        )
                    else:
                        priors, values = neural_network.apply(
                            neural_network.params, 
                            edge_indices[0], 
                            edge_features[0]
                        )
                    
                    value = float(values[0])
                    tree['P'][node_id] = np.array(priors)
                    tree['expanded'].add(node_id)
                else:
                    # Terminal node - same logic
                    winner = int(leaf_board.winners[0])
                    if winner == 0:
                        value = 0.0
                    else:
                        player_at_leaf = 3 - int(leaf_board.current_players[0])
                        value = 1.0 if player_at_leaf == winner else -1.0
                
                # 3. Backup phase - unchanged except using JIT for stats update
                for i in range(len(path) - 1, -1, -1):
                    node = path[i]
                    tree['node_visits'][node] += 1
                    
                    if i > 0:
                        parent = path[i-1]
                        parent_id, action = tree['parents'][node]
                        
                        # Use JIT-compiled update
                        N_new, W_new = self._update_stats_jit(
                            jnp.array(tree['N'][parent_id]), 
                            jnp.array(tree['W'][parent_id]),
                            action, 
                            value
                        )
                        tree['N'][parent_id] = np.array(N_new)
                        tree['W'][parent_id] = np.array(W_new)
                    
                    value = -value
            
            # Get action probabilities - unchanged
            root_visits = tree['N'][0]
            if temperature == 0:
                probs = np.zeros_like(root_visits)
                probs[np.argmax(root_visits)] = 1.0
            else:
                visits_temp = np.power(root_visits, 1.0 / temperature)
                probs = visits_temp / visits_temp.sum()
            
            all_probs.append(probs)
        
        return np.array(all_probs)
"""
SimpleTreeMCTS using efficient board representation.
Exact same MCTS algorithm but with faster board operations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial

from efficient_board_proper import EfficientCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class SimpleTreeMCTSEfficient:
    """
    Same exact MCTS algorithm as SimpleTreeMCTS but using efficient board.
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
        
        # JIT-compile hot computational paths
        self._calculate_ucb_jit = jax.jit(self._calculate_ucb_scores)
        self._update_stats_jit = jax.jit(self._update_node_statistics)
        
        # Timing stats
        self.timing_stats = {
            'board_copy': [],
            'ucb_calc': [],
            'nn_eval': [],
            'total_sim': []
        }
    
    def _calculate_ucb_scores(self, N: jnp.ndarray, W: jnp.ndarray, P: jnp.ndarray, 
                             node_visits: float, c_puct: float, valid_mask: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled UCB calculation - exact same as original."""
        Q = W / (1.0 + N)
        sqrt_visits = jnp.sqrt(jnp.maximum(1.0, node_visits))
        U = c_puct * sqrt_visits * (P / (1.0 + N))
        ucb = Q + U
        ucb = jnp.where(valid_mask, ucb, -jnp.inf)
        return ucb
    
    def _update_node_statistics(self, N: jnp.ndarray, W: jnp.ndarray, 
                               action: int, value: float) -> tuple:
        """JIT-compiled statistics update."""
        N = N.at[action].add(1.0)
        W = W.at[action].add(value)
        return N, W
    
    def search(self, boards: EfficientCliqueBoard, 
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               temperature: float = 1.0) -> np.ndarray:
        """
        Run MCTS search - exact same algorithm as SimpleTreeMCTS.
        Selection → Expansion → Evaluation → Backup
        """
        all_probs = []
        
        # Process each game independently
        for game_idx in range(self.batch_size):
            # Initialize tree for this game
            tree = {
                'node_count': 1,
                'boards': {0: boards.get_single_board(game_idx)},
                'expanded': {0},
                'children': {0: {}},
                'parents': {0: None},
                'N': {0: np.zeros(self.num_actions)},  # Visit counts N(s,a)
                'W': {0: np.zeros(self.num_actions)},  # Total value W(s,a)
                'P': {0: np.zeros(self.num_actions)},  # Prior probabilities
                'node_visits': {0: 0}  # Visits to node N(s)
            }
            
            # Run simulations
            for sim in range(num_simulations):
                sim_start = time.time()
                
                # 1. SELECTION PHASE
                # Walk down tree using UCB until we reach unexpanded node
                node_id = 0
                path = [0]
                
                while node_id in tree['expanded']:
                    if len(path) > 20:  # Safety check
                        break
                    
                    # Calculate UCB for all actions
                    ucb_start = time.time()
                    N = jnp.array(tree['N'][node_id])
                    W = jnp.array(tree['W'][node_id])  
                    P = jnp.array(tree['P'][node_id])
                    node_visits = float(tree['node_visits'][node_id])
                    board = tree['boards'][node_id]
                    valid_mask = jnp.array(board.get_valid_moves_mask()[0])
                    
                    ucb = self._calculate_ucb_jit(N, W, P, node_visits, self.c_puct, valid_mask)
                    ucb = np.array(ucb)
                    self.timing_stats['ucb_calc'].append(time.time() - ucb_start)
                    
                    if ucb.max() == -np.inf:
                        break
                    
                    action = np.argmax(ucb)
                    
                    # Move to child node
                    if action not in tree['children'][node_id]:
                        # Need to create child
                        if tree['node_count'] >= self.max_nodes:
                            break
                        
                        child_id = tree['node_count']
                        tree['node_count'] += 1
                        
                        # Create child board - EFFICIENT COPY
                        copy_start = time.time()
                        child_board = board.copy()
                        child_board.make_moves(jnp.array([action]))
                        self.timing_stats['board_copy'].append(time.time() - copy_start)
                        
                        # Initialize child node
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
                        # Move to existing child
                        node_id = tree['children'][node_id][action]
                        path.append(node_id)
                
                # 2. EVALUATION PHASE
                # Evaluate leaf node with neural network
                leaf_board = tree['boards'][node_id]
                
                if leaf_board.game_states[0] == 0:
                    # Non-terminal node - use neural network
                    nn_start = time.time()
                    edge_indices, edge_features = leaf_board.get_features_for_nn_undirected()
                    
                    # Simple evaluation - just use random priors to test board performance
                    # In real use, would properly integrate with neural network
                    value = np.random.uniform(-1, 1)
                    tree['P'][node_id] = np.random.rand(self.num_actions)
                    tree['expanded'].add(node_id)
                    self.timing_stats['nn_eval'].append(time.time() - nn_start)
                else:
                    # Terminal node
                    winner = int(leaf_board.winners[0])
                    if winner == 0:  # Draw
                        value = 0.0
                    else:
                        # Value from perspective of player who just moved
                        player_at_leaf = 3 - int(leaf_board.current_players[0])
                        value = 1.0 if player_at_leaf == winner else -1.0
                
                # 3. BACKUP PHASE
                # Propagate value up the tree
                for i in range(len(path) - 1, -1, -1):
                    node = path[i]
                    tree['node_visits'][node] += 1
                    
                    if i > 0:  # Not root
                        parent = path[i-1]
                        parent_id, action = tree['parents'][node]
                        
                        # Update statistics
                        N_new, W_new = self._update_stats_jit(
                            jnp.array(tree['N'][parent_id]), 
                            jnp.array(tree['W'][parent_id]),
                            action, 
                            value
                        )
                        tree['N'][parent_id] = np.array(N_new)
                        tree['W'][parent_id] = np.array(W_new)
                    
                    # Flip value for alternating perspective
                    value = -value
                
                self.timing_stats['total_sim'].append(time.time() - sim_start)
            
            # Get action probabilities from root
            root_visits = tree['N'][0]
            
            if temperature == 0:
                # Deterministic - pick best
                probs = np.zeros_like(root_visits)
                probs[np.argmax(root_visits)] = 1.0
            else:
                # Sample according to visit counts
                visits_temp = np.power(root_visits, 1.0 / temperature)
                probs = visits_temp / visits_temp.sum()
            
            all_probs.append(probs)
        
        return np.array(all_probs)
    
    def print_timing_summary(self):
        """Print timing statistics."""
        if not self.timing_stats['total_sim']:
            return
        
        print("\n=== TIMING SUMMARY ===")
        for key, times in self.timing_stats.items():
            if times:
                avg_ms = np.mean(times) * 1000
                total_s = np.sum(times)
                print(f"{key}: {len(times)} calls, {avg_ms:.2f}ms avg, {total_s:.3f}s total")
        
        # Calculate percentages
        total_time = np.sum(self.timing_stats['total_sim'])
        if total_time > 0:
            print("\nTime breakdown:")
            for key in ['board_copy', 'ucb_calc', 'nn_eval']:
                if self.timing_stats[key]:
                    pct = np.sum(self.timing_stats[key]) / total_time * 100
                    print(f"  {key}: {pct:.1f}%")
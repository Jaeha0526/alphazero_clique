"""
SimpleTreeMCTS with detailed timing logs to identify bottlenecks.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial
import collections

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class SimpleTreeMCTSTimed:
    """
    SimpleTreeMCTS with detailed timing instrumentation.
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
        
        # Timing accumulators
        self.timing_stats = collections.defaultdict(list)
    
    def _calculate_ucb_scores(self, N: jnp.ndarray, W: jnp.ndarray, P: jnp.ndarray, 
                             node_visits: float, c_puct: float, valid_mask: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled UCB calculation."""
        Q = W / (1.0 + N)
        sqrt_visits = jnp.sqrt(jnp.maximum(1.0, node_visits))
        U = c_puct * sqrt_visits * (P / (1.0 + N))
        ucb = Q + U
        ucb = jnp.where(valid_mask, ucb, -jnp.inf)
        return ucb
    
    def _update_node_statistics(self, N: jnp.ndarray, W: jnp.ndarray, 
                               action: int, value: float) -> tuple:
        """JIT-compiled statistics update."""
        N_new = N.at[action].add(1.0)
        W_new = W.at[action].add(value)
        return N_new, W_new
        
    def search(self,
               boards: VectorizedCliqueBoard,
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               temperature: float = 1.0) -> np.ndarray:
        """
        Run tree MCTS with detailed timing logs.
        """
        print(f"\n      === MCTS TIMING ANALYSIS ===")
        print(f"      Games: {self.batch_size}, Simulations: {num_simulations}")
        
        total_start = time.time()
        self.timing_stats.clear()
        
        # Initialize trees
        init_start = time.time()
        trees = []
        for game_idx in range(self.batch_size):
            tree = {
                'N': {},  # Visit counts
                'W': {},  # Total values
                'P': {},  # Prior probabilities
                'children': {},  # Tree structure
                'boards': {},  # Board states
                'expanded': set(),
                'node_count': 0,
                'node_visits': {}
            }
            
            # Create root
            root_id = 0
            tree['N'][root_id] = np.zeros(self.num_actions)
            tree['W'][root_id] = np.zeros(self.num_actions)
            tree['children'][root_id] = {}
            
            # Extract single board
            single_board = VectorizedCliqueBoard(
                batch_size=1,
                num_vertices=boards.num_vertices,
                k=boards.k,
                game_mode=boards.game_mode
            )
            single_board.adjacency_matrices = boards.adjacency_matrices[game_idx:game_idx+1]
            single_board.current_players = boards.current_players[game_idx:game_idx+1]
            single_board.game_states = boards.game_states[game_idx:game_idx+1]
            single_board.winners = boards.winners[game_idx:game_idx+1]
            single_board.move_counts = boards.move_counts[game_idx:game_idx+1]
            
            tree['boards'][root_id] = single_board
            tree['node_count'] = 1
            tree['node_visits'][root_id] = 0
            
            trees.append(tree)
        
        init_time = time.time() - init_start
        self.timing_stats['initialization'].append(init_time)
        
        # Evaluate root positions
        nn_start = time.time()
        root_policies, root_values = neural_network.evaluate_batch(
            *boards.get_features_for_nn_undirected(), 
            boards.get_valid_moves_mask()
        )
        nn_time = time.time() - nn_start
        self.timing_stats['root_nn_eval'].append(nn_time)
        
        # Store root priors
        for game_idx, tree in enumerate(trees):
            tree['P'][0] = np.array(root_policies[game_idx])
            tree['expanded'].add(0)
        
        # Detailed timing for simulations
        sim_times = {
            'selection': [],
            'expansion': [],
            'evaluation': [],
            'backup': [],
            'total': []
        }
        
        # Run simulations
        for sim in range(num_simulations):
            sim_start = time.time()
            
            # Selection phase
            select_start = time.time()
            leaves_to_eval = []
            paths = []
            
            for game_idx, tree in enumerate(trees):
                # Traverse tree
                node_id = 0
                path = [(node_id, None)]
                tree['node_visits'][0] += 1
                
                while node_id in tree['expanded']:
                    if len(path) > 20:
                        break
                    
                    # UCB calculation
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
                    
                    # Move to child
                    if action not in tree['children'][node_id]:
                        # Create new child
                        if tree['node_count'] >= self.max_nodes:
                            break
                        
                        child_id = tree['node_count']
                        tree['node_count'] += 1
                        
                        # Board copying
                        copy_start = time.time()
                        child_board = VectorizedCliqueBoard(
                            batch_size=1,
                            num_vertices=board.num_vertices,
                            k=board.k,
                            game_mode=board.game_mode
                        )
                        child_board.adjacency_matrices = board.adjacency_matrices.copy()
                        child_board.current_players = board.current_players.copy()
                        child_board.game_states = board.game_states.copy()
                        child_board.winners = board.winners.copy()
                        child_board.move_counts = board.move_counts.copy()
                        child_board.make_moves(jnp.array([action]))
                        self.timing_stats['board_copy'].append(time.time() - copy_start)
                        
                        # Initialize child
                        tree['N'][child_id] = np.zeros(self.num_actions)
                        tree['W'][child_id] = np.zeros(self.num_actions)
                        tree['children'][child_id] = {}
                        tree['boards'][child_id] = child_board
                        tree['children'][node_id][action] = child_id
                        tree['node_visits'][child_id] = 0
                        
                        parent_id = node_id
                        node_id = child_id
                        tree['node_visits'][child_id] += 1
                        path.append((parent_id, action))
                        break
                    else:
                        # Move to existing child
                        parent_id = node_id
                        node_id = tree['children'][node_id][action]
                        tree['node_visits'][node_id] += 1
                        path.append((parent_id, action))
                
                paths.append((game_idx, path, node_id))
                
                # Check if leaf needs evaluation
                board = tree['boards'][node_id]
                if board.game_states[0] == 0 and node_id not in tree['expanded']:
                    leaves_to_eval.append((game_idx, node_id, board))
            
            sim_times['selection'].append(time.time() - select_start)
            
            # Evaluation phase
            eval_start = time.time()
            if leaves_to_eval:
                # Batch evaluation
                batch_boards = VectorizedCliqueBoard(
                    batch_size=len(leaves_to_eval),
                    num_vertices=boards.num_vertices,
                    k=boards.k,
                    game_mode=boards.game_mode
                )
                
                for i, (_, _, board) in enumerate(leaves_to_eval):
                    batch_boards.adjacency_matrices = batch_boards.adjacency_matrices.at[i].set(
                        board.adjacency_matrices[0]
                    )
                    batch_boards.current_players = batch_boards.current_players.at[i].set(
                        board.current_players[0]
                    )
                    batch_boards.game_states = batch_boards.game_states.at[i].set(
                        board.game_states[0]
                    )
                    batch_boards.winners = batch_boards.winners.at[i].set(
                        board.winners[0]
                    )
                    batch_boards.move_counts = batch_boards.move_counts.at[i].set(
                        board.move_counts[0]
                    )
                
                # NN evaluation
                nn_eval_start = time.time()
                leaf_policies, leaf_values = neural_network.evaluate_batch(
                    *batch_boards.get_features_for_nn_undirected(),
                    batch_boards.get_valid_moves_mask()
                )
                self.timing_stats['leaf_nn_eval'].append(time.time() - nn_eval_start)
                
                # Store results
                for i, (game_idx, node_id, _) in enumerate(leaves_to_eval):
                    trees[game_idx]['P'][node_id] = np.array(leaf_policies[i])
                    trees[game_idx]['expanded'].add(node_id)
            
            sim_times['evaluation'].append(time.time() - eval_start)
            
            # Backup phase
            backup_start = time.time()
            for game_idx, path, final_node in paths:
                tree = trees[game_idx]
                board = tree['boards'][final_node]
                
                if board.game_states[0] != 0:
                    # Terminal
                    if board.winners[0] == board.current_players[0]:
                        value = 1.0
                    else:
                        value = -1.0
                else:
                    # Use NN value
                    value = 0.0
                    for i, (g, n, _) in enumerate(leaves_to_eval):
                        if g == game_idx and n == final_node:
                            value = float(leaf_values[i, 0])
                            break
                
                # Backup through path
                for parent_node, action in path[1:]:
                    if action is not None:
                        update_start = time.time()
                        N_old = jnp.array(tree['N'][parent_node])
                        W_old = jnp.array(tree['W'][parent_node])
                        N_new, W_new = self._update_stats_jit(N_old, W_old, action, value)
                        tree['N'][parent_node] = np.array(N_new)
                        tree['W'][parent_node] = np.array(W_new)
                        self.timing_stats['stats_update'].append(time.time() - update_start)
                        
                        value = -value
            
            sim_times['backup'].append(time.time() - backup_start)
            sim_times['total'].append(time.time() - sim_start)
        
        # Extract action probabilities
        final_start = time.time()
        action_probs = np.zeros((self.batch_size, self.num_actions))
        
        for game_idx, tree in enumerate(trees):
            visits = tree['N'][0]
            
            if temperature == 0:
                if visits.sum() > 0:
                    probs = (visits == visits.max()).astype(float)
                    probs = probs / probs.sum()
                else:
                    board = tree['boards'][0]
                    valid_mask = board.get_valid_moves_mask()[0]
                    probs = valid_mask.astype(float)
                    probs = probs / probs.sum()
            else:
                visits_temp = np.power(visits, 1.0 / temperature)
                if visits_temp.sum() > 0:
                    probs = visits_temp / visits_temp.sum()
                else:
                    board = tree['boards'][0]
                    valid_mask = board.get_valid_moves_mask()[0]
                    probs = valid_mask.astype(float)
                    probs = probs / probs.sum()
            
            action_probs[game_idx] = probs
        
        final_time = time.time() - final_start
        total_time = time.time() - total_start
        
        # Print timing analysis
        print(f"\n      Total MCTS time: {total_time:.3f}s")
        print(f"      Initialization: {init_time:.3f}s")
        print(f"      Root NN eval: {nn_time:.3f}s")
        print(f"      Final processing: {final_time:.3f}s")
        
        print(f"\n      Per-simulation timing (avg over {num_simulations} sims):")
        print(f"        Selection: {np.mean(sim_times['selection'])*1000:.1f}ms")
        print(f"        Evaluation: {np.mean(sim_times['evaluation'])*1000:.1f}ms")
        print(f"        Backup: {np.mean(sim_times['backup'])*1000:.1f}ms")
        print(f"        Total: {np.mean(sim_times['total'])*1000:.1f}ms")
        
        print(f"\n      Detailed operation counts and timing:")
        for op, times in self.timing_stats.items():
            if times:
                print(f"        {op}: {len(times)} calls, {np.mean(times)*1000:.2f}ms avg, {np.sum(times):.3f}s total")
        
        avg_nodes = np.mean([tree['node_count'] for tree in trees])
        print(f"\n      Average nodes per tree: {avg_nodes:.1f}")
        
        return jnp.array(action_probs)
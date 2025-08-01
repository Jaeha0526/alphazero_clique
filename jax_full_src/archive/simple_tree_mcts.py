"""
Simplified Vectorized Tree MCTS that properly tracks board states.
Now with JIT compilation for hot computational paths.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class SimpleTreeMCTS:
    """
    A simpler tree MCTS that maintains proper game states.
    
    Key simplification: We store actual board states at each node,
    rather than trying to reconstruct from move sequences.
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
    
    def _calculate_ucb_scores(self, N: jnp.ndarray, W: jnp.ndarray, P: jnp.ndarray, 
                             node_visits: float, c_puct: float, valid_mask: jnp.ndarray) -> jnp.ndarray:
        """
        JIT-compiled UCB calculation. Preserves exact computation from original.
        
        Args:
            N: Visit counts for actions (num_actions,)
            W: Total values for actions (num_actions,)  
            P: Prior probabilities (num_actions,)
            node_visits: Total visits to this node
            c_puct: Exploration constant
            valid_mask: Valid moves mask (num_actions,)
        
        Returns:
            UCB scores (num_actions,)
        """
        # Q values - using (1 + N) denominator like original
        Q = W / (1.0 + N)
        
        # U values - use visits TO this node
        sqrt_visits = jnp.sqrt(jnp.maximum(1.0, node_visits))
        U = c_puct * sqrt_visits * (P / (1.0 + N))
        
        # UCB scores
        ucb = Q + U
        
        # Mask invalid actions
        ucb = jnp.where(valid_mask, ucb, -jnp.inf)
        
        return ucb
    
    def _update_node_statistics(self, N: jnp.ndarray, W: jnp.ndarray, 
                               action: int, value: float) -> tuple:
        """
        JIT-compiled statistics update.
        
        Args:
            N: Visit counts (num_actions,)
            W: Total values (num_actions,)
            action: Action to update
            value: Value to add
            
        Returns:
            Updated (N, W) arrays
        """
        N_new = N.at[action].add(1.0)
        W_new = W.at[action].add(value)
        return N_new, W_new
        
    def search(self,
               boards: VectorizedCliqueBoard,
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               temperature: float = 1.0) -> np.ndarray:
        """
        Run tree MCTS for multiple games in parallel.
        
        This version maintains separate trees for each game but evaluates
        positions in batches for efficiency.
        """
        print(f"      Starting simple tree MCTS with {num_simulations} simulations for {self.batch_size} games")
        start_time = time.time()
        
        # Progress tracking
        progress_interval = max(1, num_simulations // 10)  # Print every 10%
        
        # Initialize trees for each game
        trees = []
        for game_idx in range(self.batch_size):
            tree = {
                'N': {},  # Visit counts N[node_id][action]
                'W': {},  # Total values W[node_id][action]
                'P': {},  # Prior probabilities
                'children': {},  # children[node_id][action] = child_id
                'boards': {},  # Board state at each node
                'expanded': set(),  # Which nodes have been expanded
                'node_count': 0,
                'node_visits': {}  # Visits TO each node
            }
            
            # Create root node
            root_id = 0
            tree['N'][root_id] = np.zeros(self.num_actions)
            tree['W'][root_id] = np.zeros(self.num_actions)
            tree['children'][root_id] = {}
            
            # Extract single board for this game
            single_board = VectorizedCliqueBoard(
                batch_size=1,
                num_vertices=boards.num_vertices,
                k=boards.k,
                game_mode=boards.game_mode
            )
            # Copy state from vectorized board
            single_board.adjacency_matrices = boards.adjacency_matrices[game_idx:game_idx+1]
            single_board.current_players = boards.current_players[game_idx:game_idx+1]
            single_board.game_states = boards.game_states[game_idx:game_idx+1]
            single_board.winners = boards.winners[game_idx:game_idx+1]
            single_board.move_counts = boards.move_counts[game_idx:game_idx+1]
            
            tree['boards'][root_id] = single_board
            tree['node_count'] = 1
            tree['node_visits'][root_id] = 0  # Will be incremented each simulation
            
            trees.append(tree)
        
        # Evaluate all root positions at once
        root_policies, root_values = neural_network.evaluate_batch(
            *boards.get_features_for_nn_undirected(), 
            boards.get_valid_moves_mask()
        )
        
        # Store root priors
        for game_idx, tree in enumerate(trees):
            tree['P'][0] = np.array(root_policies[game_idx])
            tree['expanded'].add(0)
        
        # Run simulations
        for sim in range(num_simulations):
            if sim % progress_interval == 0:
                elapsed = time.time() - start_time
                print(f"        Simulation {sim}/{num_simulations} ({elapsed:.1f}s elapsed)")
            
            # For each game, traverse tree and collect leaves to evaluate
            leaves_to_eval = []
            paths = []  # For backup
            
            for game_idx, tree in enumerate(trees):
                # Selection: traverse from root to leaf
                node_id = 0
                path = [(node_id, None)]
                
                # Increment visit count to root
                tree['node_visits'][0] += 1
                
                while node_id in tree['expanded']:
                    if len(path) > 20:  # Safety check
                        print(f"WARNING: Path too long in game {game_idx}, breaking")
                        break
                    # Calculate UCB for all actions using JIT-compiled function
                    N = jnp.array(tree['N'][node_id])
                    W = jnp.array(tree['W'][node_id])  
                    P = jnp.array(tree['P'][node_id])
                    node_visits = float(tree['node_visits'][node_id])
                    board = tree['boards'][node_id]
                    valid_mask = jnp.array(board.get_valid_moves_mask()[0])  # Single board
                    
                    # JIT-compiled UCB calculation (preserves exact computation)
                    ucb = self._calculate_ucb_jit(N, W, P, node_visits, self.c_puct, valid_mask)
                    ucb = np.array(ucb)  # Convert back to numpy for compatibility
                    
                    # Select best action
                    if ucb.max() == -np.inf:
                        break  # No valid moves
                    
                    action = np.argmax(ucb)
                    
                    # Move to child or create new node
                    if action not in tree['children'][node_id]:
                        # Create new child
                        if tree['node_count'] >= self.max_nodes:
                            break
                        
                        child_id = tree['node_count']
                        tree['node_count'] += 1
                        
                        # Create child board
                        child_board = VectorizedCliqueBoard(
                            batch_size=1,
                            num_vertices=board.num_vertices,
                            k=board.k,
                            game_mode=board.game_mode
                        )
                        # Copy parent state
                        child_board.adjacency_matrices = board.adjacency_matrices.copy()
                        child_board.current_players = board.current_players.copy()
                        child_board.game_states = board.game_states.copy()
                        child_board.winners = board.winners.copy()
                        child_board.move_counts = board.move_counts.copy()
                        
                        # Make move
                        child_board.make_moves(jnp.array([action]))
                        
                        # Initialize child node
                        tree['N'][child_id] = np.zeros(self.num_actions)
                        tree['W'][child_id] = np.zeros(self.num_actions)
                        tree['children'][child_id] = {}
                        tree['boards'][child_id] = child_board
                        tree['children'][node_id][action] = child_id
                        tree['node_visits'][child_id] = 0  # Initialize visit count
                        
                        parent_id = node_id
                        node_id = child_id
                        tree['node_visits'][child_id] += 1  # Increment visit
                        path.append((parent_id, action))
                        break
                    else:
                        # Move to existing child
                        parent_id = node_id
                        node_id = tree['children'][node_id][action]
                        tree['node_visits'][node_id] += 1  # Increment visit
                        path.append((parent_id, action))
                
                # We've reached a leaf
                paths.append((game_idx, path, node_id))  # Include final node_id
                
                # If not terminal and not expanded, add to evaluation list
                board = tree['boards'][node_id]
                if board.game_states[0] == 0 and node_id not in tree['expanded']:
                    leaves_to_eval.append((game_idx, node_id, board))
            
            # Batch evaluate all leaves
            if leaves_to_eval:
                # Combine boards for batch evaluation
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
                
                # Evaluate
                leaf_policies, leaf_values = neural_network.evaluate_batch(
                    *batch_boards.get_features_for_nn_undirected(),
                    batch_boards.get_valid_moves_mask()
                )
                
                # Store results
                for i, (game_idx, node_id, _) in enumerate(leaves_to_eval):
                    trees[game_idx]['P'][node_id] = np.array(leaf_policies[i])
                    trees[game_idx]['expanded'].add(node_id)
            
            # Backup phase
            for game_idx, path, final_node in paths:
                tree = trees[game_idx]
                
                # Get value for this path
                board = tree['boards'][final_node]
                
                if board.game_states[0] != 0:
                    # Terminal node
                    if board.winners[0] == board.current_players[0]:
                        value = 1.0
                    else:
                        value = -1.0
                else:
                    # Use NN value
                    # Find in leaves_to_eval
                    value = 0.0
                    for i, (g, n, _) in enumerate(leaves_to_eval):
                        if g == game_idx and n == final_node:
                            value = float(leaf_values[i, 0])  # Extract scalar from shape (1,)
                            break
                
                # Backup value through the path
                # path contains (parent_node, action) pairs
                for parent_node, action in path[1:]:  # Skip first entry which is (0, None)
                    if action is not None:
                        # JIT-compiled statistics update
                        N_old = jnp.array(tree['N'][parent_node])
                        W_old = jnp.array(tree['W'][parent_node])
                        N_new, W_new = self._update_stats_jit(N_old, W_old, action, value)
                        tree['N'][parent_node] = np.array(N_new)
                        tree['W'][parent_node] = np.array(W_new)
                        
                        # Flip value for opponent
                        value = -value
        
        # Extract action probabilities from root visit counts
        action_probs = np.zeros((self.batch_size, self.num_actions))
        
        for game_idx, tree in enumerate(trees):
            visits = tree['N'][0]  # Root visits
            
            if temperature == 0:
                # Deterministic
                if visits.sum() > 0:
                    probs = (visits == visits.max()).astype(float)
                    probs = probs / probs.sum()
                else:
                    # No visits, use uniform over valid moves
                    board = tree['boards'][0]
                    valid_mask = board.get_valid_moves_mask()[0]
                    probs = valid_mask.astype(float)
                    probs = probs / probs.sum()
            else:
                # Apply temperature
                visits_temp = np.power(visits, 1.0 / temperature)
                if visits_temp.sum() > 0:
                    probs = visits_temp / visits_temp.sum()
                else:
                    # No visits, use uniform over valid moves
                    board = tree['boards'][0]
                    valid_mask = board.get_valid_moves_mask()[0]
                    probs = valid_mask.astype(float)
                    probs = probs / probs.sum()
            
            action_probs[game_idx] = probs
        
        elapsed = time.time() - start_time
        avg_nodes = np.mean([tree['node_count'] for tree in trees])
        print(f"      Tree MCTS complete in {elapsed:.2f}s. Avg nodes: {avg_nodes:.1f}")
        
        return jnp.array(action_probs)
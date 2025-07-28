"""
Vectorized Tree MCTS - Modified from SimpleTreeMCTS to add vectorization.
This implements synchronized tree traversal across all games.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class VectorizedTreeMCTSv2:
    """
    Tree MCTS with vectorized operations.
    Key change: All games traverse trees synchronously.
    """
    
    def __init__(self, 
                 batch_size: int,
                 num_actions: int = 15,
                 c_puct: float = 3.0,
                 max_nodes_per_game: int = 500):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.c_puct = c_puct
        self.max_nodes = max_nodes_per_game
        
    def search(self,
               boards: VectorizedCliqueBoard,
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               temperature: float = 1.0) -> np.ndarray:
        """
        Run tree MCTS for multiple games in parallel.
        Key difference: Vectorized tree traversal!
        """
        print(f"      Starting vectorized tree MCTS v2 with {num_simulations} simulations")
        start_time = time.time()
        
        # Initialize arrays for all games at once (instead of Python dicts)
        # Shape: (batch_size, max_nodes, num_actions)
        N = jnp.zeros((self.batch_size, self.max_nodes, self.num_actions))
        W = jnp.zeros((self.batch_size, self.max_nodes, self.num_actions))
        P = jnp.zeros((self.batch_size, self.max_nodes, self.num_actions))
        
        # Tree structure arrays
        children = jnp.ones((self.batch_size, self.max_nodes, self.num_actions), dtype=jnp.int32) * -1
        expanded = jnp.zeros((self.batch_size, self.max_nodes), dtype=bool)
        terminal = jnp.zeros((self.batch_size, self.max_nodes), dtype=bool)
        
        # Node count per game
        num_nodes = jnp.ones(self.batch_size, dtype=jnp.int32)
        
        # Store board states efficiently (as edge masks)
        edge_masks = jnp.zeros((self.batch_size, self.max_nodes, self.num_actions), dtype=bool)
        current_players = jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32)
        
        # Initialize root nodes
        print("      Getting root features...")
        root_features = boards.get_features_for_nn_undirected()
        root_valid = boards.get_valid_moves_mask()
        print("      Evaluating root with NN...")
        root_policies, root_values = neural_network.evaluate_batch(*root_features, root_valid)
        print("      Root evaluation complete")
        
        # Set root node data
        P = P.at[:, 0, :].set(root_policies)
        expanded = expanded.at[:, 0].set(True)
        current_players = current_players.at[:, 0].set(boards.current_players)
        
        # Run simulations
        for sim in range(num_simulations):
            if sim % 1 == 0:  # Print every simulation for debugging
                print(f"        Simulation {sim}/{num_simulations}")
            
            # VECTORIZED SELECTION PHASE
            # All games start at root
            current_nodes = jnp.zeros(self.batch_size, dtype=jnp.int32)
            active_games = jnp.ones(self.batch_size, dtype=bool)
            paths = []
            
            # Maximum depth = number of vertices (game must end)
            max_depth = boards.num_vertices
            print(f"          Starting selection, max_depth={max_depth}")
            
            for depth in range(max_depth):
                print(f"            Depth {depth}")
                # Check which games should continue
                is_expanded_current = expanded[jnp.arange(self.batch_size), current_nodes]
                is_terminal_current = terminal[jnp.arange(self.batch_size), current_nodes]
                should_continue = active_games & is_expanded_current & ~is_terminal_current
                
                print(f"              active_games: {active_games}")
                print(f"              is_expanded: {is_expanded_current}")
                print(f"              is_terminal: {is_terminal_current}")
                print(f"              should_continue: {should_continue}")
                
                # Convert to Python bool for the check
                if not bool(jnp.any(should_continue)):
                    print("              All games reached leaves, breaking")
                    break  # All games reached leaves
                
                # VECTORIZED UCB CALCULATION
                # Get data for current nodes
                batch_idx = jnp.arange(self.batch_size)
                current_N = N[batch_idx, current_nodes]  # (batch, actions)
                current_W = W[batch_idx, current_nodes]
                current_P = P[batch_idx, current_nodes]
                
                # Calculate Q values
                Q = jnp.where(current_N > 0, current_W / current_N, 0.0)
                
                # Calculate U values
                total_N = current_N.sum(axis=1, keepdims=True)
                sqrt_total = jnp.sqrt(total_N + 1)
                U = self.c_puct * current_P * sqrt_total / (1 + current_N)
                
                # UCB = Q + U
                UCB = Q + U
                
                # Mask invalid actions and inactive games
                valid_masks = self._get_valid_masks_for_nodes(
                    boards, current_nodes, edge_masks
                )
                UCB = jnp.where(valid_masks, UCB, -jnp.inf)
                UCB = jnp.where(should_continue[:, None], UCB, -jnp.inf)
                
                # Select best actions
                best_actions = jnp.argmax(UCB, axis=1)
                
                # Store path
                paths.append((current_nodes.copy(), best_actions, should_continue.copy()))
                
                # Move to children or create new nodes
                child_indices = children[batch_idx, current_nodes, best_actions]
                need_new_child = should_continue & (child_indices == -1)
                
                # Create new children where needed
                print(f"              need_new_child: {need_new_child}")
                if bool(jnp.any(need_new_child)):
                    game_indices = jnp.where(need_new_child)[0]
                    print(f"              Creating new children for {len(game_indices)} games: {game_indices}")
                    for i, game_idx in enumerate(game_indices):
                        new_idx = int(num_nodes[game_idx])
                        if new_idx < self.max_nodes:
                            # Set parent-child relationship
                            children = children.at[game_idx, current_nodes[game_idx], best_actions[game_idx]].set(new_idx)
                            
                            # Initialize child state
                            parent_edges = edge_masks[game_idx, current_nodes[game_idx]]
                            edge_masks = edge_masks.at[game_idx, new_idx].set(parent_edges)
                            edge_masks = edge_masks.at[game_idx, new_idx, best_actions[game_idx]].set(True)
                            
                            # Update player
                            parent_player = current_players[game_idx, current_nodes[game_idx]]
                            current_players = current_players.at[game_idx, new_idx].set(1 - parent_player)
                            
                            # Update node count
                            num_nodes = num_nodes.at[game_idx].add(1)
                            
                            # Update child index
                            child_indices = child_indices.at[game_idx].set(new_idx)
                
                # Move to children
                current_nodes = jnp.where(
                    should_continue & (child_indices >= 0),
                    child_indices,
                    current_nodes
                )
                
                # Update active games
                # Games are active if they should continue and have valid children
                # (either existing or newly created)
                active_games = should_continue
            
            # VECTORIZED EXPANSION & EVALUATION
            # Find which nodes need expansion
            need_expansion = ~expanded[batch_idx, current_nodes] & ~terminal[batch_idx, current_nodes]
            
            if need_expansion.any():
                # Get board states for nodes needing evaluation
                eval_boards = self._reconstruct_boards_batch(
                    boards, current_nodes, edge_masks, need_expansion, current_players
                )
                
                # Batch NN evaluation
                if eval_boards is not None:
                    eval_features = eval_boards.get_features_for_nn_undirected()
                    eval_valid = eval_boards.get_valid_moves_mask()
                    eval_policies, eval_values = neural_network.evaluate_batch(
                        *eval_features, eval_valid
                    )
                    
                    # Store policies for expanded nodes
                    eval_idx = 0
                    for game_idx in range(self.batch_size):
                        if need_expansion[game_idx]:
                            node_idx = current_nodes[game_idx]
                            P = P.at[game_idx, node_idx].set(eval_policies[eval_idx])
                            expanded = expanded.at[game_idx, node_idx].set(True)
                            
                            # Check if terminal
                            if eval_boards.game_states[eval_idx] != 0:
                                terminal = terminal.at[game_idx, node_idx].set(True)
                            
                            eval_idx += 1
            
            # VECTORIZED BACKUP
            # Get values for all games
            values = jnp.zeros(self.batch_size)
            for game_idx in range(self.batch_size):
                node_idx = current_nodes[game_idx]
                if terminal[game_idx, node_idx]:
                    # Terminal value
                    board = self._reconstruct_single_board(
                        boards, game_idx, node_idx, edge_masks, current_players
                    )
                    if board.winners[0] == current_players[game_idx, node_idx]:
                        values = values.at[game_idx].set(1.0)
                    else:
                        values = values.at[game_idx].set(-1.0)
                elif expanded[game_idx, node_idx]:
                    # Use NN value (would be stored from evaluation)
                    values = values.at[game_idx].set(0.0)  # Placeholder
            
            # Backup through paths
            for path_nodes, path_actions, was_active in reversed(paths):
                # Update N and W for all active games
                batch_idx = jnp.arange(self.batch_size)
                
                # Only update if was active
                update_mask = was_active
                N = jnp.where(
                    update_mask[:, None, None],
                    N.at[batch_idx, path_nodes, path_actions].add(1),
                    N
                )
                W = jnp.where(
                    update_mask[:, None, None],
                    W.at[batch_idx, path_nodes, path_actions].add(values),
                    W
                )
                
                # Flip values
                values = -values
        
        # Extract action probabilities from root
        root_visits = N[:, 0, :]
        
        # Apply temperature
        if temperature == 0:
            probs = (root_visits == root_visits.max(axis=1, keepdims=True)).astype(float)
        else:
            visits_temp = jnp.power(root_visits + 1e-8, 1.0 / temperature)
            probs = visits_temp / visits_temp.sum(axis=1, keepdims=True)
        
        # Mask invalid moves
        probs = jnp.where(root_valid, probs, 0.0)
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
        
        elapsed = time.time() - start_time
        avg_nodes = float(num_nodes.mean())
        print(f"      Vectorized tree MCTS complete in {elapsed:.2f}s. Avg nodes: {avg_nodes:.1f}")
        
        return probs
    
    def _get_valid_masks_for_nodes(self, root_boards, node_indices, edge_masks):
        """Get valid move masks for current nodes."""
        # Start with all moves valid
        valid_masks = jnp.ones((self.batch_size, self.num_actions), dtype=bool)
        
        # Mark edges already taken as invalid
        batch_idx = jnp.arange(self.batch_size)
        taken_edges = edge_masks[batch_idx, node_indices]
        valid_masks = ~taken_edges
        
        return valid_masks
    
    def _reconstruct_boards_batch(self, root_boards, node_indices, edge_masks, need_mask, current_players):
        """Reconstruct boards for nodes needing evaluation."""
        num_to_eval = need_mask.sum()
        if num_to_eval == 0:
            return None
        
        # Create new board batch for evaluation
        eval_boards = VectorizedCliqueBoard(
            batch_size=int(num_to_eval),
            num_vertices=root_boards.num_vertices,
            k=root_boards.k,
            game_mode=root_boards.game_mode
        )
        
        # Fill in board states
        eval_idx = 0
        for game_idx in range(self.batch_size):
            if need_mask[game_idx]:
                node_idx = node_indices[game_idx]
                # Get edges taken up to this node
                edges_taken = edge_masks[game_idx, node_idx]
                
                # Apply edges to board
                adj_matrix = jnp.zeros((root_boards.num_vertices, root_boards.num_vertices))
                edge_idx = 0
                for i in range(root_boards.num_vertices):
                    for j in range(i + 1, root_boards.num_vertices):
                        if edges_taken[edge_idx]:
                            adj_matrix = adj_matrix.at[i, j].set(1)
                            adj_matrix = adj_matrix.at[j, i].set(1)
                        edge_idx += 1
                
                eval_boards.adjacency_matrices = eval_boards.adjacency_matrices.at[eval_idx].set(adj_matrix)
                eval_boards.current_players = eval_boards.current_players.at[eval_idx].set(
                    current_players[game_idx, node_idx]
                )
                eval_boards.move_counts = eval_boards.move_counts.at[eval_idx].set(
                    edges_taken.sum()
                )
                
                eval_idx += 1
        
        # Check for wins after reconstructing boards
        eval_boards._check_wins_batch()
        
        return eval_boards
    
    def _reconstruct_single_board(self, root_boards, game_idx, node_idx, edge_masks, current_players):
        """Reconstruct a single board state."""
        board = VectorizedCliqueBoard(
            batch_size=1,
            num_vertices=root_boards.num_vertices,
            k=root_boards.k,
            game_mode=root_boards.game_mode
        )
        
        # Apply edges
        edges_taken = edge_masks[game_idx, node_idx]
        adj_matrix = jnp.zeros((root_boards.num_vertices, root_boards.num_vertices))
        
        edge_idx = 0
        for i in range(root_boards.num_vertices):
            for j in range(i + 1, root_boards.num_vertices):
                if edges_taken[edge_idx]:
                    adj_matrix = adj_matrix.at[i, j].set(1)
                    adj_matrix = adj_matrix.at[j, i].set(1)
                edge_idx += 1
        
        board.adjacency_matrices = board.adjacency_matrices.at[0].set(adj_matrix)
        board.current_players = board.current_players.at[0].set(
            current_players[game_idx, node_idx]
        )
        board._check_wins_batch()
        
        return board
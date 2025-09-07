"""
True MCTX Implementation with Full Vectorization
This implementation follows MCTX's actual design principles:
1. NO Python loops in the hot path
2. Fully vectorized tree operations
3. Single batched neural network evaluation
4. Complete JIT compilation
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial
from typing import Tuple, NamedTuple

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


class TreeArrays(NamedTuple):
    """Pre-allocated arrays for MCTS tree"""
    # Tree statistics
    visit_counts: jnp.ndarray     # [batch, num_nodes, num_actions]
    total_values: jnp.ndarray     # [batch, num_nodes, num_actions]
    prior_probs: jnp.ndarray      # [batch, num_nodes, num_actions]
    
    # Tree structure
    children: jnp.ndarray         # [batch, num_nodes, num_actions]
    parents: jnp.ndarray          # [batch, num_nodes]
    parent_actions: jnp.ndarray   # [batch, num_nodes]
    
    # Node state
    node_indices: jnp.ndarray     # [batch, num_nodes] - for easy indexing
    node_visits: jnp.ndarray      # [batch, num_nodes]
    is_expanded: jnp.ndarray      # [batch, num_nodes]
    num_nodes: jnp.ndarray        # [batch]
    
    # Game state
    edge_states: jnp.ndarray      # [batch, num_nodes, num_edges]
    current_players: jnp.ndarray  # [batch, num_nodes]
    rewards: jnp.ndarray          # [batch, num_nodes]


class SearchState(NamedTuple):
    """State for MCTS search iteration"""
    current_nodes: jnp.ndarray    # [batch] - current node for each game
    found_leaves: jnp.ndarray     # [batch] - whether leaf was found
    selected_actions: jnp.ndarray # [batch] - action to take
    depths: jnp.ndarray           # [batch] - current depth


class MCTXTrueJAX:
    """
    True MCTX-style MCTS with full vectorization.
    
    Key differences from previous attempts:
    1. Vectorized tree traversal using vmap + while_loop
    2. No Python loops anywhere in search
    3. Batched operations across entire tree
    4. Single NN evaluation per iteration
    """
    
    def __init__(self,
                 batch_size: int,
                 num_actions: int = 15,
                 max_nodes: int = 500,
                 c_puct: float = 3.0,
                 discount: float = 1.0,
                 num_vertices: int = 6):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.max_nodes = max_nodes
        self.c_puct = c_puct
        self.discount = discount
        self.num_vertices = num_vertices
        
        # Pre-compile all major functions
        self._init_tree = jax.jit(self._init_tree_impl)
        self._compute_ucb_scores = jax.jit(self._compute_ucb_scores_impl)
        
    def _init_tree_impl(self) -> TreeArrays:
        """Initialize pre-allocated tree arrays"""
        # Create node indices for easier gathering
        node_indices = jnp.arange(self.max_nodes)[None, :].repeat(self.batch_size, axis=0)
        
        return TreeArrays(
            visit_counts=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            total_values=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            prior_probs=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions)),
            children=jnp.full((self.batch_size, self.max_nodes, self.num_actions), -1, dtype=jnp.int32),
            parents=jnp.full((self.batch_size, self.max_nodes), -1, dtype=jnp.int32),
            parent_actions=jnp.full((self.batch_size, self.max_nodes), -1, dtype=jnp.int32),
            node_indices=node_indices,
            node_visits=jnp.zeros((self.batch_size, self.max_nodes)),
            is_expanded=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.bool_),
            num_nodes=jnp.ones(self.batch_size, dtype=jnp.int32),
            edge_states=jnp.zeros((self.batch_size, self.max_nodes, self.num_actions), dtype=jnp.int32),
            current_players=jnp.zeros((self.batch_size, self.max_nodes), dtype=jnp.int32),
            rewards=jnp.zeros((self.batch_size, self.max_nodes))
        )
    
    def _compute_ucb_scores_impl(self,
                                 visit_counts: jnp.ndarray,
                                 total_values: jnp.ndarray,
                                 prior_probs: jnp.ndarray,
                                 parent_visits: jnp.ndarray,
                                 valid_actions: jnp.ndarray) -> jnp.ndarray:
        """Compute UCB scores for all actions. Shape: [batch, num_actions]"""
        # Avoid division by zero
        visit_counts_safe = jnp.maximum(visit_counts, 1e-8)
        
        # Q-values
        q_values = total_values / visit_counts_safe
        
        # Exploration term
        sqrt_parent = jnp.sqrt(parent_visits)[:, None]
        exploration = self.c_puct * prior_probs * sqrt_parent / (1.0 + visit_counts)
        
        # UCB score
        ucb_scores = q_values + exploration
        
        # Mask invalid actions
        ucb_scores = jnp.where(valid_actions, ucb_scores, -jnp.inf)
        
        return ucb_scores
    
    @partial(jax.jit, static_argnums=(0,))
    def _select_leaf_vectorized(self, tree: TreeArrays) -> SearchState:
        """
        Vectorized leaf selection for all games simultaneously.
        Uses jax.lax.while_loop for tree traversal.
        """
        def cond_fn(state: SearchState) -> jnp.ndarray:
            """Continue while any game hasn't found a leaf"""
            # Check if nodes are expanded
            batch_indices = jnp.arange(self.batch_size)
            expanded = tree.is_expanded[batch_indices, state.current_nodes]
            
            # Continue if expanded and haven't found leaf
            continue_search = expanded & ~state.found_leaves
            return jnp.any(continue_search)
        
        def body_fn(state: SearchState) -> SearchState:
            """Select actions and traverse tree"""
            batch_indices = jnp.arange(self.batch_size)
            
            # Get current node data
            current_visits = tree.visit_counts[batch_indices, state.current_nodes]
            current_values = tree.total_values[batch_indices, state.current_nodes]
            current_priors = tree.prior_probs[batch_indices, state.current_nodes]
            parent_visits = tree.node_visits[batch_indices, state.current_nodes]
            
            # Valid actions mask
            edge_states = tree.edge_states[batch_indices, state.current_nodes]
            valid_actions = edge_states == 0
            
            # Compute UCB scores
            ucb_scores = self._compute_ucb_scores(
                current_visits, current_values, current_priors,
                parent_visits, valid_actions
            )
            
            # Select best actions
            selected_actions = jnp.argmax(ucb_scores, axis=1)
            
            # Get child nodes
            children = tree.children[batch_indices, state.current_nodes, selected_actions]
            
            # Check which need expansion (no child exists)
            need_expansion = (children == -1) & (selected_actions != -1)
            
            # Update state
            # If we need expansion or node isn't expanded, we found a leaf
            expanded = tree.is_expanded[batch_indices, state.current_nodes]
            found_leaves = state.found_leaves | need_expansion | ~expanded
            
            # Move to child if it exists and we haven't found leaf
            next_nodes = jnp.where(
                found_leaves | (children == -1),
                state.current_nodes,
                children
            )
            
            return SearchState(
                current_nodes=next_nodes,
                found_leaves=found_leaves,
                selected_actions=jnp.where(found_leaves, selected_actions, state.selected_actions),
                depths=state.depths + ~found_leaves
            )
        
        # Initialize search state
        init_state = SearchState(
            current_nodes=jnp.zeros(self.batch_size, dtype=jnp.int32),
            found_leaves=jnp.zeros(self.batch_size, dtype=jnp.bool_),
            selected_actions=jnp.full(self.batch_size, -1, dtype=jnp.int32),
            depths=jnp.zeros(self.batch_size, dtype=jnp.int32)
        )
        
        # Run vectorized selection
        final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
        
        return final_state
    
    @partial(jax.jit, static_argnums=(0,))
    def _expand_nodes(self, tree: TreeArrays, search_state: SearchState) -> TreeArrays:
        """Expand nodes that need expansion (vectorized)"""
        batch_indices = jnp.arange(self.batch_size)
        
        # Which nodes need expansion
        need_expansion = (search_state.selected_actions >= 0) & \
                        ~tree.is_expanded[batch_indices, search_state.current_nodes]
        
        # New node indices
        new_node_indices = tree.num_nodes
        
        # Create children where needed
        def update_single_tree(carry, inputs):
            tree_arrays, game_idx = carry
            needs_exp, current_node, action, new_idx = inputs
            
            # Update only if expansion needed and space available
            can_expand = needs_exp & (new_idx < self.max_nodes)
            
            # Update children
            tree_arrays = tree_arrays._replace(
                children=tree_arrays.children.at[game_idx, current_node, action].set(
                    jnp.where(can_expand, new_idx, -1)
                ),
                parents=tree_arrays.parents.at[game_idx, new_idx].set(
                    jnp.where(can_expand, current_node, -1)
                ),
                parent_actions=tree_arrays.parent_actions.at[game_idx, new_idx].set(
                    jnp.where(can_expand, action, -1)
                ),
                num_nodes=tree_arrays.num_nodes.at[game_idx].set(
                    jnp.where(can_expand, new_idx + 1, tree_arrays.num_nodes[game_idx])
                )
            )
            
            # Copy and update edge states
            parent_edges = tree_arrays.edge_states[game_idx, current_node]
            new_edges = parent_edges.at[action].set(1)
            tree_arrays = tree_arrays._replace(
                edge_states=tree_arrays.edge_states.at[game_idx, new_idx].set(
                    jnp.where(can_expand, new_edges, tree_arrays.edge_states[game_idx, new_idx])
                ),
                current_players=tree_arrays.current_players.at[game_idx, new_idx].set(
                    jnp.where(can_expand, 
                             1 - tree_arrays.current_players[game_idx, current_node],
                             tree_arrays.current_players[game_idx, new_idx])
                )
            )
            
            return (tree_arrays, game_idx + 1), None
        
        # Vectorized update
        inputs = (need_expansion, search_state.current_nodes, 
                 search_state.selected_actions, new_node_indices)
        (tree, _), _ = jax.lax.scan(
            update_single_tree, (tree, 0), 
            jax.tree.map(lambda x: x, inputs)
        )
        
        return tree
    
    def _evaluate_batch(self, tree: TreeArrays, leaf_mask: jnp.ndarray,
                       leaf_indices: jnp.ndarray, neural_network) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate all leaf nodes in a single batch"""
        batch_indices = jnp.arange(self.batch_size)
        
        # Extract edge states for leaf nodes
        leaf_edge_states = tree.edge_states[batch_indices, leaf_indices]
        leaf_players = tree.current_players[batch_indices, leaf_indices]
        
        # Convert edge states to features for NN
        # Create edge indices (same for all boards)
        edge_list = []
        for i in range(self.num_vertices):
            for j in range(i+1, self.num_vertices):
                edge_list.append([i, j])
        edge_indices = jnp.array(edge_list, dtype=jnp.int32).T
        
        # Expand edge indices for batch
        batch_edge_indices = jnp.tile(edge_indices[None, :, :], (self.batch_size, 1, 1))
        
        # Convert edge states to features (one-hot encoding)
        edge_features = jnp.zeros((self.batch_size, self.num_actions, 3))
        edge_features = edge_features.at[:, :, 0].set(leaf_edge_states == 0)  # Empty
        edge_features = edge_features.at[:, :, 1].set(leaf_edge_states == 1)  # Player 1
        edge_features = edge_features.at[:, :, 2].set(leaf_edge_states == 2)  # Player 2
        
        # Get valid moves mask
        valid_moves = leaf_edge_states == 0
        
        # Evaluate with neural network
        if neural_network.asymmetric_mode:
            # For asymmetric mode, pass player roles
            policies, values = neural_network.evaluate_batch(
                batch_edge_indices, 
                edge_features,
                valid_moves,
                player_roles=leaf_players
            )
        else:
            policies, values = neural_network.evaluate_batch(
                batch_edge_indices,
                edge_features,
                valid_moves
            )
        
        # Mask out invalid leaves
        policies = jnp.where(leaf_mask[:, None], policies, 0.0)
        values = jnp.where(leaf_mask, values[:, 0], 0.0)
        
        return policies, values
    
    def _backup_values(self, tree: TreeArrays, search_state: SearchState,
                      values: jnp.ndarray) -> TreeArrays:
        """Backup values through the tree (vectorized)"""
        batch_indices = jnp.arange(self.batch_size)
        
        # We need to backup from leaf to root
        # Start with the leaf nodes
        current_nodes = search_state.current_nodes
        current_values = values
        
        # Maximum backup depth (same as max tree depth)
        max_depth = 20
        
        def backup_step(carry, _):
            tree_arrays, nodes, vals, actions = carry
            
            # Update statistics for current nodes and actions
            # Only update if action is valid (>= 0)
            valid_mask = actions >= 0
            
            # Update visit counts
            indices = (batch_indices, nodes, actions)
            tree_arrays = tree_arrays._replace(
                visit_counts=tree_arrays.visit_counts.at[indices].add(
                    jnp.where(valid_mask, 1, 0)
                ),
                total_values=tree_arrays.total_values.at[indices].add(
                    jnp.where(valid_mask, vals, 0.0)
                ),
                node_visits=tree_arrays.node_visits.at[batch_indices, nodes].add(
                    jnp.where(valid_mask, 1, 0)
                )
            )
            
            # Move to parent nodes
            parent_nodes = tree_arrays.parents[batch_indices, nodes]
            parent_actions = tree_arrays.parent_actions[batch_indices, nodes]
            
            # Stop at root (parent = -1)
            continue_mask = parent_nodes >= 0
            next_nodes = jnp.where(continue_mask, parent_nodes, nodes)
            next_actions = jnp.where(continue_mask, parent_actions, -1)
            
            # Negate values for opponent
            next_vals = -vals
            
            return (tree_arrays, next_nodes, next_vals, next_actions), None
        
        # Initialize with leaf nodes and their selected actions
        init_carry = (tree, current_nodes, current_values, search_state.selected_actions)
        
        # Run backup for maximum depth
        (tree, _, _, _), _ = jax.lax.scan(
            backup_step, init_carry, None, length=max_depth
        )
        
        return tree
    
    def _run_simulations_impl(self, tree: TreeArrays, neural_network,
                             num_simulations: int) -> TreeArrays:
        """Run MCTS simulations - fully vectorized"""
        
        def simulation_step(tree: TreeArrays, _) -> Tuple[TreeArrays, None]:
            # 1. Select leaves for all games
            search_state = self._select_leaf_vectorized(tree)
            
            # 2. Expand nodes if needed
            tree = self._expand_nodes(tree, search_state)
            
            # 3. Determine which nodes to evaluate
            batch_indices = jnp.arange(self.batch_size)
            
            # Get the actual leaf nodes (either current or newly created children)
            leaf_indices = jnp.where(
                search_state.selected_actions >= 0,
                tree.children[batch_indices, search_state.current_nodes, search_state.selected_actions],
                search_state.current_nodes
            )
            
            # Mask for valid leaves
            leaf_mask = (leaf_indices >= 0) & ~tree.is_expanded[batch_indices, leaf_indices]
            
            # 4. Evaluate leaves
            policies, values = self._evaluate_batch(
                tree, leaf_mask, leaf_indices, neural_network
            )
            
            # 5. Update tree with evaluations
            tree = tree._replace(
                prior_probs=tree.prior_probs.at[batch_indices, leaf_indices].set(
                    jnp.where(leaf_mask[:, None], policies, tree.prior_probs[batch_indices, leaf_indices])
                ),
                is_expanded=tree.is_expanded.at[batch_indices, leaf_indices].set(
                    leaf_mask | tree.is_expanded[batch_indices, leaf_indices]
                )
            )
            
            # 6. Backup values
            tree = self._backup_values(tree, search_state, values)
            
            return tree, None
        
        # Run all simulations
        tree, _ = jax.lax.scan(simulation_step, tree, None, length=num_simulations)
        
        return tree
    
    def search(self, boards: VectorizedCliqueBoard,
               neural_network: ImprovedBatchedNeuralNetwork,
               num_simulations: int,
               temperature: float = 1.0) -> jnp.ndarray:
        """
        Run MCTS search - fully vectorized and JIT-compiled.
        
        Returns:
            Action probabilities [batch_size, num_actions]
        """
        print(f"Starting True MCTX Implementation with {num_simulations} simulations")
        start_time = time.time()
        
        # Initialize tree
        tree = self._init_tree()
        
        # Set up root nodes
        # Convert board states to edge representation
        # Convert board edge states to flat array
        edge_states = jnp.zeros((self.batch_size, self.num_actions), dtype=jnp.int32)
        edge_idx = 0
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                edge_played = boards.edge_states[:, i, j]
                edge_states = edge_states.at[:, edge_idx].set(edge_played)
                edge_idx += 1
        
        tree = tree._replace(
            edge_states=tree.edge_states.at[:, 0].set(edge_states),
            current_players=tree.current_players.at[:, 0].set(boards.current_players),
            is_expanded=tree.is_expanded.at[:, 0].set(True)
        )
        
        # Evaluate root
        root_policies, root_values = neural_network.evaluate_batch(
            *boards.get_features_for_nn_undirected(),
            boards.get_valid_moves_mask()
        )
        
        # Add Dirichlet noise to root policies for exploration
        # Use different noise weights for self-play vs evaluation
        if temperature > 0:  
            # Self-play: use standard noise weight
            noise_weight = 0.25
        else:  
            # Evaluation: use smaller noise weight for variety while maintaining strong play
            noise_weight = 0.1
        
        if noise_weight > 0:
            noise_alpha = 0.3  # Standard AlphaGo/AlphaZero value
            key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
            dirichlet_noise = jax.random.dirichlet(key, jnp.ones(self.num_actions) * noise_alpha, shape=(self.batch_size,))
            
            # Mix original priors with noise (only for valid moves)
            valid_mask = boards.get_valid_moves_mask()
            noisy_policies = (1 - noise_weight) * root_policies + noise_weight * dirichlet_noise
            # Re-mask and normalize
            noisy_policies = jnp.where(valid_mask, noisy_policies, 0.0)
            noisy_policies = noisy_policies / jnp.sum(noisy_policies, axis=1, keepdims=True)
            root_policies = noisy_policies
        
        tree = tree._replace(
            prior_probs=tree.prior_probs.at[:, 0].set(root_policies)
        )
        
        # Run simulations
        tree = self._run_simulations_impl(
            tree, neural_network, num_simulations
        )
        
        # Extract visit counts from root
        root_visits = tree.visit_counts[:, 0]
        
        # Apply temperature and normalize
        if temperature == 0:
            # Deterministic
            is_max = (root_visits == jnp.max(root_visits, axis=1, keepdims=True)).astype(jnp.float32)
            action_probs = is_max / jnp.sum(is_max, axis=1, keepdims=True)
        else:
            # Stochastic with temperature
            root_visits_temp = jnp.power(root_visits + 1e-8, 1.0 / temperature)
            action_probs = root_visits_temp / jnp.sum(root_visits_temp, axis=1, keepdims=True)
        
        elapsed = time.time() - start_time
        print(f"True MCTX search complete in {elapsed:.3f}s ({elapsed/self.batch_size*1000:.1f}ms per game)")
        
        # Return both action probabilities and raw visit counts
        return action_probs, root_visits


# Test implementation
if __name__ == "__main__":
    print("Testing True MCTX Implementation...")
    
    # Create test instance
    mcts = MCTXTrueJAX(
        batch_size=8,
        num_actions=15,
        max_nodes=500,
        c_puct=3.0,
        num_vertices=6
    )
    
    # Create test boards
    boards = VectorizedCliqueBoard(8, 6, 3, "symmetric")
    
    # Create neural network
    nn = ImprovedBatchedNeuralNetwork(6, 128, 4)
    
    # Test search
    print("\nRunning search test...")
    action_probs = mcts.search(boards, nn, num_simulations=50, temperature=1.0)
    
    print(f"\nAction probabilities shape: {action_probs.shape}")
    print(f"Probabilities sum to 1: {jnp.allclose(jnp.sum(action_probs, axis=1), 1.0)}")
    print("\nTrue MCTX implementation complete!")
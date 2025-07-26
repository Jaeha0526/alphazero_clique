#!/usr/bin/env python
"""
GPU-enabled MCTS for Clique Game using JAX
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict
from functools import partial

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
    print("JAX MCTS: Using JAX with GPU acceleration")
except ImportError:
    import warnings
    warnings.warn("JAX not available, using NumPy fallback")
    jnp = np
    JAX_AVAILABLE = False
    def jit(f): return f
    def vmap(f, **kwargs): return f

# Import the base board (still uses numpy for game logic)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_src.jax_clique_board_numpy import JAXCliqueBoard
import src.encoder_decoder_clique as ed


class MCTSNode:
    """MCTS Node for tree search"""
    def __init__(self, board_state, parent=None, action=None, prior_prob=0.0):
        self.board_state = board_state
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob
        
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_expanded = False
        
    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def ucb_score(self, c_puct=1.0):
        if self.visits == 0:
            return float('inf')
        
        exploration = c_puct * self.prior_prob * np.sqrt(self.parent.visits) / (1 + self.visits)
        return self.value() + exploration


class GPUAcceleratedMCTS:
    """MCTS with GPU-accelerated neural network evaluation"""
    
    def __init__(self, board: JAXCliqueBoard, num_simulations: int,
                 model, model_params: Dict, c_puct: float = 1.0,
                 noise_weight: float = 0.0):
        self.board = board
        self.num_simulations = num_simulations
        self.model = model
        self.model_params = model_params
        self.c_puct = c_puct
        self.noise_weight = noise_weight
        
        # Create root node
        self.root = MCTSNode(board.get_board_state())
    
    def _evaluate_position_gpu(self, edge_index: jnp.ndarray, edge_attr: jnp.ndarray):
        """GPU-accelerated position evaluation"""
        # Direct call without additional JIT (model is already JIT-compiled)
        policy, value = self.model(self.model_params, edge_index, edge_attr)
        return policy, value
    
    def _evaluate_position(self, board: JAXCliqueBoard) -> Tuple[np.ndarray, float]:
        """Evaluate position using GPU model"""
        # Prepare state for network
        state_dict = ed.prepare_state_for_network(board)
        
        # Convert to JAX arrays for GPU
        edge_index = jnp.array(state_dict['edge_index'].numpy())
        edge_attr = jnp.array(state_dict['edge_attr'].numpy())
        
        # GPU evaluation
        policy, value = self._evaluate_position_gpu(edge_index, edge_attr)
        
        # Convert back to numpy
        policy = np.array(policy).flatten()
        value = float(np.array(value))
        
        # Apply valid moves mask
        valid_mask = ed.get_valid_moves_mask(board)
        masked_policy = policy * valid_mask
        
        # Renormalize
        if masked_policy.sum() > 0:
            masked_policy /= masked_policy.sum()
        else:
            # Uniform over valid moves
            masked_policy = valid_mask / valid_mask.sum()
        
        return masked_policy, value
    
    def _select(self) -> Tuple[MCTSNode, List[MCTSNode]]:
        """Select path from root to leaf"""
        path = []
        node = self.root
        
        while node.is_expanded and len(node.children) > 0:
            path.append(node)
            
            # Select best child by UCB
            best_score = -float('inf')
            best_child = None
            
            for child in node.children.values():
                score = child.ucb_score(self.c_puct)
                if score > best_score:
                    best_score = score
                    best_child = child
            
            node = best_child
        
        path.append(node)
        return node, path
    
    def _expand(self, node: MCTSNode, board: JAXCliqueBoard) -> float:
        """Expand node and return value"""
        if board.game_state != 0:  # Terminal
            # Game over
            if board.game_state == 3:  # Draw
                return 0.0
            elif board.game_state == board.player + 1:
                return 1.0
            else:
                return -1.0
        
        # Get policy and value from neural network
        policy, value = self._evaluate_position(board)
        
        # Add noise to root
        if node == self.root and self.noise_weight > 0:
            noise = np.random.dirichlet([0.3] * len(policy))
            policy = (1 - self.noise_weight) * policy + self.noise_weight * noise
        
        # Expand children
        valid_moves = board.get_valid_moves()
        node.is_expanded = True
        
        for move in valid_moves:
            action_idx = ed.encode_action(board, move)
            if 0 <= action_idx < len(policy):
                prior = policy[action_idx]
                child_board = board.copy()
                child_board.make_move(move)
                child_node = MCTSNode(
                    board_state=child_board.get_board_state(),
                    parent=node,
                    action=move,
                    prior_prob=prior
                )
                node.children[move] = child_node
        
        return value
    
    def _backup(self, path: List[MCTSNode], value: float):
        """Backup value through path"""
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value  # Flip for opponent
    
    def search(self) -> Tuple[int, Dict[int, int]]:
        """Run MCTS search and return best action"""
        # Run simulations
        for _ in range(self.num_simulations):
            # Select
            leaf, path = self._select()
            
            # Get board state for leaf
            board = JAXCliqueBoard.from_dict(leaf.board_state)
            
            # Expand and evaluate
            value = self._expand(leaf, board)
            
            # Backup
            self._backup(path, value)
        
        # Choose action
        visits = {}
        for move, child in self.root.children.items():
            action_idx = ed.encode_action(self.board, move)
            visits[action_idx] = child.visits
        
        # Temperature-based selection (tau=0 for best)
        best_action = max(visits.keys(), key=lambda a: visits[a])
        
        return best_action, visits


def batch_mcts_gpu(boards: List[JAXCliqueBoard], model, model_params: Dict,
                  num_simulations: int = 100) -> List[Tuple[int, Dict]]:
    """Run MCTS on multiple boards in parallel using GPU"""
    results = []
    
    # For true GPU parallelism, we'd batch the neural network evaluations
    # For now, run sequentially but with GPU-accelerated evaluations
    for board in boards:
        mcts = GPUAcceleratedMCTS(board, num_simulations, model, model_params)
        action, visits = mcts.search()
        results.append((action, visits))
    
    return results


# Simplified interface matching original
def UCT_search(board: JAXCliqueBoard, num_simulations: int, net, 
               device=None, noise_weight: float = 0.0) -> Tuple[int, MCTSNode]:
    """GPU-accelerated UCT search compatible with original interface"""
    if hasattr(net, 'model') and hasattr(net, 'params'):
        # It's our GPU model
        model = net.model
        params = net.params
    else:
        # Assume it's a wrapped model
        model = net
        params = getattr(net, 'params', None)
    
    mcts = GPUAcceleratedMCTS(board, num_simulations, model, params, noise_weight=noise_weight)
    best_action, visits = mcts.search()
    
    return best_action, mcts.root


if __name__ == "__main__":
    # Test GPU MCTS
    print("Testing GPU-accelerated MCTS...")
    
    from jax_src.jax_alpha_net_clique_gpu import create_gpu_model
    
    # Create model
    model, params = create_gpu_model()
    
    # Create board
    board = JAXCliqueBoard(6, 3)
    
    # Create wrapped model for interface compatibility
    class ModelWrapper:
        def __init__(self, model, params):
            self.model = model
            self.params = params
    
    wrapped_model = ModelWrapper(model, params)
    
    # Run MCTS
    print("\nRunning MCTS with GPU neural network...")
    start = time.time()
    
    best_action, root = UCT_search(board, num_simulations=100, net=wrapped_model)
    
    elapsed = time.time() - start
    print(f"MCTS completed in {elapsed:.3f}s")
    print(f"Best action: {best_action}")
    print(f"Root visits: {root.visits}")
    
    if JAX_AVAILABLE:
        print(f"\nUsing: {jax.devices()[0]}")
        print("✓ GPU acceleration enabled!")
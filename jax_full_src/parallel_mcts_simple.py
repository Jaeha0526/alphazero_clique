"""
Simple parallel MCTS that actually runs games in parallel.
Instead of a sequential for loop, we'll use multiprocessing or threading.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple
import multiprocessing as mp
from functools import partial
import time

from tree_based_mcts import TreeBasedMCTS
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork


def run_single_mcts(args):
    """Run MCTS for a single game (for multiprocessing)."""
    game_idx, board_state, nn_params, nn_config, num_simulations, temperature = args
    
    # Recreate board from state
    board = VectorizedCliqueBoard(batch_size=1)
    board.edge_states = board_state['edge_states']
    board.current_players = board_state['current_players'] 
    board.game_states = board_state['game_states']
    board.move_counts = board_state['move_counts']
    
    # Recreate neural network
    nn = ImprovedBatchedNeuralNetwork(**nn_config)
    nn.params = nn_params
    
    # Run MCTS
    mcts = TreeBasedMCTS(num_actions=15, c_puct=3.0)
    action_probs = mcts.search(board, nn, num_simulations, 0, temperature)
    
    return game_idx, action_probs


class SimpleParallelMCTS:
    """Actually parallel MCTS using multiprocessing."""
    
    def __init__(self, batch_size: int, num_workers: int = None, **mcts_kwargs):
        self.batch_size = batch_size
        self.num_workers = num_workers or min(batch_size, mp.cpu_count())
        self.mcts_kwargs = mcts_kwargs
    
    def search_parallel(self, boards: VectorizedCliqueBoard, 
                       neural_network: ImprovedBatchedNeuralNetwork,
                       num_simulations: jnp.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        """Run MCTS for multiple games truly in parallel."""
        
        # Prepare arguments for each game
        nn_config = {
            'num_vertices': neural_network.num_vertices,
            'hidden_dim': neural_network.hidden_dim,
            'num_layers': neural_network.num_layers,
            'asymmetric_mode': neural_network.asymmetric_mode
        }
        
        args_list = []
        for game_idx in range(self.batch_size):
            if boards.game_states[game_idx] == 0:  # Game still active
                board_state = {
                    'edge_states': boards.edge_states[game_idx:game_idx+1],
                    'current_players': boards.current_players[game_idx:game_idx+1],
                    'game_states': boards.game_states[game_idx:game_idx+1],
                    'move_counts': boards.move_counts[game_idx:game_idx+1]
                }
                args = (game_idx, board_state, neural_network.params, nn_config,
                       int(num_simulations[game_idx]), temperature)
                args_list.append(args)
        
        # Run in parallel
        action_probs = np.zeros((self.batch_size, 15))
        
        if len(args_list) > 0:
            with mp.Pool(self.num_workers) as pool:
                results = pool.map(run_single_mcts, args_list)
            
            for game_idx, probs in results:
                action_probs[game_idx] = probs
        
        return jnp.array(action_probs)


class ThreadedParallelMCTS:
    """Parallel MCTS using threads (shares memory, might be faster for small batches)."""
    
    def __init__(self, batch_size: int, **mcts_kwargs):
        self.batch_size = batch_size
        self.mcts_kwargs = mcts_kwargs
        
    def search_parallel(self, boards: VectorizedCliqueBoard, 
                       neural_network: ImprovedBatchedNeuralNetwork,
                       num_simulations: jnp.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        """Run MCTS for multiple games in parallel using threads."""
        import concurrent.futures
        
        action_probs = np.zeros((self.batch_size, 15))
        
        def run_game_mcts(game_idx):
            if boards.game_states[game_idx] != 0:
                return game_idx, np.zeros(15)
            
            # Create a single-game board view
            single_board = VectorizedCliqueBoard(batch_size=1)
            single_board.edge_states = boards.edge_states[game_idx:game_idx+1]
            single_board.current_players = boards.current_players[game_idx:game_idx+1]
            single_board.game_states = boards.game_states[game_idx:game_idx+1]
            single_board.move_counts = boards.move_counts[game_idx:game_idx+1]
            
            # Run MCTS
            mcts = TreeBasedMCTS(**self.mcts_kwargs)
            probs = mcts.search(
                single_board, 
                neural_network,
                int(num_simulations[game_idx]),
                0,
                temperature
            )
            return game_idx, probs
        
        # Run games in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(run_game_mcts, i) for i in range(self.batch_size)]
            
            for future in concurrent.futures.as_completed(futures):
                game_idx, probs = future.result()
                action_probs[game_idx] = probs
        
        return jnp.array(action_probs)
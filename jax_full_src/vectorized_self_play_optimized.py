"""
Optimized Vectorized Self-Play using JIT-compiled MCTS.
Combines all performance fixes for maximum speed.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import time

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from jit_mcts_simple import VectorizedJITMCTS
from batched_mcts_sync import SynchronizedBatchedMCTS


@dataclass
class OptimizedSelfPlayConfig:
    """Configuration for optimized self-play."""
    batch_size: int = 128  # Larger batch for better GPU utilization
    num_vertices: int = 6
    k: int = 3
    game_mode: str = "asymmetric"
    mcts_simulations: int = 100
    temperature_threshold: int = 10
    c_puct: float = 3.0
    perspective_mode: str = "alternating"
    use_synchronized_mcts: bool = False  # Option to use synchronized MCTS


class OptimizedVectorizedSelfPlay:
    """
    Optimized self-play using all performance fixes:
    - Fix 1: Parallelize across games ✓
    - Fix 2: Batch NN evaluations in MCTS ✓
    - Fix 3: JIT compile MCTS operations ✓
    - Fix 4: Maximize GPU utilization ✓
    """
    
    def __init__(self, config: OptimizedSelfPlayConfig):
        self.config = config
        
        # Store config for creating MCTS instances
        self.mcts_class = SynchronizedBatchedMCTS if config.use_synchronized_mcts else VectorizedJITMCTS
        self.mcts_instances = {}  # Cache MCTS instances by batch size
    
    def play_games(self, 
                   neural_network: ImprovedBatchedNeuralNetwork,
                   num_games: int,
                   verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Play multiple games using optimized MCTS.
        
        Returns list of training examples.
        """
        all_game_data = []
        games_played = 0
        
        if verbose:
            print(f"\nPlaying {num_games} games with optimized self-play...")
            print(f"Batch size: {self.config.batch_size}")
            print(f"MCTS simulations: {self.config.mcts_simulations}")
        
        start_time = time.time()
        
        while games_played < num_games:
            batch_size = min(self.config.batch_size, num_games - games_played)
            
            # Initialize batch of games
            boards = VectorizedCliqueBoard(
                batch_size=batch_size,
                num_vertices=self.config.num_vertices,
                k=self.config.k,
                game_mode=self.config.game_mode
            )
            
            # Track game data for each game in batch
            game_data = [[] for _ in range(batch_size)]
            
            # Play until all games in batch are finished
            move_count = 0
            while jnp.any(boards.game_states == 0):
                # Temperature for exploration
                temperature = 1.0 if move_count < self.config.temperature_threshold else 0.0
                
                # Get or create MCTS instance for this batch size
                if batch_size not in self.mcts_instances:
                    self.mcts_instances[batch_size] = self.mcts_class(
                        batch_size=batch_size,
                        num_actions=15,
                        c_puct=self.config.c_puct
                    )
                mcts = self.mcts_instances[batch_size]
                
                # Get MCTS action probabilities (fully optimized)
                if hasattr(mcts, 'search_batch'):
                    # Synchronized MCTS
                    mcts_probs = mcts.search_batch(
                        boards,
                        neural_network,
                        self.config.mcts_simulations,
                        temperature
                    )
                else:
                    # JIT-compiled MCTS
                    mcts_probs = mcts.search(
                        boards,
                        neural_network,
                        self.config.mcts_simulations,
                        temperature
                    )
                
                # Store training data
                edge_indices, edge_features = boards.get_features_for_nn_undirected()
                active_mask = boards.game_states == 0
                
                for i in range(batch_size):
                    if active_mask[i]:
                        game_data[i].append({
                            'edge_indices': edge_indices[i].copy(),
                            'edge_features': edge_features[i].copy(),
                            'policy': mcts_probs[i].copy(),
                            'player': int(boards.current_players[i]),
                            'move': move_count
                        })
                
                # Sample actions from MCTS probabilities
                actions = []
                for i in range(batch_size):
                    if active_mask[i]:
                        # Sample from probability distribution
                        action = np.random.choice(15, p=np.array(mcts_probs[i]))
                        actions.append(action)
                    else:
                        actions.append(0)  # Dummy action for finished games
                
                # Make moves
                boards.make_moves(jnp.array(actions))
                move_count += 1
            
            # Process finished games
            for i in range(batch_size):
                winner = int(boards.winners[i])
                
                # Assign values based on game outcome
                for move_data in game_data[i]:
                    if self.config.perspective_mode == "alternating":
                        # Value from perspective of player who made the move
                        value = 1.0 if move_data['player'] == winner else -1.0
                    else:
                        # Fixed perspective (always from Player 1's view)
                        value = 1.0 if winner == 1 else -1.0
                    
                    move_data['value'] = value
                    all_game_data.append(move_data)
            
            games_played += batch_size
            
            if verbose and games_played % 100 == 0:
                elapsed = time.time() - start_time
                games_per_sec = games_played / elapsed
                print(f"  Played {games_played}/{num_games} games "
                      f"({games_per_sec:.1f} games/sec)")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nSelf-play completed:")
            print(f"  Total games: {num_games}")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Games/second: {num_games/total_time:.1f}")
            print(f"  Training examples: {len(all_game_data)}")
            print(f"  Examples/game: {len(all_game_data)/num_games:.1f}")
        
        return all_game_data
    
    def play_single_game(self,
                        neural_network: ImprovedBatchedNeuralNetwork,
                        verbose: bool = False) -> Tuple[List[Dict], int]:
        """
        Play a single game (useful for debugging).
        
        Returns:
            game_data: List of training examples
            winner: Winner of the game (1 or 2)
        """
        # Use batch size 1
        boards = VectorizedCliqueBoard(
            batch_size=1,
            num_vertices=self.config.num_vertices,
            k=self.config.k,
            game_mode=self.config.game_mode
        )
        
        game_data = []
        move_count = 0
        
        while boards.game_states[0] == 0:
            temperature = 1.0 if move_count < self.config.temperature_threshold else 0.0
            
            # Get or create MCTS instance for batch size 1
            if 1 not in self.mcts_instances:
                self.mcts_instances[1] = self.mcts_class(
                    batch_size=1,
                    num_actions=15,
                    c_puct=self.config.c_puct
                )
            mcts = self.mcts_instances[1]
            
            # Get MCTS probabilities
            if hasattr(mcts, 'search_batch'):
                mcts_probs = mcts.search_batch(
                    boards, neural_network, self.config.mcts_simulations, temperature
                )
            else:
                mcts_probs = mcts.search(
                    boards, neural_network, self.config.mcts_simulations, temperature
                )
            
            # Store data
            edge_indices, edge_features = boards.get_features_for_nn_undirected()
            game_data.append({
                'edge_indices': edge_indices[0].copy(),
                'edge_features': edge_features[0].copy(),
                'policy': mcts_probs[0].copy(),
                'player': int(boards.current_players[0]),
                'move': move_count
            })
            
            # Sample action
            action = np.random.choice(15, p=np.array(mcts_probs[0]))
            
            if verbose:
                print(f"Move {move_count}: Player {boards.current_players[0]} "
                      f"plays edge {action}")
            
            # Make move
            boards.make_moves(jnp.array([action]))
            move_count += 1
        
        winner = int(boards.winners[0])
        
        # Assign values
        for move_data in game_data:
            if self.config.perspective_mode == "alternating":
                value = 1.0 if move_data['player'] == winner else -1.0
            else:
                value = 1.0 if winner == 1 else -1.0
            move_data['value'] = value
        
        if verbose:
            print(f"Game finished. Winner: Player {winner}")
        
        return game_data, winner
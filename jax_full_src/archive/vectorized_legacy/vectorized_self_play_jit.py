"""
JIT-optimized Vectorized Self-Play Implementation.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import time

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from vectorized_mcts_jit import JITVectorizedMCTS


@dataclass
class JITSelfPlayConfig:
    """Configuration for JIT-optimized self-play."""
    batch_size: int = 256
    num_vertices: int = 6
    k: int = 3
    game_mode: str = "asymmetric"
    mcts_simulations: int = 100
    temperature_threshold: int = 10
    c_puct: float = 1.0
    noise_weight: float = 0.25
    perspective_mode: str = "alternating"
    skill_variation: float = 0.0


class JITVectorizedSelfPlay:
    """JIT-optimized vectorized self-play implementation."""
    
    def __init__(self, config: JITSelfPlayConfig, neural_network: ImprovedBatchedNeuralNetwork):
        self.config = config
        self.nn = neural_network
        self.num_edges = config.num_vertices * (config.num_vertices - 1) // 2
        
        # Pre-compile the game step function
        self._play_step = jit(self._play_step_impl)
    
    def play_games(self, num_games: int, verbose: bool = True) -> List[List[Dict[str, Any]]]:
        """
        Play multiple games in parallel batches with JIT optimization.
        
        Args:
            num_games: Total number of games to play
            verbose: Whether to print progress
            
        Returns:
            List of game trajectories
        """
        if verbose:
            print(f"Generating {num_games} games using JIT-optimized parallel self-play")
            print(f"Batch size: {self.config.batch_size} games in parallel")
            print(f"Game mode: {self.config.game_mode}")
            print(f"Perspective: {self.config.perspective_mode}")
        
        all_games = []
        num_batches = (num_games + self.config.batch_size - 1) // self.config.batch_size
        
        total_start_time = time.time()
        
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            actual_batch_size = min(self.config.batch_size, num_games - batch_idx * self.config.batch_size)
            
            # Play one batch
            batch_games = self._play_batch_jit(actual_batch_size)
            all_games.extend(batch_games)
            
            batch_time = time.time() - batch_start_time
            if verbose:
                games_per_sec = actual_batch_size / batch_time
                print(f"Batch {batch_idx+1}/{num_batches}: {actual_batch_size} games in {batch_time:.2f}s ({games_per_sec:.1f} games/sec)")
        
        total_time = time.time() - total_start_time
        if verbose:
            total_positions = sum(len(game) for game in all_games)
            print(f"\nTotal: {len(all_games)} games in {total_time:.2f}s")
            print(f"Average: {len(all_games)/total_time:.1f} games/second")
            print(f"Generated {total_positions} positions ({total_positions/total_time:.0f} positions/sec)")
        
        return all_games
    
    def _play_batch_jit(self, batch_size: int) -> List[List[Dict[str, Any]]]:
        """Play a batch of games with JIT optimization."""
        # Initialize boards
        boards = VectorizedCliqueBoard(batch_size, self.config.num_vertices, 
                                     self.config.k, self.config.game_mode)
        
        # Initialize MCTS
        mcts = JITVectorizedMCTS(
            batch_size, 
            self.num_edges,
            c_puct=self.config.c_puct,
            noise_weight=self.config.noise_weight,
            perspective_mode=self.config.perspective_mode
        )
        
        # Generate simulation counts with skill variation
        base_sims = self.config.mcts_simulations
        if self.config.skill_variation > 0:
            rng = np.random.RandomState()
            min_sims = int(base_sims * (1 - self.config.skill_variation))
            max_sims = int(base_sims * (1 + self.config.skill_variation))
            
            # Assign different skill levels to each game
            p1_sims = rng.randint(min_sims, max_sims + 1, size=batch_size)
            p2_sims = rng.randint(min_sims, max_sims + 1, size=batch_size)
            sim_counts = jnp.stack([p1_sims, p2_sims], axis=1)
        else:
            sim_counts = jnp.full((batch_size, 2), base_sims)
        
        # Storage for game data
        game_trajectories = [[] for _ in range(batch_size)]
        
        # Play games
        move_count = 0
        max_moves = self.num_edges
        
        while jnp.any(boards.game_states == 0) and move_count < max_moves:
            # Store current position
            edge_indices, edge_features = boards.get_features_for_nn_undirected()
            valid_mask = boards.get_valid_moves_mask()
            current_players = boards.current_players
            
            # Determine simulation counts for current player
            current_sim_counts = jnp.where(
                current_players == 0,
                sim_counts[:, 0],
                sim_counts[:, 1]
            )
            
            # Get moves using JIT-optimized MCTS
            # Temperature annealing based on game progress (matching PyTorch)
            move_progress = move_count / max_moves
            if move_progress < 0.2:  # First 20% of the game
                temperature = 1.0  # High exploration
            elif move_progress < 0.4:  # Next 20% of the game
                temperature = 0.8  # Still good exploration
            elif move_progress < 0.6:  # Middle 20% of the game
                temperature = 0.5  # Balanced exploration/exploitation
            elif move_progress < 0.8:  # Next 20% of the game
                temperature = 0.2  # More exploitation
            else:  # Last 20% of the game
                temperature = 0.1  # Strong exploitation
            
            # Also reduce noise weight as the game progresses (matching PyTorch)
            current_noise_weight = self.config.noise_weight * (1.0 - move_progress)
            
            # Create MCTS with updated noise weight for this move
            mcts_with_noise = JITVectorizedMCTS(
                batch_size, 
                self.num_edges,
                c_puct=self.config.c_puct,
                noise_weight=current_noise_weight,
                perspective_mode=self.config.perspective_mode
            )
            
            action_probs = mcts_with_noise.search(boards, self.nn, current_sim_counts, temperature)
            
            # Sample actions
            if temperature > 0:
                rng_key = jax.random.PRNGKey(np.random.randint(0, 2**32))
                actions = jax.random.categorical(rng_key, jnp.log(action_probs + 1e-8))
            else:
                actions = jnp.argmax(action_probs, axis=1)
            
            # Store experiences for active games
            for game_idx in range(batch_size):
                if boards.game_states[game_idx] == 0:
                    # Prepare board state for storage
                    board_state = {
                        'edge_index': np.array(edge_indices[game_idx]),
                        'edge_attr': np.array(edge_features[game_idx])
                    }
                    
                    # Get value from current player's perspective
                    value = self._get_game_value(boards, game_idx, current_players[game_idx])
                    
                    experience = {
                        'board_state': board_state,
                        'policy': np.array(action_probs[game_idx]),
                        'value': value,
                        'player_role': int(current_players[game_idx])
                    }
                    
                    game_trajectories[game_idx].append(experience)
            
            # Make moves
            boards.make_moves(actions)
            move_count += 1
        
        # Assign final values
        for game_idx in range(batch_size):
            if len(game_trajectories[game_idx]) > 0:
                final_value = self._get_final_value(boards, game_idx)
                
                # Update all positions with final value based on perspective mode
                for exp in game_trajectories[game_idx]:
                    if self.config.perspective_mode == "alternating":
                        # Alternating: value from the player's perspective
                        if boards.game_states[game_idx] == 1:  # Player 0 wins
                            exp['value'] = 1.0 if exp['player_role'] == 0 else -1.0
                        elif boards.game_states[game_idx] == 2:  # Player 1 wins
                            exp['value'] = -1.0 if exp['player_role'] == 0 else 1.0
                        else:  # Draw
                            exp['value'] = 0.0
                    else:
                        # Fixed perspective: always from player 0
                        exp['value'] = final_value
        
        return [traj for traj in game_trajectories if len(traj) > 0]
    
    def _play_step_impl(self, boards: VectorizedCliqueBoard, mcts: JITVectorizedMCTS,
                        sim_counts: jnp.ndarray, temperature: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single game step (JIT-compiled)."""
        # This is called from within the JIT-compiled MCTS
        action_probs = mcts.search(boards, self.nn, sim_counts, temperature)
        
        # Sample or select actions
        if temperature > 0:
            rng_key = jax.random.PRNGKey(0)  # In practice, thread this through
            actions = jax.random.categorical(rng_key, jnp.log(action_probs + 1e-8))
        else:
            actions = jnp.argmax(action_probs, axis=1)
        
        return actions, action_probs
    
    def _get_game_value(self, boards: VectorizedCliqueBoard, game_idx: int, 
                        current_player: int) -> float:
        """Get intermediate game value (always from player 0's perspective)."""
        if self.config.perspective_mode == "alternating":
            # From current player's perspective, 0 = likely win, 1 = likely loss
            return 0.0
        else:
            # Fixed perspective (player 0)
            return 0.0
    
    def _get_final_value(self, boards: VectorizedCliqueBoard, game_idx: int) -> float:
        """Get final game value from player 0's perspective."""
        game_state = int(boards.game_states[game_idx])
        
        if game_state == 1:  # Player 0 wins
            return 1.0
        elif game_state == 2:  # Player 1 wins
            return -1.0
        else:  # Draw or unfinished
            return 0.0
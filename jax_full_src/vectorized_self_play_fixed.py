"""
Fixed Vectorized Self-Play using proper tree-based MCTS.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import time

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from tree_based_mcts import ParallelTreeBasedMCTS


@dataclass
class FixedSelfPlayConfig:
    """Configuration for fixed self-play."""
    batch_size: int = 64  # Reduced for tree-based MCTS
    num_vertices: int = 6
    k: int = 3
    game_mode: str = "asymmetric"
    mcts_simulations: int = 100
    temperature_threshold: int = 10
    c_puct: float = 3.0  # Matching PyTorch
    noise_weight: float = 0.25
    perspective_mode: str = "alternating"
    skill_variation: float = 0.0


class FixedVectorizedSelfPlay:
    """Fixed vectorized self-play using proper tree-based MCTS."""
    
    def __init__(self, config: FixedSelfPlayConfig, neural_network: ImprovedBatchedNeuralNetwork):
        self.config = config
        self.nn = neural_network
        self.num_edges = config.num_vertices * (config.num_vertices - 1) // 2
    
    def play_games(self, num_games: int, verbose: bool = True) -> List[List[Dict[str, Any]]]:
        """
        Play multiple games using proper tree-based MCTS.
        
        Args:
            num_games: Total number of games to play
            verbose: Whether to print progress
            
        Returns:
            List of game trajectories
        """
        if verbose:
            print(f"Generating {num_games} games using fixed tree-based MCTS")
            print(f"Batch size: {self.config.batch_size} games in parallel")
            print(f"Game mode: {self.config.game_mode}")
            print(f"Perspective: {self.config.perspective_mode}")
            print(f"C_PUCT: {self.config.c_puct}")
        
        all_games = []
        num_batches = (num_games + self.config.batch_size - 1) // self.config.batch_size
        
        total_start_time = time.time()
        
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            actual_batch_size = min(self.config.batch_size, num_games - batch_idx * self.config.batch_size)
            
            # Play one batch
            batch_games = self._play_batch(actual_batch_size)
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
    
    def _play_batch(self, batch_size: int) -> List[List[Dict[str, Any]]]:
        """Play a batch of games using tree-based MCTS."""
        # Initialize boards
        boards = VectorizedCliqueBoard(batch_size, self.config.num_vertices, 
                                     self.config.k, self.config.game_mode)
        
        # Generate simulation counts with skill variation
        base_sims = self.config.mcts_simulations
        if self.config.skill_variation > 0:
            rng = np.random.RandomState()
            min_sims = int(base_sims * (1 - self.config.skill_variation))
            max_sims = int(base_sims * (1 + self.config.skill_variation))
            
            # Assign different skill levels to each game
            p1_sims = rng.randint(min_sims, max_sims + 1, size=batch_size)
            p2_sims = rng.randint(min_sims, max_sims + 1, size=batch_size)
            sim_counts = np.stack([p1_sims, p2_sims], axis=1)
        else:
            sim_counts = np.full((batch_size, 2), base_sims)
        
        # Storage for game data
        game_trajectories = [[] for _ in range(batch_size)]
        
        # Play games
        move_count = 0
        max_moves = self.num_edges
        
        while np.any(boards.game_states == 0) and move_count < max_moves:
            # Store current position
            edge_indices, edge_features = boards.get_features_for_nn_undirected()
            valid_mask = boards.get_valid_moves_mask()
            current_players = boards.current_players
            
            # Determine simulation counts for current player
            current_sim_counts = np.where(
                current_players == 0,
                sim_counts[:, 0],
                sim_counts[:, 1]
            )
            
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
            mcts = ParallelTreeBasedMCTS(
                batch_size,
                num_actions=self.num_edges,
                c_puct=self.config.c_puct,
                noise_weight=current_noise_weight,
                perspective_mode=self.config.perspective_mode
            )
            
            # Run MCTS for all active games
            action_probs = mcts.search(boards, self.nn, current_sim_counts, temperature)
            
            # Sample actions
            actions = np.zeros(batch_size, dtype=np.int32)
            for i in range(batch_size):
                if boards.game_states[i] == 0:  # Active game
                    probs = action_probs[i]
                    # Ensure probabilities are normalized
                    probs = probs / (probs.sum() + 1e-8)
                    
                    if temperature > 0 and probs.sum() > 0:
                        # Sample from distribution
                        try:
                            actions[i] = np.random.choice(self.num_edges, p=probs)
                        except ValueError:
                            # Fallback to uniform if probabilities are invalid
                            valid_mask = boards.get_valid_moves_mask()[i]
                            valid_actions = np.where(valid_mask)[0]
                            if len(valid_actions) > 0:
                                actions[i] = np.random.choice(valid_actions)
                    else:
                        # Deterministic
                        actions[i] = np.argmax(probs)
            
            # Store experiences for active games
            for game_idx in range(batch_size):
                if boards.game_states[game_idx] == 0:
                    # Prepare board state for storage
                    board_state = {
                        'edge_index': np.array(edge_indices[game_idx]),
                        'edge_attr': np.array(edge_features[game_idx])
                    }
                    
                    # Get value placeholder (will be filled at game end)
                    value = 0.0
                    
                    experience = {
                        'board_state': board_state,
                        'policy': np.array(action_probs[game_idx]),
                        'value': value,
                        'player_role': int(current_players[game_idx])
                    }
                    
                    game_trajectories[game_idx].append(experience)
            
            # Make moves
            boards.make_moves(jnp.array(actions))
            move_count += 1
        
        # Assign final values based on game outcomes
        for game_idx in range(batch_size):
            if len(game_trajectories[game_idx]) > 0:
                game_state = int(boards.game_states[game_idx])
                
                # Update all positions with final value based on perspective mode
                for exp in game_trajectories[game_idx]:
                    if self.config.perspective_mode == "alternating":
                        # Alternating: value from the player's perspective
                        if game_state == 1:  # Player 0 wins
                            exp['value'] = 1.0 if exp['player_role'] == 0 else -1.0
                        elif game_state == 2:  # Player 1 wins
                            exp['value'] = -1.0 if exp['player_role'] == 0 else 1.0
                        else:  # Draw
                            exp['value'] = 0.0
                    else:
                        # Fixed perspective: always from player 0
                        if game_state == 1:  # Player 0 wins
                            exp['value'] = 1.0
                        elif game_state == 2:  # Player 1 wins
                            exp['value'] = -1.0
                        else:  # Draw
                            exp['value'] = 0.0
        
        return [traj for traj in game_trajectories if len(traj) > 0]
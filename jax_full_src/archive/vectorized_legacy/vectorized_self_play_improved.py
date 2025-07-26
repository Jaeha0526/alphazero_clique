#!/usr/bin/env python
"""
Improved Vectorized Self-Play Implementation with all features from improved-alphazero branch.
Key improvements:
- Skill variation for reducing draws
- Perspective modes (fixed/alternating)
- Support for asymmetric game mode
- Player role tracking
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import time
import pickle
import os
from datetime import datetime

from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from vectorized_mcts_improved import ImprovedVectorizedMCTS


@dataclass
class ImprovedSelfPlayConfig:
    """Configuration for improved vectorized self-play."""
    batch_size: int = 256  # Number of games to play in parallel
    num_vertices: int = 6
    k: int = 3
    game_mode: str = "asymmetric"
    mcts_simulations: int = 100
    temperature_threshold: int = 10  # Use temperature=1 for first N moves
    c_puct: float = 1.0
    max_moves: int = 50  # Maximum moves per game
    perspective_mode: str = "alternating"  # "fixed" or "alternating"
    skill_variation: float = 0.0  # Variation in MCTS sims (0.0 = no variation)
    dirichlet_alpha: float = 0.3
    noise_weight: float = 0.25


def get_varied_mcts_sims(base_sims: int, skill_variation: float, 
                        batch_size: int, rng_key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get varied MCTS simulation counts for each player in each game.
    
    Args:
        base_sims: Base number of simulations
        skill_variation: Variation factor (0.0 to 1.0)
        batch_size: Number of games
        rng_key: JAX random key
        
    Returns:
        player1_sims: (batch_size,) simulation counts for player 1
        player2_sims: (batch_size,) simulation counts for player 2
    """
    if skill_variation <= 0:
        # No variation - all games use same counts
        return (jnp.full(batch_size, base_sims), 
                jnp.full(batch_size, base_sims))
    
    # Clamp variation to reasonable bounds
    skill_variation = jnp.minimum(skill_variation, 0.8)  # Max 80% variation
    
    # Calculate bounds
    min_factor = 1.0 - skill_variation
    max_factor = 1.0 + skill_variation
    
    # Generate random factors for each game and player
    key1, key2 = jax.random.split(rng_key)
    p1_factors = jax.random.uniform(key1, (batch_size,), minval=min_factor, maxval=max_factor)
    p2_factors = jax.random.uniform(key2, (batch_size,), minval=min_factor, maxval=max_factor)
    
    # Calculate simulation counts
    player1_sims = jnp.round(base_sims * p1_factors).astype(jnp.int32)
    player2_sims = jnp.round(base_sims * p2_factors).astype(jnp.int32)
    
    # Ensure minimum simulations
    player1_sims = jnp.maximum(player1_sims, 10)
    player2_sims = jnp.maximum(player2_sims, 10)
    
    return player1_sims, player2_sims


class ImprovedVectorizedSelfPlay:
    """
    Improved parallel self-play with all features from improved-alphazero branch.
    """
    
    def __init__(self, config: ImprovedSelfPlayConfig, neural_network: ImprovedBatchedNeuralNetwork):
        self.config = config
        self.neural_network = neural_network
        
        # Initialize RNG
        self.rng = jax.random.PRNGKey(42)
        
        # Storage for experiences
        self.reset_storage()
    
    def reset_storage(self):
        """Reset experience storage."""
        self.all_experiences = []
        self.game_experiences = [[] for _ in range(self.config.batch_size)]
    
    def play_games(self, num_games: int, verbose: bool = True) -> List[List[Dict]]:
        """
        Play multiple batches of games in parallel.
        
        Args:
            num_games: Total number of games to generate
            verbose: Whether to print progress
            
        Returns:
            List of experience lists, one per game
        """
        if verbose:
            print(f"Generating {num_games} games using IMPROVED parallel self-play")
            print(f"Batch size: {self.config.batch_size} games in parallel")
            print(f"Game mode: {self.config.game_mode}")
            print(f"Perspective: {self.config.perspective_mode}")
            if self.config.skill_variation > 0:
                print(f"Skill variation: ±{self.config.skill_variation*100:.0f}%")
        
        num_batches = (num_games + self.config.batch_size - 1) // self.config.batch_size
        all_experiences = []
        
        total_start = time.time()
        
        for batch_idx in range(num_batches):
            batch_start = time.time()
            
            # Determine actual batch size (last batch might be smaller)
            actual_batch_size = min(self.config.batch_size, 
                                  num_games - batch_idx * self.config.batch_size)
            
            if actual_batch_size < self.config.batch_size:
                # For last batch, create smaller config
                config = ImprovedSelfPlayConfig(
                    batch_size=actual_batch_size,
                    num_vertices=self.config.num_vertices,
                    k=self.config.k,
                    game_mode=self.config.game_mode,
                    mcts_simulations=self.config.mcts_simulations,
                    temperature_threshold=self.config.temperature_threshold,
                    c_puct=self.config.c_puct,
                    perspective_mode=self.config.perspective_mode,
                    skill_variation=self.config.skill_variation
                )
                temp_self_play = ImprovedVectorizedSelfPlay(config, self.neural_network)
                batch_experiences = temp_self_play.play_batch()
            else:
                batch_experiences = self.play_batch()
            
            all_experiences.extend(batch_experiences)
            
            batch_time = time.time() - batch_start
            if verbose:
                games_in_batch = len(batch_experiences)
                print(f"Batch {batch_idx + 1}/{num_batches}: "
                      f"{games_in_batch} games in {batch_time:.2f}s "
                      f"({games_in_batch/batch_time:.1f} games/sec)")
        
        total_time = time.time() - total_start
        
        if verbose:
            print(f"\nTotal: {len(all_experiences)} games in {total_time:.2f}s")
            print(f"Average: {len(all_experiences)/total_time:.1f} games/second")
            
            # Calculate positions per second
            total_positions = sum(len(exp) for exp in all_experiences)
            print(f"Generated {total_positions} positions "
                  f"({total_positions/total_time:.0f} positions/sec)")
        
        return all_experiences
    
    def play_batch(self) -> List[List[Dict]]:
        """
        Play a single batch of games completely in parallel with improved features.
        
        Returns:
            List of experience lists, one per game
        """
        # Initialize boards
        boards = VectorizedCliqueBoard(
            self.config.batch_size,
            self.config.num_vertices,
            self.config.k,
            self.config.game_mode
        )
        
        # Reset storage
        self.reset_storage()
        
        # Track game statistics
        games_active = jnp.ones(self.config.batch_size, dtype=jnp.bool_)
        move_counts = jnp.zeros(self.config.batch_size, dtype=jnp.int32)
        
        # Get skill-varied MCTS simulations for each player
        self.rng, sim_key = jax.random.split(self.rng)
        player1_sims, player2_sims = get_varied_mcts_sims(
            self.config.mcts_simulations, 
            self.config.skill_variation,
            self.config.batch_size,
            sim_key
        )
        
        # Play all games until completion
        for move in range(self.config.max_moves):
            if not jnp.any(games_active):
                break
            
            # Get current players for each game
            current_players = boards.current_players
            
            # Get player roles for asymmetric mode
            if self.config.game_mode == "asymmetric":
                player_roles = current_players  # 0 = attacker, 1 = defender
            else:
                player_roles = None
            
            # Get features for neural network (using improved undirected format)
            edge_indices, edge_features = boards.get_features_for_nn_undirected()
            
            # Get valid moves
            valid_moves_mask = boards.get_valid_moves_mask()
            
            # Only process active games
            active_mask = games_active[:, None]
            masked_valid_moves = valid_moves_mask * active_mask
            
            # Run MCTS for each game with appropriate simulation count
            # For each game, use the simulation count for the current player
            current_sims = jnp.where(current_players == 0, player1_sims, player2_sims)
            
            # Create MCTS with perspective mode
            mcts = ImprovedVectorizedMCTS(
                self.config.batch_size,
                boards.num_edges,
                c_puct=self.config.c_puct,
                dirichlet_alpha=self.config.dirichlet_alpha,
                noise_weight=self.config.noise_weight if move == 0 else 0.0,
                perspective_mode=self.config.perspective_mode
            )
            
            # Determine temperature for this move
            temperature = jnp.where(move_counts < self.config.temperature_threshold, 1.0, 0.0)
            
            # Run MCTS search with per-game simulation counts
            action_probs = mcts.search(
                boards,
                self.neural_network,
                current_sims,  # Now supports per-game counts!
                temperature=float(jnp.mean(temperature))  # Use average temperature for now
            )
            
            # Store experiences before making moves
            self._store_experiences(
                boards, action_probs, games_active, 
                edge_indices, edge_features, player_roles
            )
            
            # Apply temperature to action selection
            temperature = jnp.where(move_counts < self.config.temperature_threshold, 1.0, 0.0)
            
            # Sample actions based on temperature
            self.rng, action_key = jax.random.split(self.rng)
            selected_actions = self._sample_actions(
                action_probs, temperature, games_active, action_key
            )
            
            # Make moves
            boards.make_moves(selected_actions * games_active)
            
            # Update game states
            games_active = (boards.game_states == 0)
            move_counts = move_counts + games_active
        
        # Process final experiences with game outcomes
        self._finalize_experiences(boards)
        
        # Return experiences for each game
        return [exp for exp in self.game_experiences if len(exp) > 0]
    
    def _store_experiences(self, boards: VectorizedCliqueBoard, 
                          action_probs: jnp.ndarray,
                          games_active: jnp.ndarray,
                          edge_indices: jnp.ndarray,
                          edge_features: jnp.ndarray,
                          player_roles: Optional[jnp.ndarray]):
        """Store experiences from current positions."""
        for game_idx in range(self.config.batch_size):
            if not games_active[game_idx]:
                continue
            
            # Create experience
            experience = {
                'board_state': {
                    'edge_index': np.array(edge_indices[game_idx]),
                    'edge_attr': np.array(edge_features[game_idx]),
                    'num_vertices': boards.num_vertices,
                    'edge_states': np.array(boards.edge_states[game_idx])
                },
                'policy': np.array(action_probs[game_idx]),
                'value': None,  # Will be filled after game ends
                'player': int(boards.current_players[game_idx]),
                'player_role': int(player_roles[game_idx]) if player_roles is not None else None
            }
            
            self.game_experiences[game_idx].append(experience)
    
    def _sample_actions(self, action_probs: jnp.ndarray, 
                       temperature: jnp.ndarray,
                       games_active: jnp.ndarray,
                       rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """Sample actions based on temperature."""
        actions = jnp.zeros(self.config.batch_size, dtype=jnp.int32)
        
        for game_idx in range(self.config.batch_size):
            if not games_active[game_idx]:
                continue
            
            probs = action_probs[game_idx]
            
            if temperature[game_idx] > 0:
                # Sample from distribution
                key = jax.random.fold_in(rng_key, game_idx)
                action = jax.random.choice(key, len(probs), p=probs)
            else:
                # Select best action
                action = jnp.argmax(probs)
            
            actions = actions.at[game_idx].set(action)
        
        return actions
    
    def _finalize_experiences(self, boards: VectorizedCliqueBoard):
        """Add final values to experiences based on game outcomes."""
        for game_idx in range(self.config.batch_size):
            if len(self.game_experiences[game_idx]) == 0:
                continue
            
            # Get final game state
            game_state = int(boards.game_states[game_idx])
            
            # Calculate values for each position
            for exp in self.game_experiences[game_idx]:
                player = exp['player']
                
                if self.config.perspective_mode == "fixed":
                    # Value from Player 1's perspective
                    if game_state == 1:  # Player 1 wins
                        value = 1.0
                    elif game_state == 2:  # Player 2 wins
                        value = -1.0
                    else:  # Draw
                        value = 0.0
                else:  # alternating
                    # Value from current player's perspective
                    if game_state == 0 or game_state == 3:  # Ongoing or draw
                        value = 0.0
                    elif game_state == player + 1:  # Current player wins
                        value = 1.0
                    else:  # Current player loses
                        value = -1.0
                
                exp['value'] = value


# Backward compatibility
SelfPlayConfig = ImprovedSelfPlayConfig
VectorizedSelfPlay = ImprovedVectorizedSelfPlay


if __name__ == "__main__":
    print("Testing Improved Vectorized Self-Play...")
    print("="*60)
    
    # Test configuration
    config = ImprovedSelfPlayConfig(
        batch_size=32,
        num_vertices=6,
        k=3,
        game_mode="asymmetric",
        mcts_simulations=50,
        perspective_mode="alternating",
        skill_variation=0.3
    )
    
    # Create neural network
    nn = ImprovedBatchedNeuralNetwork(asymmetric_mode=True)
    
    # Create self-play
    self_play = ImprovedVectorizedSelfPlay(config, nn)
    
    # Play games
    print("Playing games with improved features...")
    experiences = self_play.play_games(64, verbose=True)
    
    print(f"\nGenerated {len(experiences)} games")
    print(f"First game has {len(experiences[0])} positions")
    
    # Check for role tracking
    if experiences[0][0]['player_role'] is not None:
        print("✓ Player roles tracked (asymmetric mode)")
    
    print("\n" + "="*60)
    print("✓ Improved Self-Play Implementation Complete!")
    print("="*60)
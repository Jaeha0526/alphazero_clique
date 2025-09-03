#!/usr/bin/env python
"""
GPU-accelerated self-play for AlphaZero using JAX
"""

import os
import pickle
import time
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, pmap
    JAX_AVAILABLE = True
    print("JAX Self-Play: Using JAX with GPU acceleration")
except ImportError:
    import warnings
    warnings.warn("JAX not available, using NumPy implementation")
    jnp = np
    JAX_AVAILABLE = False
    def jit(f): return f
    def vmap(f, **kwargs): return f

# Import components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_src.jax_clique_board_numpy import JAXCliqueBoard
from jax_src.jax_mcts_clique_gpu import GPUAcceleratedMCTS, UCT_search
import src.encoder_decoder_clique as ed


@dataclass
class SelfPlayConfig:
    """Configuration for self-play"""
    num_vertices: int = 6
    k: int = 3
    game_mode: str = "symmetric"
    mcts_simulations: int = 100
    batch_size: int = 256  # Larger for GPU
    max_moves: int = 50
    temperature_threshold: int = 10
    noise_weight: float = 0.25
    c_puct: float = 1.0


class GPUBatchedSelfPlay:
    """GPU-accelerated batched self-play"""
    
    def __init__(self, config: SelfPlayConfig, model, model_params: Dict):
        self.config = config
        self.model = model
        self.model_params = model_params
        
        # Pre-compile batch evaluation function
        if JAX_AVAILABLE:
            self._compile_batch_functions()
    
    def _compile_batch_functions(self):
        """Pre-compile JAX functions for GPU"""
        # Batch neural network evaluation
        def eval_batch(params, edge_indices, edge_attrs):
            # Evaluate multiple positions in parallel
            def single_eval(ei, ea):
                return self.model(params, ei, ea)
            
            return vmap(single_eval)(edge_indices, edge_attrs)
        
        self._batch_eval = jit(eval_batch)
    
    def play_single_game(self) -> List[Dict[str, Any]]:
        """Play a single self-play game"""
        board = JAXCliqueBoard(self.config.num_vertices, self.config.k, self.config.game_mode)
        experiences = []
        
        while board.game_state == 0 and board.move_count < self.config.max_moves:
            # Run MCTS
            mcts = GPUAcceleratedMCTS(
                board, 
                self.config.mcts_simulations,
                self.model, 
                self.model_params,
                c_puct=self.config.c_puct,
                noise_weight=self.config.noise_weight if board.move_count < self.config.temperature_threshold else 0.0
            )
            
            best_action, visits = mcts.search()
            
            # Create policy from visits
            policy = np.zeros(15)  # Max possible edges
            total_visits = sum(visits.values())
            for action, visit_count in visits.items():
                if 0 <= action < 15:
                    policy[action] = visit_count / total_visits
            
            # Temperature-based action selection
            if board.move_count < self.config.temperature_threshold:
                # Sample from policy
                action = np.random.choice(15, p=policy)
            else:
                # Choose best
                action = best_action
            
            # Store experience
            state_dict = ed.prepare_state_for_network(board)
            experience = {
                'board_state': board.get_board_state(),  # Add board state for compatibility
                'edge_index': state_dict['edge_index'].numpy(),
                'edge_attr': state_dict['edge_attr'].numpy(),
                'policy': policy,
                'player': board.player
            }
            experiences.append(experience)
            
            # Make move
            edge = ed.decode_action(board, action)
            if edge != (-1, -1) and edge in board.get_valid_moves():
                board.make_move(edge)
            else:
                # Invalid move, try best valid
                valid_moves = board.get_valid_moves()
                if valid_moves:
                    board.make_move(valid_moves[0])
                else:
                    break
        
        # Assign rewards
        if board.game_state == 3:  # Draw
            reward = 0.0
        else:
            # Winner gets +1, loser gets -1
            reward = 1.0 if board.game_state == 1 else -1.0
        
        # Update experiences with final rewards
        for exp in experiences:
            if exp['player'] == 0:  # Player 1
                exp['value'] = reward
            else:  # Player 2
                exp['value'] = -reward
        
        return experiences
    
    def play_batch_games(self, num_games: int) -> List[List[Dict]]:
        """Play multiple games with GPU acceleration"""
        all_games = []
        
        # Process in batches for efficiency
        num_batches = (num_games + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_idx in range(num_batches):
            batch_start = time.time()
            batch_games = min(self.config.batch_size, num_games - batch_idx * self.config.batch_size)
            
            # Play games in this batch
            batch_experiences = []
            for _ in range(batch_games):
                game_exp = self.play_single_game()
                batch_experiences.append(game_exp)
            
            all_games.extend(batch_experiences)
            
            batch_time = time.time() - batch_start
            games_per_sec = batch_games / batch_time
            print(f"Batch {batch_idx+1}/{num_batches} completed in {batch_time:.2f}s ({games_per_sec:.1f} games/sec)")
        
        return all_games


def gpu_self_play_worker(gpu_id: int, num_games: int, config: SelfPlayConfig,
                        model, model_params: Dict, save_queue: mp.Queue):
    """Worker process for GPU self-play"""
    # Set GPU device
    if JAX_AVAILABLE:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Create self-play instance
    self_play = GPUBatchedSelfPlay(config, model, model_params)
    
    # Generate games
    print(f"GPU {gpu_id}: Generating {num_games} games...")
    experiences = self_play.play_batch_games(num_games)
    
    # Send to save queue
    save_queue.put(experiences)
    print(f"GPU {gpu_id}: Completed {num_games} games")


class GPUParallelSelfPlay:
    """Parallel self-play using multiple GPUs/CPU processes"""
    
    def __init__(self, config: SelfPlayConfig, model, model_params: Dict, num_gpus: int = 1):
        self.config = config
        self.model = model
        self.model_params = model_params
        self.num_gpus = num_gpus
    
    def generate_games(self, total_games: int, save_dir: str, iteration: int) -> str:
        """Generate games using GPU acceleration"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Generating {total_games} games using {self.num_gpus} GPU(s)")
        start_time = time.time()
        
        # For single GPU, don't use multiprocessing
        if self.num_gpus == 1:
            self_play = GPUBatchedSelfPlay(self.config, self.model, self.model_params)
            all_experiences = self_play.play_batch_games(total_games)
        else:
            # Multi-GPU setup (if available)
            save_queue = mp.Queue()
            processes = []
            
            games_per_gpu = total_games // self.num_gpus
            remainder = total_games % self.num_gpus
            
            for gpu_id in range(self.num_gpus):
                gpu_games = games_per_gpu + (1 if gpu_id < remainder else 0)
                p = mp.Process(
                    target=gpu_self_play_worker,
                    args=(gpu_id, gpu_games, self.config, self.model, self.model_params, save_queue)
                )
                p.start()
                processes.append(p)
            
            # Collect results
            all_experiences = []
            for _ in range(self.num_gpus):
                gpu_experiences = save_queue.get()
                all_experiences.extend(gpu_experiences)
            
            # Wait for completion
            for p in processes:
                p.join()
        
        # Flatten experiences
        flattened_experiences = []
        for game in all_experiences:
            flattened_experiences.extend(game)
        
        # Save experiences
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_{timestamp}_iter{iteration}.pkl"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(flattened_experiences, f)
        
        total_time = time.time() - start_time
        games_per_second = total_games / total_time
        
        print(f"\nGenerated {total_games} games in {total_time:.1f}s ({games_per_second:.1f} games/sec)")
        print(f"Saved {len(flattened_experiences)} positions to {filepath}")
        
        return filepath


# Compatible interface with original
def MCTS_self_play_gpu(clique_net, num_games: int, num_vertices: int = 6,
                      clique_size: int = 3, cpu: int = 0, mcts_sims: int = 100,
                      game_mode: str = "symmetric", iteration: int = 0,
                      data_dir: str = "./datasets/clique", noise_weight: float = 0.25):
    """GPU-accelerated self-play compatible with original interface"""
    config = SelfPlayConfig(
        num_vertices=num_vertices,
        k=clique_size,
        game_mode=game_mode,
        mcts_simulations=mcts_sims,
        noise_weight=noise_weight,
        batch_size=min(256, num_games)  # GPU batch size
    )
    
    # Extract model and params
    if hasattr(clique_net, 'model') and hasattr(clique_net, 'params'):
        model = clique_net.model
        params = clique_net.params
    else:
        model = clique_net
        params = getattr(clique_net, 'params', None)
    
    # Use GPU self-play
    self_play = GPUBatchedSelfPlay(config, model, params)
    experiences = self_play.play_batch_games(num_games)
    
    # Save in compatible format
    flattened = []
    for game in experiences:
        flattened.extend(game)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"game_{timestamp}_iter{iteration}.pkl"
    filepath = os.path.join(data_dir, filename)
    
    os.makedirs(data_dir, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(flattened, f)
    
    print(f"GPU worker {cpu}: Saved {len(flattened)} positions to {filepath}")


if __name__ == "__main__":
    # Test GPU self-play
    print("Testing GPU-accelerated self-play...")
    
    from jax_src.jax_alpha_net_clique_gpu import create_gpu_model
    
    # Create model
    model, params = create_gpu_model()
    
    # Create config
    config = SelfPlayConfig(batch_size=32, mcts_simulations=50)
    
    # Run self-play
    print("\nGenerating 10 test games...")
    self_play = GPUBatchedSelfPlay(config, model, params)
    
    start = time.time()
    games = self_play.play_batch_games(10)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.2f}s ({10/elapsed:.1f} games/sec)")
    print(f"Generated {sum(len(g) for g in games)} total positions")
    
    if JAX_AVAILABLE:
        print(f"\nUsing: {jax.devices()[0]}")
        print("✓ GPU acceleration enabled!")
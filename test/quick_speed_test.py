#!/usr/bin/env python
"""
Quick speed comparison between PyTorch and JAX implementations
Focuses on core operations without full pipeline
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

# Add paths for imports
sys.path.append('/workspace/alphazero_clique/src')
sys.path.append('/workspace/alphazero_clique/jax_full_src')

def test_pytorch_core():
    """Test PyTorch core operations"""
    print("\n" + "="*60)
    print("PYTORCH CORE TEST (n=6, k=3)")
    print("="*60)
    
    try:
        from clique_board import CliqueBoard
        from alpha_net_clique import CliqueGNN
        from MCTS_clique import MCTS
        import torch
        
        # Setup
        game = CliqueBoard(6, 3)
        model = CliqueGNN(6, hidden_dim=64, num_layers=2)
        model.eval()
        
        # Test 1: Board operations
        board = game.get_init_board()
        
        start = time.time()
        for _ in range(100):
            valid_moves = game.get_valid_moves(board)
            if valid_moves.any():
                move = np.random.choice(np.where(valid_moves)[0])
                board = game.get_next_state(board, move)
                game.get_game_ended(board)
        board_time = time.time() - start
        print(f"✓ Board operations (100 steps): {board_time*1000:.1f}ms")
        
        # Test 2: Neural network inference
        board = game.get_init_board()
        board_tensor = torch.FloatTensor(board).unsqueeze(0)
        
        with torch.no_grad():
            start = time.time()
            for _ in range(100):
                policy, value = model(board_tensor)
            nn_time = time.time() - start
        print(f"✓ Neural network (100 inferences): {nn_time*1000:.1f}ms")
        
        # Test 3: Single MCTS search
        board = game.get_init_board()
        mcts = MCTS(game, model, cpuct=3.0)
        
        start = time.time()
        probs = mcts.get_action_prob(board, temp=1, mcts_simulations=10)
        mcts_time = time.time() - start
        print(f"✓ MCTS search (10 simulations): {mcts_time*1000:.1f}ms")
        
        return {
            "board_ops_ms": board_time * 1000,
            "nn_inference_ms": nn_time * 1000,
            "mcts_10sims_ms": mcts_time * 1000,
            "success": True
        }
        
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return {"success": False, "error": str(e)}

def test_jax_core():
    """Test JAX core operations"""
    print("\n" + "="*60)
    print("JAX CORE TEST (n=6, k=3)")
    print("="*60)
    
    # Set CPU mode for fair comparison
    os.environ['JAX_PLATFORMS'] = 'cpu'
    
    try:
        import jax
        import jax.numpy as jnp
        from vectorized_board import VectorizedCliqueBoard
        from vectorized_nn import ImprovedBatchedNeuralNetwork
        from mctx_true_jax import MCTXTrueJAX
        
        # Setup
        model = ImprovedBatchedNeuralNetwork(
            num_vertices=6,
            hidden_dim=64,
            num_layers=2
        )
        
        # Test 1: Board operations (vectorized)
        board = VectorizedCliqueBoard(batch_size=1, num_vertices=6, k=3)
        
        start = time.time()
        for _ in range(100):
            valid_moves = board.get_valid_moves()
            if valid_moves[0].any():
                move_idx = np.random.choice(np.where(valid_moves[0])[0])
                board.make_move(0, move_idx)
                board.get_game_ended()
        board_time = time.time() - start
        print(f"✓ Board operations (100 steps): {board_time*1000:.1f}ms")
        
        # Test 2: Neural network inference
        board = VectorizedCliqueBoard(batch_size=1, num_vertices=6, k=3)
        state = board.get_observation()
        
        start = time.time()
        for _ in range(100):
            policy, value = model(state)
            policy.block_until_ready()  # Ensure computation completes
        nn_time = time.time() - start
        print(f"✓ Neural network (100 inferences): {nn_time*1000:.1f}ms")
        
        # Test 3: Single MCTS search
        board = VectorizedCliqueBoard(batch_size=1, num_vertices=6, k=3)
        mcts = MCTXTrueJAX(
            batch_size=1,
            num_actions=15,
            max_nodes=11,
            c_puct=3.0,
            num_vertices=6
        )
        
        start = time.time()
        probs = mcts.search(board, model, 10, temperature=1.0)
        probs.block_until_ready()  # Ensure computation completes
        mcts_time = time.time() - start
        print(f"✓ MCTS search (10 simulations): {mcts_time*1000:.1f}ms")
        
        # Test 4: Batch processing advantage
        board8 = VectorizedCliqueBoard(batch_size=8, num_vertices=6, k=3)
        mcts8 = MCTXTrueJAX(
            batch_size=8,
            num_actions=15,
            max_nodes=11,
            c_puct=3.0,
            num_vertices=6
        )
        
        start = time.time()
        probs = mcts8.search(board8, model, 10, temperature=1.0)
        probs.block_until_ready()
        batch_time = time.time() - start
        print(f"✓ Batch MCTS (8 games, 10 sims): {batch_time*1000:.1f}ms")
        print(f"  Per-game average: {batch_time*1000/8:.1f}ms")
        
        return {
            "board_ops_ms": board_time * 1000,
            "nn_inference_ms": nn_time * 1000,
            "mcts_10sims_ms": mcts_time * 1000,
            "batch_8games_ms": batch_time * 1000,
            "batch_per_game_ms": batch_time * 1000 / 8,
            "success": True
        }
        
    except Exception as e:
        print(f"✗ JAX test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def test_self_play_speed():
    """Compare self-play game generation speed"""
    print("\n" + "="*60)
    print("SELF-PLAY SPEED TEST (5 games, 10 MCTS sims)")
    print("="*60)
    
    os.environ['JAX_PLATFORMS'] = 'cpu'
    
    # PyTorch self-play
    print("\nPyTorch self-play:")
    try:
        from clique_board import CliqueBoard
        from alpha_net_clique import CliqueGNN
        from MCTS_clique import MCTS
        import torch
        
        game = CliqueBoard(6, 3)
        model = CliqueGNN(6, hidden_dim=64, num_layers=2)
        model.eval()
        
        start = time.time()
        games_data = []
        
        for _ in range(5):
            board = game.get_init_board()
            mcts = MCTS(game, model, cpuct=3.0)
            game_data = []
            
            while game.get_game_ended(board) == 0:
                probs = mcts.get_action_prob(board, temp=1, mcts_simulations=10)
                action = np.random.choice(len(probs), p=probs)
                board = game.get_next_state(board, action)
                game_data.append((board, probs))
            
            games_data.append(game_data)
        
        pytorch_time = time.time() - start
        total_moves = sum(len(g) for g in games_data)
        print(f"  ✓ Generated 5 games in {pytorch_time:.2f}s")
        print(f"    Total moves: {total_moves}")
        print(f"    Time per game: {pytorch_time/5:.2f}s")
        print(f"    Time per move: {pytorch_time/total_moves*1000:.1f}ms")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        pytorch_time = None
    
    # JAX self-play (batched)
    print("\nJAX self-play (batched):")
    try:
        from vectorized_board import VectorizedCliqueBoard
        from vectorized_nn import ImprovedBatchedNeuralNetwork
        from mctx_true_jax import MCTXTrueJAX
        import jax.numpy as jnp
        
        model = ImprovedBatchedNeuralNetwork(
            num_vertices=6,
            hidden_dim=64,
            num_layers=2
        )
        
        # Process 5 games in parallel
        board = VectorizedCliqueBoard(batch_size=5, num_vertices=6, k=3)
        mcts = MCTXTrueJAX(
            batch_size=5,
            num_actions=15,
            max_nodes=11,
            c_puct=3.0,
            num_vertices=6
        )
        
        start = time.time()
        games_data = [[] for _ in range(5)]
        active_games = np.ones(5, dtype=bool)
        move_count = 0
        
        while active_games.any():
            # Get actions for all active games
            probs = mcts.search(board, model, 10, temperature=1.0)
            probs.block_until_ready()
            
            # Apply moves for active games
            for i in range(5):
                if active_games[i]:
                    game_probs = probs[i]
                    action = np.random.choice(15, p=game_probs)
                    board.make_move(i, action)
                    games_data[i].append((None, game_probs))
                    move_count += 1
            
            # Check which games ended
            ended = board.get_game_ended()
            active_games = (ended == 0)
        
        jax_time = time.time() - start
        print(f"  ✓ Generated 5 games in {jax_time:.2f}s")
        print(f"    Total moves: {move_count}")
        print(f"    Time per game: {jax_time/5:.2f}s")
        print(f"    Time per move: {jax_time/move_count*1000:.1f}ms")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        jax_time = None
    
    if pytorch_time and jax_time:
        speedup = pytorch_time / jax_time
        print(f"\n  Speedup: {speedup:.1f}x")
        return {"pytorch_time": pytorch_time, "jax_time": jax_time, "speedup": speedup}
    return {}

def main():
    """Run all speed comparisons"""
    print("="*60)
    print("QUICK SPEED COMPARISON")
    print("PyTorch vs JAX - n=6, k=3")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Test core operations
    pytorch_results = test_pytorch_core()
    jax_results = test_jax_core()
    
    # Test self-play speed
    self_play_results = test_self_play_speed()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if pytorch_results["success"] and jax_results["success"]:
        print("\n📊 Core Operations Comparison:")
        
        # Board operations
        board_speedup = pytorch_results["board_ops_ms"] / jax_results["board_ops_ms"]
        print(f"\nBoard Operations (100 steps):")
        print(f"  PyTorch: {pytorch_results['board_ops_ms']:.1f}ms")
        print(f"  JAX:     {jax_results['board_ops_ms']:.1f}ms")
        print(f"  Speedup: {board_speedup:.1f}x")
        
        # Neural network
        nn_speedup = pytorch_results["nn_inference_ms"] / jax_results["nn_inference_ms"]
        print(f"\nNeural Network (100 inferences):")
        print(f"  PyTorch: {pytorch_results['nn_inference_ms']:.1f}ms")
        print(f"  JAX:     {jax_results['nn_inference_ms']:.1f}ms")
        print(f"  Speedup: {nn_speedup:.1f}x")
        
        # MCTS
        mcts_speedup = pytorch_results["mcts_10sims_ms"] / jax_results["mcts_10sims_ms"]
        print(f"\nMCTS Search (10 simulations):")
        print(f"  PyTorch:      {pytorch_results['mcts_10sims_ms']:.1f}ms")
        print(f"  JAX (single): {jax_results['mcts_10sims_ms']:.1f}ms")
        print(f"  JAX (batch8): {jax_results['batch_per_game_ms']:.1f}ms per game")
        print(f"  Speedup (single): {mcts_speedup:.1f}x")
        
        batch_speedup = pytorch_results["mcts_10sims_ms"] / jax_results["batch_per_game_ms"]
        print(f"  Speedup (batched): {batch_speedup:.1f}x")
    
    if self_play_results.get("speedup"):
        print(f"\n📊 Self-Play Performance (5 games, 10 MCTS sims):")
        print(f"  PyTorch: {self_play_results['pytorch_time']:.2f}s")
        print(f"  JAX:     {self_play_results['jax_time']:.2f}s")
        print(f"  Speedup: {self_play_results['speedup']:.1f}x")
    
    # Conclusions
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    
    if jax_results["success"]:
        if jax_results.get("batch_per_game_ms"):
            batch_efficiency = jax_results["mcts_10sims_ms"] / jax_results["batch_per_game_ms"]
            print(f"✅ JAX batch processing provides {batch_efficiency:.1f}x speedup")
    
    if self_play_results.get("speedup", 0) > 1:
        print(f"✅ JAX self-play is {self_play_results['speedup']:.1f}x faster than PyTorch")
    
    print("\n💡 Note: This comparison uses CPU only for fairness.")
    print("   JAX performance improves significantly with GPU and larger batches.")

if __name__ == "__main__":
    main()
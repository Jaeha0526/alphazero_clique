#!/usr/bin/env python
"""
Comprehensive speed comparison between PyTorch and JAX implementations
Testing with n=6, k=3 configuration
"""

import sys
import os
import time
import subprocess
import json
from datetime import datetime

# Add paths for imports
sys.path.append('src')
sys.path.append('jax_full_src')

def test_pytorch_mcts():
    """Test PyTorch MCTS speed"""
    print("\n" + "="*60)
    print("PYTORCH MCTS TEST (n=6, k=3, 10 simulations)")
    print("="*60)
    
    try:
        from clique_board import CliqueBoard
        from alpha_net_clique import CliqueGNN
        from MCTS_clique import MCTS
        import torch
        import numpy as np
        
        # Setup
        game = CliqueBoard(6, 3)
        model = CliqueGNN(6, hidden_dim=64, num_layers=2)
        model.eval()
        
        # Warmup
        board = game.get_init_board()
        mcts = MCTS(game, model, cpuct=3.0)
        _ = mcts.get_action_prob(board, temp=1, mcts_simulations=5)
        
        # Time multiple searches
        times = []
        for i in range(5):
            board = game.get_init_board()
            mcts = MCTS(game, model, cpuct=3.0)
            
            start = time.time()
            probs = mcts.get_action_prob(board, temp=1, mcts_simulations=10)
            times.append(time.time() - start)
        
        avg_time = sum(times[1:]) / len(times[1:])  # Skip first (warmup)
        
        print(f"✓ PyTorch MCTS: {avg_time*1000:.1f}ms per search")
        print(f"  Min: {min(times[1:])*1000:.1f}ms, Max: {max(times[1:])*1000:.1f}ms")
        
        return {"mcts_time_ms": avg_time * 1000, "success": True}
        
    except Exception as e:
        print(f"✗ PyTorch MCTS failed: {e}")
        return {"mcts_time_ms": None, "success": False, "error": str(e)}

def test_jax_mcts():
    """Test JAX MCTS speed"""
    print("\n" + "="*60)
    print("JAX MCTS TEST (n=6, k=3, 10 simulations)")
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
        
        # Warmup
        board = VectorizedCliqueBoard(batch_size=1, num_vertices=6, k=3)
        mcts = MCTXTrueJAX(
            batch_size=1,
            num_actions=15,
            max_nodes=11,
            c_puct=3.0,
            num_vertices=6
        )
        _ = mcts.search(board, model, 5, temperature=1.0)
        
        # Time multiple searches
        times = []
        for i in range(5):
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
            times.append(time.time() - start)
        
        avg_time = sum(times[1:]) / len(times[1:])  # Skip first (warmup)
        
        print(f"✓ JAX MCTS: {avg_time*1000:.1f}ms per search")
        print(f"  Min: {min(times[1:])*1000:.1f}ms, Max: {max(times[1:])*1000:.1f}ms")
        
        return {"mcts_time_ms": avg_time * 1000, "success": True}
        
    except Exception as e:
        print(f"✗ JAX MCTS failed: {e}")
        return {"mcts_time_ms": None, "success": False, "error": str(e)}

def test_batch_performance():
    """Test batch processing performance"""
    print("\n" + "="*60)
    print("BATCH PROCESSING TEST (8 games parallel)")
    print("="*60)
    
    os.environ['JAX_PLATFORMS'] = 'cpu'
    
    try:
        import jax
        from vectorized_board import VectorizedCliqueBoard
        from vectorized_nn import ImprovedBatchedNeuralNetwork
        from mctx_true_jax import MCTXTrueJAX
        
        model = ImprovedBatchedNeuralNetwork(
            num_vertices=6,
            hidden_dim=64,
            num_layers=2
        )
        
        # Test batch=1
        board1 = VectorizedCliqueBoard(batch_size=1, num_vertices=6, k=3)
        mcts1 = MCTXTrueJAX(batch_size=1, num_actions=15, max_nodes=11, c_puct=3.0, num_vertices=6)
        
        start = time.time()
        for _ in range(8):
            _ = mcts1.search(board1, model, 10, temperature=1.0)
        time_sequential = time.time() - start
        
        # Test batch=8
        board8 = VectorizedCliqueBoard(batch_size=8, num_vertices=6, k=3)
        mcts8 = MCTXTrueJAX(batch_size=8, num_actions=15, max_nodes=11, c_puct=3.0, num_vertices=6)
        
        start = time.time()
        _ = mcts8.search(board8, model, 10, temperature=1.0)
        time_batch = time.time() - start
        
        speedup = time_sequential / time_batch
        
        print(f"✓ Sequential (8x1): {time_sequential*1000:.1f}ms")
        print(f"✓ Batched (1x8): {time_batch*1000:.1f}ms")
        print(f"✓ Batch speedup: {speedup:.1f}x")
        
        return {
            "sequential_ms": time_sequential * 1000,
            "batch_ms": time_batch * 1000,
            "speedup": speedup
        }
        
    except Exception as e:
        print(f"✗ Batch test failed: {e}")
        return {"speedup": None, "error": str(e)}

def run_mini_pipeline_test():
    """Run a minimal pipeline test for both implementations"""
    print("\n" + "="*60)
    print("MINI PIPELINE TEST (2 games, 5 MCTS sims, 1 epoch)")
    print("="*60)
    
    config = {
        "games": 2,
        "mcts_sims": 5,
        "epochs": 1,
        "batch_size": 2
    }
    
    results = {}
    
    # PyTorch mini pipeline
    print("\n1. PyTorch Pipeline:")
    cmd = [
        "python", "src/pipeline_clique.py",
        "--mode", "pipeline",
        "--vertices", "6",
        "--k", "3",
        "--iterations", "1",
        "--self-play-games", str(config["games"]),
        "--mcts-sims", str(config["mcts_sims"]),
        "--num-cpus", "1",
        "--batch-size", str(config["batch_size"]),
        "--epochs", str(config["epochs"]),
        "--experiment-name", "test_pytorch_speed"
    ]
    
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        pytorch_time = time.time() - start
        print(f"  ✓ Completed in {pytorch_time:.1f}s")
        results["pytorch"] = pytorch_time
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timed out after 60s")
        results["pytorch"] = None
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["pytorch"] = None
    
    # JAX mini pipeline
    print("\n2. JAX Pipeline:")
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    
    cmd = [
        "python", "jax_full_src/run_jax_optimized.py",
        "--num_iterations", "1",
        "--num_episodes", str(config["games"]),
        "--game_batch_size", str(config["games"]),
        "--training_batch_size", str(config["batch_size"]),
        "--num_epochs", str(config["epochs"]),
        "--vertices", "6",
        "--k", "3",
        "--mcts_sims", str(config["mcts_sims"]),
        "--experiment_name", "test_jax_speed"
    ]
    
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        jax_time = time.time() - start
        print(f"  ✓ Completed in {jax_time:.1f}s")
        results["jax"] = jax_time
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timed out after 60s")
        results["jax"] = None
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["jax"] = None
    
    if results.get("pytorch") and results.get("jax"):
        speedup = results["pytorch"] / results["jax"]
        print(f"\n  Pipeline speedup: {speedup:.1f}x")
        results["speedup"] = speedup
    
    return results

def main():
    """Run all speed comparisons"""
    print("="*60)
    print("COMPREHENSIVE SPEED COMPARISON")
    print("PyTorch vs JAX - n=6, k=3")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": "n=6, k=3"
    }
    
    # Test MCTS speed
    pytorch_results = test_pytorch_mcts()
    jax_results = test_jax_mcts()
    
    results["pytorch_mcts"] = pytorch_results
    results["jax_mcts"] = jax_results
    
    if pytorch_results["success"] and jax_results["success"]:
        mcts_speedup = pytorch_results["mcts_time_ms"] / jax_results["mcts_time_ms"]
        results["mcts_speedup"] = mcts_speedup
    
    # Test batch performance
    batch_results = test_batch_performance()
    results["batch_performance"] = batch_results
    
    # Test mini pipelines
    pipeline_results = run_mini_pipeline_test()
    results["pipeline"] = pipeline_results
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if pytorch_results["success"] and jax_results["success"]:
        print(f"\n📊 MCTS Performance (10 simulations):")
        print(f"  PyTorch: {pytorch_results['mcts_time_ms']:.1f}ms")
        print(f"  JAX:     {jax_results['mcts_time_ms']:.1f}ms")
        print(f"  Speedup: {results.get('mcts_speedup', 0):.1f}x")
    
    if batch_results.get("speedup"):
        print(f"\n📊 Batch Processing (8 games):")
        print(f"  Sequential: {batch_results['sequential_ms']:.1f}ms")
        print(f"  Parallel:   {batch_results['batch_ms']:.1f}ms")
        print(f"  Speedup:    {batch_results['speedup']:.1f}x")
    
    if pipeline_results.get("speedup"):
        print(f"\n📊 Full Pipeline (2 games, 5 sims, 1 epoch):")
        print(f"  PyTorch: {pipeline_results['pytorch']:.1f}s")
        print(f"  JAX:     {pipeline_results['jax']:.1f}s")
        print(f"  Speedup: {pipeline_results['speedup']:.1f}x")
    
    # Save results
    output_file = "test/speed_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📝 Results saved to {output_file}")
    
    # Conclusions
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    
    if results.get("mcts_speedup", 0) > 1:
        print(f"✅ JAX MCTS is {results['mcts_speedup']:.1f}x faster than PyTorch")
    else:
        print("⚠️ PyTorch MCTS is faster or equal to JAX")
    
    if batch_results.get("speedup", 0) > 2:
        print(f"✅ JAX batch processing provides {batch_results['speedup']:.1f}x speedup")
    
    if pipeline_results.get("speedup", 0) > 1:
        print(f"✅ JAX pipeline is {pipeline_results['speedup']:.1f}x faster overall")
    
    print("\n💡 Note: This comparison uses CPU only for fairness.")
    print("   JAX performance improves significantly with GPU and larger batches.")

if __name__ == "__main__":
    main()
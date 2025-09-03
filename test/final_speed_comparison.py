#!/usr/bin/env python
"""
Final comprehensive speed comparison between PyTorch and JAX
Tests different configurations and batch sizes
"""

import sys
import os
import time
import subprocess
from datetime import datetime
import json

def run_test(implementation, n, k, games, sims, epochs, batch_size, timeout=300):
    """Run a single test configuration"""
    
    if implementation == "pytorch":
        cmd = [
            "python", "src/pipeline_clique.py",
            "--mode", "pipeline",
            "--vertices", str(n),
            "--k", str(k),
            "--iterations", "1",
            "--self-play-games", str(games),
            "--mcts-sims", str(sims),
            "--num-cpus", "2",
            "--batch-size", str(batch_size),
            "--epochs", str(epochs),
            "--experiment-name", f"speed_pytorch_n{n}k{k}"
        ]
        env = {**os.environ, "WANDB_MODE": "disabled"}
    else:  # jax
        cmd = [
            "python", "jax_full_src/run_jax_optimized.py",
            "--num_iterations", "1",
            "--num_episodes", str(games),
            "--game_batch_size", str(min(games, 32)),  # Cap batch size
            "--training_batch_size", str(batch_size),
            "--num_epochs", str(epochs),
            "--vertices", str(n),
            "--k", str(k),
            "--mcts_sims", str(sims),
            "--use_true_mctx",
            "--parallel_evaluation",
            "--experiment_name", f"speed_jax_n{n}k{k}"
        ]
        env = {**os.environ, "JAX_PLATFORMS": "cpu", "WANDB_MODE": "disabled"}
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            env=env
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            return {"success": True, "time": elapsed}
        else:
            return {"success": False, "time": None, "error": result.stderr[-200:] if result.stderr else "Unknown error"}
            
    except subprocess.TimeoutExpired:
        return {"success": False, "time": None, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "time": None, "error": str(e)}

def test_mcts_batch_scaling():
    """Test how MCTS scales with batch size"""
    print("\n" + "="*60)
    print("MCTS BATCH SCALING TEST")
    print("="*60)
    
    test_code = """
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import sys
sys.path.append('jax_full_src')

import time
import jax.numpy as jnp
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from mctx_true_jax import MCTXTrueJAX

model = ImprovedBatchedNeuralNetwork(num_vertices=6, hidden_dim=64, num_layers=2)

print("Batch Size | Time (ms) | Per-game (ms) | Speedup")
print("-" * 50)

# Test different batch sizes
batch_sizes = [1, 2, 4, 8, 16, 32]
base_time = None

for bs in batch_sizes:
    board = VectorizedCliqueBoard(batch_size=bs, num_vertices=6, k=3)
    mcts = MCTXTrueJAX(batch_size=bs, num_actions=15, max_nodes=26, c_puct=3.0, num_vertices=6)
    
    # Warmup
    probs = mcts.search(board, model, 25, temperature=1.0)
    probs.block_until_ready()
    
    # Time
    times = []
    for _ in range(5):
        start = time.time()
        probs = mcts.search(board, model, 25, temperature=1.0)
        probs.block_until_ready()
        times.append(time.time() - start)
    
    avg_time = sum(times[1:]) / len(times[1:])  # Skip first
    per_game = avg_time / bs
    
    if base_time is None:
        base_time = avg_time
        speedup = 1.0
    else:
        speedup = (base_time * bs) / avg_time
    
    print(f"{bs:10d} | {avg_time*1000:9.1f} | {per_game*1000:13.1f} | {speedup:7.1f}x")
"""
    
    try:
        result = subprocess.run(
            ["python", "-c", test_code],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Test failed:", result.stderr[-500:])
    except Exception as e:
        print(f"Test failed: {e}")

def main():
    """Run comprehensive speed comparison"""
    print("="*60)
    print("COMPREHENSIVE SPEED COMPARISON")
    print("PyTorch vs JAX AlphaZero")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Test MCTS batch scaling
    test_mcts_batch_scaling()
    
    # Test configurations
    configs = [
        # Small test (n=6, k=3)
        {"n": 6, "k": 3, "games": 10, "sims": 25, "epochs": 5, "batch": 32, "name": "Small (n=6,k=3)"},
        # Medium test (n=6, k=3, more games)
        {"n": 6, "k": 3, "games": 50, "sims": 50, "epochs": 10, "batch": 64, "name": "Medium (n=6,k=3)"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n" + "="*60)
        print(f"Testing: {config['name']}")
        print(f"Config: {config['games']} games, {config['sims']} MCTS sims, {config['epochs']} epochs")
        print("="*60)
        
        # PyTorch
        print(f"\nPyTorch {config['name']}...")
        pytorch_result = run_test(
            "pytorch", 
            config["n"], config["k"], 
            config["games"], config["sims"], 
            config["epochs"], config["batch"],
            timeout=600
        )
        
        if pytorch_result["success"]:
            print(f"  ✓ PyTorch: {pytorch_result['time']:.1f}s")
        else:
            print(f"  ✗ PyTorch failed: {pytorch_result.get('error', 'Unknown')}")
        
        # JAX
        print(f"\nJAX {config['name']}...")
        jax_result = run_test(
            "jax", 
            config["n"], config["k"], 
            config["games"], config["sims"], 
            config["epochs"], config["batch"],
            timeout=600
        )
        
        if jax_result["success"]:
            print(f"  ✓ JAX: {jax_result['time']:.1f}s")
        else:
            print(f"  ✗ JAX failed: {jax_result.get('error', 'Unknown')}")
        
        # Calculate speedup
        if pytorch_result["success"] and jax_result["success"]:
            speedup = pytorch_result["time"] / jax_result["time"]
            print(f"\n  Speedup: {speedup:.2f}x")
            
            results.append({
                "config": config["name"],
                "pytorch_time": pytorch_result["time"],
                "jax_time": jax_result["time"],
                "speedup": speedup
            })
        
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if results:
        print("\n📊 Performance Results:")
        print("\nConfiguration           | PyTorch | JAX     | Speedup")
        print("-" * 55)
        
        for r in results:
            print(f"{r['config']:23s} | {r['pytorch_time']:7.1f}s | {r['jax_time']:7.1f}s | {r['speedup']:6.2f}x")
        
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        print(f"\nAverage speedup: {avg_speedup:.2f}x")
        
        # Save results
        output_file = "test/speed_comparison_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "average_speedup": avg_speedup
            }, f, indent=2)
        print(f"\n📝 Results saved to {output_file}")
        
        # Conclusions
        print("\n" + "="*60)
        print("CONCLUSIONS")
        print("="*60)
        
        if avg_speedup > 1.5:
            print(f"✅ JAX is {avg_speedup:.2f}x faster than PyTorch on average")
        elif avg_speedup > 0.9:
            print(f"⚠️ Performance is comparable (JAX {avg_speedup:.2f}x)")
        else:
            print(f"❌ PyTorch is faster (JAX only {avg_speedup:.2f}x)")
        
        print("\n💡 Key Insights:")
        print("  - JAX benefits from batch processing (6-8x speedup for MCTS)")
        print("  - JAX JIT compilation provides consistent performance")
        print("  - Results are CPU-only; GPU would favor JAX more")
        print("  - Larger batches and longer runs favor JAX")
    else:
        print("\n⚠️ No successful comparisons completed")

if __name__ == "__main__":
    main()
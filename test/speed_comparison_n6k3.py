#!/usr/bin/env python
"""
Speed comparison between PyTorch and JAX implementations
Testing with n=6, k=3 configuration
"""

import sys
import os
import time
import subprocess
from datetime import datetime

def run_pytorch_test():
    """Run PyTorch pipeline test"""
    print("\n" + "="*60)
    print("PYTORCH PIPELINE TEST (n=6, k=3)")
    print("="*60)
    
    cmd = [
        "python", "src/pipeline_clique.py",
        "--mode", "pipeline",
        "--vertices", "6",
        "--k", "3",
        "--iterations", "1",
        "--self-play-games", "10",
        "--mcts-sims", "25",
        "--num-cpus", "2",
        "--batch-size", "32",
        "--epochs", "5",
        "--experiment-name", "speed_test_pytorch_n6k3"
    ]
    
    print("Running PyTorch with:")
    print("  - 10 self-play games")
    print("  - 25 MCTS simulations")
    print("  - 5 training epochs")
    print("  - Batch size 32")
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,
            env={**os.environ, "WANDB_MODE": "disabled"}
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"✓ PyTorch completed in {elapsed:.1f}s")
            
            # Extract some metrics from output
            lines = result.stdout.split('\n')
            for line in lines[-50:]:  # Check last 50 lines
                if "Self-play" in line or "Training" in line or "Evaluation" in line:
                    print(f"  {line.strip()}")
            
            return {"success": True, "time": elapsed}
        else:
            print(f"✗ PyTorch failed with return code {result.returncode}")
            print("Error output:", result.stderr[-500:] if result.stderr else "None")
            return {"success": False, "time": None}
            
    except subprocess.TimeoutExpired:
        print(f"✗ PyTorch timed out after 300s")
        return {"success": False, "time": None}
    except Exception as e:
        print(f"✗ PyTorch failed: {e}")
        return {"success": False, "time": None}

def run_jax_test():
    """Run JAX pipeline test"""
    print("\n" + "="*60)
    print("JAX PIPELINE TEST (n=6, k=3)")
    print("="*60)
    
    cmd = [
        "python", "jax_full_src/run_jax_optimized.py",
        "--num_iterations", "1",
        "--num_episodes", "10",
        "--game_batch_size", "10",
        "--training_batch_size", "32",
        "--num_epochs", "5",
        "--vertices", "6",
        "--k", "3",
        "--mcts_sims", "25",
        "--use_true_mctx",
        "--parallel_evaluation",
        "--experiment_name", "speed_test_jax_n6k3"
    ]
    
    print("Running JAX with:")
    print("  - 10 self-play games (batched)")
    print("  - 25 MCTS simulations")
    print("  - 5 training epochs")
    print("  - Batch size 32")
    print("  - True MCTX (JIT compiled)")
    print("  - Parallel evaluation")
    
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["WANDB_MODE"] = "disabled"
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,
            env=env
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"✓ JAX completed in {elapsed:.1f}s")
            
            # Extract some metrics from output
            lines = result.stdout.split('\n')
            for line in lines[-50:]:  # Check last 50 lines
                if "Self-play" in line or "Training" in line or "Evaluation" in line:
                    print(f"  {line.strip()}")
            
            return {"success": True, "time": elapsed}
        else:
            print(f"✗ JAX failed with return code {result.returncode}")
            print("Error output:", result.stderr[-500:] if result.stderr else "None")
            return {"success": False, "time": None}
            
    except subprocess.TimeoutExpired:
        print(f"✗ JAX timed out after 300s")
        return {"success": False, "time": None}
    except Exception as e:
        print(f"✗ JAX failed: {e}")
        return {"success": False, "time": None}

def test_mcts_only():
    """Test just MCTS performance"""
    print("\n" + "="*60)
    print("MCTS-ONLY PERFORMANCE TEST")
    print("="*60)
    
    # JAX MCTS test
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

# Setup
model = ImprovedBatchedNeuralNetwork(
    num_vertices=6,
    hidden_dim=64,
    num_layers=2
)

# Test single game
board1 = VectorizedCliqueBoard(batch_size=1, num_vertices=6, k=3)
mcts1 = MCTXTrueJAX(batch_size=1, num_actions=15, max_nodes=26, c_puct=3.0, num_vertices=6)

# Warmup
probs = mcts1.search(board1, model, 25, temperature=1.0)
probs.block_until_ready()

# Time single game
start = time.time()
for _ in range(10):
    probs = mcts1.search(board1, model, 25, temperature=1.0)
    probs.block_until_ready()
single_time = (time.time() - start) / 10

# Test batch of 8
board8 = VectorizedCliqueBoard(batch_size=8, num_vertices=6, k=3)
mcts8 = MCTXTrueJAX(batch_size=8, num_actions=15, max_nodes=26, c_puct=3.0, num_vertices=6)

# Warmup
probs = mcts8.search(board8, model, 25, temperature=1.0)
probs.block_until_ready()

# Time batch
start = time.time()
for _ in range(10):
    probs = mcts8.search(board8, model, 25, temperature=1.0)
    probs.block_until_ready()
batch_time = (time.time() - start) / 10

print(f"Single game (25 sims): {single_time*1000:.1f}ms")
print(f"Batch of 8 (25 sims): {batch_time*1000:.1f}ms total, {batch_time*1000/8:.1f}ms per game")
print(f"Batch speedup: {single_time*8/batch_time:.1f}x")
"""
    
    try:
        result = subprocess.run(
            ["python", "-c", test_code],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("JAX MCTS Performance:")
            print(result.stdout)
        else:
            print("JAX MCTS test failed:")
            print(result.stderr)
    except Exception as e:
        print(f"MCTS test failed: {e}")

def main():
    """Run all speed comparisons"""
    print("="*60)
    print("SPEED COMPARISON - PyTorch vs JAX")
    print("Configuration: n=6, k=3")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Test MCTS performance first (quick)
    test_mcts_only()
    
    # Run full pipeline tests
    pytorch_results = run_pytorch_test()
    jax_results = run_jax_test()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if pytorch_results["success"] and jax_results["success"]:
        speedup = pytorch_results["time"] / jax_results["time"]
        
        print(f"\n📊 Full Pipeline Performance (10 games, 25 sims, 5 epochs):")
        print(f"  PyTorch: {pytorch_results['time']:.1f}s")
        print(f"  JAX:     {jax_results['time']:.1f}s")
        print(f"  Speedup: {speedup:.1f}x")
        
        if speedup > 1.5:
            print(f"\n✅ JAX is {speedup:.1f}x faster than PyTorch!")
        elif speedup > 0.8:
            print(f"\n⚠️ Performance is comparable (JAX {speedup:.1f}x)")
        else:
            print(f"\n❌ PyTorch is faster (JAX only {speedup:.1f}x)")
    else:
        print("\n⚠️ One or both tests failed - cannot compare performance")
        if pytorch_results["success"]:
            print(f"  PyTorch succeeded: {pytorch_results['time']:.1f}s")
        else:
            print(f"  PyTorch failed")
        if jax_results["success"]:
            print(f"  JAX succeeded: {jax_results['time']:.1f}s")
        else:
            print(f"  JAX failed")
    
    print("\n💡 Notes:")
    print("  - This comparison uses CPU only for fairness")
    print("  - JAX performance improves significantly with GPU")
    print("  - JAX benefits more from larger batch sizes")
    print("  - First JAX run includes JIT compilation time")

if __name__ == "__main__":
    main()
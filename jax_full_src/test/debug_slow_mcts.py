#!/usr/bin/env python
"""Debug why MCTS is still slow even with JIT."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from simple_tree_mcts import SimpleTreeMCTS

print("Setting up minimal test...")

# Very minimal setup
batch_size = 1
num_vertices = 6
k = 3
num_actions = 15
mcts_sims = 5  # Very small

print(f"Creating board (batch_size={batch_size})...")
start = time.time()
boards = VectorizedCliqueBoard(
    batch_size=batch_size,
    num_vertices=num_vertices,
    k=k,
    game_mode="symmetric"
)
print(f"Board created in {time.time() - start:.2f}s")

print("Creating neural network...")
start = time.time()
nn = ImprovedBatchedNeuralNetwork(
    num_vertices=num_vertices,
    hidden_dim=32,
    num_layers=2,
    asymmetric_mode=False
)
print(f"NN created in {time.time() - start:.2f}s")

print("Creating MCTS...")
start = time.time()
mcts = SimpleTreeMCTS(
    batch_size=batch_size,
    num_actions=num_actions,
    c_puct=3.0,
    max_nodes_per_game=20  # Very small
)
print(f"MCTS created in {time.time() - start:.2f}s")

print(f"Starting MCTS search with {mcts_sims} simulations...")
start = time.time()

# Set timeout
import signal
def timeout_handler(signum, frame):
    raise TimeoutError("MCTS took too long")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)  # 1 minute timeout

try:
    probs = mcts.search(boards, nn, mcts_sims, temperature=1.0)
    elapsed = time.time() - start
    print(f"✓ MCTS completed in {elapsed:.2f}s")
    print(f"  Time per simulation: {elapsed/mcts_sims:.3f}s")
    print(f"  Probabilities shape: {probs.shape}")
except TimeoutError:
    print("✗ TIMEOUT! MCTS hanging again")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    signal.alarm(0)
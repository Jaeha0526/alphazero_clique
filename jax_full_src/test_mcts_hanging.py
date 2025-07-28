#!/usr/bin/env python
"""Debug where MCTS is hanging."""

import numpy as np
from vectorized_board import VectorizedCliqueBoard
from vectorized_nn import ImprovedBatchedNeuralNetwork
from simple_tree_mcts import SimpleTreeMCTS
import time

boards = VectorizedCliqueBoard(batch_size=1, num_vertices=9, k=4, game_mode="symmetric")
nn = ImprovedBatchedNeuralNetwork(num_vertices=9, hidden_dim=64, num_layers=3, asymmetric_mode=False)
mcts = SimpleTreeMCTS(batch_size=1, num_actions=36, c_puct=3.0, max_nodes_per_game=10)

print("Starting MCTS search...")
start = time.time()

# Set a timeout
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("MCTS took too long")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    probs = mcts.search(boards, nn, num_simulations=2, temperature=1.0)
    elapsed = time.time() - start
    print(f"Success! Took {elapsed:.2f}s")
    print(f"Probs: {probs}")
except TimeoutError:
    print("TIMEOUT! MCTS is hanging")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    signal.alarm(0)  # Cancel the alarm
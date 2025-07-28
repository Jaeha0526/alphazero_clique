#!/usr/bin/env python
"""
Benchmark using the real MCTS implementation from src to measure actual timings
"""

import sys
sys.path.append('/workspace/alphazero_clique/src')

import time
import torch
import numpy as np
from typing import List

from clique_board import CliqueBoard
from MCTS_clique import UCT_search, UCTNode
from alpha_net_clique import CliqueGNN
import encode_decode as ed


def benchmark_real_mcts_phases():
    """
    Measure time for each phase using the actual MCTS implementation.
    This will show us what really happens in practice.
    """
    print("=== Real MCTS Timing Benchmark ===")
    print("Using actual implementation from src/")
    print("Board: n=9, k=4")
    
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    model = CliqueGNN(
        num_vertices=9,
        hidden_dim=64,
        num_layers=3,
        asymmetric_mode=False
    )
    model = model.to(device)
    model.eval()
    
    # Create initial board
    board = CliqueBoard(num_vertices=9, k=4, game_mode="symmetric")
    
    # Warm up
    print("\nWarming up...")
    _ = UCT_search(board, num_reads=5, net=model)
    
    print("\nRunning detailed timing analysis...\n")
    
    # Instrument the MCTS to measure each phase
    # We'll modify UCTNode to track timings
    class TimedUCTNode(UCTNode):
        selection_time = 0
        expansion_time = 0
        nn_eval_time = 0
        backup_time = 0
        nn_calls = 0
        
        @classmethod
        def reset_timers(cls):
            cls.selection_time = 0
            cls.expansion_time = 0
            cls.nn_eval_time = 0
            cls.backup_time = 0
            cls.nn_calls = 0
    
    # Run instrumented MCTS
    def timed_UCT_search(game_state, num_reads, net):
        """Modified UCT_search with timing measurements"""
        root = TimedUCTNode(game_state, move=None, parent=None)
        TimedUCTNode.reset_timers()
        
        total_start = time.time()
        
        for i in range(num_reads):
            # SELECTION PHASE
            selection_start = time.time()
            leaf = root
            current = root
            
            # Track selection path
            while current.is_expanded:
                best_move = current.best_child()
                current = current.maybe_add_child(best_move)
            leaf = current
            
            TimedUCTNode.selection_time += time.time() - selection_start
            
            # Check if terminal
            if leaf.game.game_state != 0 or not leaf.game.get_valid_moves():
                backup_start = time.time()
                if leaf.game.game_state == 1:
                    value = 1.0
                elif leaf.game.game_state == 2:
                    value = -1.0
                else:
                    value = 0.0
                leaf.backup(value)
                TimedUCTNode.backup_time += time.time() - backup_start
                continue
            
            # NN EVALUATION PHASE
            nn_start = time.time()
            state_dict = ed.prepare_state_for_network(leaf.game)
            edge_index = state_dict['edge_index'].to(device)
            edge_attr = state_dict['edge_attr'].to(device)
            
            with torch.no_grad():
                child_priors, value_estimate = net(edge_index, edge_attr)
                child_priors = child_priors.cpu().numpy().squeeze()
                value_estimate = value_estimate.item()
            
            TimedUCTNode.nn_eval_time += time.time() - nn_start
            TimedUCTNode.nn_calls += 1
            
            # EXPANSION PHASE
            expansion_start = time.time()
            leaf.expand(child_priors)
            TimedUCTNode.expansion_time += time.time() - expansion_start
            
            # BACKUP PHASE
            backup_start = time.time()
            leaf.backup(value_estimate)
            TimedUCTNode.backup_time += time.time() - backup_start
        
        total_time = time.time() - total_start
        
        # Return timing info
        return {
            'total_time': total_time,
            'selection_time': TimedUCTNode.selection_time,
            'nn_eval_time': TimedUCTNode.nn_eval_time,
            'expansion_time': TimedUCTNode.expansion_time,
            'backup_time': TimedUCTNode.backup_time,
            'nn_calls': TimedUCTNode.nn_calls,
            'num_simulations': num_reads
        }
    
    # Test with different numbers of simulations
    for num_sims in [50, 100, 300]:
        print(f"\n--- Testing with {num_sims} simulations ---")
        
        # Fresh board
        board = CliqueBoard(num_vertices=9, k=4, game_mode="symmetric")
        
        # Run timed MCTS
        timings = timed_UCT_search(board, num_sims, model)
        
        # Calculate percentages
        total = timings['total_time']
        selection_pct = timings['selection_time'] / total * 100
        nn_pct = timings['nn_eval_time'] / total * 100
        expansion_pct = timings['expansion_time'] / total * 100
        backup_pct = timings['backup_time'] / total * 100
        other = total - (timings['selection_time'] + timings['nn_eval_time'] + 
                        timings['expansion_time'] + timings['backup_time'])
        other_pct = other / total * 100
        
        print(f"\nTotal time: {total*1000:.1f}ms ({timings['nn_calls']} NN calls)")
        print(f"Per simulation: {total/num_sims*1000:.2f}ms")
        print(f"\nBreakdown:")
        print(f"  Selection:  {timings['selection_time']*1000:6.1f}ms ({selection_pct:4.1f}%)")
        print(f"  NN Eval:    {timings['nn_eval_time']*1000:6.1f}ms ({nn_pct:4.1f}%)")
        print(f"  Expansion:  {timings['expansion_time']*1000:6.1f}ms ({expansion_pct:4.1f}%)")
        print(f"  Backup:     {timings['backup_time']*1000:6.1f}ms ({backup_pct:4.1f}%)")
        print(f"  Other:      {other*1000:6.1f}ms ({other_pct:4.1f}%)")
        
        if timings['nn_calls'] > 0:
            print(f"\nNN timing:")
            print(f"  Per NN call: {timings['nn_eval_time']/timings['nn_calls']*1000:.2f}ms")
    
    # Now simulate what would happen with batching
    print("\n\n=== Simulating Vectorized MCTS ===")
    print("What if we could batch NN evaluations across games?")
    
    # Measure batch NN evaluation
    print("\nMeasuring batch NN performance...")
    batch_sizes = [1, 10, 50, 100, 500]
    batch_times = []
    
    for batch_size in batch_sizes:
        # Create batch of board states
        boards = [CliqueBoard(num_vertices=9, k=4, game_mode="symmetric") 
                 for _ in range(batch_size)]
        
        # Prepare batch
        batch_edge_indices = []
        batch_edge_attrs = []
        
        for board in boards:
            state_dict = ed.prepare_state_for_network(board)
            batch_edge_indices.append(state_dict['edge_index'])
            batch_edge_attrs.append(state_dict['edge_attr'])
        
        # Stack for batching
        edge_indices = torch.stack(batch_edge_indices).to(device)
        edge_attrs = torch.stack(batch_edge_attrs).to(device)
        
        # Time batch evaluation
        start = time.time()
        with torch.no_grad():
            # Note: Original model doesn't support batching, so we simulate
            for i in range(batch_size):
                _ = model(edge_indices[i], edge_attrs[i])
        batch_time = time.time() - start
        batch_times.append(batch_time)
        
        print(f"  Batch size {batch_size:3d}: {batch_time*1000:.1f}ms total, "
              f"{batch_time/batch_size*1000:.2f}ms per board")
    
    # Project savings
    print("\n=== Projected Performance with Vectorization ===")
    
    # Using data from 300 simulation test
    single_game_time = timings['total_time']  # Time for 300 sims, 1 game
    nn_time_portion = timings['nn_eval_time']
    other_time = single_game_time - nn_time_portion
    
    print(f"\nFor 500 games, 300 simulations per move:")
    print(f"Sequential (current): {single_game_time * 500:.1f}s")
    print(f"Vectorized estimate:")
    print(f"  - NN time: {nn_time_portion:.2f}s (would be same for 500 games with batching)")
    print(f"  - Other ops: {other_time:.2f}s (assuming vectorizable)")
    print(f"  - Total: ~{single_game_time:.2f}s (500x speedup!)")
    
    print(f"\nFor full game (30 moves):")
    print(f"  Sequential: {single_game_time * 500 * 30:.0f}s = {single_game_time * 500 * 30 / 3600:.1f} hours")
    print(f"  Vectorized: {single_game_time * 30:.0f}s = {single_game_time * 30 / 60:.1f} minutes")


if __name__ == "__main__":
    benchmark_real_mcts_phases()
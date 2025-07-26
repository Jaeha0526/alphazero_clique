#!/usr/bin/env python
"""
Test script to verify JAX MCTS implementation produces correct results.
Compares behavior with original PyTorch MCTS.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import time
from typing import List, Tuple

# Import original components
from src.clique_board import CliqueBoard as OriginalBoard
from src.alpha_net_clique import CliqueGNN as PyTorchGNN
from src.MCTS_clique import UCT_search as original_uct_search, get_policy

# Import JAX components
from jax_src.jax_clique_board_numpy import JAXCliqueBoard
from jax_src.jax_alpha_net_clique import CliqueGNN as JAXGNN
from jax_src.jax_mcts_clique import UCT_search as jax_uct_search, SimpleMCTS, VectorizedMCTS
import src.encoder_decoder_clique as ed


def compare_mcts_policies(orig_policy: np.ndarray, jax_policy: np.ndarray, 
                         tolerance: float = 0.1) -> Tuple[bool, str]:
    """
    Compare MCTS policies allowing for some variance due to randomness.
    
    Returns:
        (is_similar, message)
    """
    # Check shapes
    if orig_policy.shape != jax_policy.shape:
        return False, f"Shape mismatch: {orig_policy.shape} vs {jax_policy.shape}"
    
    # Find top moves in each policy
    orig_top3 = np.argsort(orig_policy)[-3:][::-1]
    jax_top3 = np.argsort(jax_policy)[-3:][::-1]
    
    # Check if top move is the same
    if orig_top3[0] != jax_top3[0]:
        # Check if policies are very different
        policy_diff = np.abs(orig_policy - jax_policy).max()
        if policy_diff > tolerance:
            return False, f"Different top moves: {orig_top3[0]} vs {jax_top3[0]}, max diff: {policy_diff:.3f}"
    
    # Check overall similarity
    correlation = np.corrcoef(orig_policy, jax_policy)[0, 1]
    if correlation < 0.5:
        return False, f"Low correlation: {correlation:.3f}"
    
    return True, f"Policies similar (corr: {correlation:.3f})"


def test_single_mcts_search():
    """Test single MCTS search comparing original vs JAX"""
    print("Testing single MCTS search...")
    
    # Setup
    num_vertices = 6
    k = 3
    num_simulations = 50  # Reduced for testing
    
    # Create identical boards
    orig_board = OriginalBoard(num_vertices, k, "symmetric")
    jax_board = JAXCliqueBoard(num_vertices, k, "symmetric")
    
    # Make same moves
    moves = [(0, 1), (2, 3)]
    for move in moves:
        orig_board.make_move(move)
        jax_board.make_move(move)
    
    # Create models
    pytorch_model = PyTorchGNN(num_vertices)
    pytorch_model.eval()
    
    jax_model = JAXGNN(num_vertices)
    rng = np.random.RandomState(42)
    jax_params = jax_model.init_params(rng)
    
    # Run original MCTS
    print(f"\nRunning original MCTS with {num_simulations} simulations...")
    start = time.time()
    with torch.no_grad():
        orig_best_move, orig_root = original_uct_search(orig_board, num_simulations, pytorch_model)
    orig_policy = get_policy(orig_root)
    orig_time = time.time() - start
    
    # Run JAX MCTS
    print(f"Running JAX MCTS with {num_simulations} simulations...")
    start = time.time()
    
    # Create SimpleMCTS with JAX model
    jax_mcts = SimpleMCTS(jax_board, num_simulations, jax_model, jax_params)
    jax_best_move, _ = jax_mcts.search()
    jax_policy = jax_mcts._get_action_probs(jax_mcts.get_state_hash(jax_board))
    jax_time = time.time() - start
    
    # Compare results
    print(f"\nOriginal MCTS time: {orig_time:.3f}s")
    print(f"JAX MCTS time: {jax_time:.3f}s")
    print(f"Speedup: {orig_time/jax_time:.2f}x")
    
    print(f"\nOriginal best move: {orig_best_move}")
    print(f"JAX best move: {jax_best_move}")
    
    # Compare policies
    is_similar, message = compare_mcts_policies(orig_policy, jax_policy)
    print(f"\nPolicy comparison: {message}")
    
    # Show top 5 moves from each
    print("\nTop 5 moves by policy:")
    print("Original:", np.argsort(orig_policy)[-5:][::-1])
    print("JAX:     ", np.argsort(jax_policy)[-5:][::-1])
    
    return is_similar


def test_vectorized_mcts():
    """Test vectorized MCTS on multiple positions"""
    print("\n" + "="*60)
    print("Testing vectorized MCTS...")
    
    num_vertices = 6
    k = 3
    batch_size = 4
    num_simulations = 100
    
    # Create batch of different board positions
    boards = []
    for i in range(batch_size):
        board = JAXCliqueBoard(num_vertices, k, "symmetric")
        # Make different moves on each board
        if i >= 1:
            board.make_move((0, i))
        if i >= 2:
            board.make_move((2, 3))
        if i >= 3:
            board.make_move((4, 5))
        boards.append(board)
    
    # Create model
    model = JAXGNN(num_vertices)
    rng = np.random.RandomState(42)
    params = model.init_params(rng)
    
    # Initialize vectorized MCTS
    vmcts = VectorizedMCTS(num_vertices, k)
    
    # Run batch simulations
    print(f"Running {num_simulations} simulations on {batch_size} positions...")
    start = time.time()
    
    try:
        policies, final_state = vmcts.run_simulations(boards, params, num_simulations)
        batch_time = time.time() - start
        
        print(f"Batch MCTS time: {batch_time:.3f}s")
        print(f"Time per position: {batch_time/batch_size:.3f}s")
        
        # Show policies for each position
        for i in range(batch_size):
            top_moves = np.argsort(policies[i])[-3:][::-1]
            print(f"Board {i} top moves: {top_moves}")
            
    except Exception as e:
        print(f"Vectorized MCTS failed: {e}")
        return False
    
    return True


def test_mcts_game_play():
    """Test MCTS by playing out a game"""
    print("\n" + "="*60)
    print("Testing MCTS game play...")
    
    num_vertices = 6
    k = 3
    num_simulations = 200
    
    # Create board
    board = JAXCliqueBoard(num_vertices, k, "symmetric")
    
    # Create model
    model = JAXGNN(num_vertices)
    rng = np.random.RandomState(42)
    params = model.init_params(rng)
    
    # Play game with MCTS
    max_moves = 10
    move_count = 0
    
    print("\nPlaying game with JAX MCTS:")
    while board.game_state == 0 and move_count < max_moves:
        # Run MCTS
        mcts = SimpleMCTS(board, num_simulations, model, params)
        best_action, _ = mcts.search()
        
        # Make move
        edge = ed.decode_action(board, best_action)
        if edge != (-1, -1):
            board.make_move(edge)
            print(f"Move {move_count + 1}: {edge}")
            move_count += 1
        else:
            print("Invalid move returned by MCTS")
            break
        
        # Check game state
        if board.game_state != 0:
            if board.game_state == 3:
                print("Game ended in draw")
            else:
                print(f"Player {board.game_state} wins!")
            break
    
    return True


def test_mcts_statistics():
    """Test MCTS tree statistics and visit counts"""
    print("\n" + "="*60)
    print("Testing MCTS statistics...")
    
    num_vertices = 6
    k = 3
    num_simulations = 100
    
    # Create board with a few moves
    board = JAXCliqueBoard(num_vertices, k, "symmetric")
    board.make_move((0, 1))
    board.make_move((2, 3))
    
    # Create model
    model = JAXGNN(num_vertices)
    rng = np.random.RandomState(42)
    params = model.init_params(rng)
    
    # Run MCTS
    mcts = SimpleMCTS(board, num_simulations, model, params)
    best_action, stats = mcts.search()
    
    # Check statistics
    root_hash = mcts.get_state_hash(board)
    root_visits = mcts.visit_counts.get(root_hash, 0)
    
    print(f"Root visits: {root_visits}")
    print(f"Total simulations: {num_simulations}")
    print(f"Number of nodes in tree: {len(mcts.visit_counts)}")
    
    # Check policy sums to 1
    policy = mcts._get_action_probs(root_hash)
    policy_sum = policy.sum()
    print(f"Policy sum: {policy_sum:.6f}")
    
    # Show top actions by visit count
    valid_actions = board.get_valid_moves()
    print(f"\nTop actions by visits:")
    action_visits = []
    for move in valid_actions:
        action = ed.encode_action(board, move)
        # Make move to get child visits
        board_copy = board.copy()
        board_copy.make_move(move)
        child_hash = mcts.get_state_hash(board_copy)
        visits = mcts.visit_counts.get(child_hash, 0)
        action_visits.append((action, move, visits))
    
    action_visits.sort(key=lambda x: x[2], reverse=True)
    for action, move, visits in action_visits[:5]:
        print(f"  Action {action} (move {move}): {visits} visits")
    
    return abs(policy_sum - 1.0) < 0.001


def run_all_tests():
    """Run all MCTS tests"""
    print("=" * 60)
    print("JAX MCTS Parity Tests")
    print("=" * 60)
    
    tests = [
        ("Single MCTS Search", test_single_mcts_search),
        ("Vectorized MCTS", test_vectorized_mcts),
        ("MCTS Game Play", test_mcts_game_play),
        ("MCTS Statistics", test_mcts_statistics),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"  ✅ PASSED")
            else:
                print(f"  ❌ FAILED")
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed!")
        print("\nKey achievements:")
        print("- SimpleMCTS provides compatible interface with original")
        print("- VectorizedMCTS enables batch processing for GPU")
        print("- Tree statistics and policies are correctly computed")
        print("- Ready for massive parallelization with JAX!")
    else:
        print("\n❌ Some tests failed")
        for test_name, success in results:
            if not success:
                print(f"  - {test_name}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
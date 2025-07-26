#!/usr/bin/env python
"""
Test script to verify JAX implementation produces identical results to original.
This ensures we maintain exact feature parity.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from src.clique_board import CliqueBoard
from jax_src.jax_clique_board_numpy import JAXCliqueBoard
import time


def compare_boards(orig_board, jax_board, test_name=""):
    """Compare all attributes of original and JAX boards"""
    errors = []
    
    # Compare basic attributes
    if orig_board.num_vertices != jax_board.num_vertices:
        errors.append(f"{test_name}: num_vertices mismatch: {orig_board.num_vertices} vs {jax_board.num_vertices}")
    
    if orig_board.k != jax_board.k:
        errors.append(f"{test_name}: k mismatch: {orig_board.k} vs {jax_board.k}")
    
    if orig_board.game_mode != jax_board.game_mode:
        errors.append(f"{test_name}: game_mode mismatch: {orig_board.game_mode} vs {jax_board.game_mode}")
    
    if orig_board.player != jax_board.player:
        errors.append(f"{test_name}: player mismatch: {orig_board.player} vs {jax_board.player}")
    
    if orig_board.move_count != jax_board.move_count:
        errors.append(f"{test_name}: move_count mismatch: {orig_board.move_count} vs {jax_board.move_count}")
    
    if orig_board.game_state != jax_board.game_state:
        errors.append(f"{test_name}: game_state mismatch: {orig_board.game_state} vs {jax_board.game_state}")
    
    # Compare adjacency matrices
    if not np.allclose(orig_board.adjacency_matrix, jax_board.adjacency_matrix):
        errors.append(f"{test_name}: adjacency_matrix mismatch")
    
    # Compare edge states
    if not np.array_equal(orig_board.edge_states, jax_board.edge_states):
        errors.append(f"{test_name}: edge_states mismatch")
        print(f"Original edge_states:\n{orig_board.edge_states}")
        print(f"JAX edge_states:\n{jax_board.edge_states}")
    
    # Compare valid moves
    orig_moves = sorted(orig_board.get_valid_moves())
    jax_moves = sorted(jax_board.get_valid_moves())
    if orig_moves != jax_moves:
        errors.append(f"{test_name}: valid_moves mismatch: {orig_moves} vs {jax_moves}")
    
    # Compare string representations
    if str(orig_board) != str(jax_board):
        errors.append(f"{test_name}: string representation mismatch")
        print("Original:\n", str(orig_board))
        print("JAX:\n", str(jax_board))
    
    return errors


def test_initialization():
    """Test board initialization"""
    print("Testing board initialization...")
    errors = []
    
    # Test different board sizes and game modes
    test_configs = [
        (6, 3, "symmetric"),
        (6, 3, "asymmetric"),
        (8, 4, "symmetric"),
        (10, 5, "asymmetric"),
    ]
    
    for num_vertices, k, game_mode in test_configs:
        orig = CliqueBoard(num_vertices, k, game_mode)
        jax = JAXCliqueBoard(num_vertices, k, game_mode)
        
        test_name = f"Init(v={num_vertices}, k={k}, mode={game_mode})"
        errors.extend(compare_boards(orig, jax, test_name))
    
    return errors


def test_single_moves():
    """Test making individual moves"""
    print("Testing single moves...")
    errors = []
    
    # Test on 6-vertex board
    orig = CliqueBoard(6, 3, "symmetric")
    jax = JAXCliqueBoard(6, 3, "symmetric")
    
    # Test valid moves
    test_moves = [(0, 1), (2, 3), (1, 4), (0, 5)]
    
    for i, move in enumerate(test_moves):
        # Make move on both boards
        orig_result = orig.make_move(move)
        jax_result = jax.make_move(move)
        
        if orig_result != jax_result:
            errors.append(f"Move {i} result mismatch: {orig_result} vs {jax_result}")
        
        errors.extend(compare_boards(orig, jax, f"After move {i}: {move}"))
    
    # Test invalid move
    invalid_move = (0, 1)  # Already taken
    orig_result = orig.make_move(invalid_move)
    jax_result = jax.make_move(invalid_move)
    
    if orig_result != jax_result:
        errors.append(f"Invalid move result mismatch: {orig_result} vs {jax_result}")
    
    return errors


def test_win_conditions():
    """Test win condition detection"""
    print("Testing win conditions...")
    errors = []
    
    # Test symmetric mode - Player 1 forms a triangle
    orig = CliqueBoard(6, 3, "symmetric")
    jax = JAXCliqueBoard(6, 3, "symmetric")
    
    # Player 1 moves to form triangle (0,1,2)
    moves = [(0, 1), (3, 4), (0, 2), (4, 5), (1, 2)]  # Last move completes triangle
    
    for i, move in enumerate(moves):
        orig.make_move(move)
        jax.make_move(move)
        
        if i == len(moves) - 1:  # After last move
            if orig.game_state != 1 or jax.game_state != 1:
                errors.append(f"Win detection failed: orig={orig.game_state}, jax={jax.game_state}")
    
    # Test asymmetric mode
    orig2 = CliqueBoard(6, 3, "asymmetric")
    jax2 = JAXCliqueBoard(6, 3, "asymmetric")
    
    # Fill board without letting Player 1 form triangle
    # This should result in Player 2 winning
    moves2 = [(0, 1), (0, 2), (0, 3), (1, 2), (0, 4), (1, 3), 
              (0, 5), (1, 4), (2, 3), (1, 5), (2, 4), (3, 4), 
              (2, 5), (3, 5), (4, 5)]
    
    for move in moves2:
        orig2.make_move(move)
        jax2.make_move(move)
    
    if orig2.game_state != jax2.game_state:
        errors.append(f"Asymmetric endgame mismatch: orig={orig2.game_state}, jax={jax2.game_state}")
    
    return errors


def test_random_games(num_games=10):
    """Test complete random games"""
    print(f"Testing {num_games} random games...")
    errors = []
    
    for game_num in range(num_games):
        # Random configuration
        num_vertices = random.choice([6, 7, 8])
        k = random.choice([3, 4])
        game_mode = random.choice(["symmetric", "asymmetric"])
        
        orig = CliqueBoard(num_vertices, k, game_mode)
        jax = JAXCliqueBoard(num_vertices, k, game_mode)
        
        # Play random game
        move_count = 0
        max_moves = num_vertices * (num_vertices - 1) // 2
        
        while orig.game_state == 0 and move_count < max_moves:
            valid_moves = orig.get_valid_moves()
            if not valid_moves:
                break
            
            move = random.choice(valid_moves)
            
            orig_result = orig.make_move(move)
            jax_result = jax.make_move(move)
            
            if orig_result != jax_result:
                errors.append(f"Game {game_num}, move {move_count}: move result mismatch")
            
            # Compare states after each move
            move_errors = compare_boards(orig, jax, f"Game {game_num}, move {move_count}")
            if move_errors:
                errors.extend(move_errors)
                break  # Stop this game on first error
            
            move_count += 1
        
        # Final comparison
        if not errors:  # Only do final check if no errors so far
            final_errors = compare_boards(orig, jax, f"Game {game_num} final")
            errors.extend(final_errors)
    
    return errors


def test_copy_functionality():
    """Test board copying"""
    print("Testing copy functionality...")
    errors = []
    
    # Create boards and make some moves
    orig = CliqueBoard(6, 3, "symmetric")
    jax = JAXCliqueBoard(6, 3, "symmetric")
    
    moves = [(0, 1), (2, 3), (4, 5)]
    for move in moves:
        orig.make_move(move)
        jax.make_move(move)
    
    # Copy boards
    orig_copy = orig.copy()
    jax_copy = jax.copy()
    
    # Compare copies
    errors.extend(compare_boards(orig_copy, jax_copy, "After copy"))
    
    # Make more moves on originals
    orig.make_move((1, 2))
    jax.make_move((1, 2))
    
    # Ensure copies weren't affected
    errors.extend(compare_boards(orig_copy, jax_copy, "Copy after original modified"))
    
    return errors


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("JAX Clique Board Parity Tests")
    print("=" * 60)
    
    all_errors = []
    
    # Run each test suite
    test_suites = [
        ("Initialization", test_initialization),
        ("Single Moves", test_single_moves),
        ("Win Conditions", test_win_conditions),
        ("Copy Functionality", test_copy_functionality),
        ("Random Games", test_random_games),
    ]
    
    for test_name, test_func in test_suites:
        print(f"\n{test_name}:")
        errors = test_func()
        all_errors.extend(errors)
        
        if errors:
            print(f"  ❌ FAILED - {len(errors)} errors found")
            for error in errors:
                print(f"    - {error}")
        else:
            print(f"  ✅ PASSED")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    # Time board creation and moves
    num_iterations = 1000
    
    # Original version
    start = time.time()
    for _ in range(num_iterations):
        board = CliqueBoard(8, 4)
        moves = board.get_valid_moves()
        if moves:
            board.make_move(moves[0])
    orig_time = time.time() - start
    
    # JAX version
    start = time.time()
    for _ in range(num_iterations):
        board = JAXCliqueBoard(8, 4)
        moves = board.get_valid_moves()
        if moves:
            board.make_move(moves[0])
    jax_time = time.time() - start
    
    print(f"Original version: {orig_time:.4f}s for {num_iterations} iterations")
    print(f"JAX version: {jax_time:.4f}s for {num_iterations} iterations")
    print(f"Ratio: {orig_time/jax_time:.2f}x")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    if all_errors:
        print(f"❌ FAILED - Total {len(all_errors)} errors found")
        print("\nFirst 10 errors:")
        for error in all_errors[:10]:
            print(f"  - {error}")
    else:
        print("✅ ALL TESTS PASSED - JAX implementation matches original exactly!")
    
    return len(all_errors) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
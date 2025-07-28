#!/usr/bin/env python
"""Test basic pipeline functionality."""

import sys
import os

print("1. Testing Python path...")
print(f"   Current dir: {os.getcwd()}")
print(f"   Script path: {__file__}")

print("\n2. Testing imports...")
try:
    from run_jax_optimized import main
    print("   run_jax_optimized imported successfully")
except Exception as e:
    print(f"   Error importing: {e}")
    sys.exit(1)

print("\n3. Testing argparse...")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_iterations', type=int, default=1)
parser.add_argument('--num_episodes', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--checkpoint_dir', type=str, default='test_checkpoints')
parser.add_argument('--asymmetric', action='store_true')
parser.add_argument('--experiment_name', type=str, default='test_run')
parser.add_argument('--resume_from', type=str, default=None)

args = parser.parse_args(['--num_iterations', '1', '--num_episodes', '2', '--batch_size', '2', '--experiment_name', 'test_minimal'])
print(f"   Args parsed: {args}")

print("\nAll basic tests passed!")
#!/usr/bin/env python
"""Test running the pipeline directly."""

import sys
sys.path.append('/workspace/alphazero_clique/jax_full_src')

# Set args before importing
sys.argv = ['run_jax_optimized.py', '--num_iterations', '1', '--num_episodes', '2', 
            '--batch_size', '2', '--experiment_name', 'test_direct_import', '--num_epochs', '1']

print("Importing and running main...")
try:
    from run_jax_optimized import main
    main()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
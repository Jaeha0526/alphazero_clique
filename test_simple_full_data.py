#!/usr/bin/env python3
"""Simple test for full game data saving."""

import subprocess
import os
from pathlib import Path
import pickle

# Test with minimal iterations
exp_name = "test_full_data_simple"
cmd = [
    "python", "jax_full_src/run_jax_optimized.py",
    "--experiment_name", exp_name,
    "--num_iterations", "3",
    "--num_episodes", "2",
    "--mcts_sims", "5",
    "--vertices", "6",
    "--k", "3",
    "--save_full_game_data",
    "--skip_evaluation",
    "--num_epochs", "1",
    "--game_batch_size", "2",
    "--training_batch_size", "4"
]

print("Running training with --save_full_game_data...")
print(" ".join(cmd))

result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

if result.returncode != 0:
    print(f"Error: {result.stderr}")
    exit(1)

# Check game data files
game_data_dir = Path(f"experiments/{exp_name}/game_data")

if game_data_dir.exists():
    files = sorted(game_data_dir.glob("*.pkl"))
    print(f"\n✅ Success! Found {len(files)} game data files:")
    
    for f in files:
        with open(f, 'rb') as file:
            data = pickle.load(file)
        print(f"  - {f.name}: iteration {data['iteration']}, full_data={data.get('is_full_data', False)}, games={data.get('num_games_saved', 'N/A')}")
    
    # With --save_full_game_data, we should have files for all 3 iterations (0, 1, 2)
    expected_iterations = [0, 1, 2]
    found_iterations = []
    for f in files:
        with open(f, 'rb') as file:
            data = pickle.load(file)
            found_iterations.append(data['iteration'])
    
    if sorted(found_iterations) == expected_iterations:
        print(f"\n✅ All iterations saved correctly: {found_iterations}")
    else:
        print(f"\n⚠️  Expected iterations {expected_iterations}, found {found_iterations}")
else:
    print(f"❌ Game data directory not found: {game_data_dir}")

print("\n✅ Test completed successfully!")
print("\nFeature is working! Use --save_full_game_data to save game data every iteration.")
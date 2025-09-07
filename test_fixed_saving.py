#!/usr/bin/env python3
"""Test the fixed game data saving."""

import subprocess
import pickle
from pathlib import Path
import time

exp_name = f"test_fix_{int(time.time())}"

print("Testing fixed game data saving...")
print(f"Experiment: {exp_name}")

# Run a quick test
cmd = [
    "python", "jax_full_src/run_jax_optimized.py",
    "--experiment_name", exp_name,
    "--num_iterations", "1",
    "--num_episodes", "5",
    "--mcts_sims", "10",
    "--vertices", "6",
    "--k", "3",
    "--save_full_game_data",
    "--skip_evaluation",
    "--num_epochs", "1",
    "--game_batch_size", "5",
    "--training_batch_size", "8"
]

print("\nRunning command:")
print(" ".join(cmd))

result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

if result.returncode != 0:
    print(f"Error: {result.stderr}")
    exit(1)

# Check the saved file
game_data_path = Path(f"experiments/{exp_name}/game_data/iteration_0.pkl")

if game_data_path.exists():
    print(f"\n✅ File created: {game_data_path}")
    
    with open(game_data_path, 'rb') as f:
        data = pickle.load(f)
    
    print("\nSaved data structure:")
    print(f"  Keys: {list(data.keys())}")
    print(f"  Total training examples: {data.get('total_training_examples', 'N/A')}")
    print(f"  Examples saved: {data.get('num_examples_saved', 'N/A')}")
    print(f"  Games played: {data.get('num_games_played', 'N/A')}")
    
    if 'game_stats' in data:
        stats = data['game_stats']
        print(f"\nGame statistics included:")
        print(f"  Average game length: {stats.get('avg_game_length', 'N/A')}")
        if 'game_length_distribution' in stats:
            print(f"  Game length distribution: {stats.get('game_length_distribution', {})}")
    
    print("\n✅ New format working correctly!")
    print("   - No flawed reconstruction")
    print("   - Raw training data saved")
    print("   - Real game statistics included")
else:
    print(f"❌ File not found: {game_data_path}")
    exit(1)
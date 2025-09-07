# Game Data Viewing Guide

## Overview
The AlphaZero training pipeline now saves comprehensive game data for analysis, including both self-play games and evaluation games.

## Self-Play Game Data

### Saving Options
- **Default**: Game data saved every 5 iterations (iteration_0.pkl, iteration_5.pkl, etc.)
- **Full saving**: Use `--save_full_game_data` flag to save every iteration

### Viewing Self-Play Games
```bash
# View a specific game (e.g., game 1 from iteration 1)
python view_game_replay.py experiments/YOUR_EXP/game_data/iteration_1.pkl 1

# List all games in a file
python view_game_replay.py experiments/YOUR_EXP/game_data/iteration_1.pkl list

# View most recent game data automatically
python view_game_replay.py

# Analyze game statistics and learning progress
python analyze_game_data.py experiments/YOUR_EXP/game_data/iteration_1.pkl

# Compare learning across iterations
python analyze_game_data.py experiments/YOUR_EXP/game_data/ --compare
```

### What's Saved in Self-Play Games
- Complete move sequence with actual actions taken
- MCTS policies (visit distributions) for each move
- Value estimates from neural network
- Exact game boundaries (no reconstruction needed)
- Player information and final outcomes

## Evaluation Game Data

### When Evaluation Games Are Saved
- Automatically saved during model evaluation
- Two files per iteration:
  - `iteration_N_vs_initial.pkl` - Games vs initial model
  - `iteration_N_vs_best.pkl` - Games vs best model (if applicable)

### Viewing Evaluation Games
```bash
# View specific evaluation game
python view_eval_games.py experiments/YOUR_EXP/eval_games/iteration_5_vs_initial.pkl 1

# List all evaluation files in experiment
python view_eval_games.py experiments/YOUR_EXP

# View with game number (1-indexed)
python view_eval_games.py experiments/YOUR_EXP/eval_games/iteration_10_vs_best.pkl 3
```

### What's Saved in Evaluation Games
- Which model (current/baseline) made each move
- Complete move sequences with actions
- MCTS policies from both models
- Model assignments (who plays first)
- Win/loss/draw statistics

## Ramsey Counterexample Analysis

For avoid_clique mode experiments:

### Finding Potential Counterexamples
```bash
# Check for games that filled all edges (potential counterexamples)
python read_games_from_data.py experiments/ramsey_n10k4/game_data/iteration_1.pkl

# View a complete 45-move game in K_10
python view_game_replay.py experiments/ramsey_n10k4/game_data/iteration_1.pkl 2
```

### Key Indicators
- Games reaching maximum moves (e.g., 45 for K_10) are potential Ramsey counterexamples
- These are automatically saved to `experiments/YOUR_EXP/ramsey_counterexamples/`
- Look for "DRAW" outcomes in avoid_clique mode

## Example Commands for Common Tasks

### Track Learning Progress
```bash
# See how game quality improves over iterations
python analyze_game_data.py experiments/my_exp/game_data/ --compare

# Check win rates in evaluation
python view_eval_games.py experiments/my_exp
```

### Debug Specific Games
```bash
# Find out why a game ended early
python view_game_replay.py experiments/my_exp/game_data/iteration_10.pkl 5

# Compare how current vs baseline models play
python view_eval_games.py experiments/my_exp/eval_games/iteration_15_vs_best.pkl 1
```

### Analyze Ramsey Search
```bash
# Find all games that avoided k-cliques to completion
python read_games_from_data.py experiments/ramsey_exp/game_data/iteration_20.pkl | grep "45 moves"

# Replay a potential counterexample
python view_game_replay.py experiments/ramsey_exp/game_data/iteration_20.pkl 8
```

## File Formats

### Self-Play Game Data (iteration_N.pkl)
```python
{
    'training_data': [...],  # Raw training examples
    'games_info': [...],     # Game boundaries and metadata
    'vertices': N,           # Graph size
    'k': K,                  # Clique size
    'game_mode': 'avoid_clique' or 'symmetric',
    'iteration': N
}
```

### Evaluation Game Data (iteration_N_vs_TYPE.pkl)
```python
{
    'games_data': [...],     # All moves from all games
    'games_info': [...],     # Game boundaries and results
    'current_wins': N,
    'baseline_wins': N,
    'draws': N,
    'vertices': N,
    'k': K,
    'mcts_sims': N,
    'models': {...}          # Model identifiers
}
```
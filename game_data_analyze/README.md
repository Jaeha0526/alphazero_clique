# Game Data Analysis Tools

This directory contains tools for analyzing and visualizing AlphaZero Clique Game training data.

## Interactive Game Viewer

An interactive matplotlib-based tool to visualize game trajectories from saved game data.

### Features

- **Load any game data file** from training (iteration_0.pkl, iteration_5.pkl, etc.)
- **Select specific games** to visualize from the saved samples
- **Step through moves** manually with Previous/Next buttons
- **Auto-play** game trajectories with adjustable speed
- **Visual graph display** showing:
  - Red edges: Player 1's moves
  - Blue edges: Player 2's moves  
  - Yellow highlight: Current move
  - Gray edges: Uncolored edges
- **Move probability display** showing top 10 action probabilities
- **Game information** including player, value, and best move details

### Usage

```bash
# View a specific game data file
python game_data_analyze/interactive_game_viewer.py experiments/your_experiment/game_data/iteration_0.pkl

# Example with the n13k4_ramsey experiment
python game_data_analyze/interactive_game_viewer.py experiments/n13k4_ramsey/game_data/iteration_0.pkl
```

### Controls

- **Game selector**: Enter game index (0 to N-1) to switch between games
- **Move slider**: Drag to jump to any move in the game
- **Previous/Next buttons**: Step through moves one at a time
- **Play/Pause button**: Auto-play the game trajectory
- **Reset button**: Return to the first move
- **Speed slider**: Adjust auto-play speed (0.5x to 3x)

### Understanding the Display

1. **Graph View** (left panel):
   - Shows the complete graph with n vertices
   - Edges are colored as players make moves
   - Current move is highlighted in yellow
   - Node labels show vertex indices

2. **Probability Chart** (right panel):
   - Bar chart of top 10 move probabilities
   - Yellow bar indicates the selected move
   - Higher probability = more confident move

3. **Information Panel** (bottom):
   - Current iteration, game mode
   - Active player and their role (if asymmetric)
   - Final value for this position (+1 for win, -1 for loss)
   - Best move details with probability

### Game Modes

- **Symmetric**: Both players try to form k-cliques
- **Asymmetric**: Attacker forms cliques, Defender prevents
- **Avoid_clique**: Both players avoid forming k-cliques (forming one = losing)

### Interpreting Avoid_Clique Games

In avoid_clique mode:
- Players take turns coloring edges
- If a player creates a k-clique in their color, they LOSE
- Successful games (draws) mean both players avoided k-cliques
- These draws are potential Ramsey counterexamples

### Understanding Temperature Annealing and Move Probabilities

**Why do later moves show very high probabilities (near 1.0)?**

This is completely normal and expected due to **temperature annealing** in MCTS during self-play:

#### Temperature Schedule Used:
- **Early game (0-20% of moves)**: τ = 1.0 (high exploration)
- **Early-mid game (20-40%)**: τ = 0.8 (good exploration) 
- **Mid game (40-60%)**: τ = 0.5 (balanced)
- **Late-mid game (60-80%)**: τ = 0.2 (more exploitation)
- **End game (80-100%)**: τ = 0.1 (strong exploitation)

#### What This Means:
- **Low temperature** (τ → 0) makes the policy very deterministic, heavily favoring the most-visited MCTS action
- **High temperature** (τ = 1) creates more uniform exploration across viable moves
- **By iteration 5**: The model has learned strong strategic preferences, so MCTS visit counts become concentrated on the "best" moves

#### Comparison to AlphaZero Original:
- **Original AlphaZero**: τ = 1.0 for first 30 moves, then τ ≈ 0.0
- **Our implementation**: Gradual 5-phase annealing (potentially superior for learning)
- **Both approaches**: Essential for generating quality training data

#### This is Training Success:
- **Early iterations**: More random, diverse move probabilities
- **Later iterations**: Confident, strategic move selection
- **High probabilities**: Indicate the model has learned clear preferences and strategies

### Tips for Analysis

1. **Temperature effects**: Early moves (~0.04 max prob) vs later moves (1.0 prob) show learning progression
2. **Watch for forced moves**: Low entropy (one high probability) indicates forced moves or strong strategic preferences
3. **Observe learning progress**: Compare early iterations (random) vs later (strategic)
4. **Identify patterns**: Look for common opening moves or defensive strategies
5. **Check game endings**: See if players are learning to avoid losing moves
6. **Iteration comparison**: Compare probability distributions between iteration_0.pkl and iteration_5.pkl

### Requirements

```bash
pip install matplotlib networkx numpy
```

### Troubleshooting

If the viewer doesn't open:
- Ensure you have a display available (X11 forwarding for SSH)
- Try saving figures instead: modify script to use `plt.savefig()`
- Use a Jupyter notebook version (can be created on request)
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

### Tips for Analysis

1. **Watch for forced moves**: Low entropy (one high probability) indicates forced moves
2. **Observe learning progress**: Compare early iterations (random) vs later (strategic)
3. **Identify patterns**: Look for common opening moves or defensive strategies
4. **Check game endings**: See if players are learning to avoid losing moves

### Requirements

```bash
pip install matplotlib networkx numpy
```

### Troubleshooting

If the viewer doesn't open:
- Ensure you have a display available (X11 forwarding for SSH)
- Try saving figures instead: modify script to use `plt.savefig()`
- Use a Jupyter notebook version (can be created on request)
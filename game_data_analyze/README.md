# Game Data Web Visualizer

A Flask-based web application for visualizing AlphaZero Clique Game training and evaluation data.

## Features

- **Interactive Graph Visualization**: View game states as colored graphs with real-time updates
- **Game Replay Controls**: Step through moves manually or auto-play with adjustable speed
- **Move Probability Display**: See the top action probabilities for each move
- **Support for Both Data Types**:
  - Training games from self-play
  - Evaluation games showing model comparisons
- **Game Statistics**: View win rates, game lengths, and complete game counts
- **File Browser**: Easy navigation through all experiments and iterations

## Usage

### Starting the Web Server

```bash
cd game_data_analyze
python app.py
```

Then open your browser and navigate to: http://localhost:5000

### Interface Overview

1. **Left Sidebar**: Lists all available game data files
   - Green badges = training games
   - Orange badges = evaluation games
   - Click any file to load it

2. **Main Canvas**: Interactive graph visualization
   - Red edges = Player 0 moves
   - Blue edges = Player 1 moves  
   - Yellow highlight = Current move
   - Gray edges = Uncolored edges

3. **Controls**:
   - Game selector dropdown
   - Move slider for quick navigation
   - Previous/Next buttons for step-by-step viewing
   - Play/Pause for automatic replay
   - Speed control (0.5x to 3x)

4. **Info Panel**: 
   - Game statistics (wins, draws, average length)
   - Current move details
   - Move probabilities with visual bars
   - Game result on final move

## Data Format Support

The visualizer works with the new data format:
- **Training games**: `training_data` + `games_info` structure
- **Evaluation games**: `games_data` + `games_info` structure
- Automatically detects and handles both formats

## Understanding Game Modes

- **Symmetric**: Both players try to form k-cliques
- **Asymmetric**: Attacker forms cliques, Defender prevents
- **Avoid_clique**: Both players avoid forming k-cliques (forming one = losing)

In avoid_clique mode, complete games (all edges colored) are potential Ramsey counterexamples!

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

## Requirements

```bash
pip install flask networkx numpy
```

## Tips

1. **Identify Critical Moves**: Low entropy (one high probability) indicates forced defensive moves
2. **Compare Iterations**: Load different iterations to see learning progress
3. **Spot Patterns**: Look for common opening moves or defensive strategies
4. **Check Endings**: In avoid_clique mode, see if games end with 4-clique formation or complete coloring
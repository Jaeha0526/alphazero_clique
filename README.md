# AlphaZero Clique Game

This project implements the AlphaZero algorithm for the Clique Game, using Graph Neural Networks and Monte Carlo Tree Search.
Main structure of the code originated from https://github.com/geochri/AlphaZero_Chess

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/alphazero_clique.git
cd alphazero_clique
```

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

The requirements.txt file includes the following key dependencies:
- numpy>=1.19.5: For numerical computations
- matplotlib>=3.5.1: For visualization
- networkx>=2.6.3: For graph operations
- flask>=2.0.1: For web interface
- torch>=2.0.0: For deep learning
- torchvision>=0.15.0: For computer vision utilities
- tqdm>=4.65.0: For progress bars
- torch-geometric>=2.0.4: For graph neural networks
- pandas>=1.3.5: For data manipulation

## Game Rules

The Clique Game is played on an undirected graph with N vertices. There are two game modes:

### Symmetric Mode (Default)
- Both players try to form a k-clique (a complete subgraph with k vertices)
- Players take turns adding edges to the graph
- First player to form a k-clique wins
- If no more moves are possible and no k-clique is formed, the game is a draw
- Example: With 6 vertices and k=3, players try to form triangles

### Asymmetric Mode
- Player 1 tries to form a k-clique
- Player 2 tries to prevent Player 1 from forming a k-clique
- Players take turns adding edges to the graph
- Player 1 wins by forming a k-clique
- Player 2 wins by preventing Player 1 from forming a k-clique
- If no more moves are possible and no k-clique is formed, Player 2 wins

## Interactive Game Interface

Test the Clique Game, visualize the board, and interact with trained AI models using the web interface:

```bash
python src/interactive_clique_game.py
```

This starts a Flask web server (usually at `http://127.0.0.1:8080` or `http://localhost:8080`). Open this address in your web browser.

**Features:**

*   Select the number of vertices (N) and clique size (k).
*   Choose between Symmetric and Asymmetric game modes.
*   Start new games and reset the current game.
*   Make moves by clicking available edges on the board or from the valid moves list.
*   View game state, move history, and valid moves.
*   Visualize the game board with colored edges (blue for Player 1, red for Player 2).
*   **AI Prediction (New!):**
    *   Toggle the "Show AI Predictions" checkbox.
    *   If enabled, a dropdown lists compatible trained models found in the `playable_models/` directory.
    *   **Compatibility:** A model is compatible if the `num_vertices` and `clique_size` (k) saved within the model file match the current game's settings.
    *   Select a model and click "Load Model".
    *   If loaded successfully, the interface will display the model's predicted game outcome (Value) and the probability distribution over valid moves (Policy) for the current state. This shows the model's raw evaluation *without* running MCTS search.
    *   Use the "Show All Found Models" button for debugging to list all `.pth.tar` files in `playable_models/` and their detected V/k parameters.

**Using Your Own Models:**

1.  Ensure your models were trained with the corrected pipeline/training scripts that save `num_vertices`, `clique_size`, and `hidden_dim` within the `.pth.tar` file.
2.  Create a directory named `playable_models` in the project's root directory (`alphazero_clique/playable_models/`).
3.  Copy your trained model files (e.g., `clique_net.pth.tar`, `clique_net_iterN.pth.tar`) into this `playable_models` directory.
4.  Relaunch the interactive game server.

## Running the AlphaZero Pipeline

Train your own models using the full AlphaZero pipeline:

**Example Command:**

```bash
python src/pipeline_clique.py --mode pipeline \
                             --vertices 6 \
                             --k 3 \
                             --iterations 20 \
                             --self-play-games 48 \
                             --mcts-sims 200 \
                             --num-cpus 6 \
                             --hidden-dim 128 \
                             --num-layers 3 \
                             --initial-lr 0.0003 \
                             --lr-factor 0.7 \
                             --lr-patience 5 \
                             --lr-threshold 0.003 \
                             --batch-size 128 \
                             --epochs 10 \
                             --use-legacy-policy-loss \
                             --experiment-name n6k3_h128_l3 \
```

**Pipeline Arguments Detailed:**

Above are main arguments we want to change for the experiment. You can find details on arguments in Command-Line Arguments section. 

*(Other modes like `selfplay`, `train`, `evaluate` are available via `--mode` for running specific parts of the pipeline - see `src/pipeline_clique.py` for details)*

The pipeline will:

1.  Create experiment-specific directories.
2.  Run multiple iterations, generating self-play data, training the network, and evaluating.
3.  Save the best model (e.g., `clique_net.pth.tar`) and per-iteration models (e.g., `clique_net_iterN.pth.tar`) in `./model_data/<experiment_name>/`.
    *   **Model Compatibility:** Models saved by the pipeline now include `num_vertices`, `clique_size`, `hidden_dim`, and `num_layers`, allowing them to be loaded by the interactive game's AI feature.
4.  Store training examples (`.pkl` files) in `./datasets/clique/<experiment_name>/`.
5.  Log training progress and results in `./model_data/<experiment_name>/experiment_log.json`.

## Project Structure

```
alphazero_clique/
├── playable_models/        # Directory to place trained models for the interactive UI
├── model_data/             # Stores models and logs from pipeline runs (organized by experiment)
├── datasets/
│   └── clique/             # Stores self-play game data (organized by experiment)
├── src/
│   ├── pipeline_clique.py      # Main AlphaZero pipeline script & modes
│   ├── interactive_clique_game.py  # Web interface server (Flask)
│   ├── alpha_net_clique.py     # Neural network architecture (CliqueGNN)
│   ├── clique_board.py         # Game logic and board state
│   ├── MCTS_clique.py          # Monte Carlo Tree Search implementation
│   ├── train_clique.py         # Standalone training script & utilities (used by pipeline)
│   ├── encoder_decoder_clique.py # Helpers for board state/action encoding
│   ├── visualize_clique.py     # Board visualization helpers
│   └── check_model_keys.py     # Utility script to inspect saved model files
├── templates/
│   └── index.html            # HTML template for the web interface
├── static/
│   └── ...                   # CSS/JS for the web interface (if separated)
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## License

[Add your license information here]

## Command-Line Arguments

The `pipeline_clique.py` script accepts various command-line arguments to configure the training process:

| Argument                      | Type    | Default     | Description                                                                      |
|-------------------------------|---------|-------------|----------------------------------------------------------------------------------|
| `--mode`                      | str     | `pipeline`  | Execution mode (`pipeline`, `selfplay`, `train`, `evaluate`, `play`).            |
| `--vertices`                  | int     | 6           | Number of vertices in the graph.                                                 |
| `--k`                         | int     | 3           | Size of the clique (`k`) required to win.                                        |
| `--game-mode`                 | str     | `symmetric` | Game rules (`symmetric` or `asymmetric`).                                        |
| `--iterations`                | int     | 5           | Total number of training iterations to run in pipeline mode.                     |
| `--self-play-games`           | int     | 100         | Number of self-play games generated per iteration.                               |
| `--mcts-sims`                 | int     | 200         | Number of MCTS simulations per move during self-play.                            |
| `--eval-threshold`            | float   | 0.55        | Win rate threshold against the previous best model to accept the new model.        |
| `--num-cpus`                  | int     | 4           | Number of CPU processes for parallel self-play game generation.                  |
| `--experiment-name`           | str     | `default`   | Directory name under `./experiments/` to store models, data, and logs.           |
| `--hidden-dim`                | int     | 64          | Hidden dimension size within the GNN layers.                                     |
| `--num-layers`                | int     | 2           | Number of GNN layers (each consisting of a node and edge update block).        |
| `--initial-lr`                | float   | 1e-5        | Initial learning rate for the Adam optimizer.                                    |
| `--lr-factor`                 | float   | 0.7         | Factor by which the learning rate is reduced by the ReduceLROnPlateau scheduler. | 
| `--lr-patience`               | int     | 5           | Number of epochs with no improvement after which learning rate will be reduced.  |
| `--lr-threshold`              | float   | 1e-3        | Threshold for measuring the new optimum, to only focus on significant changes.   |
| `--min-lr`                    | float   | 1e-7        | Lower bound on the learning rate for ReduceLROnPlateau.                          |
| `--batch-size`                | int     | 32          | Batch size used during the training phase.                                       |
| `--epochs`                    | int     | 30          | Number of training epochs performed on the collected data each iteration.      |
| `--use-legacy-policy-loss`    | flag    | `False`     | If set, use the older (potentially unstable) policy loss calculation method.     |
| `--min-alpha`                 | float   | 0.5         | Minimum clipping value for the dynamically calculated value loss weight (`alpha`). | 
| `--max-alpha`                 | float   | 100.0       | Maximum clipping value for the dynamically calculated value loss weight (`alpha`). | 
| `--iteration`                 | int     | 0           | Iteration number (used specifically for `train` mode).                           |
| `--num-games`                 | int     | 21          | Number of games to play (used for `evaluate` and `play` modes).                  |
| `--eval-mcts-sims`            | int     | 30          | Number of MCTS simulations per move during evaluation/play modes.                |

**Note:** The `min-alpha` and `max-alpha` arguments define the clipping range for the value loss weight, which is dynamically calculated as `alpha = policy_loss.detach() / (value_loss.detach() + 1e-6)` during training in `alpha_net_clique.py`.

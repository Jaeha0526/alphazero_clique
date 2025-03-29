# AlphaZero Clique Game

This project implements the AlphaZero algorithm for the Clique Game, using Graph Neural Networks and Monte Carlo Tree Search.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/alphazero_clique.git
cd alphazero_clique
```

2. Install the required dependencies:
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

## Testing the Game

To test the Clique Game with an interactive web interface:

```bash
python src/interactive_clique_game.py
```

This will:
1. Start a Flask web server
2. Open your default web browser to `http://localhost:8080`
3. Allow you to:
   - Select the number of vertices (N) and clique size (k)
   - Start a new game
   - Make moves by clicking on available edges
   - View game state, move history, and valid moves
   - See the game board with colored edges (blue for Player 1, red for Player 2)

## Running the AlphaZero Pipeline

To train the model using the AlphaZero pipeline:

```bash
python src/pipeline_clique.py --mode pipeline --vertices 6 --clique-size 3 --iterations 3 --self-play-games 2 --mcts-sims 50 --eval-threshold 0.55
```

This command runs the full AlphaZero training pipeline with the following parameters:
- `--mode pipeline`: Runs the complete AlphaZero training pipeline
- `--vertices 6`: Sets the number of vertices in the graph to 6
- `--clique-size 3`: Sets the required clique size for Player 1 to win to 3
- `--iterations 3`: Runs 3 training iterations
- `--self-play-games 2`: Plays 2 self-play games per iteration
- `--mcts-sims 50`: Uses 50 Monte Carlo Tree Search simulations per move
- `--eval-threshold 0.55`: Updates the best model if win rate exceeds 55%

The pipeline will:
1. Create necessary directories for model data and datasets
2. Run multiple iterations of:
   - Self-play games to generate training data
   - Neural network training on collected examples
   - Model evaluation against the best model
3. Save the best model in `./model_data/`
4. Store training examples in `./datasets/clique/`

## Project Structure

```
alphazero_clique/
├── src/
│   ├── pipeline_clique.py      # Main AlphaZero pipeline
│   ├── interactive_clique_game.py  # Web interface
│   ├── alpha_net_clique.py     # Neural network architecture
│   ├── clique_board.py         # Game board implementation
│   ├── MCTS_clique.py          # Monte Carlo Tree Search
│   └── train_clique.py         # Training utilities
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Game Rules

The Clique Game is played on an undirected graph with N vertices:
1. Players take turns adding edges to the graph
2. Player 1 wins by creating a clique of size k
3. Player 2 wins by preventing Player 1 from creating a k-clique
4. The game ends when either player wins or the graph is complete

## License

[Add your license information here]

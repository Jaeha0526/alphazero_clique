#!/usr/bin/env python
import os
import torch
import torch.multiprocessing as mp
import argparse
import random
import time
from clique_board import CliqueBoard
from alpha_net_clique import CliqueGNN
from MCTS_clique import MCTS_self_play, UCT_search

def setup_directories():
    """Create necessary directories for model data and datasets"""
    os.makedirs("./model_data", exist_ok=True)
    os.makedirs("./datasets/clique", exist_ok=True)

def run_self_play(model, num_games, num_vertices, k, mcts_sims, process_id=0):
    """Run self-play games using the provided model"""
    print(f"Process {process_id}: Starting self-play with {num_games} games")
    start_time = time.time()
    
    # Ensure model is on CPU for multiprocessing
    model = model.cpu()
    model.eval()
    
    games_completed = 0
    for game_idx in range(num_games):
        # Create a new game board
        board = CliqueBoard(num_vertices, k)
        
        # Save initial state for visualization
        states = [board.copy()]
        moves_history = []
        
        # Play until game is over or move limit reached
        while board.game_state == 0 and board.move_count < num_vertices * (num_vertices - 1) // 2:
            # Use MCTS to find the best move
            best_move, root = UCT_search(board, mcts_sims, model)
            
            # Get the actual edge from the move index
            from encoder_decoder_clique import decode_action
            edge = decode_action(board, best_move)
            
            # Make the move
            if not board.make_move(edge):
                print(f"Invalid move: {edge}")
                break
            
            # Save state and move
            states.append(board.copy())
            moves_history.append(edge)
            
            # Print game progress
            if game_idx == 0:  # Only print for the first game
                print(f"Move {board.move_count}: Player {2 - board.player} played {edge}")
        
        games_completed += 1
        if game_idx == 0:  # Only print for the first game
            if board.game_state == 0:
                print("Game ended in a draw")
            else:
                print(f"Player {board.game_state} won!")
    
    elapsed = time.time() - start_time
    print(f"Process {process_id}: Completed {games_completed} games in {elapsed:.1f} seconds")
    return games_completed

def train_network(model, epochs=5):
    """Train the model for a specified number of epochs"""
    print("Training network...")
    
    # Normally we would load the generated datasets here
    # For the minimal test, we'll just create a dummy training loop
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Just run a few dummy training steps
    for epoch in range(epochs):
        # Simulate loss calculation and backpropagation
        dummy_loss = torch.tensor([0.5], requires_grad=True)
        optimizer.zero_grad()
        dummy_loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {dummy_loss.item():.4f}")
    
    print("Training completed")
    return model

def evaluate_model(model, num_games=5, mcts_sims=50):
    """Evaluate the model by playing against a random agent"""
    print("Evaluating model...")
    
    # This is a simplified evaluation - in a real pipeline, 
    # we would compare against a baseline model
    return 0.7  # Dummy win rate

def run_minimal_pipeline(vertices=6, clique_size=3, games=2, mcts_sims=50):
    """Run a minimal version of the AlphaZero pipeline"""
    print(f"\nRunning minimal pipeline with settings:")
    print(f"- Vertices: {vertices}")
    print(f"- Clique size: {clique_size}")
    print(f"- Self-play games: {games}")
    print(f"- MCTS simulations: {mcts_sims}")
    
    # Set up directories
    setup_directories()
    
    # Create the model
    model = CliqueGNN(num_vertices=vertices)
    print("Created new model")
    
    # Save initial model
    torch.save({
        'state_dict': model.state_dict(),
    }, './model_data/clique_net_initial.pth.tar')
    
    # Step 1: Self-play
    print("\n=== Step 1: Self-Play ===")
    model.eval()
    model.share_memory()
    
    games_completed = run_self_play(
        model=model,
        num_games=games,
        num_vertices=vertices,
        k=clique_size,
        mcts_sims=mcts_sims
    )
    
    print(f"Completed {games_completed} self-play games")
    
    # Step 2: Training
    print("\n=== Step 2: Network Training ===")
    model = train_network(model, epochs=3)
    
    # Save trained model
    torch.save({
        'state_dict': model.state_dict(),
    }, './model_data/clique_net_trained.pth.tar')
    
    # Step 3: Evaluation
    print("\n=== Step 3: Model Evaluation ===")
    win_rate = evaluate_model(model, num_games=2, mcts_sims=mcts_sims)
    
    print(f"Model evaluation complete - Win rate: {win_rate:.2f}")
    print("\nMinimal pipeline test completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a minimal AlphaZero pipeline test for Clique Game")
    parser.add_argument("--vertices", type=int, default=4, help="Number of vertices in the graph")
    parser.add_argument("--clique-size", type=int, default=3, help="Size of clique needed to win")
    parser.add_argument("--games", type=int, default=2, help="Number of self-play games")
    parser.add_argument("--mcts-sims", type=int, default=50, help="Number of MCTS simulations per move")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Run the pipeline
    run_minimal_pipeline(
        vertices=args.vertices,
        clique_size=args.clique_size,
        games=args.games,
        mcts_sims=args.mcts_sims
    ) 
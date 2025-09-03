#!/usr/bin/env python3
"""
Simple Game Trajectory Viewer for AlphaZero Clique Game
A text-based viewer with optional matplotlib visualization
"""

import pickle
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple


class SimpleGameViewer:
    def __init__(self, filepath: str):
        """Initialize the viewer with a game data file."""
        self.filepath = filepath
        self.load_game_data()
        self.current_game_idx = 0
        self.current_move_idx = 0
    
    def load_game_data(self):
        """Load and parse the game data file."""
        print(f"\nLoading game data from: {self.filepath}")
        
        with open(self.filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.iteration = data.get('iteration', 'unknown')
        self.n = data['vertices']
        self.k = data['k']
        self.game_mode = data.get('game_mode', 'symmetric')
        self.games = data['sample_games']
        self.num_games = len(self.games)
        
        print(f"Loaded {self.num_games} games from iteration {self.iteration}")
        print(f"Graph: n={self.n}, k={self.k}, mode={self.game_mode}")
        
        # Precompute edge mappings
        self.edge_to_idx = {}
        self.idx_to_edge = {}
        idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.edge_to_idx[(i, j)] = idx
                self.edge_to_idx[(j, i)] = idx  # Undirected
                self.idx_to_edge[idx] = (i, j)
                idx += 1
        
        self.num_edges = idx
    
    def display_game_state(self, game_idx: int, move_idx: int):
        """Display the game state at a specific move."""
        game = self.games[game_idx]
        
        print("\n" + "="*60)
        print(f"Game {game_idx + 1}/{self.num_games} - Move {move_idx + 1}/{len(game['moves'])}")
        print("="*60)
        
        # Build edge state
        edge_colors = ['_'] * self.num_edges  # _ = uncolored
        
        for m_idx in range(min(move_idx + 1, len(game['moves']))):
            move = game['moves'][m_idx]
            if move['top_actions']:
                action = move['top_actions'][0][0]
                player = move['player']
                edge_colors[action] = str(player + 1)  # 1 or 2
        
        # Display as adjacency matrix
        print("\nAdjacency Matrix (1=Player1, 2=Player2, _=uncolored):")
        print("    ", end="")
        for j in range(self.n):
            print(f"{j:3}", end="")
        print()
        
        for i in range(self.n):
            print(f"{i:3}: ", end="")
            for j in range(self.n):
                if i == j:
                    print("  .", end="")
                elif i < j:
                    edge_idx = self.edge_to_idx[(i, j)]
                    print(f"  {edge_colors[edge_idx]}", end="")
                else:
                    edge_idx = self.edge_to_idx[(j, i)]
                    print(f"  {edge_colors[edge_idx]}", end="")
            print()
        
        # Display current move info
        if move_idx < len(game['moves']):
            move = game['moves'][move_idx]
            print(f"\nCurrent Move Info:")
            print(f"  Player: {move['player'] + 1}", end="")
            if move.get('player_role') is not None:
                role = "Attacker" if move['player_role'] == 0 else "Defender"
                print(f" ({role})")
            else:
                print()
            
            print(f"  Final Value: {move['value']:+.2f}")
            
            if move['top_actions']:
                print(f"\n  Top 5 Move Probabilities:")
                for i, (action, prob) in enumerate(move['top_actions'][:5]):
                    edge = self.idx_to_edge[action]
                    print(f"    {i+1}. Edge {edge}: {prob:.3f}")
        
        # Count edges
        p1_edges = edge_colors.count('1')
        p2_edges = edge_colors.count('2')
        uncolored = edge_colors.count('_')
        
        print(f"\nEdge Count:")
        print(f"  Player 1 (Red): {p1_edges}")
        print(f"  Player 2 (Blue): {p2_edges}")
        print(f"  Uncolored: {uncolored}")
        print(f"  Total: {self.num_edges}")
    
    def interactive_mode(self):
        """Run interactive text-based viewer."""
        print("\n" + "="*60)
        print("INTERACTIVE GAME VIEWER")
        print("="*60)
        print("\nCommands:")
        print("  g <num>  - Go to game number (0-based)")
        print("  m <num>  - Go to move number (0-based)")
        print("  n        - Next move")
        print("  p        - Previous move")
        print("  r        - Reset to first move")
        print("  ng       - Next game")
        print("  pg       - Previous game")
        print("  s        - Show current state")
        print("  info     - Show game info")
        print("  q        - Quit")
        print("="*60)
        
        self.display_game_state(self.current_game_idx, self.current_move_idx)
        
        while True:
            try:
                cmd = input("\n> ").strip().lower().split()
                
                if not cmd:
                    continue
                
                if cmd[0] == 'q':
                    break
                
                elif cmd[0] == 'g' and len(cmd) > 1:
                    game_idx = int(cmd[1])
                    if 0 <= game_idx < self.num_games:
                        self.current_game_idx = game_idx
                        self.current_move_idx = 0
                        self.display_game_state(self.current_game_idx, self.current_move_idx)
                    else:
                        print(f"Invalid game index. Must be 0-{self.num_games-1}")
                
                elif cmd[0] == 'm' and len(cmd) > 1:
                    move_idx = int(cmd[1])
                    game = self.games[self.current_game_idx]
                    if 0 <= move_idx < len(game['moves']):
                        self.current_move_idx = move_idx
                        self.display_game_state(self.current_game_idx, self.current_move_idx)
                    else:
                        print(f"Invalid move index. Must be 0-{len(game['moves'])-1}")
                
                elif cmd[0] == 'n':
                    game = self.games[self.current_game_idx]
                    if self.current_move_idx < len(game['moves']) - 1:
                        self.current_move_idx += 1
                        self.display_game_state(self.current_game_idx, self.current_move_idx)
                    else:
                        print("Already at last move")
                
                elif cmd[0] == 'p':
                    if self.current_move_idx > 0:
                        self.current_move_idx -= 1
                        self.display_game_state(self.current_game_idx, self.current_move_idx)
                    else:
                        print("Already at first move")
                
                elif cmd[0] == 'r':
                    self.current_move_idx = 0
                    self.display_game_state(self.current_game_idx, self.current_move_idx)
                
                elif cmd[0] == 'ng':
                    if self.current_game_idx < self.num_games - 1:
                        self.current_game_idx += 1
                        self.current_move_idx = 0
                        self.display_game_state(self.current_game_idx, self.current_move_idx)
                    else:
                        print("Already at last game")
                
                elif cmd[0] == 'pg':
                    if self.current_game_idx > 0:
                        self.current_game_idx -= 1
                        self.current_move_idx = 0
                        self.display_game_state(self.current_game_idx, self.current_move_idx)
                    else:
                        print("Already at first game")
                
                elif cmd[0] == 's':
                    self.display_game_state(self.current_game_idx, self.current_move_idx)
                
                elif cmd[0] == 'info':
                    print(f"\nFile: {self.filepath}")
                    print(f"Iteration: {self.iteration}")
                    print(f"Game Mode: {self.game_mode}")
                    print(f"Graph: n={self.n}, k={self.k}")
                    print(f"Total Games: {self.num_games}")
                    
                    # Show game lengths
                    lengths = [len(g['moves']) for g in self.games]
                    print(f"Game Lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
                
                else:
                    print("Unknown command. Type 'q' to quit.")
                    
            except (ValueError, IndexError) as e:
                print(f"Error: {e}")
            except KeyboardInterrupt:
                print("\nUse 'q' to quit")
    
    def export_game_to_text(self, game_idx: int, output_file: str = None):
        """Export a game's full trajectory to text file."""
        game = self.games[game_idx]
        
        lines = []
        lines.append(f"Game {game_idx} from {self.filepath}")
        lines.append(f"Iteration: {self.iteration}")
        lines.append(f"Graph: n={self.n}, k={self.k}, mode={self.game_mode}")
        lines.append(f"Total Moves: {len(game['moves'])}")
        lines.append("="*60)
        
        for move_idx, move in enumerate(game['moves']):
            lines.append(f"\nMove {move_idx + 1}:")
            lines.append(f"  Player: {move['player'] + 1}")
            if move.get('player_role') is not None:
                role = "Attacker" if move['player_role'] == 0 else "Defender"
                lines.append(f"  Role: {role}")
            lines.append(f"  Value: {move['value']:+.2f}")
            
            if move['top_actions']:
                lines.append("  Top Actions:")
                for i, (action, prob) in enumerate(move['top_actions'][:5]):
                    edge = self.idx_to_edge[action]
                    lines.append(f"    {i+1}. Edge {edge}: {prob:.3f}")
        
        text = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(text)
            print(f"Exported game {game_idx} to {output_file}")
        else:
            print(text)


def main():
    parser = argparse.ArgumentParser(description='Simple Game Trajectory Viewer')
    parser.add_argument('filepath', help='Path to game data pickle file')
    parser.add_argument('--game', type=int, help='Specific game index to view')
    parser.add_argument('--export', type=str, help='Export game to text file')
    parser.add_argument('--all-moves', action='store_true', help='Show all moves (non-interactive)')
    
    args = parser.parse_args()
    
    if not Path(args.filepath).exists():
        print(f"Error: File {args.filepath} not found!")
        return
    
    viewer = SimpleGameViewer(args.filepath)
    
    if args.export and args.game is not None:
        viewer.export_game_to_text(args.game, args.export)
    elif args.all_moves and args.game is not None:
        # Show all moves for a specific game
        game = viewer.games[args.game]
        for move_idx in range(len(game['moves'])):
            viewer.display_game_state(args.game, move_idx)
            input("Press Enter for next move...")
    elif args.game is not None:
        # Show specific game at first move
        viewer.current_game_idx = args.game
        viewer.display_game_state(args.game, 0)
    else:
        # Interactive mode
        viewer.interactive_mode()


if __name__ == "__main__":
    main()
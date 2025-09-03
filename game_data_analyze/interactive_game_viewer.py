#!/usr/bin/env python3
"""
Interactive Game Trajectory Viewer for AlphaZero Clique Game

This tool allows you to:
1. Load game data from pickle files
2. Select a specific game to visualize
3. Step through moves manually or auto-play
4. See the graph evolution and move probabilities
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, TextBox
import networkx as nx
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import matplotlib.patches as mpatches


class GameTrajectoryViewer:
    def __init__(self, filepath: str):
        """Initialize the viewer with a game data file."""
        self.filepath = filepath
        self.load_game_data()
        self.current_game_idx = 0
        self.current_move_idx = 0
        self.is_playing = False
        self.animation = None
        
        # Setup the figure and UI
        self.setup_figure()
        self.setup_ui()
        self.update_display()
    
    def load_game_data(self):
        """Load and parse the game data file."""
        print(f"Loading game data from: {self.filepath}")
        
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
    
    def setup_figure(self):
        """Setup the matplotlib figure with subplots."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(f'Game Trajectory Viewer - n={self.n}, k={self.k}, mode={self.game_mode}')
        
        # Create grid for subplots
        gs = self.fig.add_gridspec(3, 3, height_ratios=[4, 1, 0.5], width_ratios=[2, 2, 1])
        
        # Main graph display
        self.ax_graph = self.fig.add_subplot(gs[0, :2])
        self.ax_graph.set_title('Game Board')
        self.ax_graph.axis('off')
        
        # Move probabilities
        self.ax_probs = self.fig.add_subplot(gs[0, 2])
        self.ax_probs.set_title('Top Move Probabilities')
        
        # Game info text
        self.ax_info = self.fig.add_subplot(gs[1, :])
        self.ax_info.axis('off')
        
        # Controls area (row 2)
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        self.ax_controls.axis('off')
        
        # Initialize graph
        self.G = nx.complete_graph(self.n)
        self.pos = nx.spring_layout(self.G, seed=42)  # Fixed layout
    
    def setup_ui(self):
        """Setup UI controls."""
        # Move slider
        self.move_slider = Slider(
            plt.axes([0.15, 0.15, 0.5, 0.03]),
            'Move', 0, 1, valinit=0, valstep=1
        )
        self.move_slider.on_changed(self.on_move_change)
        
        # Game selector
        self.game_input = TextBox(
            plt.axes([0.15, 0.08, 0.1, 0.04]),
            'Game:', initial=str(self.current_game_idx)
        )
        self.game_input.on_submit(self.on_game_change)
        
        # Control buttons
        self.btn_prev_move = Button(plt.axes([0.30, 0.08, 0.08, 0.04]), '← Prev')
        self.btn_prev_move.on_clicked(self.prev_move)
        
        self.btn_next_move = Button(plt.axes([0.39, 0.08, 0.08, 0.04]), 'Next →')
        self.btn_next_move.on_clicked(self.next_move)
        
        self.btn_play = Button(plt.axes([0.48, 0.08, 0.08, 0.04]), '▶ Play')
        self.btn_play.on_clicked(self.toggle_play)
        
        self.btn_reset = Button(plt.axes([0.57, 0.08, 0.08, 0.04]), '↺ Reset')
        self.btn_reset.on_clicked(self.reset_game)
        
        # Speed control
        self.speed_slider = Slider(
            plt.axes([0.70, 0.08, 0.15, 0.03]),
            'Speed', 0.5, 3.0, valinit=1.0
        )
    
    def get_game_state_at_move(self, game_idx: int, move_idx: int) -> Dict:
        """Reconstruct game state at a specific move."""
        game = self.games[game_idx]
        
        # Initialize edge colors: 0=uncolored, 1=player1(red), 2=player2(blue)
        edge_colors = {}
        for i in range(self.num_edges):
            edge_colors[i] = 0
        
        # Apply moves up to move_idx
        for m_idx in range(min(move_idx + 1, len(game['moves']))):
            move = game['moves'][m_idx]
            # Get the action from top_actions (highest probability)
            if move['top_actions']:
                action = move['top_actions'][0][0]
                player = move['player']
                edge_colors[action] = player + 1
        
        return {
            'edge_colors': edge_colors,
            'current_move': game['moves'][move_idx] if move_idx < len(game['moves']) else None,
            'total_moves': len(game['moves'])
        }
    
    def update_display(self):
        """Update the entire display for current game and move."""
        game = self.games[self.current_game_idx]
        state = self.get_game_state_at_move(self.current_game_idx, self.current_move_idx)
        
        # Clear and redraw graph
        self.ax_graph.clear()
        self.ax_graph.set_title(f'Game {self.current_game_idx + 1}/{self.num_games} - Move {self.current_move_idx + 1}/{state["total_moves"]}')
        self.ax_graph.axis('off')
        
        # Draw edges with colors
        edge_list = []
        edge_colors_list = []
        edge_widths = []
        
        for action_idx, color in state['edge_colors'].items():
            edge = self.idx_to_edge[action_idx]
            edge_list.append(edge)
            
            if color == 0:
                edge_colors_list.append('lightgray')
                edge_widths.append(1)
            elif color == 1:
                edge_colors_list.append('red')
                edge_widths.append(3)
            else:  # color == 2
                edge_colors_list.append('blue')
                edge_widths.append(3)
        
        # Highlight the current move
        if state['current_move'] and state['current_move']['top_actions']:
            current_action = state['current_move']['top_actions'][0][0]
            current_edge = self.idx_to_edge[current_action]
            
            # Draw all edges first
            nx.draw_networkx_edges(self.G, self.pos, edgelist=edge_list,
                                  edge_color=edge_colors_list, width=edge_widths,
                                  ax=self.ax_graph)
            
            # Highlight current edge
            nx.draw_networkx_edges(self.G, self.pos, edgelist=[current_edge],
                                  edge_color='yellow', width=5, alpha=0.5,
                                  ax=self.ax_graph)
        else:
            nx.draw_networkx_edges(self.G, self.pos, edgelist=edge_list,
                                  edge_color=edge_colors_list, width=edge_widths,
                                  ax=self.ax_graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, self.pos, node_color='lightblue',
                             node_size=500, ax=self.ax_graph)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax_graph)
        
        # Add legend
        red_patch = mpatches.Patch(color='red', label='Player 1')
        blue_patch = mpatches.Patch(color='blue', label='Player 2')
        yellow_patch = mpatches.Patch(color='yellow', alpha=0.5, label='Current Move')
        self.ax_graph.legend(handles=[red_patch, blue_patch, yellow_patch], loc='upper right')
        
        # Update probability bar chart
        self.ax_probs.clear()
        self.ax_probs.set_title('Top Move Probabilities')
        
        if state['current_move']:
            move = state['current_move']
            if move['top_actions']:
                actions = [f"Edge {a}" for a, _ in move['top_actions'][:10]]
                probs = [p for _, p in move['top_actions'][:10]]
                
                colors = ['yellow' if i == 0 else 'gray' for i in range(len(actions))]
                self.ax_probs.barh(range(len(actions)), probs, color=colors)
                self.ax_probs.set_yticks(range(len(actions)))
                self.ax_probs.set_yticklabels(actions)
                self.ax_probs.set_xlabel('Probability')
                self.ax_probs.set_xlim(0, max(probs) * 1.1 if probs else 1)
                self.ax_probs.invert_yaxis()
        
        # Update info text
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_text = f"Iteration: {self.iteration}\n"
        info_text += f"Game Mode: {self.game_mode}\n"
        
        if state['current_move']:
            move = state['current_move']
            player_name = f"Player {move['player'] + 1}"
            if move.get('player_role') is not None:
                role = "Attacker" if move['player_role'] == 0 else "Defender"
                player_name += f" ({role})"
            
            info_text += f"Current Player: {player_name}\n"
            info_text += f"Final Value: {move['value']:+.2f}\n"
            
            # Add edge info
            if move['top_actions']:
                best_action = move['top_actions'][0][0]
                best_edge = self.idx_to_edge[best_action]
                best_prob = move['top_actions'][0][1]
                info_text += f"Best Move: Edge {best_edge} (prob={best_prob:.3f})"
        
        self.ax_info.text(0.5, 0.5, info_text, ha='center', va='center',
                         fontsize=11, transform=self.ax_info.transAxes)
        
        # Update slider range
        self.move_slider.set_val(self.current_move_idx)
        self.move_slider.valmax = len(game['moves']) - 1
        self.move_slider.ax.set_xlim(0, len(game['moves']) - 1)
        
        plt.draw()
    
    def on_move_change(self, val):
        """Handle move slider change."""
        self.current_move_idx = int(val)
        self.update_display()
    
    def on_game_change(self, text):
        """Handle game selection change."""
        try:
            game_idx = int(text)
            if 0 <= game_idx < self.num_games:
                self.current_game_idx = game_idx
                self.current_move_idx = 0
                self.update_display()
            else:
                print(f"Game index {game_idx} out of range (0-{self.num_games-1})")
        except ValueError:
            print(f"Invalid game index: {text}")
    
    def prev_move(self, event):
        """Go to previous move."""
        if self.current_move_idx > 0:
            self.current_move_idx -= 1
            self.update_display()
    
    def next_move(self, event):
        """Go to next move."""
        game = self.games[self.current_game_idx]
        if self.current_move_idx < len(game['moves']) - 1:
            self.current_move_idx += 1
            self.update_display()
    
    def reset_game(self, event):
        """Reset to first move."""
        self.current_move_idx = 0
        self.update_display()
    
    def toggle_play(self, event):
        """Toggle auto-play."""
        if self.is_playing:
            self.stop_play()
        else:
            self.start_play()
    
    def start_play(self):
        """Start auto-playing moves."""
        self.is_playing = True
        self.btn_play.label.set_text('⏸ Pause')
        
        def animate(frame):
            if not self.is_playing:
                return
            
            game = self.games[self.current_game_idx]
            if self.current_move_idx < len(game['moves']) - 1:
                self.current_move_idx += 1
                self.update_display()
            else:
                self.stop_play()
        
        interval = int(1000 / self.speed_slider.val)  # Convert speed to interval
        self.animation = animation.FuncAnimation(
            self.fig, animate, interval=interval, repeat=False
        )
        plt.draw()
    
    def stop_play(self):
        """Stop auto-playing."""
        self.is_playing = False
        self.btn_play.label.set_text('▶ Play')
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
    
    def show(self):
        """Display the viewer."""
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Interactive Game Trajectory Viewer')
    parser.add_argument('filepath', help='Path to game data pickle file')
    
    args = parser.parse_args()
    
    if not Path(args.filepath).exists():
        print(f"Error: File {args.filepath} not found!")
        return
    
    viewer = GameTrajectoryViewer(args.filepath)
    viewer.show()


if __name__ == "__main__":
    main()
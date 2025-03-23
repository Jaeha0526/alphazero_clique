#!/usr/bin/env python

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def view_clique_board(board):
    """
    Visualize the clique game board as a graph
    
    Parameters:
    board (CliqueBoard): The clique game board to visualize
    
    Returns:
    fig: The matplotlib figure
    """
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(board.num_vertices):
        G.add_node(i)
    
    # Add edges with colors based on edge states
    edge_colors = []
    for i in range(board.num_vertices):
        for j in range(i+1, board.num_vertices):
            G.add_edge(i, j)
            state = board.edge_states[i,j]
            if state == 0:  # Unselected
                edge_colors.append('lightgray')
            elif state == 1:  # Player 1
                edge_colors.append('blue')
            else:  # Player 2
                edge_colors.append('red')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Position nodes in a circle
    pos = nx.circular_layout(G)
    
    # Draw nodes (always black)
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=700, alpha=0.8)
    
    # Draw edges with colors
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2.0, alpha=0.8)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold', font_color='white')
    
    # Set title
    current_player = "Player 1 (Blue)" if board.player == 0 else "Player 2 (Red)"
    game_state = "Ongoing" if board.game_state == 0 else f"Player {board.game_state} Wins!"
    plt.title(f"Clique Game (n={board.num_vertices}) - Move {board.move_count}\nCurrent: {current_player} - {game_state}")
    
    # Remove axis
    plt.axis('off')
    
    return fig

def animate_game(game_states):
    """
    Create multiple visualizations for a sequence of game states
    
    Parameters:
    game_states (list): List of CliqueBoard objects representing game states
    
    Returns:
    figs: List of matplotlib figures
    """
    figs = []
    for board in game_states:
        fig = view_clique_board(board)
        figs.append(fig)
    return figs

# Example usage (will be replaced by clique_board.py)
if __name__ == "__main__":
    from clique_board import CliqueBoard
    
    # Create a game with 6 vertices
    board = CliqueBoard(6)
    
    # Make some moves
    board.make_move((0,1))  # Player 1 selects edge (0,1)
    board.make_move((1,2))  # Player 2 selects edge (1,2)
    board.make_move((2,3))  # Player 1 selects edge (2,3)
    
    # Visualize the board
    fig = view_clique_board(board)
    plt.show() 
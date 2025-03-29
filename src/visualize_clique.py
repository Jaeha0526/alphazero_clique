#!/usr/bin/env python

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def get_edge_positions(G, pos):
    """
    Calculate midpoint positions of all edges in the graph for click detection
    
    Parameters:
    G (nx.Graph): The graph
    pos (dict): Node positions dictionary
    
    Returns:
    list: List of dictionaries with edge positions and vertices
    """
    edge_positions = []
    
    for edge in G.edges():
        v1, v2 = edge
        # Get node positions
        x1, y1 = pos[v1]
        x2, y2 = pos[v2]
        
        # Calculate midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Store edge position and vertices
        edge_positions.append({
            'v1': int(v1),
            'v2': int(v2),
            'x1': float(x1),
            'y1': float(y1),
            'x2': float(x2),
            'y2': float(y2),
            'mid_x': float(mid_x),
            'mid_y': float(mid_y)
        })
    
    return edge_positions

def view_clique_board(board, return_edge_positions=False):
    """
    Visualize the clique game board as a graph
    
    Parameters:
    board (CliqueBoard): The clique game board to visualize
    return_edge_positions (bool): Whether to return edge positions for click detection
    
    Returns:
    fig: The matplotlib figure
    edge_positions (optional): List of edge positions if return_edge_positions is True
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
    
    # Get game state text
    if board.game_state == 0:
        game_state = "Ongoing"
    elif board.game_state == 1:
        game_state = "Player 1 (Blue) Wins!"
    elif board.game_state == 2:
        game_state = "Player 2 (Red) Wins!"
    else:
        game_state = "Draw!"
    
    # Get game mode text
    game_mode = "Asymmetric" if board.game_mode == "asymmetric" else "Symmetric"
    if board.game_mode == "asymmetric":
        mode_info = "P1: Form k-clique, P2: Prevent"
    else:
        mode_info = "Both players try to form k-clique"
    
    plt.title(f"Clique Game (n={board.num_vertices}, k={board.k}, {game_mode})\n"
              f"Move {board.move_count} - Current: {current_player}\n"
              f"{game_state} - {mode_info}")
    
    # Add instruction for clicking
    plt.figtext(0.5, 0.01, "Click on an edge to make a move", 
                ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Remove axis
    plt.axis('off')
    
    # Return edge positions if requested
    if return_edge_positions:
        edge_pos = get_edge_positions(G, pos)
        return fig, edge_pos
    
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
    board = CliqueBoard(6, 3, "asymmetric")
    
    # Make some moves
    board.make_move((0,1))  # Player 1 selects edge (0,1)
    board.make_move((1,2))  # Player 2 selects edge (1,2)
    board.make_move((2,3))  # Player 1 selects edge (2,3)
    
    # Visualize the board
    fig, edge_positions = view_clique_board(board, return_edge_positions=True)
    print(f"Edge positions: {edge_positions}")
    plt.show()
    
    # Create a symmetric game
    print("\nSymmetric mode:")
    board = CliqueBoard(6, 3, "symmetric")
    
    # Make some moves
    board.make_move((0,1))  # Player 1 selects edge (0,1)
    board.make_move((1,2))  # Player 2 selects edge (1,2)
    board.make_move((2,0))  # Player 1 selects edge (2,0) to form a triangle (wins)
    
    # Visualize the board
    fig = view_clique_board(board)
    plt.show() 
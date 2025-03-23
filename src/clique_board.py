#!/usr/bin/env python

import numpy as np
import copy
from itertools import combinations

class CliqueBoard:
    def __init__(self, num_vertices, k):
        """
        Initialize a complete graph with given number of vertices
        num_vertices: number of vertices in the graph
        k: size of clique needed for Player 1 to win
        """
        self.num_vertices = num_vertices
        self.k = k  # Size of clique needed for Player 1 to win
        # Initialize adjacency matrix (complete graph)
        self.adjacency_matrix = np.ones((num_vertices, num_vertices)) - np.eye(num_vertices)
        # Initialize edge states (0: unselected, 1: player1, 2: player2)
        self.edge_states = np.zeros((num_vertices, num_vertices), dtype=int)
        # Current player (0: player1, 1: player2)
        self.player = 0
        # Move count
        self.move_count = 0
        # Game state (0: ongoing, 1: player1 wins, 2: player2 wins)
        self.game_state = 0
        
    def get_valid_moves(self):
        """Get list of valid moves (unselected edges)"""
        valid_moves = []
        for i in range(self.num_vertices):
            for j in range(i+1, self.num_vertices):
                if self.edge_states[i,j] == 0:
                    valid_moves.append((i,j))
        return valid_moves
    
    def make_move(self, edge):
        """
        Make a move by selecting an edge
        edge: tuple of (vertex1, vertex2)
        Returns: True if move is valid, False otherwise
        """
        if edge not in self.get_valid_moves():
            return False
            
        v1, v2 = edge
        self.edge_states[v1,v2] = self.player + 1
        self.edge_states[v2,v1] = self.player + 1  # Symmetric matrix
        self.move_count += 1
        
        # Check for win condition (forming a clique)
        if self.check_win_condition():
            self.game_state = self.player + 1
        # Switch player
        self.player = 1 - self.player
        return True
    
    def check_win_condition(self):
        """
        Check if current player has won
        Player 1 wins by forming a k-clique
        Player 2 wins by preventing Player 1 from forming a k-clique
        Returns: True if current player has won, False otherwise
        """
        if self.player == 0:  # Player 1's turn
            # Check if Player 1 has formed a k-clique
            player1_edges = []
            for i in range(self.num_vertices):
                for j in range(i+1, self.num_vertices):
                    if self.edge_states[i,j] == 1:
                        player1_edges.append((i,j))
            
            # Get all vertices involved in Player 1's edges
            player1_vertices = set()
            for v1, v2 in player1_edges:
                player1_vertices.add(v1)
                player1_vertices.add(v2)
            
            # Check if any subset of vertices forms a k-clique
            for vertices in self._get_combinations(list(player1_vertices), self.k):
                if self._is_clique(vertices, 1):
                    self.game_state = 1  # Player 1 wins
                    return True
        else:  # Player 2's turn
            # Check if Player 2 has prevented Player 1 from forming a k-clique
            # by checking if there are any remaining valid moves for Player 1
            valid_moves = self.get_valid_moves()
            if not valid_moves:  # No more moves possible
                self.game_state = 2  # Player 2 wins by preventing k-clique
                return True
        return False
    
    def _get_combinations(self, lst, r):
        """Helper function to get all combinations of size r from a list"""
        return list(combinations(lst, r))
    
    def _is_clique(self, vertices, player):
        """Check if given vertices form a clique with player's edges"""
        for i in vertices:
            for j in vertices:
                if i != j and self.edge_states[i,j] != player:
                    return False
        return True
    
    def get_board_state(self):
        """Get current board state as a dictionary"""
        return {
            'adjacency_matrix': self.adjacency_matrix.copy(),
            'edge_states': self.edge_states.copy(),
            'player': self.player,
            'move_count': self.move_count,
            'game_state': self.game_state
        }
    
    def copy(self):
        """Create a deep copy of the board"""
        new_board = CliqueBoard(self.num_vertices, self.k)
        new_board.adjacency_matrix = self.adjacency_matrix.copy()
        new_board.edge_states = self.edge_states.copy()
        new_board.player = self.player
        new_board.move_count = self.move_count
        new_board.game_state = self.game_state
        return new_board
    
    def __str__(self):
        """String representation of the board"""
        s = f"Clique Game Board (n={self.num_vertices}, k={self.k})\n"
        s += f"Current Player: {'Player 1' if self.player == 0 else 'Player 2'}\n"
        s += f"Move Count: {self.move_count}\n"
        s += f"Game State: {'Ongoing' if self.game_state == 0 else f'Player {self.game_state} Wins'}\n\n"
        
        # Print edge states
        s += "Edge States:\n"
        for i in range(self.num_vertices):
            for j in range(i+1, self.num_vertices):
                state = self.edge_states[i,j]
                state_str = "Unselected" if state == 0 else f"Player {state}"
                s += f"Edge ({i},{j}): {state_str}\n"
        
        return s

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        from visualize_clique import view_clique_board
        has_visualization = True
    except ImportError:
        print("Visualization not available. Install networkx and matplotlib for visualization.")
        has_visualization = False
    
    # Create a game with 6 vertices and k=3 (need to form a triangle to win)
    board = CliqueBoard(6, 3)
    print(board)
    
    # Store game states for visualization
    game_states = [board.copy()]
    
    # Play a sample game
    sample_edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (0,5)]  # Sample edges
    
    for edge in sample_edges:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            break
            
        player = "Player 1" if board.player == 0 else "Player 2"
        print(f"\n{player} selects edge {edge}")
        
        board.make_move(edge)
        print(board)
        
        # Store game state for visualization
        game_states.append(board.copy())
        
        # Check if game is over
        if board.game_state != 0:
            print(f"Player {board.game_state} wins!")
            break
    
    # Visualize the game if visualization is available
    if has_visualization:
        for i, state in enumerate(game_states):
            fig = view_clique_board(state)
            plt.savefig(f"./src/graphs/clique_game_state_{i}.png")
            print(f"Saved visualization to clique_game_state_{i}.png")
        
        # Show the final state
        plt.figure(figsize=(10, 8))
        fig = view_clique_board(game_states[-1])
        plt.show() 
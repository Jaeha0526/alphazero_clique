#!/usr/bin/env python

import numpy as np
import copy
from itertools import combinations

class CliqueBoard:
    def __init__(self, num_vertices, k, game_mode="asymmetric"):
        """
        Initialize a complete graph with given number of vertices
        num_vertices: number of vertices in the graph
        k: size of clique needed for Player 1 to win
        game_mode: "asymmetric" or "symmetric"
            - "asymmetric": Player 1 tries to form a k-clique, Player 2 tries to prevent it
            - "symmetric": Both players try to form a k-clique
        """
        self.num_vertices = num_vertices
        self.k = k  # Size of clique needed to win
        self.game_mode = game_mode  # Game mode: "asymmetric" or "symmetric"
        # Initialize adjacency matrix (complete graph)
        self.adjacency_matrix = np.ones((num_vertices, num_vertices)) - np.eye(num_vertices)
        # Initialize edge states (0: unselected, 1: player1, 2: player2)
        self.edge_states = np.zeros((num_vertices, num_vertices), dtype=int)
        # Current player (0: player1, 1: player2)
        self.player = 0
        # Move count
        self.move_count = 0
        # Game state (0: ongoing, 1: player1 wins, 2: player2 wins, 3: draw)
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
        # If no more moves are possible and no winner, it's a draw in symmetric mode
        elif not self.get_valid_moves() and self.game_mode == "symmetric":
            self.game_state = 3  # Draw
        # In asymmetric mode, if no more moves are possible, Player 2 wins
        elif not self.get_valid_moves() and self.game_mode == "asymmetric":
            self.game_state = 2  # Player 2 wins
            
        # Switch player
        self.player = 1 - self.player
        return True
    
    def check_win_condition(self):
        """
        Check if current player has won
        In asymmetric mode:
            - Player 1 wins by forming a k-clique
            - Player 2 wins by preventing Player 1 from forming a k-clique
        In symmetric mode:
            - Both players win by forming a k-clique
        Returns: True if current player has won, False otherwise
        """
        current_player = self.player + 1  # Convert to 1-indexed player number
        
        if self.game_mode == "symmetric" or (self.game_mode == "asymmetric" and current_player == 1):
            # Check if current player has formed a k-clique
            player_edges = []
            for i in range(self.num_vertices):
                for j in range(i+1, self.num_vertices):
                    if self.edge_states[i,j] == current_player:
                        player_edges.append((i,j))
            
            # Get all vertices involved in player's edges
            player_vertices = set()
            for v1, v2 in player_edges:
                player_vertices.add(v1)
                player_vertices.add(v2)
            
            # Check if any subset of vertices forms a k-clique
            for vertices in self._get_combinations(list(player_vertices), self.k):
                if self._is_clique(vertices, current_player):
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
            'num_vertices': self.num_vertices,
            'player': self.player,
            'move_count': self.move_count,
            'game_state': self.game_state,
            'game_mode': self.game_mode,
            'k': self.k
        }
    
    def copy(self):
        """Create a deep copy of the board"""
        new_board = CliqueBoard(self.num_vertices, self.k, self.game_mode)
        new_board.adjacency_matrix = self.adjacency_matrix.copy()
        new_board.edge_states = self.edge_states.copy()
        new_board.player = self.player
        new_board.move_count = self.move_count
        new_board.game_state = self.game_state
        return new_board
    
    def __str__(self):
        """String representation of the board"""
        s = f"Clique Game Board (n={self.num_vertices}, k={self.k}, mode={self.game_mode})\n"
        s += f"Current Player: {'Player 1' if self.player == 0 else 'Player 2'}\n"
        s += f"Move Count: {self.move_count}\n"
        
        if self.game_state == 0:
            state_str = "Ongoing"
        elif self.game_state == 1:
            state_str = "Player 1 Wins"
        elif self.game_state == 2:
            state_str = "Player 2 Wins"
        else:
            state_str = "Draw"
        
        s += f"Game State: {state_str}\n\n"
        
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
    print("=== Asymmetric Mode ===")
    board = CliqueBoard(6, 3, "asymmetric")
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
            print(f"Game over! Game state: {board.game_state}")
            break
    
    print("\n=== Symmetric Mode ===")
    board = CliqueBoard(6, 3, "symmetric")
    print(board)
    
    # Play a sample game in symmetric mode
    game_states = [board.copy()]
    
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
            if board.game_state == 3:
                print("Game drawn!")
            else:
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
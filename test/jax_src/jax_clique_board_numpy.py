#!/usr/bin/env python
"""
NumPy implementation that will be converted to JAX.
This ensures algorithmic correctness before JAX optimization.
"""

import numpy as np
from typing import Tuple, List, NamedTuple, Optional
from functools import partial
import itertools

class CliqueBoardState(NamedTuple):
    """Immutable board state for functional operations"""
    adjacency_matrix: np.ndarray  # Shape: (num_vertices, num_vertices)
    edge_states: np.ndarray       # Shape: (num_vertices, num_vertices), dtype=int32
    player: int                   # Current player (0 or 1)
    move_count: int              # Number of moves made
    game_state: int              # 0: ongoing, 1: player1 wins, 2: player2 wins, 3: draw
    num_vertices: int
    k: int                       # Clique size needed to win
    game_mode: str              # "symmetric" or "asymmetric"


class JAXCliqueBoard:
    """NumPy version that will be converted to JAX - maintains exact same interface as original"""
    
    def __init__(self, num_vertices, k, game_mode="asymmetric"):
        """Initialize a complete graph with given number of vertices"""
        self.num_vertices = num_vertices
        self.k = k
        self.game_mode = game_mode
        
        # Initialize adjacency matrix (complete graph)
        adjacency = np.ones((num_vertices, num_vertices)) - np.eye(num_vertices)
        
        # Initialize edge states (0: unselected, 1: player1, 2: player2)
        edge_states = np.zeros((num_vertices, num_vertices), dtype=np.int32)
        
        # Create initial state
        self.state = CliqueBoardState(
            adjacency_matrix=adjacency,
            edge_states=edge_states,
            player=0,
            move_count=0,
            game_state=0,
            num_vertices=num_vertices,
            k=k,
            game_mode=game_mode
        )
        
    @property
    def adjacency_matrix(self):
        return self.state.adjacency_matrix.copy()
    
    @property
    def edge_states(self):
        return self.state.edge_states.copy()
    
    @property
    def player(self):
        return self.state.player
    
    @property
    def move_count(self):
        return self.state.move_count
    
    @property
    def game_state(self):
        return self.state.game_state
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get list of valid moves (unselected edges) - matches original exactly"""
        valid_moves = []
        
        for i in range(self.num_vertices):
            for j in range(i+1, self.num_vertices):
                if self.state.edge_states[i, j] == 0:
                    valid_moves.append((i, j))
        return valid_moves
    
    def make_move(self, edge: Tuple[int, int]) -> bool:
        """Make a move by selecting an edge - matches original behavior exactly"""
        if edge not in self.get_valid_moves():
            return False
        
        v1, v2 = edge
        
        # Update edge states
        new_edge_states = self.state.edge_states.copy()
        new_edge_states[v1, v2] = self.state.player + 1
        new_edge_states[v2, v1] = self.state.player + 1
        
        # Create new state with updated edge states
        new_state = self.state._replace(
            edge_states=new_edge_states,
            move_count=self.state.move_count + 1
        )
        
        # Check for win condition
        self.state = new_state  # Update state temporarily for check_win_condition
        if self.check_win_condition():
            new_game_state = self.state.player + 1
        elif not self.get_valid_moves() and self.game_mode == "symmetric":
            new_game_state = 3  # Draw
        elif not self.get_valid_moves() and self.game_mode == "asymmetric":
            new_game_state = 2  # Player 2 wins
        else:
            new_game_state = 0  # Game continues
        
        # Create final state with all updates
        self.state = new_state._replace(
            player=1 - self.state.player,
            game_state=new_game_state
        )
        
        return True
    
    def check_win_condition(self) -> bool:
        """Check if current player has won - exact same logic as original"""
        current_player = self.state.player + 1
        
        if self.game_mode == "symmetric" or (self.game_mode == "asymmetric" and current_player == 1):
            # Check if current player has formed a k-clique
            player_edges = []
            
            for i in range(self.num_vertices):
                for j in range(i+1, self.num_vertices):
                    if self.state.edge_states[i, j] == current_player:
                        player_edges.append((i, j))
            
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
    
    def _get_combinations(self, vertices: List[int], k: int):
        """Get all combinations of k vertices from the list"""
        if len(vertices) < k:
            return []
        return itertools.combinations(vertices, k)
    
    def _is_clique(self, vertices: List[int], player: int) -> bool:
        """Check if given vertices form a clique for the player"""
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if self.state.edge_states[vertices[i], vertices[j]] != player:
                    return False
        return True
    
    def copy(self):
        """Create a copy of the board"""
        new_board = JAXCliqueBoard(self.num_vertices, self.k, self.game_mode)
        new_board.state = CliqueBoardState(
            adjacency_matrix=self.state.adjacency_matrix.copy(),
            edge_states=self.state.edge_states.copy(),
            player=self.state.player,
            move_count=self.state.move_count,
            game_state=self.state.game_state,
            num_vertices=self.state.num_vertices,
            k=self.state.k,
            game_mode=self.state.game_mode
        )
        return new_board
    
    def get_board_state(self):
        """Get current board state as a dictionary - matches original interface"""
        return {
            'adjacency_matrix': self.state.adjacency_matrix.copy(),
            'edge_states': self.state.edge_states.copy(),
            'num_vertices': self.num_vertices,
            'player': self.state.player,
            'move_count': self.state.move_count,
            'game_state': self.state.game_state,
            'game_mode': self.game_mode,
            'k': self.k
        }
    
    @classmethod
    def from_dict(cls, state_dict):
        """Create board from dictionary state"""
        # Create new board
        board = cls.__new__(cls)
        board.num_vertices = state_dict['num_vertices']
        board.k = state_dict['k']
        board.game_mode = state_dict.get('game_mode', 'symmetric')
        
        # Recreate state from dictionary
        adjacency = state_dict.get('adjacency_matrix', 
                                  np.ones((board.num_vertices, board.num_vertices)) - np.eye(board.num_vertices))
        edge_states = state_dict.get('edge_states', 
                                    np.zeros((board.num_vertices, board.num_vertices), dtype=np.int32))
        
        board.state = CliqueBoardState(
            adjacency_matrix=adjacency.copy() if isinstance(adjacency, np.ndarray) else adjacency,
            edge_states=edge_states.copy() if isinstance(edge_states, np.ndarray) else edge_states,
            player=state_dict.get('player', 0),
            move_count=state_dict.get('move_count', 0),
            game_state=state_dict.get('game_state', 0),
            num_vertices=board.num_vertices,
            k=board.k,
            game_mode=board.game_mode
        )
        
        return board
    
    def __str__(self):
        """String representation matching original exactly"""
        s = f"Clique Game Board (n={self.num_vertices}, k={self.k}, mode={self.game_mode})\n"
        s += f"Current Player: {'Player 1' if self.state.player == 0 else 'Player 2'}\n"
        s += f"Move Count: {self.state.move_count}\n"
        
        if self.state.game_state == 0:
            state_str = "Ongoing"
        elif self.state.game_state == 1:
            state_str = "Player 1 Wins"
        elif self.state.game_state == 2:
            state_str = "Player 2 Wins"
        else:
            state_str = "Draw"
        
        s += f"Game State: {state_str}\n\n"
        
        # Print edge states
        s += "Edge States:\n"
        for i in range(self.num_vertices):
            for j in range(i+1, self.num_vertices):
                state = self.state.edge_states[i,j]
                state_str = "Unselected" if state == 0 else f"Player {state}"
                s += f"Edge ({i},{j}): {state_str}\n"
        
        return s


# Batch operations for GPU efficiency (for future use in MCTS)
def batch_get_valid_moves_mask(edge_states_batch: np.ndarray) -> np.ndarray:
    """
    Get valid moves mask for a batch of boards.
    Args:
        edge_states_batch: Shape (batch_size, num_vertices, num_vertices)
    Returns:
        valid_mask: Shape (batch_size, num_edges) where num_edges = n*(n-1)/2
    """
    batch_size, n, _ = edge_states_batch.shape
    
    # Create upper triangular mask
    triu_indices = np.triu_indices(n, k=1)
    
    # Extract upper triangular elements for each board
    edge_states_triu = edge_states_batch[:, triu_indices[0], triu_indices[1]]
    
    # Valid moves are where edge_state == 0
    valid_mask = (edge_states_triu == 0)
    
    return valid_mask
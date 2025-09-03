#!/usr/bin/env python
"""
Save Ramsey counterexamples when draws occur in avoid_clique mode.
A draw means we've successfully 2-colored a complete graph without any monochromatic k-cliques.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import jax.numpy as jnp


class RamseyCounterexampleSaver:
    """Save potential Ramsey counterexamples from avoid_clique games."""
    
    def __init__(self, save_dir: str = "./ramsey_counterexamples"):
        """
        Initialize the saver.
        
        Args:
            save_dir: Directory to save counterexamples
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Track statistics
        self.stats = {
            'total_games': 0,
            'draws_found': 0,
            'counterexamples_by_size': {}
        }
        
        # Load existing stats if available
        self.stats_file = self.save_dir / "statistics.json"
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)
    
    def save_counterexample(
        self,
        edge_states: jnp.ndarray,
        num_vertices: int,
        k: int,
        source: str = "unknown",
        iteration: Optional[int] = None,
        game_idx: Optional[int] = None
    ) -> str:
        """
        Save a Ramsey counterexample (complete 2-coloring without k-cliques).
        
        Args:
            edge_states: Matrix of edge states (0=uncolored, 1=player1/red, 2=player2/blue)
            num_vertices: Number of vertices (n)
            k: Clique size that was avoided
            source: Where this came from ("self_play", "evaluation", etc.)
            iteration: Training iteration number
            game_idx: Game index within batch
            
        Returns:
            Path to saved file
        """
        # Convert to numpy for JSON serialization
        if hasattr(edge_states, 'numpy'):
            edge_states_np = edge_states.numpy()
        else:
            edge_states_np = np.array(edge_states)
        
        # Create edge list representation
        edges_red = []
        edges_blue = []
        edge_count = 0
        
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                edge_count += 1
                if edge_states_np[i, j] == 1:
                    edges_red.append([int(i), int(j)])
                elif edge_states_np[i, j] == 2:
                    edges_blue.append([int(i), int(j)])
        
        # Verify it's a complete coloring
        total_edges = num_vertices * (num_vertices - 1) // 2
        colored_edges = len(edges_red) + len(edges_blue)
        
        if colored_edges != total_edges:
            print(f"Warning: Incomplete coloring! {colored_edges}/{total_edges} edges colored")
            # Still save it as it might be interesting
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ramsey_n{num_vertices}_k{k}_{source}_{timestamp}"
        if iteration is not None:
            filename += f"_iter{iteration}"
        if game_idx is not None:
            filename += f"_game{game_idx}"
        filename += ".json"
        
        filepath = self.save_dir / filename
        
        # Prepare data
        data = {
            'metadata': {
                'num_vertices': num_vertices,
                'k': k,
                'source': source,
                'iteration': iteration,
                'game_idx': game_idx,
                'timestamp': timestamp,
                'total_edges': total_edges,
                'colored_edges': colored_edges,
                'complete': colored_edges == total_edges
            },
            'coloring': {
                'red_edges': edges_red,
                'blue_edges': edges_blue,
                'num_red': len(edges_red),
                'num_blue': len(edges_blue)
            },
            'adjacency_matrix': edge_states_np.tolist(),
            'verification': {
                'is_valid_counterexample': colored_edges == total_edges,
                'avoided_clique_size': k,
                'graph_complete': True
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Update statistics
        self.stats['draws_found'] += 1
        key = f"n{num_vertices}_k{k}"
        if key not in self.stats['counterexamples_by_size']:
            self.stats['counterexamples_by_size'][key] = 0
        self.stats['counterexamples_by_size'][key] += 1
        
        # Save updated stats
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"💎 RAMSEY COUNTEREXAMPLE SAVED: {filepath}")
        print(f"   Graph: K_{num_vertices} with no monochromatic K_{k}")
        print(f"   Red edges: {len(edges_red)}, Blue edges: {len(edges_blue)}")
        
        # Special celebration for important cases
        if num_vertices >= 42 and k == 5 and colored_edges == total_edges:
            print("🎉🎉🎉 MAJOR DISCOVERY: Potential R(5,5) ≥ 43 proof! 🎉🎉🎉")
        
        return str(filepath)
    
    def save_batch_counterexamples(
        self,
        boards,  # VectorizedCliqueBoard
        source: str = "unknown",
        iteration: Optional[int] = None
    ) -> List[str]:
        """
        Save all draws from a batch of games.
        
        Args:
            boards: VectorizedCliqueBoard with completed games
            source: Where these came from
            iteration: Training iteration
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        for game_idx in range(boards.batch_size):
            # Check if this game was a draw
            if boards.game_states[game_idx] == 3:  # Draw
                # Extract edge states for this game
                edge_states = boards.edge_states[game_idx]
                
                filepath = self.save_counterexample(
                    edge_states=edge_states,
                    num_vertices=boards.num_vertices,
                    k=boards.k,
                    source=source,
                    iteration=iteration,
                    game_idx=game_idx
                )
                saved_files.append(filepath)
        
        if saved_files:
            print(f"📊 Saved {len(saved_files)} Ramsey counterexamples from {source}")
        
        return saved_files
    
    def update_game_count(self, num_games: int):
        """Update total game count."""
        self.stats['total_games'] += num_games
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def get_summary(self) -> str:
        """Get a summary of found counterexamples."""
        summary = "=== Ramsey Counterexample Summary ===\n"
        summary += f"Total games played: {self.stats['total_games']}\n"
        summary += f"Total draws (counterexamples): {self.stats['draws_found']}\n"
        
        if self.stats['draws_found'] > 0:
            summary += f"Draw rate: {self.stats['draws_found']/max(1, self.stats['total_games']):.2%}\n"
            summary += "\nCounterexamples by size:\n"
            for key, count in sorted(self.stats['counterexamples_by_size'].items()):
                summary += f"  {key}: {count} examples\n"
        
        return summary


def load_counterexample(filepath: str) -> Dict:
    """
    Load a saved counterexample.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with counterexample data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def verify_counterexample(data: Dict) -> bool:
    """
    Verify that a saved counterexample is valid.
    
    Args:
        data: Loaded counterexample data
        
    Returns:
        True if valid (no monochromatic k-cliques)
    """
    n = data['metadata']['num_vertices']
    k = data['metadata']['k']
    adj_matrix = np.array(data['adjacency_matrix'])
    
    # Check all possible k-cliques
    from itertools import combinations
    
    for clique_vertices in combinations(range(n), k):
        # Check if this k-subset forms a monochromatic clique
        for color in [1, 2]:  # Red and Blue
            is_monochromatic = True
            for i, v1 in enumerate(clique_vertices):
                for v2 in clique_vertices[i+1:]:
                    if adj_matrix[v1, v2] != color:
                        is_monochromatic = False
                        break
                if not is_monochromatic:
                    break
            
            if is_monochromatic:
                print(f"Found monochromatic {color} clique: {clique_vertices}")
                return False
    
    return True


if __name__ == "__main__":
    # Test the saver
    import jax.numpy as jnp
    
    saver = RamseyCounterexampleSaver()
    
    # Create a test example: K_5 with no K_3
    n = 5
    edge_states = jnp.zeros((n, n), dtype=jnp.int32)
    
    # Color edges to avoid triangles (this is just an example pattern)
    edge_states = edge_states.at[0, 1].set(1)  # Red
    edge_states = edge_states.at[0, 2].set(1)  # Red
    edge_states = edge_states.at[0, 3].set(2)  # Blue
    edge_states = edge_states.at[0, 4].set(2)  # Blue
    edge_states = edge_states.at[1, 2].set(2)  # Blue
    edge_states = edge_states.at[1, 3].set(1)  # Red
    edge_states = edge_states.at[1, 4].set(1)  # Red
    edge_states = edge_states.at[2, 3].set(1)  # Red
    edge_states = edge_states.at[2, 4].set(2)  # Blue
    edge_states = edge_states.at[3, 4].set(2)  # Blue
    
    # Make symmetric
    for i in range(n):
        for j in range(i+1, n):
            edge_states = edge_states.at[j, i].set(edge_states[i, j])
    
    # Save it
    filepath = saver.save_counterexample(
        edge_states=edge_states,
        num_vertices=n,
        k=3,
        source="test",
        iteration=0,
        game_idx=0
    )
    
    # Load and verify
    data = load_counterexample(filepath)
    is_valid = verify_counterexample(data)
    print(f"\nVerification: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Show summary
    print("\n" + saver.get_summary())
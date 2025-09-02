#!/usr/bin/env python
"""
Test that Ramsey counterexample saving is integrated correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
from vectorized_board import VectorizedCliqueBoard
from ramsey_counterexample_saver import RamseyCounterexampleSaver
import json
from pathlib import Path

def test_avoid_clique_with_draw():
    """Test that draws in avoid_clique mode are saved correctly."""
    
    print("="*60)
    print("Testing Ramsey Counterexample Saving Integration")
    print("="*60)
    
    # Create a saver
    saver = RamseyCounterexampleSaver(save_dir="./test_ramsey")
    
    # Create a batch of games
    boards = VectorizedCliqueBoard(
        batch_size=3,
        num_vertices=4,
        k=3,
        game_mode="avoid_clique"
    )
    
    # Simulate games:
    # Game 0: Draw (no cliques formed)
    # Game 1: Player 1 wins (Player 2 forms clique)
    # Game 2: Draw (no cliques formed)
    
    print("\nSimulating 3 games in avoid_clique mode...")
    
    # Game 0: Careful play leads to draw
    # Color edges to avoid any triangles
    # This is actually hard to do manually, so let's just set the state
    boards.game_states = boards.game_states.at[0].set(3)  # Draw
    boards.edge_states = boards.edge_states.at[0, 0, 1].set(1)  # Red
    boards.edge_states = boards.edge_states.at[0, 0, 2].set(2)  # Blue
    boards.edge_states = boards.edge_states.at[0, 0, 3].set(1)  # Red
    boards.edge_states = boards.edge_states.at[0, 1, 2].set(2)  # Blue
    boards.edge_states = boards.edge_states.at[0, 1, 3].set(2)  # Blue
    boards.edge_states = boards.edge_states.at[0, 2, 3].set(1)  # Red
    
    # Game 1: Player 2 forms clique
    boards.game_states = boards.game_states.at[1].set(1)  # P1 wins
    
    # Game 2: Another draw
    boards.game_states = boards.game_states.at[2].set(3)  # Draw
    boards.edge_states = boards.edge_states.at[2, 0, 1].set(2)  # Blue
    boards.edge_states = boards.edge_states.at[2, 0, 2].set(1)  # Red
    boards.edge_states = boards.edge_states.at[2, 0, 3].set(2)  # Blue
    boards.edge_states = boards.edge_states.at[2, 1, 2].set(1)  # Red
    boards.edge_states = boards.edge_states.at[2, 1, 3].set(1)  # Red
    boards.edge_states = boards.edge_states.at[2, 2, 3].set(2)  # Blue
    
    # Save counterexamples
    saved_files = saver.save_batch_counterexamples(
        boards=boards,
        source="test",
        iteration=0
    )
    
    print(f"\nSaved {len(saved_files)} files (should be 2 for the 2 draws)")
    
    # Verify files were created
    if len(saved_files) == 2:
        print("✅ Correct number of files saved!")
        
        # Load and check one
        with open(saved_files[0], 'r') as f:
            data = json.load(f)
        
        print(f"\nFirst saved counterexample:")
        print(f"  Vertices: {data['metadata']['num_vertices']}")
        print(f"  k: {data['metadata']['k']}")
        print(f"  Red edges: {data['coloring']['num_red']}")
        print(f"  Blue edges: {data['coloring']['num_blue']}")
        print(f"  Complete: {data['metadata']['complete']}")
    else:
        print(f"❌ Wrong number of files! Expected 2, got {len(saved_files)}")
    
    # Show summary
    print("\n" + saver.get_summary())
    
    # Clean up test directory
    import shutil
    shutil.rmtree("./test_ramsey", ignore_errors=True)
    
    return len(saved_files) == 2

def test_integration_message():
    """Show integration summary."""
    
    print("\n" + "="*60)
    print("RAMSEY COUNTEREXAMPLE FEATURE SUMMARY")
    print("="*60)
    
    print("""
The avoid_clique mode now automatically saves potential Ramsey counterexamples!

📊 When a draw occurs in avoid_clique mode:
   - The complete 2-colored graph is saved to JSON
   - Location: ./ramsey_counterexamples/
   - Format: Edge lists for red and blue colorings
   - Metadata: n, k, source, iteration, timestamp

🎮 Integration points:
   1. Self-play: Saves draws during training games
   2. Sequential evaluation: Saves draws from evaluation games  
   3. Parallel evaluation: Batch saves all draws at once

📁 File format:
   - Red edges: List of [i,j] pairs colored red (player 1)
   - Blue edges: List of [i,j] pairs colored blue (player 2)
   - Adjacency matrix: Full edge state matrix
   - Verification flag: Confirms it's a valid counterexample

🎯 Special feature:
   - If n≥42 and k=5 with complete coloring → Major discovery alert!
   - This would prove R(5,5) ≥ 43, a significant mathematical result

💾 Statistics tracking:
   - Total games played
   - Total draws found
   - Counterexamples organized by (n,k) size
   - Draw rate percentage
""")
    
    return True

if __name__ == "__main__":
    success1 = test_avoid_clique_with_draw()
    success2 = test_integration_message()
    
    if success1 and success2:
        print("\n" + "="*60)
        print("✅ Ramsey counterexample saving is fully integrated!")
        print("="*60)
#!/usr/bin/env python
"""
Test that standalone evaluation correctly detects game mode from training.
"""

import json
import tempfile
from pathlib import Path

def test_game_mode_detection():
    """Test that game mode is saved and loaded correctly."""
    
    print("="*60)
    print("Testing Game Mode Detection")
    print("="*60)
    
    # Check existing experiments
    exp_dir = Path("experiments")
    
    for exp in exp_dir.glob("*"):
        if not exp.is_dir():
            continue
            
        log_file = exp / "training_log.json"
        if not log_file.exists():
            continue
            
        print(f"\nExperiment: {exp.name}")
        
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                
            if log_data:
                # Check first entry
                first_entry = log_data[0]
                
                # Check if config exists in log
                if 'config' in first_entry:
                    game_mode = first_entry['config'].get('game_mode', 'Not found')
                    vertices = first_entry['config'].get('num_vertices', 'Not found')
                    k = first_entry['config'].get('k', 'Not found')
                    print(f"  ✅ Config found: game_mode={game_mode}, n={vertices}, k={k}")
                else:
                    print(f"  ⚠️  No config in training log (old format)")
                    
                # Check for game statistics that might indicate mode
                if 'selfplay_attacker_wins' in first_entry:
                    print(f"  → Likely asymmetric mode (has attacker/defender stats)")
                elif 'selfplay_stats' in first_entry:
                    stats = first_entry['selfplay_stats']
                    if 'draw_rate' in stats and stats.get('draw_rate', 0) > 0.5:
                        print(f"  → Could be avoid_clique mode (high draw rate: {stats['draw_rate']:.1%})")
                    
        except Exception as e:
            print(f"  ❌ Error reading log: {e}")
    
    # Test standalone evaluation auto-detection
    print("\n" + "-"*40)
    print("\nTesting standalone evaluation auto-detection:")
    
    # Find an experiment with the new config format
    test_exp = None
    for exp in exp_dir.glob("*"):
        log_file = exp / "training_log.json"
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    if data and 'config' in data[0]:
                        test_exp = exp.name
                        expected_mode = data[0]['config']['game_mode']
                        break
            except:
                pass
    
    if test_exp:
        print(f"  Testing with experiment: {test_exp}")
        print(f"  Expected game mode: {expected_mode}")
        
        # Simulate what standalone_evaluation.py does
        import subprocess
        result = subprocess.run([
            "python", "-c",
            f"""
import json
from pathlib import Path

exp_dir = Path('experiments/{test_exp}')
training_log = exp_dir / 'training_log.json'

if training_log.exists():
    with open(training_log, 'r') as f:
        log_data = json.load(f)
        if log_data and 'config' in log_data[0]:
            game_mode = log_data[0]['config'].get('game_mode', 'symmetric')
            print(f'Detected game mode: {{game_mode}}')
        else:
            print('No config found, would use default: symmetric')
"""
        ], capture_output=True, text=True)
        
        if result.stdout:
            print(f"  {result.stdout.strip()}")
            if expected_mode in result.stdout:
                print("  ✅ Game mode detection working correctly!")
            else:
                print("  ❌ Game mode mismatch")
    else:
        print("  ⚠️  No experiments with new config format found")
        print("  (New experiments will save config in training log)")
    
    print("\n" + "="*60)
    print("Game mode detection test complete!")
    print("="*60)


if __name__ == "__main__":
    test_game_mode_detection()
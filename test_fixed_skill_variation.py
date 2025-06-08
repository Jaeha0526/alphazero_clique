import numpy as np
import sys
sys.path.append('src')
from MCTS_clique import get_varied_mcts_sims

def test_skill_variation():
    """Test the fixed skill variation implementation"""
    base_sims = 200
    skill_variation = 0.3
    num_tests = 10000
    
    print(f"Testing fixed skill variation implementation:")
    print(f"Base sims: {base_sims}")
    print(f"Skill variation: {skill_variation} (±30%)")
    print(f"Expected range: {int(base_sims * 0.7)} to {int(base_sims * 1.3)} sims")
    print()
    
    # Collect results
    results = []
    for _ in range(num_tests):
        p1_sims, p2_sims = get_varied_mcts_sims(base_sims, skill_variation)
        results.append((p1_sims, p2_sims))
    
    # Analyze results
    all_sims = [sim for p1, p2 in results for sim in [p1, p2]]
    ratios = [max(p1, p2) / min(p1, p2) for p1, p2 in results]
    
    print("Results:")
    print(f"  Min sims: {min(all_sims)}")
    print(f"  Max sims: {max(all_sims)}")
    print(f"  Mean sims: {np.mean(all_sims):.1f}")
    print(f"  Std sims: {np.std(all_sims):.1f}")
    print()
    
    print("Skill ratios (stronger/weaker player):")
    print(f"  Min ratio: {min(ratios):.2f}x")
    print(f"  Max ratio: {max(ratios):.2f}x")
    print(f"  Mean ratio: {np.mean(ratios):.2f}x")
    print()
    
    # Check if results are within expected bounds
    expected_min = int(base_sims * 0.7)
    expected_max = int(base_sims * 1.3)
    
    within_bounds = all(expected_min <= sim <= expected_max for sim in all_sims)
    print(f"All simulations within expected bounds [{expected_min}, {expected_max}]: {within_bounds}")
    
    # Show some examples
    print("\nSample results:")
    for i in range(10):
        p1, p2 = results[i]
        ratio = max(p1, p2) / min(p1, p2)
        print(f"  Game {i+1}: P1={p1:3d} sims, P2={p2:3d} sims (ratio: {ratio:.2f}x)")

if __name__ == "__main__":
    test_skill_variation()
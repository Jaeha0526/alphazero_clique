import numpy as np

# Simulate the current skill variation implementation
def get_varied_mcts_sims_current(base_sims: int, skill_variation: float, num_samples: int = 10000):
    """Current implementation from the codebase"""
    results = []
    
    for _ in range(num_samples):
        player1_multiplier = max(0.1, np.random.normal(1.0, skill_variation))
        player2_multiplier = max(0.1, np.random.normal(1.0, skill_variation))
        
        player1_sims = max(1, int(base_sims * player1_multiplier))
        player2_sims = max(1, int(base_sims * player2_multiplier))
        
        results.append((player1_sims, player2_sims))
    
    return results

# Test with base_sims=200 and skill_variation=0.3
base_sims = 200
skill_variation = 0.3

results = get_varied_mcts_sims_current(base_sims, skill_variation)
player1_sims = [r[0] for r in results]
player2_sims = [r[1] for r in results]

print(f"Base MCTS simulations: {base_sims}")
print(f"Skill variation: {skill_variation}")
print(f"\nPlayer 1 simulations:")
print(f"  Min: {min(player1_sims)}")
print(f"  Max: {max(player1_sims)}")
print(f"  Mean: {np.mean(player1_sims):.1f}")
print(f"  Std: {np.std(player1_sims):.1f}")

print(f"\nPlayer 2 simulations:")
print(f"  Min: {min(player2_sims)}")
print(f"  Max: {max(player2_sims)}")
print(f"  Mean: {np.mean(player2_sims):.1f}")
print(f"  Std: {np.std(player2_sims):.1f}")

# Check extreme cases
extreme_ratios = []
for p1, p2 in results:
    ratio = max(p1, p2) / min(p1, p2)
    extreme_ratios.append(ratio)

print(f"\nSimulation count ratios (stronger/weaker player):")
print(f"  Min ratio: {min(extreme_ratios):.1f}x")
print(f"  Max ratio: {max(extreme_ratios):.1f}x")
print(f"  Mean ratio: {np.mean(extreme_ratios):.1f}x")

# Show some extreme examples
extreme_indices = np.argsort(extreme_ratios)[-10:]  # Top 10 most extreme
print(f"\nTop 10 most extreme simulation differences:")
for i in extreme_indices:
    p1, p2 = results[i]
    ratio = max(p1, p2) / min(p1, p2)
    print(f"  Player 1: {p1:3d} sims, Player 2: {p2:3d} sims (ratio: {ratio:.1f}x)")

# Check percentiles
print(f"\nPercentiles of simulation counts:")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
all_sims = player1_sims + player2_sims
for p in percentiles:
    value = np.percentile(all_sims, p)
    print(f"  {p:2d}th percentile: {value:.0f} sims")
import numpy as np

# Let's trace through the exact implementation
base_sims = 200
skill_variation = 0.3

print("Current implementation analysis:")
print(f"base_sims = {base_sims}")
print(f"skill_variation = {skill_variation}")
print()

# The current implementation
print("Current code:")
print("player1_multiplier = max(0.1, np.random.normal(1.0, skill_variation))")
print("player2_multiplier = max(0.1, np.random.normal(1.0, skill_variation))")
print()

# Simulate what happens
print("What the normal distribution generates:")
samples = np.random.normal(1.0, skill_variation, 10000)
print(f"Normal(1.0, {skill_variation}) range:")
print(f"  Min: {samples.min():.3f}")
print(f"  Max: {samples.max():.3f}")
print(f"  1st percentile: {np.percentile(samples, 1):.3f}")
print(f"  99th percentile: {np.percentile(samples, 99):.3f}")
print()

# Show the effect of max(0.1, ...) clipping
clipped_samples = np.maximum(0.1, samples)
print(f"After max(0.1, ...) clipping:")
print(f"  Min: {clipped_samples.min():.3f}")
print(f"  Max: {clipped_samples.max():.3f}")
print(f"  1st percentile: {np.percentile(clipped_samples, 1):.3f}")
print(f"  99th percentile: {np.percentile(clipped_samples, 99):.3f}")
print()

# Convert to simulation counts
sim_counts = np.maximum(1, (base_sims * clipped_samples).astype(int))
print(f"Final simulation counts:")
print(f"  Min: {sim_counts.min()}")
print(f"  Max: {sim_counts.max()}")
print(f"  1st percentile: {np.percentile(sim_counts, 1):.0f}")
print(f"  99th percentile: {np.percentile(sim_counts, 99):.0f}")
print()

# Show percentage differences from baseline
percentage_diff = ((sim_counts - base_sims) / base_sims) * 100
print(f"Percentage difference from baseline ({base_sims} sims):")
print(f"  Min: {percentage_diff.min():.1f}%")
print(f"  Max: {percentage_diff.max():.1f}%")
print(f"  1st percentile: {np.percentile(percentage_diff, 1):.1f}%")
print(f"  99th percentile: {np.percentile(percentage_diff, 99):.1f}%")

print("\n" + "="*50)
print("WHY 20 SIMS IS POSSIBLE:")
print("="*50)

print("\n1. Normal distribution can generate negative values!")
print("   Normal(1.0, 0.3) occasionally produces values < 0")

print("\n2. max(0.1, ...) clips ALL negative values to 0.1")
print("   Any normal sample < 0.1 becomes exactly 0.1")

print("\n3. 0.1 * 200 base_sims = 20 sims")
print("   So the minimum is always 20 sims, not a 30% variation!")

print("\n4. The 'skill_variation=0.3' is NOT a percentage limit")
print("   It's the standard deviation of the normal distribution")
print("   This allows for MUCH larger variations than 30%")

# Show what percentage-based variation would look like
print("\n" + "="*50)
print("WHAT PERCENTAGE-BASED VARIATION SHOULD LOOK LIKE:")
print("="*50)

print("\nFor true 30% variation:")
percentage_var = 0.3
min_multiplier = 1.0 - percentage_var  # 0.7
max_multiplier = 1.0 + percentage_var  # 1.3

print(f"Should be: uniform({min_multiplier}, {max_multiplier})")
print(f"Simulation range: {int(base_sims * min_multiplier)} to {int(base_sims * max_multiplier)}")
print(f"That's {int(base_sims * min_multiplier)} to {int(base_sims * max_multiplier)} sims")
print(f"Max ratio: {max_multiplier/min_multiplier:.2f}x")
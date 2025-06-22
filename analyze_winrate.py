#!/usr/bin/env python3
"""
Statistical analysis focused on maximizing evaluation_win_rate_vs_initial
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('experiments/wandb_export_2025-06-18T12_08_52.699-07_00.csv')

# Clean and prepare the data
hyperparams = ['batch-size', 'initial-lr', 'epochs', 'self-play-games', 'mcts-sims', 'skill-variation', 'lr-factor', 'lr-patience']
metrics = ['validation_policy_loss', 'validation_value_loss', 'evaluation_win_rate_vs_initial']

# Create clean dataframe
data = df[hyperparams + metrics].copy()

# Handle missing values and convert to numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Focus on win rate analysis - drop rows with missing win rates
data_winrate = data.dropna(subset=['evaluation_win_rate_vs_initial'])

print("Win Rate Analysis Dataset Info:")
print(f"Total experiments with win rate data: {len(data_winrate)}")
print(f"Win rate range: {data_winrate['evaluation_win_rate_vs_initial'].min():.1%} - {data_winrate['evaluation_win_rate_vs_initial'].max():.1%}")
print(f"Mean win rate: {data_winrate['evaluation_win_rate_vs_initial'].mean():.1%}")

print("\n" + "="*70)
print("ANALYSIS FOR MAXIMIZING WIN RATE VS INITIAL MODEL")
print("="*70)

# 1. Correlation Analysis for Win Rate
print("\n1. CORRELATION ANALYSIS (Win Rate Focus)")
print("-" * 45)

correlations_winrate = data_winrate[hyperparams].corrwith(data_winrate['evaluation_win_rate_vs_initial'])
print("Correlation with evaluation_win_rate_vs_initial:")
for param, corr in correlations_winrate.sort_values(ascending=False).items():
    strength = "STRONG" if abs(corr) > 0.5 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
    direction = "positive" if corr > 0 else "negative"
    print(f"  {param:20s}: {corr:6.3f} ({strength} {direction})")

# 2. Feature Importance for Win Rate
print("\n2. FEATURE IMPORTANCE FOR WIN RATE (Random Forest)")
print("-" * 50)

# Prepare data for Random Forest
X_wr = data_winrate[hyperparams].fillna(data_winrate[hyperparams].median())
y_wr = data_winrate['evaluation_win_rate_vs_initial']

# Fit Random Forest
rf_wr = RandomForestRegressor(n_estimators=100, random_state=42)
rf_wr.fit(X_wr, y_wr)

# Feature importance
importance_wr_df = pd.DataFrame({
    'parameter': hyperparams,
    'importance': rf_wr.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance for evaluation_win_rate_vs_initial:")
for _, row in importance_wr_df.iterrows():
    print(f"  {row['parameter']:20s}: {row['importance']:6.3f}")

# 3. Top Win Rate Performers
print("\n3. TOP WIN RATE PERFORMERS")
print("-" * 30)

top_winrate = data_winrate.nlargest(5, 'evaluation_win_rate_vs_initial')
print("Top 5 win rate performers:")
for i, (idx, row) in enumerate(top_winrate.iterrows(), 1):
    print(f"\nRank {i}: Win Rate = {row['evaluation_win_rate_vs_initial']:.1%} (Policy Loss = {row['validation_policy_loss']:.4f})")
    for param in hyperparams:
        print(f"  {param:20s}: {row[param]}")

# 4. Statistical Analysis of High Win Rate vs Low Win Rate
print("\n4. HIGH WIN RATE vs LOW WIN RATE COMPARISON")
print("-" * 50)

# Split into high and low win rate groups
winrate_threshold = data_winrate['evaluation_win_rate_vs_initial'].quantile(0.6)  # Top 40%
high_winrate = data_winrate[data_winrate['evaluation_win_rate_vs_initial'] >= winrate_threshold]
low_winrate = data_winrate[data_winrate['evaluation_win_rate_vs_initial'] < winrate_threshold]

print(f"Comparing high win rate (≥{winrate_threshold:.1%}, n={len(high_winrate)}) vs low win rate (<{winrate_threshold:.1%}, n={len(low_winrate)}):")

for param in hyperparams:
    if param in data_winrate.columns:
        high_values = high_winrate[param].dropna()
        low_values = low_winrate[param].dropna()
        
        if len(high_values) > 1 and len(low_values) > 1:
            # Perform t-test
            statistic, p_value = stats.ttest_ind(high_values, low_values, equal_var=False)
            
            high_mean = high_values.mean()
            low_mean = low_values.mean()
            high_std = high_values.std()
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"\n{param:20s}:")
            print(f"  High Win Rate: {high_mean:8.4f} ± {high_std:6.4f}")
            print(f"  Low Win Rate:  {low_mean:8.4f}")
            print(f"  p-value:       {p_value:8.4f} {significance}")

# 5. Comparison: Policy Loss vs Win Rate Optimization
print("\n5. POLICY LOSS vs WIN RATE OPTIMIZATION COMPARISON")
print("-" * 55)

# Get top performers for each metric
top_policy_loss = data.nsmallest(5, 'validation_policy_loss')
top_winrate_clean = data_winrate.nlargest(5, 'evaluation_win_rate_vs_initial')

print("Optimal parameters for POLICY LOSS vs WIN RATE:")
print("\nParameter              Policy Loss Opt    Win Rate Opt       Difference")
print("-" * 70)

for param in hyperparams:
    if param in top_policy_loss.columns and param in top_winrate_clean.columns:
        policy_opt = top_policy_loss[param].median()
        winrate_opt = top_winrate_clean[param].median()
        diff_pct = ((winrate_opt - policy_opt) / policy_opt * 100) if policy_opt != 0 else 0
        
        print(f"{param:20s}   {policy_opt:10.4f}    {winrate_opt:10.4f}    {diff_pct:+7.1f}%")

# 6. Win Rate Optimization Recommendations
print("\n" + "="*70)
print("WIN RATE OPTIMIZATION RECOMMENDATIONS")
print("="*70)

print("\nBased on win rate analysis:")
print("\n🏆 PRIMARY WIN RATE BOOSTERS (High Importance):")

top_5_winrate = data_winrate.nlargest(5, 'evaluation_win_rate_vs_initial')

for param in importance_wr_df.head(4)['parameter']:  # Top 4 most important for win rate
    if param in top_5_winrate.columns:
        optimal_value = top_5_winrate[param].median()
        param_range = f"[{top_5_winrate[param].min():.4f} - {top_5_winrate[param].max():.4f}]"
        print(f"  {param:20s}: {optimal_value:8.4f} (range: {param_range})")

print("\n🎯 SECONDARY WIN RATE FACTORS:")
for param in importance_wr_df.iloc[4:]['parameter']:
    if param in top_5_winrate.columns:
        optimal_value = top_5_winrate[param].median()
        param_range = f"[{top_5_winrate[param].min():.4f} - {top_5_winrate[param].max():.4f}]"
        print(f"  {param:20s}: {optimal_value:8.4f} (range: {param_range})")

# Performance metrics
print("\n🏅 EXPECTED WIN RATE PERFORMANCE:")
best_winrate_run = data_winrate.loc[data_winrate['evaluation_win_rate_vs_initial'].idxmax()]
print(f"  Best win rate achieved:       {best_winrate_run['evaluation_win_rate_vs_initial']:.1%}")
print(f"  Policy loss at best win rate: {best_winrate_run['validation_policy_loss']:.4f}")

# Trade-off analysis
print("\n⚖️  POLICY LOSS vs WIN RATE TRADE-OFF:")
# Calculate Pareto frontier approximation
efficient_runs = []
for idx, run in data_winrate.iterrows():
    win_rate = run['evaluation_win_rate_vs_initial']
    policy_loss = run['validation_policy_loss']
    
    # Check if this run is dominated by any other run
    is_dominated = False
    for _, other_run in data_winrate.iterrows():
        if (other_run['evaluation_win_rate_vs_initial'] >= win_rate and 
            other_run['validation_policy_loss'] <= policy_loss and
            (other_run['evaluation_win_rate_vs_initial'] > win_rate or 
             other_run['validation_policy_loss'] < policy_loss)):
            is_dominated = True
            break
    
    if not is_dominated:
        efficient_runs.append((win_rate, policy_loss, idx))

efficient_runs.sort(key=lambda x: x[0], reverse=True)  # Sort by win rate
print(f"  Pareto efficient configurations: {len(efficient_runs)}")
if len(efficient_runs) > 0:
    best_efficient = efficient_runs[0]
    print(f"  Best efficient: {best_efficient[0]:.1%} win rate, {best_efficient[1]:.4f} policy loss")

# Create win rate focused visualization
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Win Rate Feature Importance
axes[0,0].barh(importance_wr_df.head(6)['parameter'], importance_wr_df.head(6)['importance'], color='green', alpha=0.7)
axes[0,0].set_title('Feature Importance for Win Rate')
axes[0,0].set_xlabel('Importance Score')

# 2. Policy Loss vs Win Rate Trade-off
axes[0,1].scatter(data_winrate['validation_policy_loss'], data_winrate['evaluation_win_rate_vs_initial'], alpha=0.7)
axes[0,1].set_xlabel('Validation Policy Loss')
axes[0,1].set_ylabel('Evaluation Win Rate vs Initial')
axes[0,1].set_title('Policy Loss vs Win Rate Trade-off')

# 3. MCTS Sims vs Win Rate
axes[1,0].scatter(data_winrate['mcts-sims'], data_winrate['evaluation_win_rate_vs_initial'], alpha=0.7, color='orange')
axes[1,0].set_xlabel('MCTS Simulations')
axes[1,0].set_ylabel('Evaluation Win Rate vs Initial')
axes[1,0].set_title('MCTS Sims vs Win Rate')

# 4. Learning Rate vs Win Rate
axes[1,1].scatter(data_winrate['initial-lr'], data_winrate['evaluation_win_rate_vs_initial'], alpha=0.7, color='red')
axes[1,1].set_xlabel('Initial Learning Rate')
axes[1,1].set_ylabel('Evaluation Win Rate vs Initial')
axes[1,1].set_title('Learning Rate vs Win Rate')
axes[1,1].set_xscale('log')

plt.tight_layout()
plt.savefig('experiments/winrate_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n📊 Win rate visualization saved to: experiments/winrate_analysis.png")

print("\n" + "="*70)
print("WIN RATE ANALYSIS COMPLETE")
print("="*70)
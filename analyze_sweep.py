#!/usr/bin/env python3
"""
Statistical analysis of wandb sweep results to identify optimal hyperparameters
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
# Extract key columns
hyperparams = ['batch-size', 'initial-lr', 'epochs', 'self-play-games', 'mcts-sims', 'skill-variation', 'lr-factor', 'lr-patience']
metrics = ['validation_policy_loss', 'validation_value_loss', 'evaluation_win_rate_vs_initial']

# Create clean dataframe with relevant columns
data = df[hyperparams + metrics].copy()

# Handle missing values and convert to numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with missing target metrics
data = data.dropna(subset=['validation_policy_loss'])

print("Dataset Info:")
print(f"Total experiments: {len(data)}")
print(f"Complete cases: {len(data.dropna())}")
print("\nDataset head:")
print(data.head())

# Statistical Analysis
print("\n" + "="*60)
print("STATISTICAL ANALYSIS OF HYPERPARAMETERS")
print("="*60)

# 1. Correlation Analysis
print("\n1. CORRELATION ANALYSIS")
print("-" * 30)

# Correlation with validation_policy_loss (primary metric)
correlations = data[hyperparams].corrwith(data['validation_policy_loss'])
print("Correlation with validation_policy_loss:")
for param, corr in correlations.sort_values().items():
    strength = "STRONG" if abs(corr) > 0.5 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
    direction = "negative" if corr < 0 else "positive"
    print(f"  {param:20s}: {corr:6.3f} ({strength} {direction})")

# 2. Feature Importance using Random Forest
print("\n2. FEATURE IMPORTANCE (Random Forest)")
print("-" * 40)

# Prepare data for Random Forest
X = data[hyperparams].fillna(data[hyperparams].median())
y = data['validation_policy_loss'].fillna(data['validation_policy_loss'].median())

# Fit Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Feature importance
importance_df = pd.DataFrame({
    'parameter': hyperparams,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance for validation_policy_loss:")
for _, row in importance_df.iterrows():
    print(f"  {row['parameter']:20s}: {row['importance']:6.3f}")

# 3. Optimal Parameter Values Analysis
print("\n3. OPTIMAL PARAMETER VALUES")
print("-" * 30)

# Get top 5 performing experiments
top_performers = data.nsmallest(5, 'validation_policy_loss')
print("Top 5 performers (lowest validation_policy_loss):")
for i, (idx, row) in enumerate(top_performers.iterrows(), 1):
    print(f"\nRank {i}: Loss = {row['validation_policy_loss']:.4f}")
    for param in hyperparams:
        print(f"  {param:20s}: {row[param]}")

# Statistical analysis of optimal ranges
print("\n4. STATISTICAL ANALYSIS OF OPTIMAL RANGES")
print("-" * 45)

# Split data into top and bottom performers
threshold = data['validation_policy_loss'].quantile(0.3)  # Top 30%
top_30_percent = data[data['validation_policy_loss'] <= threshold]
bottom_70_percent = data[data['validation_policy_loss'] > threshold]

print(f"Comparing top 30% performers (n={len(top_30_percent)}) vs bottom 70% (n={len(bottom_70_percent)}):")

for param in hyperparams:
    if param in data.columns:
        top_values = top_30_percent[param].dropna()
        bottom_values = bottom_70_percent[param].dropna()
        
        if len(top_values) > 0 and len(bottom_values) > 0:
            # Perform t-test
            statistic, p_value = stats.ttest_ind(top_values, bottom_values, equal_var=False)
            
            top_mean = top_values.mean()
            bottom_mean = bottom_values.mean()
            top_std = top_values.std()
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"\n{param:20s}:")
            print(f"  Top 30%:    {top_mean:8.4f} ± {top_std:6.4f}")
            print(f"  Bottom 70%: {bottom_mean:8.4f}")
            print(f"  p-value:    {p_value:8.4f} {significance}")

# 5. Recommendations
print("\n" + "="*60)
print("OPTIMAL HYPERPARAMETER RECOMMENDATIONS")
print("="*60)

print("\nBased on statistical analysis:")
print("\n🎯 PRIMARY RECOMMENDATIONS (High Importance):")

# Get median values from top performers for robust estimates
top_5 = data.nsmallest(5, 'validation_policy_loss')

for param in importance_df.head(4)['parameter']:  # Top 4 most important
    if param in top_5.columns:
        optimal_value = top_5[param].median()
        param_range = f"[{top_5[param].min():.4f} - {top_5[param].max():.4f}]"
        print(f"  {param:20s}: {optimal_value:8.4f} (range: {param_range})")

print("\n📊 SECONDARY RECOMMENDATIONS (Medium Importance):")
for param in importance_df.iloc[4:]['parameter']:  # Remaining parameters
    if param in top_5.columns:
        optimal_value = top_5[param].median()
        param_range = f"[{top_5[param].min():.4f} - {top_5[param].max():.4f}]"
        print(f"  {param:20s}: {optimal_value:8.4f} (range: {param_range})")

# Performance metrics of optimal configuration
print("\n📈 EXPECTED PERFORMANCE:")
best_run = data.loc[data['validation_policy_loss'].idxmin()]
print(f"  Best validation_policy_loss: {best_run['validation_policy_loss']:.4f}")
if not pd.isna(best_run['evaluation_win_rate_vs_initial']):
    print(f"  Best win rate vs initial:    {best_run['evaluation_win_rate_vs_initial']:.1%}")

# Create visualization
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Feature Importance
axes[0,0].barh(importance_df.head(6)['parameter'], importance_df.head(6)['importance'])
axes[0,0].set_title('Feature Importance (Random Forest)')
axes[0,0].set_xlabel('Importance Score')

# 2. Correlation Heatmap
corr_matrix = data[hyperparams + ['validation_policy_loss']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,1])
axes[0,1].set_title('Correlation Matrix')

# 3. Learning Rate vs Performance
axes[1,0].scatter(data['initial-lr'], data['validation_policy_loss'], alpha=0.7)
axes[1,0].set_xlabel('Initial Learning Rate')
axes[1,0].set_ylabel('Validation Policy Loss')
axes[1,0].set_title('Learning Rate vs Performance')
axes[1,0].set_xscale('log')

# 4. Batch Size vs Performance
batch_sizes = data['batch-size'].unique()
batch_performance = [data[data['batch-size']==bs]['validation_policy_loss'].mean() for bs in sorted(batch_sizes) if not pd.isna(bs)]
axes[1,1].bar([str(int(bs)) for bs in sorted(batch_sizes) if not pd.isna(bs)], batch_performance)
axes[1,1].set_xlabel('Batch Size')
axes[1,1].set_ylabel('Mean Validation Policy Loss')
axes[1,1].set_title('Batch Size vs Performance')

plt.tight_layout()
plt.savefig('experiments/hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n📊 Visualization saved to: experiments/hyperparameter_analysis.png")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
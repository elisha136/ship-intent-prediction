"""
Visualization Script for Model Evaluation Results

Generates performance comparison charts from model evaluation data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read results
df = pd.read_csv('model_comparison.csv')

# Set up the plot style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Ship Trajectory Prediction - Statistical Models Performance', 
             fontsize=16, fontweight='bold')

# Prepare data
models = df['Model'].unique()
horizons = df['Horizon (s)'].unique()

colors = {
    'Constant Velocity': '#2ecc71',  # Green
    'Constant Turn Rate': '#e74c3c',  # Red
    'Kalman Filter': '#3498db'  # Blue
}

markers = {
    'Constant Velocity': 'o',
    'Constant Turn Rate': 's',
    'Kalman Filter': '^'
}

# Plot 1: ADE vs Horizon
ax1 = axes[0, 0]
for model in models:
    model_data = df[df['Model'] == model]
    ax1.plot(model_data['Horizon (s)'].values, model_data['ADE (m)'].values,
             marker=markers[model], linewidth=2, markersize=8,
             label=model, color=colors[model])
ax1.set_xlabel('Prediction Horizon (seconds)', fontsize=11)
ax1.set_ylabel('Average Displacement Error (meters)', fontsize=11)
ax1.set_title('ADE: Average Error Across All Predicted Points', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(horizons)

# Plot 2: FDE vs Horizon
ax2 = axes[0, 1]
for model in models:
    model_data = df[df['Model'] == model]
    ax2.plot(model_data['Horizon (s)'].values, model_data['FDE (m)'].values,
             marker=markers[model], linewidth=2, markersize=8,
             label=model, color=colors[model])
ax2.set_xlabel('Prediction Horizon (seconds)', fontsize=11)
ax2.set_ylabel('Final Displacement Error (meters)', fontsize=11)
ax2.set_title('FDE: Error at Final Prediction Point', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(horizons)

# Plot 3: RMSE vs Horizon
ax3 = axes[1, 0]
for model in models:
    model_data = df[df['Model'] == model]
    ax3.plot(model_data['Horizon (s)'].values, model_data['RMSE (m)'].values,
             marker=markers[model], linewidth=2, markersize=8,
             label=model, color=colors[model])
ax3.set_xlabel('Prediction Horizon (seconds)', fontsize=11)
ax3.set_ylabel('Root Mean Squared Error (meters)', fontsize=11)
ax3.set_title('RMSE: Penalizes Large Errors', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(horizons)

# Plot 4: Model Comparison Bar Chart at 180s
ax4 = axes[1, 1]
horizon_180 = df[df['Horizon (s)'] == 180]
x = np.arange(len(models))
width = 0.2

metrics = ['ADE (m)', 'FDE (m)', 'RMSE (m)']
metric_colors = ['#3498db', '#e74c3c', '#2ecc71']

for i, metric in enumerate(metrics):
    values = [horizon_180[horizon_180['Model'] == model][metric].values[0] 
              for model in models]
    ax4.bar(x + i*width, values, width, label=metric, color=metric_colors[i], alpha=0.8)

ax4.set_xlabel('Model', fontsize=11)
ax4.set_ylabel('Error (meters)', fontsize=11)
ax4.set_title('Performance Comparison at 180s Horizon', fontsize=12, fontweight='bold')
ax4.set_xticks(x + width)
ax4.set_xticklabels(['CV', 'CTR', 'KF'], fontsize=10)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_performance_comparison.png")

# Create second figure: Error growth analysis
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle('Error Growth Analysis', fontsize=16, fontweight='bold')

# Plot 5: Relative error growth
ax5 = axes2[0]
for model in models:
    model_data = df[df['Model'] == model].sort_values('Horizon (s)')
    ade_values = model_data['ADE (m)'].values
    # Calculate percentage increase from 30s baseline
    baseline = ade_values[0]
    relative_growth = [(val / baseline - 1) * 100 for val in ade_values]
    ax5.plot(model_data['Horizon (s)'].values, relative_growth,
             marker=markers[model], linewidth=2, markersize=8,
             label=model, color=colors[model])

ax5.set_xlabel('Prediction Horizon (seconds)', fontsize=11)
ax5.set_ylabel('Error Growth from 30s Baseline (%)', fontsize=11)
ax5.set_title('Relative Error Growth Rate', fontsize=12, fontweight='bold')
ax5.legend(loc='upper left', fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_xticks(horizons)
ax5.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Plot 6: Error per second
ax6 = axes2[1]
for model in models:
    model_data = df[df['Model'] == model].sort_values('Horizon (s)')
    error_per_second = model_data['ADE (m)'].values / model_data['Horizon (s)'].values
    ax6.plot(model_data['Horizon (s)'].values, error_per_second,
             marker=markers[model], linewidth=2, markersize=8,
             label=model, color=colors[model])

ax6.set_xlabel('Prediction Horizon (seconds)', fontsize=11)
ax6.set_ylabel('Error per Second (m/s)', fontsize=11)
ax6.set_title('Error Accumulation Rate', fontsize=12, fontweight='bold')
ax6.legend(loc='upper right', fontsize=10)
ax6.grid(True, alpha=0.3)
ax6.set_xticks(horizons)

plt.tight_layout()
plt.savefig('error_growth_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: error_growth_analysis.png")

# Create summary statistics table
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for model in models:
    model_data = df[df['Model'] == model]
    print(f"\n{model}:")
    print(f"  Mean ADE: {model_data['ADE (m)'].mean():.2f}m")
    print(f"  Mean RMSE: {model_data['RMSE (m)'].mean():.2f}m")
    print(f"  Best Horizon (lowest ADE): {model_data.loc[model_data['ADE (m)'].idxmin(), 'Horizon (s)']}s")
    print(f"  Worst Horizon (highest ADE): {model_data.loc[model_data['ADE (m)'].idxmax(), 'Horizon (s)']}s")

print("\n" + "="*80)
print("MODEL RANKINGS (by mean ADE)")
print("="*80)
mean_ade = df.groupby('Model')['ADE (m)'].mean().sort_values()
for i, (model, ade) in enumerate(mean_ade.items(), 1):
    print(f"{i}. {model}: {ade:.2f}m")

print("\n✓ Visualization complete!")


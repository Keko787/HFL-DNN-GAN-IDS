import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import make_interp_spline

# Set global font size and style for better readability
plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.titlesize': 22,      # Axes title font size
    'axes.labelsize': 18,      # X and Y label font size
    'xtick.labelsize': 16,     # X tick label font size
    'ytick.labelsize': 16,     # Y tick label font size
    'legend.fontsize': 16,     # Legend font size
    'figure.titlesize': 24     # Figure title font size
})

# Set style for clean white background
plt.style.use('default')
sns.set_palette("husl")

# Set clean white background
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
})

# Sample data - typically federated learning has higher loss due to data heterogeneity
epochs = np.arange(1, 6)
centralized_loss = [0.5419, 0.2283, 0.1367, 0.1118, 0.1027]  # Faster convergence
federated_loss = [0.2663, 0.1468, 0.1177, 0.1057, 0.0996]    # Slower convergence due to data distribution

# Create DataFrame for easier plotting
data = pd.DataFrame({
    'Epoch': np.tile(epochs, 2),
    'Loss': centralized_loss + federated_loss,
    'NIDS Model': ['Federated'] * 5 + ['Hierarchically Federated'] * 5
})

# Calculate statistical closeness measures
correlation, p_value = stats.pearsonr(centralized_loss, federated_loss)
mae = np.mean(np.abs(np.array(centralized_loss) - np.array(federated_loss)))
rmse = np.sqrt(np.mean((np.array(centralized_loss) - np.array(federated_loss))**2))
max_diff = np.max(np.abs(np.array(centralized_loss) - np.array(federated_loss)))
mean_diff = np.mean(np.array(centralized_loss) - np.array(federated_loss))

# Create the enhanced figure with smooth curves - INCREASED SIZE AND ADJUSTED LAYOUT
fig, ax = plt.subplots(figsize=(14, 12))  # Increased height from 9 to 12

# Smooth curves comparison
x_smooth = np.linspace(1, 5, 100)
centralized_smooth = make_interp_spline(epochs, centralized_loss, k=3)(x_smooth)
federated_smooth = make_interp_spline(epochs, federated_loss, k=3)(x_smooth)

ax.plot(x_smooth, centralized_smooth, linewidth=4, color='#2E86AB',
         label='Federated Learning Curve', alpha=0.8)
ax.plot(x_smooth, federated_smooth, linewidth=4, color='#A23B72',
         label='Hierarchically Fed Curve', alpha=0.8)

# Add original data points with different markers
# FL points (circles)
fl_scatter = ax.scatter(epochs, centralized_loss, color='#1B5A7A', s=120, zorder=5,
           edgecolors='white', linewidth=2, label='Federated Data Points', marker='o')
# HFL points (squares)
hfl_scatter = ax.scatter(epochs, federated_loss, color='#7A1B47', s=120, zorder=5,
           edgecolors='white', linewidth=2, label='Hierarchically Fed Data Points', marker='s')

# Add value labels for each point
for i, (epoch, fl_loss) in enumerate(zip(epochs, centralized_loss)):
    ax.annotate(f'{fl_loss:.4f}',
                (epoch, fl_loss),
                textcoords="offset points",
                xytext=(0,15),
                ha='center',
                fontsize=20,
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#1B5A7A', alpha=0.9, edgecolor='white', linewidth=1))

# ADJUSTED: Moved the bottom annotations higher to avoid overlap with x-axis labels
for i, (epoch, hfl_loss) in enumerate(zip(epochs, federated_loss)):
    ax.annotate(f'{hfl_loss:.4f}',
                (epoch, hfl_loss),
                textcoords="offset points",
                xytext=(0,-25),  # Moved from -25 to -35 for more space
                ha='center',
                fontsize=20,
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#7A1B47', alpha=0.9, edgecolor='white', linewidth=1))

# Fill area between curves to show difference
ax.fill_between(x_smooth, centralized_smooth, federated_smooth,
                alpha=0.2, color='gray', label='Difference Area')

# Customize the plot
ax.set_xlabel('Rounds', fontsize=24, fontweight='bold', labelpad=20)  # Increased labelpad
ax.set_ylabel('Loss', fontsize=24, fontweight='bold', labelpad=15)
ax.legend(fontsize=20, loc='upper right', frameon=True, fancybox=False, shadow=False,
          facecolor='white', edgecolor='black', framealpha=0.95, borderpad=1)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid(True, alpha=0.25, linewidth=1, linestyle='-', color='gray')
ax.set_axisbelow(True)

# Set x-axis to only show whole numbers
ax.set_xticks(epochs)  # Only show ticks at 1, 2, 3, 4, 5
ax.set_xlim(0.5, 5.5)  # Add some padding on the sides

# Add statistical information as text box
title = "Statistical Closeness Analysis"
stats_lines = [
    f"Correlation: {correlation:.4f} (p={p_value:.6f})",
    f"Maximum Difference: {max_diff:.4f}",
    f"Mean Difference: {mean_diff:.4f}",
    f"Root Mean Square Error (RMSE): {rmse:.6f}",
    f"Mean Absolute Error (MAE): {mae:.6f}"
]

# Find the longest line to determine centering width
max_length = max(len(line) for line in stats_lines)
centered_title = title.center(max_length)

stats_text = f"""{centered_title}
{chr(10).join(stats_lines)}"""

ax.text(0.98, 0.45, stats_text, transform=ax.transAxes, fontsize=20,
         verticalalignment='center', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95,
                  edgecolor='black', linewidth=1.2))

# ADJUSTED: Added more space at the bottom to prevent overlap
plt.subplots_adjust(bottom=0.15)  # Increase bottom margin
plt.tight_layout()
plt.show()

# Enhanced summary statistics
print("=" * 80)
print("                    COMPREHENSIVE LOSS ANALYSIS")
print("=" * 80)
print(f"{'Epoch':<10} {'Federated':<15} {'Hier-Fed':<15} {'Difference':<15} {'% Difference':<15}")
print("-" * 80)
for i, epoch in enumerate(epochs):
    diff = centralized_loss[i] - federated_loss[i]
    pct_diff = (diff / centralized_loss[i]) * 100
    print(f"{epoch:<10} {centralized_loss[i]:<15.4f} {federated_loss[i]:<15.4f} {diff:<15.4f} {pct_diff:<15.2f}%")

print("\n" + "=" * 80)
print("                        STATISTICAL CLOSENESS METRICS")
print("=" * 80)
print(f"Pearson Correlation Coefficient:    {correlation:.6f} (p-value: {p_value:.6f})")
print(f"Mean Absolute Error (MAE):          {mae:.6f}")
print(f"Root Mean Square Error (RMSE):      {rmse:.6f}")
print(f"Maximum Absolute Difference:        {max_diff:.6f}")
print(f"Mean Difference:                    {mean_diff:.6f}")
print(f"Standard Deviation of Differences:  {np.std(np.array(centralized_loss) - np.array(federated_loss)):.6f}")

print("\n" + "=" * 80)
print("                        CONVERGENCE ANALYSIS")
print("=" * 80)
fed_total_improvement = centralized_loss[0] - centralized_loss[-1]
hier_total_improvement = federated_loss[0] - federated_loss[-1]
print(f"Federated Total Improvement:        {fed_total_improvement:.6f}")
print(f"Hierarchically Fed Total Improve:   {hier_total_improvement:.6f}")
print(f"Federated Learning Rate:            {fed_total_improvement/4:.6f} per epoch")
print(f"Hierarchically Fed Learning Rate:   {hier_total_improvement/4:.6f} per epoch")

print("\n" + "=" * 80)
print("                        INTERPRETATION")
print("=" * 80)
if correlation > 0.9:
    corr_strength = "Very Strong"
elif correlation > 0.7:
    corr_strength = "Strong"
elif correlation > 0.5:
    corr_strength = "Moderate"
else:
    corr_strength = "Weak"

print(f"Correlation Strength:               {corr_strength}")
print(f"Average Closeness:                  {(1 - mae/np.mean(centralized_loss))*100:.2f}%")
print("=" * 80)
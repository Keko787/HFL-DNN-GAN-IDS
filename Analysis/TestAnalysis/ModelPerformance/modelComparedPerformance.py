import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set global font sizes for better readability
plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 18,     # Title font size
    'axes.labelsize': 16,     # Axis label font size
    'xtick.labelsize': 14,    # X-tick label font size
    'ytick.labelsize': 14,    # Y-tick label font size
    'legend.fontsize': 14,    # Legend font size
    'figure.titlesize': 20    # Figure title font size
})

# Data from the two models based on the images
model1_data = {
    'Accuracy': 99.15,
    'Precision': 100,
    'Recall': 98.31,
    'F1-Score': 99.15
}

model2_data = {
    'Accuracy': 99.55,
    'Precision': 100,
    'Recall': 98.07,
    'F1-Score': 99.03
}

# Create DataFrame for easier plotting
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
model1_values = [model1_data[metric] for metric in metrics]
model2_values = [model2_data[metric] for metric in metrics]

# First plot - Enhanced matplotlib version
plt.figure(figsize=(12, 8))  # Larger figure size
x = np.arange(len(metrics))
width = 0.3  # Reduced width to create more space between bars
bar_spacing = 0.4  # Increased spacing between bar groups

# Create bars with enhanced styling and more spacing
bars1 = plt.bar(x - bar_spacing/2, model1_values, width, label='Federated Model',
                color=['#3498DB', '#3498DB', '#3498DB', '#3498DB'],
                edgecolor='black', linewidth=1.2)
bars2 = plt.bar(x + bar_spacing/2, model2_values, width, label='Central Model',
                color=['#2ECC71', '#2ECC71', '#2ECC71', '#2ECC71'],
                edgecolor='black', linewidth=1.2)

# Customize the plot with larger fonts
plt.xlabel('Metrics', fontsize=18, fontweight='bold')
plt.ylabel('Percentage (%)', fontsize=18, fontweight='bold')
plt.title('', fontsize=22, fontweight='bold', pad=20)
plt.xticks(x, metrics, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=20, loc='upper right')

# Add value labels on bars with larger font
def add_value_labels(bars, fontsize=14):
    for bar in bars:
        height = bar.get_height()
        # Format label - remove decimals if value is exactly 100
        if abs(height - 100) < 0.001:  # More robust comparison for 100
            label = '100%'
        else:
            label = f'{height:.2f}%'
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                label, ha='center', va='bottom',
                fontsize=fontsize, fontweight='bold')

add_value_labels(bars1, fontsize=16)
add_value_labels(bars2, fontsize=16)

# Set y-axis to show the range that highlights differences
plt.ylim(98, 100.25)
plt.yticks([98, 98.5, 99, 99.5, 100])
plt.grid(True, alpha=0.3, linewidth=1)
plt.tight_layout()
plt.show()

# Second plot - Enhanced seaborn version
plt.figure(figsize=(12, 8))  # Larger figure size

# Prepare data for seaborn
df = pd.DataFrame({
    'Metric': metrics + metrics,
    'Value': model1_values + model2_values,
    'NIDS Model': ['Federated Model'] * 4 + ['Central Model'] * 4
})

# Create seaborn barplot with enhanced styling and spacing
ax = sns.barplot(data=df, x='Metric', y='Value', hue='NIDS Model',
                 palette='Set2', edgecolor='black', linewidth=1.2,
                 width=0.6, dodge=True)  # Reduced width for more spacing

# Manually adjust bar positions to create more spacing
for i, patch in enumerate(ax.patches):
    # Get current position
    current_x = patch.get_x()
    width = patch.get_width()

    # Adjust positions to create more spacing
    if i < 4:  # First model bars
        new_x = current_x - 0.05  # Move left slightly
    else:  # Second model bars
        new_x = current_x + 0.05  # Move right slightly

    patch.set_x(new_x)

# Add value labels with larger font
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    # Format label - remove decimals if value is exactly 100
    if abs(height - 100) < 0.001:  # More robust comparison for 100
        label = '100%'
    else:
        label = f'{height:.2f}%'
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
            label, ha='center', va='bottom',
            fontsize=14, fontweight='bold')

# Enhanced styling
plt.title('Model Performance Comparison (Seaborn Version)',
          fontsize=22, fontweight='bold', pad=20)
plt.xlabel('Metrics', fontsize=18, fontweight='bold')
plt.ylabel('Percentage (%)', fontsize=18, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Enhance legend
legend = plt.legend(fontsize=16, loc='upper right', frameon=True,
                    fancybox=True, shadow=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

plt.ylim(97.5, 100.25)
plt.grid(True, alpha=0.3, linewidth=1)
plt.tight_layout()
plt.show()

# Reset matplotlib parameters to default (optional)
plt.rcdefaults()
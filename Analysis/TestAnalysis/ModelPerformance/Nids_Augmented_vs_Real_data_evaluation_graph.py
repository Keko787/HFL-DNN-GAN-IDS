import matplotlib.pyplot as plt
import numpy as np

# Labels for each dataset evaluation step
labels = ["AVG Real Data", "Epoch 1", "Epoch 2",
          "Epoch 3", "Epoch 4", "Epoch 5"]

# Metric values for real data (averaged) and augmented data per epoch
real_accuracy = [0.9909, 0.9903, 0.9908, 0.9915, 0.9905]
real_precision = [0.9825, 0.9813, 0.9826, 0.9835, 0.9817]
real_recall = [0.9996, 0.9997, 0.9994, 0.9997, 0.9997]
real_loss = [0.0782, 0.0804, 0.0776, 0.0744, 0.0801]

# Averaging real data metrics
avg_real_accuracy = np.mean(real_accuracy)
avg_real_precision = np.mean(real_precision)
avg_real_recall = np.mean(real_recall)
avg_real_loss = np.mean(real_loss)

# Augmented Data Metrics per Epoch
augmented_accuracy = [0.8161, 0.8157, 0.8205, 0.8188, 0.8319]
augmented_precision = [0.5761, 0.5756, 0.5822, 0.5798, 0.5979]
augmented_recall = [0.9996, 0.9997, 0.9994, 0.9997, 0.9997]
augmented_loss = [0.6091, 0.6126, 0.5439, 0.5343, 0.4979]

# Combine the real data with the augmented data metrics
bar_data = [(avg_real_accuracy, avg_real_precision, avg_real_recall, avg_real_loss)] + list(zip(augmented_accuracy, augmented_precision, augmented_recall, augmented_loss))

# Setting up bar chart parameters
bar_width = 0.2
x = np.arange(len(labels))

# Extracting individual metric lists
accuracy_vals = [entry[0] for entry in bar_data]
precision_vals = [entry[1] for entry in bar_data]
recall_vals = [entry[2] for entry in bar_data]
loss_vals = [entry[3] for entry in bar_data]

# Define unique hatch patterns for black-and-white differentiation
hatch_patterns = ['/', '\\', 'x', 'o']

# Plotting the bars with patterns
plt.figure(figsize=(12, 6))
plt.bar(x - 1.5*bar_width, accuracy_vals, bar_width, label='Accuracy', color='white', edgecolor='black', hatch=hatch_patterns[0])
plt.bar(x - 0.5*bar_width, precision_vals, bar_width, label='Precision', color='white', edgecolor='black', hatch=hatch_patterns[1])
plt.bar(x + 0.5*bar_width, recall_vals, bar_width, label='Recall', color='white', edgecolor='black', hatch=hatch_patterns[2])
plt.bar(x + 1.5*bar_width, loss_vals, bar_width, label='Loss', color='white', edgecolor='black', hatch=hatch_patterns[3])

# Add a vertical separation line between "AVG Real Data" and "Augmented Data Epoch 1"
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2)

# Formatting the chart
plt.xlabel("Dataset Evaluation")
plt.ylabel("Metric Values")
plt.title("Comparison of NIDS Performance: Real Data vs Augmented Data (5 Epochs)")
plt.xticks(ticks=x, labels=labels)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Display the chart
plt.show()

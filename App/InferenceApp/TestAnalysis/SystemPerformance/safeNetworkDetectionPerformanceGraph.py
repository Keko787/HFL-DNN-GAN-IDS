import matplotlib.pyplot as plt
import numpy as np


def plot_attack_detection_performance(true_negatives=16.61, false_positives=8.24,
                                      model_name="NIDS Model", dataset="CICIOT",
                                      attack_types=None, attack_detection_rates=None,
                                      use_log_scale=True):
    """
    Create a horizontal bar chart showing only true negatives and false positives for safe data testing.

    Parameters:
    -----------
    true_negatives : float
        Number of correctly identified normal traffic (log₂ scale if use_log_scale=True)
    false_positives : float
        Number of normal traffic incorrectly labeled as attacks (log₂ scale if use_log_scale=True)
    model_name : str
        Name of the model used for detection
    dataset : str
        Name of the dataset used (e.g., "CICIOT", "IOTBOTNET", "IOT")
    attack_types : list of str, optional
        List of attack types to show detection rates for
    attack_detection_rates : list of float, optional
        List of detection rates corresponding to attack_types
    use_log_scale : bool
        Whether the provided values are in log₂ scale
    """

    # Convert log values to actual counts if necessary
    if use_log_scale:
        tn_actual = 2 ** true_negatives
        fp_actual = 2 ** false_positives
    else:
        tn_actual = true_negatives
        fp_actual = false_positives

    # Calculate metrics based on actual counts
    total = tn_actual + fp_actual
    specificity = 100 * tn_actual / total if total > 0 else 0
    false_positive_rate = 100 * fp_actual / total if total > 0 else 0

    # Set up figure and primary axis for main metrics
    fig = plt.figure(figsize=(14, 6))  # Decreased height to bring bars closer

    # First subplot for confusion matrix data
    ax1 = plt.subplot2grid((1, 2), (0, 0))

    # Create grouped horizontal bar chart for TN, FP
    categories = ['True Negatives', 'False Positives']

    # Use the log values for the bar chart
    values = [true_negatives, false_positives]
    colors = ['#3498db', '#e74c3c']  # Blue for TN, Red for FP

    # Create horizontal bars with reduced height
    bars1 = ax1.barh(categories, values, color=colors, height=0.4)  # Reduced height from 0.6 to 0.4

    # Format the text labels
    label_texts = []
    for log_val, actual in zip(values, [tn_actual, fp_actual]):
        # Format actual values for better readability
        if actual >= 1000:
            actual_str = f"{actual / 1000:.1f}K"
        else:
            actual_str = f"{int(actual)}"

        label_texts.append(f'log₂: {log_val:.2f}\n(≈{actual_str})')

    # Add data labels to the bars
    for bar, text in zip(bars1, label_texts):
        width = bar.get_width()
        ax1.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                 text, va='center', ha='left', fontweight='bold')

    # Set labels and title
    ax1.set_xlabel('log₂ Count', fontsize=12)
    ax1.set_title('Safe Data Testing Results (log₂ scale)', fontsize=14, pad=20)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    # Reduce the y-padding to bring bars closer
    ax1.margins(y=0.15)  # Decreased y margin

    # Increase the right margin to make room for labels
    max_value = max(values)
    x_max = max_value * 1.4  # Add 40% extra space to ensure labels fit
    ax1.set_xlim(0, x_max)

    # Second subplot for performance metrics
    ax2 = plt.subplot2grid((1, 2), (0, 1))

    # Create horizontal bar chart for specificity and false positive rate
    metrics = ['Specificity', 'False Positive\nRate']
    performance = [specificity, false_positive_rate]
    colors = ['#1abc9c', '#e67e22']  # Green for Specificity, Orange for FPR

    bars2 = ax2.barh(metrics, performance, color=colors, height=0.4)  # Reduced height from 0.6 to 0.4

    # Add data labels
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{width:.1f}%', va='center', ha='left', fontweight='bold')

    # Set labels and title
    ax2.set_xlabel('Percentage (%)', fontsize=12)
    ax2.set_title('Performance Metrics for Safe Data', fontsize=14, pad=20)
    ax2.set_xlim(0, 110)  # Set x-axis to go from 0 to 110 to make room for text
    ax2.grid(axis='x', linestyle='--', alpha=0.7)

    # Reduce the y-padding to bring bars closer
    ax2.margins(y=0.15)  # Decreased y margin

    # Add overall title with dataset and model information
    plt.suptitle(f'Safe Data Testing Performance: {model_name} on {dataset} Dataset', fontsize=16, y=0.98)

    # Print the exact values used for reference
    print(f"Log₂ Values Used:")
    print(f"True Negatives (log₂): {true_negatives:.2f} → Actual: {int(tn_actual)}")
    print(f"False Positives (log₂): {false_positives:.2f} → Actual: {int(fp_actual)}")
    print(f"\nPerformance Metrics:")
    print(f"Specificity: {specificity:.2f}%")
    print(f"False Positive Rate: {false_positive_rate:.2f}%")

    # Adjust layout to ensure everything fits
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, right=0.95, wspace=0.3, hspace=0)

    return fig


# Example using the exact log base 2 values:
fig = plot_attack_detection_performance(
    true_negatives=8.714,  # log₂(99796) ≈ 16.61
    false_positives=8.353,  # log₂(303) ≈ 8.24
    model_name="ACGAN Federated Model",
    dataset="CICIOT",
    use_log_scale=True  # Indicate that we're using log₂ values
)

plt.show()
import matplotlib.pyplot as plt
import numpy as np


def plot_attack_detection_performance(true_positives=14.86, true_negatives=16.61, false_positives=8.24,
                                      false_negatives=7.67,
                                      model_name="NIDS Model", dataset="CICIOT",
                                      attack_types=None, attack_detection_rates=None,
                                      use_log_scale=True, performance_range=(97, 100),
                                      manual_accuracy=None, manual_precision=None,
                                      manual_recall=None, manual_f1_score=None):
    """
    Create a bar chart showing the performance metrics for network attack detection.

    Parameters:
    -----------
    true_positives : float
        Number of correctly identified attacks (log₂ scale if use_log_scale=True)
    true_negatives : float
        Number of correctly identified normal traffic (log₂ scale if use_log_scale=True)
    false_positives : float
        Number of normal traffic incorrectly labeled as attacks (log₂ scale if use_log_scale=True)
    false_negatives : float
        Number of attacks incorrectly labeled as normal traffic (log₂ scale if use_log_scale=True)
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
    performance_range : tuple, optional
        Range (min, max) for performance metrics y-axis. Default is (97, 100) for high-performance models.
        Set to None to use automatic scaling (0-110%).
    manual_accuracy : float, optional
        Manual accuracy percentage (0-100). If provided, overrides calculated value.
    manual_precision : float, optional
        Manual precision percentage (0-100). If provided, overrides calculated value.
    manual_recall : float, optional
        Manual recall percentage (0-100). If provided, overrides calculated value.
    manual_f1_score : float, optional
        Manual F1-score percentage (0-100). If provided, overrides calculated value.
    """

    # Convert log values to actual counts if necessary
    if use_log_scale:
        tp_actual = 2 ** true_positives
        tn_actual = 2 ** true_negatives
        fp_actual = 2 ** false_positives
        fn_actual = 2 ** false_negatives
    else:
        tp_actual = true_positives
        tn_actual = true_negatives
        fp_actual = false_positives
        fn_actual = false_negatives

    # Calculate metrics based on actual counts (will be overridden if manual values provided)
    total = tp_actual + tn_actual + fp_actual + fn_actual
    calculated_accuracy = 100 * (tp_actual + tn_actual) / total
    calculated_precision = 100 * tp_actual / (tp_actual + fp_actual) if (tp_actual + fp_actual) > 0 else 0
    calculated_recall = 100 * tp_actual / (tp_actual + fn_actual) if (tp_actual + fn_actual) > 0 else 0
    calculated_f1_score = 2 * calculated_precision * calculated_recall / (calculated_precision + calculated_recall) if (
                                                                                                                                   calculated_precision + calculated_recall) > 0 else 0

    # Use manual values if provided, otherwise use calculated values
    accuracy = manual_accuracy if manual_accuracy is not None else calculated_accuracy
    precision = manual_precision if manual_precision is not None else calculated_precision
    recall = manual_recall if manual_recall is not None else calculated_recall
    f1_score = manual_f1_score if manual_f1_score is not None else calculated_f1_score

    # Set up figure and primary axis for main metrics
    fig = plt.figure(figsize=(14, 10))

    # First subplot for confusion matrix data
    ax1 = plt.subplot2grid((2, 2), (0, 0))

    # Create grouped bar chart for TP, TN, FP, FN
    categories = ['True\nPositives', 'True\nNegatives', 'False\nPositives', 'False\nNegatives']

    # Use the log values for the bar chart
    values = [true_positives, true_negatives, false_positives, false_negatives]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

    bars1 = ax1.bar(categories, values, color=colors, width=0.6)

    # Add data labels to the bars - show both log value and actual value
    for bar, log_val, actual in zip(bars1, values, [tp_actual, tn_actual, fp_actual, fn_actual]):
        height = bar.get_height()
        # Format actual values for better readability
        if actual >= 1000:
            actual_str = f"{actual / 1000:.1f}K"
        else:
            actual_str = f"{int(actual)}"

        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'log₂: {log_val:.2f}\n(≈{actual_str})', ha='center', va='bottom', fontweight='bold')

    # Set labels and title
    ax1.set_ylabel('log₂ Count', fontsize=12)
    ax1.set_title('Confusion Matrix (log₂ scale)', fontsize=14, pad=20)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Increase the top margin to make room for labels
    y_max = max(values) * 1.15  # Add 15% extra space at the top
    ax1.set_ylim(0, y_max)

    # Second subplot for performance metrics
    ax2 = plt.subplot2grid((2, 2), (0, 1))

    # Create bar chart for accuracy, precision, recall, F1
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    performance = [accuracy, precision, recall, f1_score]
    colors = ['#9b59b6', '#1abc9c', '#34495e', '#2980b9']
    # colors = ['#FF0000','#0000FF', '#00FF00', '#800080']
    # colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

    # Use custom range if specified
    if performance_range is not None:
        min_range, max_range = performance_range

        # Check if all metrics are within the custom range
        all_in_range = all(min_range <= perf <= max_range for perf in performance)

        if all_in_range:
            bars2 = ax2.bar(metrics, performance, color=colors, width=0.6)

            # Set y-axis to custom range with some padding
            padding = (max_range - min_range) * 0.05  # 5% padding
            ax2.set_ylim(min_range - padding, max_range + padding)

            # Add a note about the custom range
            ax2.text(0.5, min_range - padding + 0.1, f"Note: Y-axis range set to {min_range}-{max_range}%",
                     ha='center', fontsize=10, style='italic', color='gray',
                     transform=ax2.get_xaxis_transform())

            # Add data labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                         f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
        else:
            # If not all metrics are in range, use automatic scaling
            bars2 = ax2.bar(metrics, performance, color=colors, width=0.6)
            ax2.set_ylim(0, 110)

            # Add data labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

            print(
                f"Warning: Some performance metrics are outside the range {min_range}-{max_range}%. Using automatic scaling.")
    else:
        # Check if all metrics are above 99% for auto-normalization
        all_high_performance = all(perf >= 99.0 for perf in performance)

        if all_high_performance:
            # Save original values for labels
            original_performance = performance.copy()

            # Normalize values to 99-100% range for better visualization
            normalized_performance = [99.0 + (perf - 99.0) * 10 for perf in performance]
            bars2 = ax2.bar(metrics, normalized_performance, color=colors, width=0.6)

            # Set y-axis to focus on 99-100% range
            ax2.set_ylim(98.5, 100.5)

            # Add a note about the normalized scale
            ax2.text(0.5, 98.8, "Note: Y-axis normalized to 99-100% range",
                     ha='center', fontsize=10, style='italic', color='gray')

            # Add data labels showing original values
            for bar, orig_height in zip(bars2, original_performance):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                         f'{orig_height:.2f}%', ha='center', va='bottom', fontweight='bold')
        else:
            # Use regular scaling if not all metrics are high
            bars2 = ax2.bar(metrics, performance, color=colors, width=0.6)
            ax2.set_ylim(0, 110)

            # Add data labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Set labels and title
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    title_suffix = f" ({performance_range[0]}-{performance_range[1]}% range)" if performance_range is not None else ""
    ax2.set_title(f'Performance Metrics{title_suffix}', fontsize=14, pad=20)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Third subplot for attack type detection rates (if provided)
    if attack_types and attack_detection_rates:
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

        # Sort attack types by detection rate for better visualization
        sorted_indices = np.argsort(attack_detection_rates)
        sorted_attack_types = [attack_types[i] for i in sorted_indices]
        sorted_rates = [attack_detection_rates[i] for i in sorted_indices]

        # Create color gradient based on detection rates
        colors = plt.cm.RdYlGn(np.array(sorted_rates) / 100)

        bars3 = ax3.bar(sorted_attack_types, sorted_rates, color=colors, width=0.6)

        # Add data labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Set labels and title
        ax3.set_title('Detection Rate by Attack Type', fontsize=14, pad=20)
        ax3.set_xlabel('Attack Type', fontsize=12)
        ax3.set_ylabel('Detection Rate (%)', fontsize=12)
        ax3.set_ylim(0, 110)  # Set y-axis to go from 0 to 110 to make room for text
        ax3.grid(axis='y', linestyle='--', alpha=0.7)

        # Rotate x-axis labels if there are many attack types
        if len(attack_types) > 5:
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

    # Add overall title with dataset and model information
    plt.suptitle(f'Attack Detection Performance: {model_name} on {dataset} Dataset', fontsize=16, y=0.98)

    # Print the exact values used for reference
    print(f"Log₂ Values Used:")
    print(f"True Positives (log₂): {true_positives:.2f} → Actual: {int(tp_actual)}")
    print(f"True Negatives (log₂): {true_negatives:.2f} → Actual: {int(tn_actual)}")
    print(f"False Positives (log₂): {false_positives:.2f} → Actual: {int(fp_actual)}")
    print(f"False Negatives (log₂): {false_negatives:.2f} → Actual: {int(fn_actual)}")

    print(f"\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.2f}%{' (manual)' if manual_accuracy is not None else ' (calculated)'}")
    print(f"Precision: {precision:.2f}%{' (manual)' if manual_precision is not None else ' (calculated)'}")
    print(f"Recall: {recall:.2f}%{' (manual)' if manual_recall is not None else ' (calculated)'}")
    print(f"F1 Score: {f1_score:.2f}%{' (manual)' if manual_f1_score is not None else ' (calculated)'}")

    if performance_range is not None:
        min_range, max_range = performance_range
        all_in_range = all(min_range <= perf <= max_range for perf in performance)
        if all_in_range:
            print(f"\nNote: Performance metrics chart uses {min_range}-{max_range}% range for better visualization.")
    elif performance_range is None and all(perf >= 99.0 for perf in performance):
        print("\nNote: All performance metrics are above 99%. The chart has been normalized to show the 99-100% range.")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return fig




# Example 7: Automatic scaling (no custom range)
print("\nExample 7: Automatic scaling with no custom range")
fig7 = plot_attack_detection_performance(
    true_positives=4.288,
    true_negatives=14.288,
    false_positives=5.044,
    false_negatives=13.283,
    model_name="Federated Model",
    dataset="CICIOT",
    use_log_scale=True,
    performance_range=(98.0, 100.25),  # Use automatic scaling
    manual_accuracy=99.15,
    manual_precision=100.00,
    manual_recall=98.31,
    manual_f1_score=99.15
)
plt.show()

# Example 7: Automatic scaling (no custom range)
print("\nExample 7: Automatic scaling with no custom range")
fig7 = plot_attack_detection_performance(
    true_positives=4.288,
    true_negatives=14.288,
    false_positives=5.044,
    false_negatives=13.283,
    model_name="Central Model",
    dataset="CICIOT",
    use_log_scale=True,
    performance_range=(98.0, 100.25),  # Use automatic scaling
    manual_accuracy=99.55,
    manual_precision=100.00,
    manual_recall=98.07,
    manual_f1_score=99.03
)
plt.show()
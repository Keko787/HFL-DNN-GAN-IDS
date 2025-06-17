import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter
import os
import argparse


def load_models(generator_path, discriminator_path):
    """
    Load the trained generator and discriminator models
    """
    print(f"Loading Generator from: {generator_path}")
    generator = tf.keras.models.load_model(generator_path)

    print(f"Loading Discriminator from: {discriminator_path}")
    discriminator = tf.keras.models.load_model(discriminator_path)

    return generator, discriminator


def generate_samples_for_specific_class(generator, num_samples, latent_dim, class_label):
    """
    Generate samples for a specific class label
    """
    # Generate random noise
    noise = tf.random.normal((num_samples, latent_dim))

    # Create array with the specified class label
    labels = tf.ones((num_samples,), dtype=tf.int32) * class_label

    # Generate fake data
    generated_data = generator.predict([noise, labels])

    return generated_data, labels.numpy()  # Convert to NumPy array here


def evaluate_samples(discriminator, generated_data, sampled_labels, num_classes, class_names=None):
    """
    Evaluate generated samples using the discriminator
    """
    # If class names not provided, use default numbering
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

    # Ensure labels are in the right format
    sampled_labels = np.array(sampled_labels)

    # Convert labels to one-hot encoding
    sampled_labels_onehot = tf.one_hot(sampled_labels, depth=num_classes)

    # Get discriminator predictions
    validity, predicted_classes = discriminator.predict(generated_data)

    # Convert predicted classes to class indices
    predicted_class_indices = np.argmax(predicted_classes, axis=1)

    # Create DataFrame for easier analysis
    results_df = pd.DataFrame({
        'true_label': sampled_labels,  # Already a NumPy array
        'predicted_label': predicted_class_indices,
        'validity_score': validity.flatten(),
        'true_class_name': [class_names[int(i)] for i in sampled_labels],
        'predicted_class_name': [class_names[int(i)] for i in predicted_class_indices]
    })

    return results_df


def analyze_class_distribution(results_df, num_classes, class_names=None):
    """
    Analyze the class distribution of the generated samples
    """
    # If class names not provided, use default numbering
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

    # Count the occurrences of each class in true and predicted labels
    true_counts = Counter(results_df['true_label'])
    pred_counts = Counter(results_df['predicted_label'])

    # Create a DataFrame for plotting
    true_df = pd.DataFrame({
        'Class': [class_names[i] for i in range(num_classes)],
        'Count': [true_counts.get(i, 0) for i in range(num_classes)],
        'Type': 'Input Label'
    })

    pred_df = pd.DataFrame({
        'Class': [class_names[i] for i in range(num_classes)],
        'Count': [pred_counts.get(i, 0) for i in range(num_classes)],
        'Type': 'Predicted Label'
    })

    distribution_df = pd.concat([true_df, pred_df])

    # Calculate accuracy overall and per class
    accuracy = accuracy_score(results_df['true_label'], results_df['predicted_label'])

    # Create confusion matrix
    cm = confusion_matrix(results_df['true_label'], results_df['predicted_label'])

    # Normalize confusion matrix by row (true label)
    with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0

    return distribution_df, accuracy, cm, cm_norm


def plot_class_distribution(distribution_df, save_path=None):
    """
    Plot the class distribution of generated samples
    """
    plt.figure(figsize=(12, 6))

    # Use seaborn for better visualization
    ax = sns.barplot(x='Class', y='Count', hue='Type', data=distribution_df)

    plt.title('Class Distribution of Generated Network Traffic', fontsize=16)
    plt.xlabel('Class Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='')

    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Class distribution plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(cm, cm_norm, num_classes, class_names=None, save_path=None):
    """
    Plot the confusion matrix
    """
    # Create class labels
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

    # Plot raw confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()

    if save_path:
        raw_path = save_path.replace('.png', '_raw.png')
        plt.savefig(raw_path)
        print(f"Raw confusion matrix saved to {raw_path}")

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()

    if save_path:
        norm_path = save_path.replace('.png', '_norm.png')
        plt.savefig(norm_path)
        print(f"Normalized confusion matrix saved to {norm_path}")

    plt.show()


def analyze_network_traffic_features(generated_data, labels, class_names=None, feature_names=None):
    """
    Analyze the features of the generated network traffic data
    """
    if feature_names is None:
        # Create default feature names if not provided
        feature_names = [f'Feature_{i}' for i in range(generated_data.shape[1])]

    if class_names is None:
        # Create default class names if not provided
        num_classes = len(np.unique(labels))
        class_names = [f'Class {i}' for i in range(num_classes)]

    # Create a DataFrame with the generated data
    gen_df = pd.DataFrame(generated_data, columns=feature_names)

    # Add class information to the DataFrame
    gen_df['class'] = labels
    gen_df['class_name'] = [class_names[int(i)] for i in labels]

    # Calculate statistics for each class
    class_stats = []

    for class_idx, class_name in enumerate(class_names):
        class_data = gen_df[gen_df['class'] == class_idx].drop(['class', 'class_name'], axis=1)

        if len(class_data) > 0:
            # Calculate mean, std, min, max for each feature
            class_mean = class_data.mean()
            class_std = class_data.std()
            class_min = class_data.min()
            class_max = class_data.max()

            class_stats.append({
                'class': class_idx,
                'class_name': class_name,
                'means': class_mean,
                'stds': class_std,
                'mins': class_min,
                'maxs': class_max,
                'count': len(class_data)
            })

    return gen_df, class_stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate ACGAN for network traffic generation')
    parser.add_argument('--generator', type=str, required=True, help='Path to the generator model')
    parser.add_argument('--discriminator', type=str, required=True, help='Path to the discriminator model')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate per class')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space')
    parser.add_argument('--class_labels', type=str, default="0,1",
                        help='Comma-separated list of class labels to generate')
    parser.add_argument('--class_names', type=str, default="Benign,Attack", help='Comma-separated list of class names')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output')

    args = parser.parse_args()

    # Parse class labels and names
    class_labels = [int(i) for i in args.class_labels.split(',')]
    class_names = args.class_names.split(',')

    if len(class_labels) != len(class_names):
        print("Warning: Number of class labels and class names don't match!")
        class_names = [f'Class {i}' for i in class_labels]

    num_classes = max(class_labels) + 1

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    generator, discriminator = load_models(args.generator, args.discriminator)

    # Generate and evaluate samples for each specified class
    all_generated_data = []
    all_labels = []

    for class_label in class_labels:
        print(f"\nGenerating {args.num_samples} samples for class {class_names[class_label]} (label {class_label})...")

        # Generate samples for this class
        generated_data, labels = generate_samples_for_specific_class(
            generator, args.num_samples, args.latent_dim, class_label
        )

        # Store data and labels
        all_generated_data.append(generated_data)
        all_labels.append(labels)

        # Analyze and save class-specific samples
        gen_df, class_stats = analyze_network_traffic_features(
            generated_data, labels, class_names, None
        )

        # Save class-specific samples
        samples_df = pd.DataFrame(generated_data)
        samples_df['class'] = labels
        samples_df['class_name'] = class_names[class_label]
        samples_file = os.path.join(args.output_dir, f'generated_samples_class_{class_label}.csv')
        samples_df.to_csv(samples_file, index=False)
        print(f"Samples for class {class_names[class_label]} saved to {samples_file}")

        # Print statistics for this class
        for stats in class_stats:
            print(f"\nStatistics for {stats['class_name']} (first 5 features):")
            print(f"  Sample count: {stats['count']}")
            if len(stats['means']) >= 5:
                print("  Means:", stats['means'][:5].values)
                print("  Stds: ", stats['stds'][:5].values)
                print("  Mins: ", stats['mins'][:5].values)
                print("  Maxs: ", stats['maxs'][:5].values)
            else:
                print("  Means:", stats['means'].values)
                print("  Stds: ", stats['stds'].values)
                print("  Mins: ", stats['mins'].values)
                print("  Maxs: ", stats['maxs'].values)

    # Combine all generated data and labels
    all_generated_data = np.vstack(all_generated_data)
    all_labels = np.concatenate(all_labels)

    # Evaluate all samples together
    print("\nEvaluating all generated samples...")
    results_df = evaluate_samples(discriminator, all_generated_data, all_labels, num_classes, class_names)

    # Save evaluation results
    results_file = os.path.join(args.output_dir, 'evaluation_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"Evaluation results saved to {results_file}")

    # Analyze class distribution
    print("\nAnalyzing class distribution...")
    distribution_df, accuracy, cm, cm_norm = analyze_class_distribution(results_df, num_classes, class_names)

    # Print overall statistics
    print(f"\nOverall accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(results_df['true_label'], results_df['predicted_label'],
                                target_names=class_names))

    # Plot and save results
    print("\nCreating plots...")

    # 1. Plot class distribution
    plot_class_distribution(distribution_df, save_path=os.path.join(args.output_dir, 'class_distribution.png'))

    # 2. Plot confusion matrix
    plot_confusion_matrix(cm, cm_norm, num_classes, class_names,
                          save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

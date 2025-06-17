import numpy as np
import pandas as pd
import math
import glob
import random
from tqdm import tqdm
import os
import time

DATASET_DIRECTORY = '../ciciot2023_archive/'          # If your dataset is within your python project directory, change this to the relative path to your dataset
csv_filepaths = [filename for filename in os.listdir(DATASET_DIRECTORY) if filename.endswith('.csv')]

# Print the list of CSV files found
print("CSV files found:", csv_filepaths)

# Dictionary to aggregate the sample amounts
aggregate_distribution = {}
label_distributions = {}
label_ranks = {}

# Iterate over each CSV file
for filename in csv_filepaths:
    # Construct the full file path
    file_path = os.path.join(DATASET_DIRECTORY, filename)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Check if the 'label' column exists
    if 'label' in df.columns:
        # Calculate the distribution of the 'label' column
        label_distribution = df['label'].value_counts()
        label_distribution_percentage = df['label'].value_counts(normalize=True) * 100

        # Rank the labels based on their distribution
        ranked_labels = label_distribution_percentage.rank(method='min', ascending=False).astype(int)

        # Print the distribution for this file
        print(f"Distribution of 'label' column in {filename}:")
        for label, count in label_distribution.items():
            percentage = label_distribution_percentage[label]
            rank = ranked_labels[label]
            print(f"Label: {label}, Count: {count}, Percentage: {percentage:.2f}%, Rank: {rank}")

            # Track the distributions and ranks for statistics
            if label not in label_distributions:
                label_distributions[label] = []
                label_ranks[label] = []
            label_distributions[label].append(percentage)
            label_ranks[label].append(rank)

        print("\n")

        # Aggregate the distribution
        for label, count in label_distribution.items():
            if label in aggregate_distribution:
                aggregate_distribution[label] += count
            else:
                aggregate_distribution[label] = count
    else:
        print(f"'label' column not found in {filename}\n")

# Calculate the total samples
total_samples = sum(aggregate_distribution.values())

# Print the aggregated distribution
print("Aggregated Distribution of 'label' column across all files:")
for label, count in aggregate_distribution.items():
    percentage = (count / total_samples) * 100
    print(f"Label: {label}, Count: {count}, Percentage: {percentage:.2f}%")

# Calculate and print statistics for each label
print("\nStatistics for 'label' distribution across all files:")
for label, percentages in label_distributions.items():
    min_distribution = min(percentages)
    avg_distribution = sum(percentages) / len(percentages)
    max_distribution = max(percentages)
    min_rank = min(label_ranks[label])
    avg_rank = sum(label_ranks[label]) / len(label_ranks[label])
    max_rank = max(label_ranks[label])
    print(f"Label: {label}, Lowest: {min_distribution:.2f}%, Average: {avg_distribution:.2f}%, Highest: {max_distribution:.2f}%, Rank Range: Min {min_rank}, Avg {avg_rank:.2f}, Max {max_rank}")

#########################################################
#    Imports / Env setup                                #
#########################################################

import os
import random
import time
from datetime import datetime
import argparse

if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']

import flwr as fl

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import expand_dims

# import math
# import glob

# from tqdm import tqdm

# import seaborn as sns

# import pickle
# import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle


################################################################################################################
#                                       Preprocessing & Assigning the dataset                                  #
################################################################################################################

def preprocess_dataset(dataset_used, ciciot_train_data=None, ciciot_test_data=None, all_attacks_train=None,
                       all_attacks_test=None, irrelevant_features_ciciot=None, relevant_features_iotbotnet=None,
                       dataset_processing="Default"):

    # --- 1 Feature Selection ---#
    print(f"\n=== Selecting Features for {dataset_used} ===\n")

    if dataset_used == "CICIOT":
        # Drop the irrelevant features (Feature selection)
        print(irrelevant_features_ciciot)
        ciciot_train_data = ciciot_train_data.drop(columns=irrelevant_features_ciciot)
        ciciot_test_data = ciciot_test_data.drop(columns=irrelevant_features_ciciot)

        # Shuffle data
        ciciot_train_data = shuffle(ciciot_train_data, random_state=47)
        ciciot_test_data = shuffle(ciciot_test_data, random_state=47)

        # initiate model training and test data
        train_data = ciciot_train_data
        test_data = ciciot_test_data

    elif dataset_used == "IOTBOTNET":
        # Select the relevant features in the dataset and labels
        all_attacks_train = all_attacks_train[relevant_features_iotbotnet + ['Label']]
        all_attacks_test = all_attacks_test[relevant_features_iotbotnet + ['Label']]

        # Shuffle data
        all_attacks_train = shuffle(all_attacks_train, random_state=47)
        all_attacks_test = shuffle(all_attacks_test, random_state=47)

        # initiate model training and test data
        train_data = all_attacks_train
        test_data = all_attacks_test

    else:
        raise ValueError("Unsupported dataset type.")

    print("=== Features Selected ===")

    # --- 2 Encoding ---#
    print("\n=== Encoding === \n")

    # conditional variable on label column name for either real dataset used
    label_column = 'Label' if dataset_used == "IOTBOTNET" else 'label'

    # Print instances before encoding and scaling
    unique_labels = train_data[label_column].unique()
    for label in unique_labels:
        print(f"\nFirst instance of {label}:")
        print(train_data[train_data[label_column] == label].iloc[0])

    # Print the amount of instances for each label
    class_counts = train_data[label_column].value_counts()
    print(f"\n{class_counts}\n")

    # Define the desired order for the encoded labels ex:[0,1,2,...]
    desired_order = ['Benign', 'Attack']

    # CREATE THE MAPPING dynamically using the desired order.
    # EX: This ensures that if 'Benign' is present in the data, it gets mapped to 0 and 'Attack' to 1.
    mapping = {label: desired_order.index(label) for label in desired_order if
               label in train_data[label_column].unique()}

    # Apply the mapping to both training and test datasets
    train_data[label_column] = train_data[label_column].map(mapping)
    test_data[label_column] = test_data[label_column].map(mapping)

    # Create an inverse mapping to display the results dynamically
    inverse_mapping = {v: k for k, v in mapping.items()}
    print("\nLabel mappings:", inverse_mapping)

    # Print the amount of instances for each label
    class_counts = train_data[label_column].value_counts()
    print(f"\n{class_counts}\n")

    print("\n=== Labels Encoded ===\n")

    # --- 3 Normalizing ---#
    print("\n=== Normalizing ===\n")

    # DEBUG DISPLAY Before
    print("\nTraining Data Before Normalization:")
    print(train_data.head())
    print(train_data.shape)
    print("\nTest Data Before Normalization:")
    print(test_data.head())
    print(test_data.shape)

    # initiate scaler and colums to scale
    if dataset_processing == "MM[-1,1]":
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))

    relevant_num_cols = train_data.columns.difference([label_column])
    relevantFeatures = relevant_num_cols if dataset_used == "CICIOT" else relevant_features_iotbotnet

    # fit scaler
    scaler.fit(train_data[relevantFeatures])

    # Normalize data
    train_data[relevantFeatures] = scaler.transform(train_data[relevantFeatures])
    test_data[relevantFeatures] = scaler.transform(test_data[relevantFeatures])

    # DEBUG DISPLAY After
    print("\nTraining Data After Normalization:")
    print(train_data.head())
    print(train_data.shape)
    print("\nTest Data After Normalization:")
    print(test_data.head())
    print(test_data.shape)

    print("\n=== Data Normalized ===\n")

    # --- 4 Assigning and Splitting ---#
    print("\n=== Assigning Data to Models ===\n")

    # Train & Validation data
    X_data = train_data.drop(columns=[label_column])
    y_data = train_data[label_column]

    # Split into Train & Validation data
    X_train_data, X_val_data, y_train_data, y_val_data = train_test_split(X_data, y_data, test_size=0.2,
                                                                          random_state=47, stratify=y_data)
    # Test data
    X_test_data = test_data.drop(columns=[label_column])
    y_test_data = test_data[label_column]

    # Print dataset distributions
    print(f"\nTraining label distribution:")
    print(pd.Series(y_train_data).value_counts())
    print(f"\nValidation label distribution:")
    print(pd.Series(y_val_data).value_counts())
    print(f"\nTest label distribution:")
    print(pd.Series(y_test_data).value_counts())

    print("\n=== Data Assigned ===\n")
    return X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data

#
# Categorical
#


def preprocess_dataset_label_encoder(dataset_used, ciciot_train_data=None, ciciot_test_data=None, all_attacks_train=None,
                       all_attacks_test=None, irrelevant_features_ciciot=None, relevant_features_iotbotnet=None):
    # --- 1 Feature Selection ---#
    print(f"\n=== Selecting Features for {dataset_used} ===\n")

    if dataset_used == "CICIOT":
        # Drop the irrelevant features (Feature selection)
        ciciot_train_data = ciciot_train_data.drop(columns=irrelevant_features_ciciot)
        ciciot_test_data = ciciot_test_data.drop(columns=irrelevant_features_ciciot)

        # Shuffle data
        ciciot_train_data = shuffle(ciciot_train_data, random_state=47)
        ciciot_test_data = shuffle(ciciot_test_data, random_state=47)

        # initiate model training and test data
        train_data = ciciot_train_data
        test_data = ciciot_test_data

    elif dataset_used == "IOTBOTNET":
        # Select the relevant features in the dataset and labels
        all_attacks_train = all_attacks_train[relevant_features_iotbotnet + ['Label']]
        all_attacks_test = all_attacks_test[relevant_features_iotbotnet + ['Label']]

        # Shuffle data
        all_attacks_train = shuffle(all_attacks_train, random_state=47)
        all_attacks_test = shuffle(all_attacks_test, random_state=47)

        # initiate model training and test data
        train_data = all_attacks_train
        test_data = all_attacks_test

    else:
        raise ValueError("Unsupported dataset type.")

    print("=== Features Selected ===")

    # --- 2 Encoding ---#
    print("\n=== Encoding === \n")

    # conditional variable on label column name for either real dataset used
    label_column = 'Label' if dataset_used == "IOTBOTNET" else 'label'

    # Print instances before encoding and scaling
    unique_labels = train_data[label_column].unique()
    for label in unique_labels:
        print(f"\nFirst instance of {label}:")
        print(train_data[train_data[label_column] == label].iloc[0])

    # Print the amount of instances for each label
    class_counts = train_data[label_column].value_counts()
    print(class_counts)

    # initiate encoder
    label_encoder = LabelEncoder()

    # fit and encode training data
    train_data[label_column] = label_encoder.fit_transform(train_data[label_column])

    # encode test data
    test_data[label_column] = label_encoder.transform(test_data[label_column])

    # showing the label mappings
    label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print("Label mappings:", label_mapping)

    print("\n=== Labels Encoded ===\n")

    # --- 3 Normalizing ---#
    print("\n=== Normalizing ===\n")

    # DEBUG DISPLAY Before
    print("\nTraining Data Before Normalization:")
    print(train_data.head())
    print(train_data.shape)
    print("\nTest Data Before Normalization:")
    print(test_data.head())
    print(test_data.shape)

    # initiate scaler and colums to scale
    # review this part - doesnt matter since encoded data is 0-1 already
    scaler = MinMaxScaler(feature_range=(0, 1))

    relevant_num_cols = train_data.columns.difference([label_column])
    relevantFeatures = relevant_num_cols if dataset_used == "CICIOT" else relevant_features_iotbotnet

    # fit scaler
    scaler.fit(train_data[relevantFeatures])

    # Normalize data
    train_data[relevantFeatures] = scaler.transform(train_data[relevantFeatures])
    test_data[relevantFeatures] = scaler.transform(test_data[relevantFeatures])

    # DEBUG DISPLAY After
    print("\nTraining Data After Normalization:")
    print(train_data.head())
    print(train_data.shape)
    print("\nTest Data After Normalization:")
    print(test_data.head())
    print(test_data.shape)

    print("\n=== Data Normalized ===\n")

    # --- 4 Assigning and Splitting ---#
    print("\n=== Assigning Data to Models ===\n")

    # Train & Validation data
    X_data = train_data.drop(columns=[label_column])
    y_data = train_data[label_column]

    # Split into Train & Validation data
    X_train_data, X_val_data, y_train_data, y_val_data = train_test_split(X_data, y_data, test_size=0.2,
                                                                          random_state=47, stratify=y_data)
    # Test data
    X_test_data = test_data.drop(columns=[label_column])
    y_test_data = test_data[label_column]

    # Print dataset distributions
    print(f"\nTraining label distribution:")
    print(pd.Series(y_train_data).value_counts())
    print(f"\nValidation label distribution:")
    print(pd.Series(y_val_data).value_counts())
    print(f"\nTest label distribution:")
    print(pd.Series(y_test_data).value_counts())

    print("\n=== Data Assigned ===\n")
    return X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data

#
# AC GAN
#


def preprocess_AC_dataset(dataset_used, ciciot_train_data=None, ciciot_test_data=None, all_attacks_train=None,
                       all_attacks_test=None, irrelevant_features_ciciot=None, relevant_features_iotbotnet=None):
    # --- 1 Feature Selection ---#
    print(f"\n=== Selecting Features for {dataset_used} ===\n")

    if dataset_used == "CICIOT":
        # Drop the irrelevant features (Feature selection)
        ciciot_train_data = ciciot_train_data.drop(columns=irrelevant_features_ciciot)
        ciciot_test_data = ciciot_test_data.drop(columns=irrelevant_features_ciciot)

        # Shuffle data
        ciciot_train_data = shuffle(ciciot_train_data, random_state=47)
        ciciot_test_data = shuffle(ciciot_test_data, random_state=47)

        # initiate model training and test data
        train_data = ciciot_train_data
        test_data = ciciot_test_data

    elif dataset_used == "IOTBOTNET":
        # Select the relevant features in the dataset and labels
        all_attacks_train = all_attacks_train[relevant_features_iotbotnet + ['Label']]
        all_attacks_test = all_attacks_test[relevant_features_iotbotnet + ['Label']]

        # Shuffle data
        all_attacks_train = shuffle(all_attacks_train, random_state=47)
        all_attacks_test = shuffle(all_attacks_test, random_state=47)

        # initiate model training and test data
        train_data = all_attacks_train
        test_data = all_attacks_test

    else:
        raise ValueError("Unsupported dataset type.")

    print("=== Features Selected ===")

    # --- 2 Encoding ---# #! ENSURE ITS BENIGN 0, ATTACK: 1
    print("\n=== Encoding ===\n")

    label_column = 'Label' if dataset_used == "IOTBOTNET" else 'label'

    # Print instances before encoding and scaling
    unique_labels = train_data[label_column].unique()
    for label in unique_labels:
        print(f"\nFirst instance of {label}:")
        print(train_data[train_data[label_column] == label].iloc[0])

    # Print the amount of instances for each label
    class_counts = train_data[label_column].value_counts()
    print(f"\n{class_counts}\n")

    # Define the desired order for the encoded labels ex:[0,1,2,...]
    desired_order = ['Benign', 'Attack']

    # CREATE THE MAPPING dynamically using the desired order.
    # EX: This ensures that if 'Benign' is present in the data, it gets mapped to 0 and 'Attack' to 1.
    mapping = {label: desired_order.index(label) for label in desired_order if
               label in train_data[label_column].unique()}

    # Apply the mapping to both training and test datasets
    train_data[label_column] = train_data[label_column].map(mapping)
    test_data[label_column] = test_data[label_column].map(mapping)

    # Create an inverse mapping to display the results dynamically
    inverse_mapping = {v: k for k, v in mapping.items()}
    print("\nLabel mappings:", inverse_mapping)

    # Print the amount of instances for each label
    class_counts = train_data[label_column].value_counts()
    print(f"\n{class_counts}\n")

    print("\n=== Labels Encoded ===\n")

    # --- 3 Normalizing ---#
    print("\n=== Normalizing ===\n")

    # DEBUG DISPLAY Before
    print("\nTraining Data Before Normalization:")
    print(train_data.head())
    print(train_data.shape)
    print("\nTest Data Before Normalization:")
    print(test_data.head())
    print(test_data.shape)

    # initiate scaler and colums to scale
    scaler = MinMaxScaler(feature_range=(0, 1))

    # make variable for releavant col for ciciot
    relevant_num_cols = train_data.columns.difference([label_column])
    # make a variable based choice on values
    relevantFeatures = relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet

    # fit scaler
    scaler.fit(train_data[relevantFeatures])

    # Normalize data
    train_data[relevantFeatures] = scaler.transform(train_data[relevantFeatures])
    test_data[relevantFeatures] = scaler.transform(test_data[relevantFeatures])

    # DEBUG DISPLAY After
    print("\nTraining Data After Normalization:")
    print(train_data.head())
    print(train_data.shape)
    print("\nMin:\n", train_data.min(), "\nMax:\n", train_data.max())

    print("\nTest Data After Normalization:")
    print(test_data.head())
    print(test_data.shape)
    print("\nMin:\n", test_data.min(), "\nMax:\n", test_data.max())

    print("\n=== Data Normalized ===\n")

    # --- 4 Assigning and Splitting ---#
    print("\n=== Assigning Data to Models ===\n")

    X_data = train_data.drop(columns=[label_column])
    y_data = train_data[label_column]
    X_train_data, X_val_data, y_train_data, y_val_data = train_test_split(X_data, y_data, test_size=0.2,
                                                                          random_state=47, stratify=y_data)
    X_test_data = test_data.drop(columns=[label_column])
    y_test_data = test_data[label_column]

    # Debug: Display label samples before converting to categorical
    print("\nBefore to_categorical conversion:")
    print("\ny_train_data sample:")
    print(y_train_data.head())
    print("\ny_val_data sample:")
    print(y_val_data.head())
    print("\ny_test_data sample:")
    print(y_test_data.head())

    # Convert labels to categorical format
    y_train_categorical = to_categorical(y_train_data, num_classes=2)
    y_val_categorical = to_categorical(y_val_data, num_classes=2)
    y_test_categorical = to_categorical(y_test_data, num_classes=2)

    # Debug: Display label shapes and a sample after conversion
    print("\nAfter to_categorical conversion:")
    print("\ny_train_categorical shape:", y_train_categorical.shape)
    print("Sample of y_train_categorical:")
    print(y_train_categorical[:5])
    print("\ny_val_categorical shape:", y_val_categorical.shape)
    print("Sample of y_val_categorical:")
    print(y_val_categorical[:5])
    print("\ny_test_categorical shape:", y_test_categorical.shape)
    print("Sample of y_test_categorical:")
    print(y_test_categorical[:5])

    print(f"\nTraining label distribution:")
    print(pd.Series(y_train_data).value_counts())
    print(f"\nValidation label distribution:")
    print(pd.Series(y_val_data).value_counts())
    print(f"\nTest label distribution:")
    print(pd.Series(y_test_data).value_counts())

    print("\n=== Data Assigned ===\n")
    return X_train_data, X_val_data, y_train_categorical, y_val_categorical, X_test_data, y_test_categorical

#
# Time Series
#


# Time-series transformation using sliding window
def create_sliding_window(X, y, time_steps):
    X_windowed, y_windowed = [], []
    for i in range(len(X) - time_steps):
        X_windowed.append(X.iloc[i:i + time_steps].values)
        y_windowed.append(y.iloc[i + time_steps])
    return np.array(X_windowed), np.array(y_windowed)


def preprocess_timeseries_dataset(X, y, time_steps=10, test_size=0.2, val_size=0.1, random_state=42):
    """
    Preprocesses a dataset for time-series analysis, including train, validation, and test splits.

    Parameters:
    - X (pd.DataFrame): Features dataframe.
    - y (pd.Series or pd.DataFrame): Target dataframe or series.
    - time_steps (int): Number of time steps for sliding window transformation.
    - test_size (float): Proportion of the dataset to include in the test split.
    - val_size (float): Proportion of the training data to include in the validation split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_train, X_val, X_test (np.array): Preprocessed train, validation, and test features.
    - y_train, y_val, y_test (np.array): Preprocessed train, validation, and test labels.
    """
    # Check for missing values and impute (mean for numerical columns)
    if X.isnull().sum().sum() > 0:
        X.fillna(X.mean(), inplace=True)
    if y.isnull().sum() > 0:
        y.fillna(y.mode()[0], inplace=True)  # Replace with mode for categorical

    # Encode categorical target labels (if needed)
    if y.dtypes == 'object' or isinstance(y.iloc[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Normalize features
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

    # Time-series transformation using sliding window
    X_ts, y_ts = create_sliding_window(X_normalized, pd.Series(y), time_steps)

    # Split into training + validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_ts, y_ts, test_size=test_size, random_state=random_state
    )

    # Split training + validation into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

#
# Live Data
#


def preprocess_live_dataset(live_data=None, irrelevant_features=None):

    # --- 1 Feature Selection ---#
    print(f"\n=== Selecting Features for Live Data ===\n")

    # Drop the irrelevant features (Feature selection)
    print(irrelevant_features)
    live_data = live_data.drop(columns=irrelevant_features)
    X_live_data = live_data

    print("=== Features Selected ===")

    # --- 2 Normalizing (Skip encoding because data there are no labels) ---#
    print("\n=== Normalizing ===\n")

    # DEBUG DISPLAY Before
    print("\nTest Data Before Normalization:")
    print(X_live_data.head())
    print(X_live_data.shape)

    scaler = MinMaxScaler(feature_range=(0, 1))

    # fit scaler
    scaler.fit(X_live_data)

    # Normalize data
    # Transform data and preserve DataFrame structure
    # Get the normalized data as a numpy array
    normalized_data = scaler.transform(X_live_data)

    # DEBUG DISPLAY After
    print("\nTest Data After Normalization:")
    # Convert back to DataFrame with original column names
    X_live_data = pd.DataFrame(normalized_data, columns=X_live_data.columns)
    print(X_live_data.head())
    print(X_live_data.shape)

    print("\n=== Data Normalized ===\n")

    # --- 4 Assigning ---#
    print("\n=== Assigning Data to Model ===\n")
    # Print dataset distributions
    print(f"\n Live Validation / Test Data Information:")
    print(f"Shape: {X_live_data.shape}")
    print(f"Column types: {X_live_data.dtypes}")

    print("\n=== Data Assigned ===\n")
    return X_live_data


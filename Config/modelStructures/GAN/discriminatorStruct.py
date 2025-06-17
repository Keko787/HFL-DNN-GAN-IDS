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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Add, BatchNormalization, Dropout, LSTM, Conv1D, MaxPooling1D, GRU, LeakyReLU, Activation, Input, Flatten, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import SpectralNormalization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Function for creating the discriminator model
def create_discriminator(input_dim):
    # Discriminator is designed to classify three classes:
    # - Normal (Benign) traffic
    # - Intrusive (Malicious) traffic
    # - Generated (Fake) traffic from the generator
    discriminator = tf.keras.Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation='softmax')  # 3 classes: Normal, Intrusive, Fake
    ])  # see if sigmoid is the issue in transfer
    return discriminator


def create_discriminator_binary(input_dim):
    # Discriminator to classify two classes:
    # - Real (Benign & Malicious) traffic
    # - Generated (Fake) traffic from the generator
    discriminator = tf.keras.Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # 2 classes: Real, Fake
    ])
    return discriminator


def create_discriminator_binary_optimized(input_dim):
    """
    Optimized Discriminator Model
    - Uses LeakyReLU for better gradient flow.
    - Applies Spectral Normalization for stable training.
    - Includes BatchNorm and Dropout for regularization.
    """
    discriminator = Sequential([
        Dense(512, input_shape=(input_dim,)),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation='sigmoid')  # Output: Probability of being real
    ])
    return discriminator

# ACGAN

def build_AC_discriminator_V0(input_dim, num_classes):
    data_input = Input(shape=(input_dim,))

    x = Dense(512)(data_input)
    x = LeakyReLU(0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)

    # Output layers
    validity = Dense(1, activation='sigmoid', name="validity")(x)  # Real/Fake classification
    label_output = Dense(num_classes, activation='softmax', name="class")(x)  # Class prediction

    return Model(data_input, [validity, label_output], name="Discriminator")


def build_AC_discriminator_ver_2(input_dim, num_classes):
    data_input = Input(shape=(input_dim,))

    x = Dense(512, kernel_regularizer=l2(0.001))(data_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    shared = Dense(128, kernel_regularizer=l2(0.001))(x)
    shared = BatchNormalization()(shared)
    shared = LeakyReLU(0.2)(shared)

    # Split into two branches:
    validity = Dense(1, activation='sigmoid', name="validity")(shared)

    # You could add additional layers for class prediction
    class_branch = Dense(64, kernel_regularizer=l2(0.001))(shared)
    class_branch = BatchNormalization()(class_branch)
    class_branch = LeakyReLU(0.2)(class_branch)

    class_branch1 = Dense(64, kernel_regularizer=l2(0.001))(shared)
    class_branch1 = BatchNormalization()(class_branch1)
    class_branch1 = LeakyReLU(0.2)(class_branch1)

    class_branch = Add()([class_branch, class_branch1])  # Adding residual connection

    class_branch = Dense(32, kernel_regularizer=l2(0.001))(class_branch)
    class_branch = BatchNormalization()(class_branch)
    class_branch = LeakyReLU(0.2)(class_branch)

    class_branch = Dense(16, kernel_regularizer=l2(0.001))(class_branch)
    class_branch = BatchNormalization()(class_branch)
    class_branch = LeakyReLU(0.2)(class_branch)

    label_output = Dense(num_classes, activation='softmax', name="class")(class_branch)

    return Model(data_input, [validity, label_output], name="Discriminator")


def build_AC_discriminator(input_dim, num_classes):
    data_input = Input(shape=(input_dim,))

    # Increase regularization and dropout in initial layers
    x = Dense(512, kernel_regularizer=l2(0.002))(data_input)  # Increased regularization
    x = BatchNormalization(momentum=0.8)(x)  # Adjust batch norm momentum
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)  # Increased dropout

    x = Dense(256, kernel_regularizer=l2(0.002))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)

    shared = Dense(128, kernel_regularizer=l2(0.002))(x)
    shared = BatchNormalization()(shared)
    shared = LeakyReLU(0.2)(shared)

    # -- validity branch (instead of just a single layer from ver 2)
    validity_branch = Dense(64, kernel_regularizer=l2(0.002))(shared)
    validity_branch = BatchNormalization()(validity_branch)
    validity_branch = LeakyReLU(0.2)(validity_branch)
    validity_branch = Dropout(0.4)(validity_branch)

    # residual connection to validity branch
    validity_branch1 = Dense(64, kernel_regularizer=l2(0.002))(shared)
    validity_branch1 = BatchNormalization()(validity_branch1)
    validity_branch1 = LeakyReLU(0.2)(validity_branch1)
    validity_branch = Add()([validity_branch, validity_branch1])

    # layers to validity branch
    validity_branch = Dense(32, kernel_regularizer=l2(0.002))(validity_branch)
    validity_branch = BatchNormalization()(validity_branch)
    validity_branch = LeakyReLU(0.2)(validity_branch)

    # Final sigmoid activation for validity
    validity = Dense(1, activation='sigmoid', name="validity")(validity_branch)

    # -- Class branch (remains mostly the same based on ver 2)
    class_branch = Dense(64, kernel_regularizer=l2(0.001))(shared)
    class_branch = BatchNormalization()(class_branch)
    class_branch = LeakyReLU(0.2)(class_branch)

    # residual connection to validity branch
    class_branch1 = Dense(64, kernel_regularizer=l2(0.001))(shared)
    class_branch1 = BatchNormalization()(class_branch1)
    class_branch1 = LeakyReLU(0.2)(class_branch1)

    class_branch = Add()([class_branch, class_branch1])

    class_branch = Dense(32, kernel_regularizer=l2(0.001))(class_branch)
    class_branch = BatchNormalization()(class_branch)
    class_branch = LeakyReLU(0.2)(class_branch)

    class_branch = Dense(16, kernel_regularizer=l2(0.001))(class_branch)
    class_branch = BatchNormalization()(class_branch)
    class_branch = LeakyReLU(0.2)(class_branch)

    label_output = Dense(num_classes, activation='softmax', name="class")(class_branch)

    return Model(data_input, [validity, label_output], name="Discriminator")


def build_AC_discriminator_ver_3b(input_dim, num_classes):
    from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout, Add, Concatenate
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2

    data_input = Input(shape=(input_dim,))

    # Increase regularization and dropout in initial layers based on ver 2
    x = Dense(512, kernel_regularizer=l2(0.002))(data_input)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)

    x = Dense(256, kernel_regularizer=l2(0.002))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)

    shared = Dense(128, kernel_regularizer=l2(0.002))(x)
    shared = BatchNormalization()(shared)
    shared = LeakyReLU(0.2)(shared)

    # -- validity branch
    validity_branch = Dense(64, kernel_regularizer=l2(0.002))(shared)
    validity_branch = BatchNormalization()(validity_branch)
    validity_branch = LeakyReLU(0.2)(validity_branch)
    validity_branch = Dropout(0.4)(validity_branch)

    # residual connection to validity branch
    # validity_branch1 = Dense(64, kernel_regularizer=l2(0.002))(shared)
    # validity_branch1 = BatchNormalization()(validity_branch1)
    # validity_branch1 = LeakyReLU(0.2)(validity_branch1)
    # validity_branch = Add()([validity_branch, validity_branch1])

    validity_branch = Dense(32, kernel_regularizer=l2(0.002))(validity_branch)
    validity_branch = BatchNormalization()(validity_branch)
    validity_branch = LeakyReLU(0.2)(validity_branch)

    # -- Class branch
    class_branch = Dense(64, kernel_regularizer=l2(0.001))(shared)
    class_branch = BatchNormalization()(class_branch)
    class_branch = LeakyReLU(0.2)(class_branch)

    # residual connection to class branch
    class_branch1 = Dense(64, kernel_regularizer=l2(0.001))(shared)
    class_branch1 = BatchNormalization()(class_branch1)
    class_branch1 = LeakyReLU(0.2)(class_branch1)

    class_branch = Add()([class_branch, class_branch1])

    class_branch = Dense(32, kernel_regularizer=l2(0.001))(class_branch)
    class_branch = BatchNormalization()(class_branch)
    class_branch = LeakyReLU(0.2)(class_branch)

    class_branch = Dense(16, kernel_regularizer=l2(0.001))(class_branch)
    class_branch = BatchNormalization()(class_branch)
    class_branch_features = LeakyReLU(0.2)(class_branch)

    # Regular class output
    label_output = Dense(num_classes, activation='softmax', name="class")(class_branch_features)

    # Optional layer (3b): Consider combining validity and class information for final validity output
    # Combine validity and class information for final validity output
    combined = Concatenate()([validity_branch, class_branch_features])
    combined = Dense(32, kernel_regularizer=l2(0.002))(combined)
    combined = BatchNormalization()(combined)
    combined = LeakyReLU(0.2)(combined)
    combined = Dropout(0.3)(combined)
    validity = Dense(1, activation='sigmoid', name="validity")(combined)

    return Model(data_input, [validity, label_output], name="Discriminator")


def build_AC_discriminator_ver_4(input_dim, num_classes):
    """
    AC-GAN version 4 discriminator with better validity detection capabilities.
    Key improvements:
    1. Simplified architecture with better gradient flow
    2. More capacity in shared layers
    3. Balanced branches for validity and class detection
    4. Better regularization strategy
    5. Modified activation functions
    """
    # Input layer
    data_input = Input(shape=(input_dim,))

    # Shared Feature Extraction - Increased capacity
    # First shared block
    x = Dense(768, kernel_regularizer=l2(0.0005))(data_input)  # Increased width
    x = BatchNormalization(momentum=0.9)(x)  # Higher momentum for stability
    x = LeakyReLU(0.1)(x)  # Lower alpha for better gradients
    x = Dropout(0.2)(x)  # Start with lighter dropout

    # Second shared block with residual connection
    res1 = Dense(384, kernel_regularizer=l2(0.0005))(x)
    res1 = BatchNormalization(momentum=0.9)(res1)
    res1 = LeakyReLU(0.1)(res1)

    x = Dense(384, kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.3)(x)

    x = Dense(384, kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0.1)(x)
    x = Add()([x, res1])  # Residual connection improves gradient flow

    # Final shared representation
    shared = Dense(192, kernel_regularizer=l2(0.0005))(x)
    shared = BatchNormalization()(shared)
    shared = LeakyReLU(0.1)(shared)
    shared = Dropout(0.3)(shared)

    # -- Validity Branch (Real vs. Fake) --
    # Simplify the validity branch for better focus
    validity_branch = Dense(128, kernel_regularizer=l2(0.001))(shared)
    validity_branch = BatchNormalization()(validity_branch)
    validity_branch = LeakyReLU(0.1)(validity_branch)
    validity_branch = Dropout(0.3)(validity_branch)

    validity_branch = Dense(64, kernel_regularizer=l2(0.001))(validity_branch)
    validity_branch = BatchNormalization()(validity_branch)
    validity_branch = LeakyReLU(0.1)(validity_branch)

    # Direct connection to output for validity
    validity = Dense(1, activation='sigmoid', name="validity",
                     kernel_regularizer=l2(0.001))(validity_branch)

    # -- Class Branch (For classifying actual classes) --
    class_branch = Dense(128, kernel_regularizer=l2(0.001))(shared)
    class_branch = BatchNormalization()(class_branch)
    class_branch = LeakyReLU(0.1)(class_branch)
    class_branch = Dropout(0.3)(class_branch)

    # Add a residual connection in class branch too
    class_res = Dense(64, kernel_regularizer=l2(0.001))(class_branch)
    class_res = BatchNormalization()(class_res)
    class_res = LeakyReLU(0.1)(class_res)

    class_branch = Dense(64, kernel_regularizer=l2(0.001))(class_branch)
    class_branch = BatchNormalization()(class_branch)
    class_branch = LeakyReLU(0.1)(class_branch)
    class_branch = Add()([class_branch, class_res])  # Residual connection

    class_branch = Dense(32, kernel_regularizer=l2(0.001))(class_branch)
    class_branch = BatchNormalization()(class_branch)
    class_branch = LeakyReLU(0.1)(class_branch)

    # Softmax output for class prediction
    label_output = Dense(num_classes, activation='softmax', name="class",
                         kernel_regularizer=l2(0.001))(class_branch)

    return Model(data_input, [validity, label_output], name="Discriminator")


def build_AC_discriminator_v5(input_dim, num_classes):
    data_input = Input(shape=(input_dim,))

    # Simplify the backbone with fewer layers and less regularization
    x = Dense(256, kernel_regularizer=l2(0.0005))(data_input)  # Reduced regularization
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)  # Reduced dropout

    x = Dense(128, kernel_regularizer=l2(0.0005))(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)

    # Shared feature extraction without batch normalization
    shared = Dense(64)(x)
    shared = LeakyReLU(0.2)(shared)

    # Simplified validity branch - remove residual connections
    validity_branch = Dense(32)(shared)
    validity_branch = LeakyReLU(0.2)(validity_branch)

    # Option 1: For better metrics tracking, use linear output with BCE from_logits=True
    validity = Dense(1, name="validity")(validity_branch)
    # Option 2: Or keep sigmoid if it works better in practice
    # validity = Dense(1, activation='sigmoid', name="validity")(validity_branch)

    # Simplified class branch - remove residual connections
    class_branch = Dense(32)(shared)
    class_branch = LeakyReLU(0.2)(class_branch)
    label_output = Dense(num_classes, activation='softmax', name="class")(class_branch)

    return Model(data_input, [validity, label_output], name="Discriminator")


# WGAN
def create_W_discriminator_binary_optimized(input_dim):
    """
    Optimized Discriminator Model for WGAN-GP
    - Uses Spectral Normalization for stable training.
    - LeakyReLU for improved gradient flow.
    - No activation in the output layer (Wasserstein loss).
    """
    model = tf.keras.Sequential([
        SpectralNormalization(Dense(512, input_shape=(input_dim,))),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),

        SpectralNormalization(Dense(256)),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),

        SpectralNormalization(Dense(128)),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),

        SpectralNormalization(Dense(1))  # No activation (raw Wasserstein score)
    ])
    return model


# Real Time

def create_discriminator_binary_optimized_spectral(input_dim):
    # Discriminator to classify two classes: Real (Benign & Malicious) traffic vs. Fake traffic
    discriminator = Sequential([
        # First Layer
        SpectralNormalization(Dense(512, use_bias=False, input_shape=(input_dim,))),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        # Second Layer
        SpectralNormalization(Dense(256, use_bias=False)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        # Third Layer
        SpectralNormalization(Dense(128, use_bias=False)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        # Final Layer (Output)
        Dense(1, activation='sigmoid')  # 2 classes: Real, Fake
    ])

    return discriminator

# too strong, wtf
def create_discriminator_binary_optimized_2(input_dim):
    discriminator = Sequential([
        # Input Layer
        Dense(512, use_bias=False, input_shape=(input_dim,)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        # Hidden Layers
        Dense(256, use_bias=False),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Dense(128, use_bias=False),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        # Final Layer (Binary Classification)
        Dense(1, activation='sigmoid')  # 2 classes: Real, Fake
    ])
    return discriminator


def create_discriminator_realtime_GRU(input_dim):
    # Discriminator to classify three classes: Normal, Intrusive, Fake
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),  # For real-time step input

        # GRU to capture temporal dependencies
        GRU(50, return_sequences=False, activation='tanh'),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Output layer for three classes
        Dense(3, activation='softmax')  # 3 classes: Normal, Intrusive, Fake
    ])
    return discriminator

def create_discriminator_realtime_GRU_binary(input_dim):
    # Discriminator to classify two classes: Real, Fake
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),  # For real-time step input

        # GRU to capture temporal dependencies
        GRU(50, return_sequences=False, activation='tanh'),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Output layer for two classes
        Dense(1, activation='sigmoid')  # 2 classes: Real, Fake
    ])
    return discriminator

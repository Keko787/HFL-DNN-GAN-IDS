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
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, ELU, Reshape, Conv1DTranspose, LSTM, GRU, Embedding, Concatenate, Flatten, Input, MaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Baseline

# Function for creating the generator model
def create_generator(input_dim, noise_dim):
    generator = tf.keras.Sequential([
        # Input layer 1
        Dense(128, activation='relu', input_shape=(noise_dim,)),
        BatchNormalization(),
        # layer 2
        Dense(256, activation='relu'),
        BatchNormalization(),
        # layer 3
        Dense(512, activation='relu'),
        BatchNormalization(),
        # output layer
        Dense(input_dim, activation='sigmoid')  # Generate traffic features
    ])
    return generator

def create_generator_optimized(input_dim, noise_dim):
    generator = Sequential([
        # Input layer (Noise to feature transformation)
        Dense(128, use_bias=False, input_shape=(noise_dim,)),
        BatchNormalization(),
        ELU(alpha=1.0),

        # Increasing non-linearity
        Dense(256, use_bias=False),
        BatchNormalization(),
        ELU(alpha=1.0),

        Dense(512, use_bias=False),
        BatchNormalization(),
        ELU(alpha=1.0),

        # Reshape for structured output
        Dense(input_dim, activation='sigmoid')  # Generate traffic features
    ])

    return generator

def build_AC_generator(latent_dim, num_classes, input_dim):
    noise_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(1,), dtype='int32')

    # Embedding layer to process labels
    label_embedding = Embedding(num_classes, latent_dim)(label_input)
    label_embedding = Flatten()(label_embedding)

    # Concatenate noise and label embedding
    merged_input = Concatenate()([noise_input, label_embedding])

    # Fully connected layers
    x = Dense(128)(merged_input)
    x = LeakyReLU(0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    output = Dense(input_dim, activation='tanh')(x)  # Output size should match dataset features

    return Model([noise_input, label_input], output, name="ACGenerator")


def build_improved_AC_generator(latent_dim, num_classes, input_dim):
    noise_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(1,), dtype='int32')

    # Using a larger embedding dimension for better class representation
    label_embedding = Embedding(num_classes, 256)(label_input)
    label_embedding = Flatten()(label_embedding)

    # Concatenate noise and label embedding
    merged_input = Concatenate()([noise_input, label_embedding])

    # Progressive growth of feature maps with BatchNormalization
    x = Dense(128)(merged_input)
    x = BatchNormalization()(x)  # Add normalization
    x = LeakyReLU(0.2)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # Final output - keep tanh activation
    output = Dense(input_dim, activation='tanh')(x)

    return Model([noise_input, label_input], output, name="ACGenerator")

# WGAN

def create_W_generator(input_dim, noise_dim):
    """
    Optimized Generator Model for WGAN-GP
    - Uses LeakyReLU to improve training stability.
    - BatchNormalization to prevent mode collapse.
    - Sigmoid activation for network traffic feature generation.
    """
    model = tf.keras.Sequential([
        Dense(256, input_shape=(noise_dim,)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Dense(512),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Dense(1024),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Dense(input_dim, activation='sigmoid')  # Output layer for network traffic data
    ])
    return model

# Real Time

def create_generator_optimized_conv(input_dim, noise_dim):
    generator = Sequential([
        # Input layer (Noise to feature transformation)
        Dense(128, use_bias=False, input_shape=(noise_dim,)),
        BatchNormalization(),
        ELU(alpha=1.0),

        # Increasing non-linearity
        Dense(256, use_bias=False),
        BatchNormalization(),
        ELU(alpha=1.0),

        Dense(512, use_bias=False),
        BatchNormalization(),
        ELU(alpha=1.0),

        # Reshape for Conv1DTranspose processing (for time-series data)
        Reshape((1, 512)),  # Reshape for 1D Conv layers

        # Upsampling layers (Using transposed convolution)
        Conv1DTranspose(256, kernel_size=3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Conv1DTranspose(input_dim, kernel_size=3, strides=2, padding='same', activation='sigmoid')  # Final output
    ])

    return generator


def create_optimized_generator_low_conv(input_dim, noise_dim):
    generator = Sequential([
        # Step 1: Transform random noise into meaningful feature representation
        Dense(256, use_bias=False, input_shape=(noise_dim,)),
        BatchNormalization(),
        ELU(alpha=1.0),

        Dense(512, use_bias=False),
        BatchNormalization(),
        ELU(alpha=1.0),

        # Step 2: Convert dense output into a spatial representation
        Reshape((4, 128)),  # (timesteps, feature maps) → Enables structured upsampling

        # Step 3: Efficient upsampling with transposed convolutions
        Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Conv1DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        # Step 4: Final layer (Generate synthetic traffic data)
        Conv1DTranspose(input_dim, kernel_size=3, strides=2, padding='same', activation='sigmoid')
    ])

    return generator


def create_generator_realtime_LSTM(input_dim, noise_dim):
    generator = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(noise_dim,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),

        # Adding an LSTM layer to better generate realistic time-series data
        tf.keras.layers.Reshape((1, 512)),  # Reshape to use LSTM after fully connected layers
        LSTM(50, return_sequences=False, activation='tanh'),

        Dense(input_dim, activation='sigmoid')  # Generate traffic features
    ])
    return generator

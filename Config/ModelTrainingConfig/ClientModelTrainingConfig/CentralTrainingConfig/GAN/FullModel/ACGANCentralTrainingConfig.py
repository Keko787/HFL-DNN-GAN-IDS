#########################################################
#    Imports / Env setup                                #
#########################################################

import os
import random
import time
import logging
from datetime import datetime
import argparse


if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']

import flwr as fl

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, classification_report
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from collections import Counter
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
#                                               FL-GAN TRAINING Setup                                         #
################################################################################################################

class CentralACGan:
    def __init__(self, discriminator, generator, nids, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 noise_dim, latent_dim, num_classes, input_dim, epochs, steps_per_epoch, learning_rate,
                 log_file="training.log"):

        # ═══════════════════════════════════════════════════════════════════════
        # MODEL INITIALIZATION
        # ═══════════════════════════════════════════════════════════════════════
        self.generator = generator
        self.discriminator = discriminator
        self.nids = nids

        # ═══════════════════════════════════════════════════════════════════════
        # MODEL I/O SPECIFICATIONS
        # ═══════════════════════════════════════════════════════════════════════
        self.batch_size = BATCH_SIZE
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_dim = input_dim

        # ═══════════════════════════════════════════════════════════════════════
        # TRAINING CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════════
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        # ═══════════════════════════════════════════════════════════════════════
        # DATA ASSIGNMENT
        # ═══════════════════════════════════════════════════════════════════════
        # ─── Features ───
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val

        # ─── Categorical Labels ───
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        # ═══════════════════════════════════════════════════════════════════════
        # LOGGING SETUP
        # ═══════════════════════════════════════════════════════════════════════
        self.setup_logger(log_file)

        # ═══════════════════════════════════════════════════════════════════════
        # OPTIMIZER CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════════
        # ─── Learning Rate Schedules ───
        # Slower learning for generator to prevent overpowering discriminator
        lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.00012, decay_steps=10000, decay_rate=0.98, staircase=False)

        # Faster learning for discriminator to maintain strength
        lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.00007, decay_steps=10000, decay_rate=0.98, staircase=False)

        # ─── Optimizer Compilation with Gradient Clipping ───
        self.gen_optimizer = Adam(learning_rate=lr_schedule_gen, beta_1=0.5, beta_2=0.999, clipnorm=1.0)
        self.disc_optimizer = Adam(learning_rate=lr_schedule_disc, beta_1=0.5, beta_2=0.999, clipnorm=1.0)

        # ═══════════════════════════════════════════════════════════════════════
        # MODEL COMPILATION
        # ═══════════════════════════════════════════════════════════════════════
        print("Discriminator Output:", self.discriminator.output_names)

        # ─── CRITICAL FIX: Get actual discriminator output names ───
        print("Discriminator Outputs:", self.discriminator.output_names)
        
        # ═══════════════════════════════════════════════════════════════════════
        # LOSS FUNCTIONS SETUP
        # ═══════════════════════════════════════════════════════════════════════
        # Define loss functions for custom training
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        # ═══════════════════════════════════════════════════════════════════════
        # METRICS SETUP
        # ═══════════════════════════════════════════════════════════════════════
        # Create metrics for tracking (optional, for compatibility with existing code)
        self.d_binary_accuracy = tf.keras.metrics.BinaryAccuracy(name='d_binary_accuracy')
        self.d_categorical_accuracy = tf.keras.metrics.CategoricalAccuracy(name='d_categorical_accuracy')
        self.g_binary_accuracy = tf.keras.metrics.BinaryAccuracy(name='g_binary_accuracy')
        self.g_categorical_accuracy = tf.keras.metrics.CategoricalAccuracy(name='g_categorical_accuracy')

        # NOTE: No model compilation needed - using custom training loops

# ═══════════════════════════════════════════════════════════════════════
# MODEL ACCESS METHODS
# ═══════════════════════════════════════════════════════════════════════
    def setACGAN(self):
        # For backward compatibility - ACGAN model no longer exists
        return None

#########################################################################
#                   CUSTOM TRAINING STEP METHODS                        #
#########################################################################
    @tf.function
    def train_discriminator_step(self, real_data, real_labels, real_validity_labels):
        """
        Custom training step for discriminator on real data.

        Args:
            real_data: Real input features
            real_labels: One-hot encoded class labels
            real_validity_labels: Validity labels (1 for real)

        Returns:
            Tuple of (total_loss, validity_loss, class_loss, validity_acc, class_acc)
        """
        # Convert inputs to float32 for type consistency
        real_data = tf.cast(real_data, tf.float32)
        real_labels = tf.cast(real_labels, tf.float32)
        real_validity_labels = tf.cast(real_validity_labels, tf.float32)

        with tf.GradientTape() as tape:
            # Forward pass with training=True
            validity_pred, class_pred = self.discriminator(real_data, training=True)

            # Calculate losses
            validity_loss = self.binary_crossentropy(real_validity_labels, validity_pred)
            class_loss = self.categorical_crossentropy(real_labels, class_pred)

            # CRITICAL FIX: Reduce validity loss weight for real data
            # Real data has huge validity loss that dominates gradients
            # This balances gradients between real (high loss) and fake (low loss)
            # Tuning history: 0.5x → 0.2x → 0.1x (too strong) → 0.15x (still too strong)
            # Using 0.3x - closer to original but still reduced
            total_loss = (0.3 * validity_loss) + class_loss

        # Calculate gradients and update weights
        gradients = tape.gradient(total_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        # Calculate accuracies
        validity_acc = self.d_binary_accuracy(real_validity_labels, validity_pred)
        class_acc = self.d_categorical_accuracy(real_labels, class_pred)

        return total_loss, validity_loss, class_loss, validity_acc, class_acc

    @tf.function
    def train_discriminator_on_fake_step(self, fake_data, fake_labels, fake_validity_labels):
        """
        Custom training step for discriminator on fake/generated data.

        Args:
            fake_data: Generated input features
            fake_labels: One-hot encoded class labels for generated data
            fake_validity_labels: Validity labels (0 for fake)

        Returns:
            Tuple of (total_loss, validity_loss, class_loss, validity_acc, class_acc)
        """
        # Convert inputs to float32 for type consistency
        fake_data = tf.cast(fake_data, tf.float32)
        fake_labels = tf.cast(fake_labels, tf.float32)
        fake_validity_labels = tf.cast(fake_validity_labels, tf.float32)

        with tf.GradientTape() as tape:
            # Forward pass with training=True
            validity_pred, class_pred = self.discriminator(fake_data, training=True)

            # Calculate losses
            validity_loss = self.binary_crossentropy(fake_validity_labels, validity_pred)
            class_loss = self.categorical_crossentropy(fake_labels, class_pred)

            # CRITICAL FIX: Increase validity loss weight for fake data
            # This balances with the reduced weight on real data validity loss
            # Helps discriminator learn to distinguish real from fake more effectively
            # Tuning history: 2.0x → 5.0x → 10.0x (too strong) → 7.0x (still too strong)
            # Back to 5.0x to balance with real data's 0.3x weight (ratio ~1:4)
            total_loss = (5.0 * validity_loss) + class_loss

        # Calculate gradients and update weights
        gradients = tape.gradient(total_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        # Calculate accuracies
        validity_acc = self.d_binary_accuracy(fake_validity_labels, validity_pred)
        class_acc = self.d_categorical_accuracy(fake_labels, class_pred)

        return total_loss, validity_loss, class_loss, validity_acc, class_acc

    @tf.function
    def train_generator_step(self, noise, labels_int, labels_onehot, validity_labels):
        """
        Custom training step for generator.
        CRITICAL: Discriminator is called with training=False to prevent BatchNorm corruption.

        Args:
            noise: Random noise input
            labels_int: Integer class labels (for generator input)
            labels_onehot: One-hot encoded class labels (for loss calculation)
            validity_labels: Target validity labels (1 - generator wants to fool discriminator)

        Returns:
            Tuple of (total_loss, validity_loss, class_loss, validity_acc, class_acc)
        """
        # Convert inputs to proper types
        noise = tf.cast(noise, tf.float32)
        labels_onehot = tf.cast(labels_onehot, tf.float32)
        validity_labels = tf.cast(validity_labels, tf.float32)

        with tf.GradientTape() as tape:
            # Generate fake data with training=True
            # Generator expects INTEGER labels
            generated_data = self.generator([noise, labels_int], training=True)

            # CRITICAL: Call discriminator with training=False
            # This prevents BatchNorm layers from updating their statistics
            validity_pred, class_pred = self.discriminator(generated_data, training=False)

            # Calculate losses (generator wants discriminator to predict "real")
            validity_loss = self.binary_crossentropy(validity_labels, validity_pred)
            # Use one-hot labels for loss calculation
            class_loss = self.categorical_crossentropy(labels_onehot, class_pred)
            total_loss = validity_loss + 2.0 * class_loss  # Give class loss MORE weight

        # Calculate gradients ONLY for generator variables
        gradients = tape.gradient(total_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        # Calculate accuracies
        validity_acc = self.g_binary_accuracy(validity_labels, validity_pred)
        class_acc = self.g_categorical_accuracy(labels_onehot, class_pred)

        return total_loss, validity_loss, class_loss, validity_acc, class_acc

    @tf.function
    def evaluate_discriminator(self, data, labels, validity_labels):
        """
        Evaluate discriminator without updating weights.

        Args:
            data: Input features
            labels: One-hot encoded class labels
            validity_labels: Validity labels

        Returns:
            Tuple of (total_loss, validity_loss, class_loss, validity_acc, class_acc)
        """
        # Convert inputs to float32 to ensure type consistency
        data = tf.cast(data, tf.float32)
        labels = tf.cast(labels, tf.float32)
        validity_labels = tf.cast(validity_labels, tf.float32)

        # Forward pass with training=False for evaluation
        validity_pred, class_pred = self.discriminator(data, training=False)

        # Calculate losses
        validity_loss = self.binary_crossentropy(validity_labels, validity_pred)
        class_loss = self.categorical_crossentropy(labels, class_pred)
        total_loss = validity_loss + class_loss

        # Calculate accuracies
        validity_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.round(validity_pred), validity_labels), tf.float32)
        )
        class_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(class_pred, axis=1), tf.argmax(labels, axis=1)), tf.float32)
        )

        return total_loss, validity_loss, class_loss, validity_acc, class_acc

    @tf.function
    def evaluate_generator(self, noise, labels_int, labels_onehot, validity_labels):
        """
        Evaluate generator without updating weights.

        Args:
            noise: Random noise input
            labels_int: Integer class labels (for generator input)
            labels_onehot: One-hot encoded class labels (for loss calculation)
            validity_labels: Target validity labels

        Returns:
            Tuple of (total_loss, validity_loss, class_loss, validity_acc, class_acc)
        """
        # Convert inputs to proper types
        noise = tf.cast(noise, tf.float32)
        labels_onehot = tf.cast(labels_onehot, tf.float32)
        validity_labels = tf.cast(validity_labels, tf.float32)

        # Generate data - generator expects INTEGER labels
        generated_data = self.generator([noise, labels_int], training=False)

        # Get discriminator predictions
        validity_pred, class_pred = self.discriminator(generated_data, training=False)

        # Calculate losses - use one-hot labels
        validity_loss = self.binary_crossentropy(validity_labels, validity_pred)
        class_loss = self.categorical_crossentropy(labels_onehot, class_pred)
        total_loss = validity_loss + class_loss

        # Calculate accuracies
        validity_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.round(validity_pred), validity_labels), tf.float32)
        )
        class_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(class_pred, axis=1), tf.argmax(labels_onehot, axis=1)), tf.float32)
        )

        return total_loss, validity_loss, class_loss, validity_acc, class_acc

#########################################################################
#                           LOGGING FUNCTIONS                          #
#########################################################################
    def setup_logger(self, log_file):
        """Set up a logger that records both to a file and to the console."""
        self.logger = logging.getLogger("CentralACGan")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # ─── File Handler Configuration ───
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # ─── Console Handler Configuration ───
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # NOTE: Avoid adding duplicate handlers if logger already has them
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def log_model_settings(self):
        """Logs model names, structures, and hyperparameters."""
        self.logger.info("=== Model Settings ===")

        # ─── Generator Model Summary ───
        self.logger.info("Generator Model Summary:")
        generator_summary = []
        self.generator.summary(print_fn=lambda x: generator_summary.append(x))
        for line in generator_summary:
            self.logger.info(line)

        # ─── Discriminator Model Summary ───
        self.logger.info("Discriminator Model Summary:")
        discriminator_summary = []
        self.discriminator.summary(print_fn=lambda x: discriminator_summary.append(x))
        for line in discriminator_summary:
            self.logger.info(line)

        # ─── NIDS Model Summary ───
        if self.nids is not None:
            self.logger.info("NIDS Model Summary:")
            nids_summary = []
            self.nids.summary(print_fn=lambda x: nids_summary.append(x))
            for line in nids_summary:
                self.logger.info(line)
        else:
            self.logger.info("NIDS Model is not defined.")

        # ─── Hyperparameters Logging ───
        self.logger.info("--- Hyperparameters ---")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Noise Dimension: {self.noise_dim}")
        self.logger.info(f"Latent Dimension: {self.latent_dim}")
        self.logger.info(f"Number of Classes: {self.num_classes}")
        self.logger.info(f"Input Dimension: {self.input_dim}")
        self.logger.info(f"Epochs: {self.epochs}")
        self.logger.info(f"Steps per Epoch: {self.steps_per_epoch}")
        self.logger.info(f"Learning Rate (Generator): {self.gen_optimizer.learning_rate}")
        self.logger.info(f"Learning Rate (Discriminator): {self.disc_optimizer.learning_rate}")
        self.logger.info("=" * 50)

    def log_epoch_metrics(self, epoch, d_metrics, g_metrics, nids_metrics=None, fusion_metrics=None):
        """Logs a formatted summary of the metrics for this epoch."""
        self.logger.info(f"=== Epoch {epoch + 1} Metrics Summary ===")

        # ─── Discriminator Metrics ───
        self.logger.info("Discriminator Metrics:")
        for key, value in d_metrics.items():
            self.logger.info(f"  {key}: {value}")

        # ─── Generator Metrics ───
        self.logger.info("Generator Metrics:")
        for key, value in g_metrics.items():
            self.logger.info(f"  {key}: {value}")

        # ─── NIDS Metrics (Optional) ───
        if nids_metrics is not None:
            self.logger.info("NIDS Metrics:")
            for key, value in nids_metrics.items():
                self.logger.info(f"  {key}: {value}")

        # ─── Fusion Metrics (Optional) ───
        if fusion_metrics is not None:
            self.logger.info("Probabilistic Fusion Metrics:")
            for key, value in fusion_metrics.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    def log_evaluation_metrics(self, d_eval, g_eval, nids_eval=None, fusion_eval=None):
        """Logs a formatted summary of evaluation metrics."""
        self.logger.info("=== Evaluation Metrics Summary ===")

        # ─── Discriminator Evaluation ───
        self.logger.info("Discriminator Evaluation:")
        for key, value in d_eval.items():
            self.logger.info(f"  {key}: {value}")

        # ─── Generator Evaluation ───
        self.logger.info("Generator Evaluation:")
        for key, value in g_eval.items():
            self.logger.info(f"  {key}: {value}")

        # ─── NIDS Evaluation (Optional) ───
        if nids_eval is not None:
            self.logger.info("NIDS Evaluation:")
            for key, value in nids_eval.items():
                self.logger.info(f"  {key}: {value}")

        # ─── Fusion Evaluation (Optional) ───
        if fusion_eval is not None:
            self.logger.info("Probabilistic Fusion Evaluation:")
            for key, value in fusion_eval.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

#########################################################################
# Helper method for TRAINING PROCESS to balanced fake label generation  #
#########################################################################
    def generate_balanced_fake_labels(self, total_samples):
        """
        Generate balanced fake labels ensuring equal distribution of classes.

        Parameters:
        -----------
        total_samples : int
            Total number of fake labels to generate

        Returns:
        --------
        tf.Tensor
            Balanced and shuffled fake labels
        """
        half_samples = total_samples // 2
        remaining_samples = total_samples - half_samples

        # Create balanced labels
        fake_labels_0 = tf.zeros(half_samples, dtype=tf.int32)  # Benign class
        fake_labels_1 = tf.ones(remaining_samples, dtype=tf.int32)  # Attack class

        # Concatenate and shuffle
        fake_labels = tf.concat([fake_labels_0, fake_labels_1], axis=0)
        fake_labels = tf.random.shuffle(fake_labels)

        return fake_labels

#########################################################################
#                      LEGACY METHODS (NO LONGER USED)                 #
#########################################################################
# NOTE: Freeze/unfreeze methods are no longer needed with custom training steps.
# The discriminator's training mode is controlled explicitly via training=True/False
# in the custom training step functions.

#########################################################################
#                      BATCH DATA PROCESSING HELPER                     #
#########################################################################
    def process_batch_data(self, data, labels, valid_smoothing_factor):
        """
        Process batch data and labels to ensure correct shapes and encoding.

        Args:
            data: Input feature data
            labels: Corresponding labels
            valid_smoothing_factor: Label smoothing factor for validity labels

        Returns:
            Tuple of (processed_data, processed_labels, validity_labels)
        """
        # • Fix shape issues - ensure 2D data
        if len(data.shape) > 2:
            data = tf.reshape(data, (data.shape[0], -1))

        # • Ensure one-hot encoding
        if len(labels.shape) == 1:
            labels_onehot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
        else:
            labels_onehot = labels

        # • Ensure correct shape for labels
        if len(labels_onehot.shape) > 2:
            labels_onehot = tf.reshape(labels_onehot, (labels_onehot.shape[0], self.num_classes))

        # • Create validity labels with smoothing
        validity_labels = tf.ones((data.shape[0], 1)) * (1 - valid_smoothing_factor)

        return data, labels_onehot, validity_labels

#########################################################################
#                            TRAINING PROCESS                          #
#########################################################################
    def fit(self, X_train=None, y_train=None, d_to_g_ratio=1):
        """
        Train the AC-GAN with a configurable ratio between discriminator and generator training steps.

        Parameters:
        -----------
        X_train : array-like, optional
            Training features. If None, uses self.x_train.
        y_train : array-like, optional
            Training labels. If None, uses self.y_train.
        d_to_g_ratio : int, optional
            Ratio of discriminator training steps to generator training steps.
            Default is 3 (train discriminator 3 times for each generator training step).
        """
        # ═══════════════════════════════════════════════════════════════════════
        # TRAINING DATA PREPARATION
        # ═══════════════════════════════════════════════════════════════════════
        if X_train is None or y_train is None:
            X_train = self.x_train
            y_train = self.y_train

        # ═══════════════════════════════════════════════════════════════════════
        # TRAINING INITIALIZATION & LOGGING
        # ═══════════════════════════════════════════════════════════════════════
        self.log_model_settings()
        self.logger.info(f"Training with discriminator-to-generator ratio: {d_to_g_ratio}:1")
        self.logger.info("CRITICAL FIX: Using custom training loops with separate optimizers")
        self.logger.info("Generator training uses discriminator with training=False to prevent BatchNorm corruption")
        self.logger.info("This addresses the mode collapse issue by ensuring BatchNorm statistics isolation")

        # ═══════════════════════════════════════════════════════════════════════
        # CLASS-SPECIFIC DATA SEPARATION
        # ═══════════════════════════════════════════════════════════════════════
        # ─── Separate Data by Class ───
        benign_indices = tf.where(tf.equal(tf.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train, 0))
        attack_indices = tf.where(tf.equal(tf.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train, 1))

        # ═══════════════════════════════════════════════════════════════════════
        # LABEL SMOOTHING CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════════
        # ─── Discriminator Label Smoothing ───
        valid_smoothing_factor = 0.12
        valid_smooth = tf.ones((self.batch_size, 1)) * (1 - valid_smoothing_factor)

        fake_smoothing_factor = 0.10
        fake_smooth = tf.zeros((self.batch_size, 1)) + fake_smoothing_factor

        # ─── Generator Label Smoothing ───
        # NOTE: Use slightly different smoothing to keep generator from becoming too confident or don't I'm not your mom
        gen_smoothing_factor = 0.05
        valid_smooth_gen = tf.ones((self.batch_size, 1)) * (1 - gen_smoothing_factor)  # Slightly less than 1.0

        self.logger.info(f"Using valid label smoothing with factor: {valid_smoothing_factor}")
        self.logger.info(f"Using fake label smoothing with factor: {fake_smoothing_factor}")
        self.logger.info(f"Using gen label smoothing with factor: {gen_smoothing_factor}")

        # ═══════════════════════════════════════════════════════════════════════
        # METRICS TRACKING INITIALIZATION
        # ═══════════════════════════════════════════════════════════════════════
        d_metrics_history = []
        g_metrics_history = []

        # ═══════════════════════════════════════════════════════════════════════
        # MAIN TRAINING LOOP
        # ═══════════════════════════════════════════════════════════════════════
        for epoch in range(self.epochs):
            print(f'\n=== Epoch {epoch + 1}/{self.epochs} ===\n')
            self.logger.info(f'=== Epoch {epoch + 1}/{self.epochs} ===')

            # ─── Epoch Metrics Initialization ───
            epoch_d_losses = []
            epoch_g_losses = []

            # ─── Determine Steps per Epoch ───
            # CRITICAL FIX: Calculate based on per-class data size, not total dataset
            # Each step consumes batch_size from BOTH benign and attack separately
            # So limiting factor is the smaller class, not total data
            # With 160K benign + 160K attack, batch_size=512: 160K/512 = 312.5 steps
            actual_steps = min(self.steps_per_epoch, min(len(benign_indices), len(attack_indices)) // self.batch_size)

            # ═══════════════════════════════════════════════════════════════════════
            # SHUFFLE INDICES ONCE PER EPOCH (NOT PER STEP)
            # ═══════════════════════════════════════════════════════════════════════
            # Shuffle ONCE at the start of the epoch to prevent duplicate sampling across steps
            shuffled_benign_indices = tf.random.shuffle(benign_indices)
            shuffled_attack_indices = tf.random.shuffle(attack_indices)

            # Initialize global offsets for tracking position in shuffled indices
            global_benign_offset = 0
            global_attack_offset = 0

            # ┌─────────────────────────────────────────────────────────────────┐
            # │                      STEP-BY-STEP TRAINING                      │
            # └─────────────────────────────────────────────────────────────────┘
            for step in range(actual_steps):
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # PRE-SAMPLE BATCHES FOR DISCRIMINATOR TRAINING
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # Pre-sample all batches before discriminator loop for efficiency
                benign_batches = []
                attack_batches = []
                effective_fake_batch_sizes = []

                for d_step in range(d_to_g_ratio):
                    # Calculate start/end for benign batch using global offset
                    benign_start = global_benign_offset
                    benign_end = min(benign_start + self.batch_size, len(benign_indices))

                    # Pre-sample benign batches with dynamic batch sizing
                    if benign_start < len(benign_indices):
                        # Take consecutive slice from shuffled indices
                        benign_idx = shuffled_benign_indices[benign_start:benign_end]
                        benign_batch_data = tf.gather(X_train, benign_idx)
                        benign_batch_labels = tf.gather(y_train, benign_idx)
                        benign_batches.append((benign_batch_data, benign_batch_labels))
                        benign_batch_size = benign_end - benign_start
                        # Advance global offset for next iteration
                        global_benign_offset = benign_end
                    else:
                        benign_batches.append(None)
                        benign_batch_size = 0

                    # Calculate start/end for attack batch using global offset
                    attack_start = global_attack_offset
                    attack_end = min(attack_start + self.batch_size, len(attack_indices))

                    # Pre-sample attack batches with dynamic batch sizing
                    if attack_start < len(attack_indices):
                        # Take consecutive slice from shuffled indices
                        attack_idx = shuffled_attack_indices[attack_start:attack_end]
                        attack_batch_data = tf.gather(X_train, attack_idx)
                        attack_batch_labels = tf.gather(y_train, attack_idx)
                        attack_batches.append((attack_batch_data, attack_batch_labels))
                        attack_batch_size = attack_end - attack_start
                        # Advance global offset for next iteration
                        global_attack_offset = attack_end
                    else:
                        attack_batches.append(None)
                        attack_batch_size = 0

                    # Calculate effective fake batch size based on available real data
                    effective_fake_batch_size = min(
                        max(benign_batch_size, attack_batch_size),
                        self.batch_size
                    )
                    # Ensure at least 1 sample for fake data if no real data available
                    if effective_fake_batch_size == 0:
                        effective_fake_batch_size = min(self.batch_size, 32)  # Fallback to smaller batch

                    effective_fake_batch_sizes.append(effective_fake_batch_size)
                
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # DISCRIMINATOR TRAINING (Multiple Steps)
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                d_step_losses = []

                for d_step in range(d_to_g_ratio):
                    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                    # ┃                  TRAIN ON REAL DATA                           ┃
                    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

                    # ▼ BATCH 1: Train on Benign Data ▼
                    if benign_batches[d_step] is not None:
                        # • Use pre-sampled benign batch
                        benign_data, benign_labels = benign_batches[d_step]

                        # • Process batch data using helper function
                        benign_data, benign_labels_onehot, valid_smooth_benign = self.process_batch_data(
                            benign_data, benign_labels, valid_smoothing_factor)

                        # • TRAIN DISCRIMINATOR ON BATCH DATA (CUSTOM TRAINING STEP)
                        d_total_benign, d_val_loss_benign, d_cls_loss_benign, d_val_acc_benign, d_cls_acc_benign = \
                            self.train_discriminator_step(benign_data, benign_labels_onehot, valid_smooth_benign)

                        # Package results in same format as before for compatibility
                        d_loss_benign = [
                            float(d_total_benign.numpy()),
                            float(d_val_loss_benign.numpy()),
                            float(d_cls_loss_benign.numpy()),
                            float(d_val_acc_benign.numpy()),
                            float(d_cls_acc_benign.numpy())
                        ]

                    # ▼ BATCH 2: Train on Attack Data ▼
                    if attack_batches[d_step] is not None:
                        # • Use pre-sampled attack batch
                        attack_data, attack_labels = attack_batches[d_step]

                        # • Process batch data using helper function
                        attack_data, attack_labels_onehot, valid_smooth_attack = self.process_batch_data(
                            attack_data, attack_labels, valid_smoothing_factor)

                        # • TRAIN DISCRIMINATOR ON ATTACK DATA (CUSTOM TRAINING STEP)
                        d_total_attack, d_val_loss_attack, d_cls_loss_attack, d_val_acc_attack, d_cls_acc_attack = \
                            self.train_discriminator_step(attack_data, attack_labels_onehot, valid_smooth_attack)

                        # Package results in same format as before for compatibility
                        d_loss_attack = [
                            float(d_total_attack.numpy()),
                            float(d_val_loss_attack.numpy()),
                            float(d_cls_loss_attack.numpy()),
                            float(d_val_acc_attack.numpy()),
                            float(d_cls_acc_attack.numpy())
                        ]

                    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                    # ┃                  TRAIN ON FAKE DATA                           ┃
                    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

                    # ▼ BATCH 3: Generate and Train on Fake Data ▼
                    # • Sample noise data with adaptive batch size
                    current_fake_batch_size = effective_fake_batch_sizes[d_step]
                    noise = tf.random.normal((current_fake_batch_size, self.latent_dim))
                    fake_labels = self.generate_balanced_fake_labels(current_fake_batch_size)
                    fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)

                    # • Generate data from noise and labels
                    # Use direct call instead of predict() for better performance during training
                    generated_data = self.generator([noise, fake_labels], training=False)

                    # • Create fake validity labels with proper batch size
                    fake_smooth_batch = tf.zeros((generated_data.shape[0], 1)) + fake_smoothing_factor

                    # • TRAIN DISCRIMINATOR ON FAKE DATA (CUSTOM TRAINING STEP)
                    d_total_fake, d_val_loss_fake, d_cls_loss_fake, d_val_acc_fake, d_cls_acc_fake = \
                        self.train_discriminator_on_fake_step(generated_data, fake_labels_onehot, fake_smooth_batch)

                    # Package results in same format as before for compatibility
                    d_loss_fake = [
                        float(d_total_fake.numpy()),
                        float(d_val_loss_fake.numpy()),
                        float(d_cls_loss_fake.numpy()),
                        float(d_val_acc_fake.numpy()),
                        float(d_cls_acc_fake.numpy())
                    ]

                    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                    # ┃           CALCULATE DISCRIMINATOR WEIGHTED LOSS               ┃
                    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                    d_loss, d_metrics = self.calculate_weighted_loss(
                        d_loss_benign,
                        d_loss_attack,
                        d_loss_fake,
                        attack_weight=0.5,  # Adjust as needed
                        benign_weight=0.5,  # Adjust as needed
                        validity_weight=0.5,  # Adjust as needed
                        class_weight=0.5  # Adjust as needed
                    )

                    d_step_losses.append(float(d_metrics['Total Loss']))

                # --- End Of Discriminator Ratio Loop --- #

                # ─── Store Average Discriminator Loss ───
                avg_d_loss = sum(d_step_losses) / len(d_step_losses)
                epoch_d_losses.append(avg_d_loss)

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # GENERATOR TRAINING (Single Step)
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                # ▼ BATCH 4: New Generator Input ▼
                # • Generate new noise and label inputs for generator training
                noise = tf.random.normal((self.batch_size, self.latent_dim))
                sampled_labels = self.generate_balanced_fake_labels(self.batch_size)
                sampled_labels_onehot = tf.one_hot(sampled_labels, depth=self.num_classes)

                # • Create validity labels for generator with proper batch size
                valid_smooth_gen_batch = tf.ones((noise.shape[0], 1)) * (1 - gen_smoothing_factor)

                # • TRAIN GENERATOR (CUSTOM TRAINING STEP)
                # CRITICAL: discriminator is called with training=False inside this function
                # Generator needs INTEGER labels, but loss calculation needs ONE-HOT labels
                g_total, g_val_loss, g_cls_loss, g_val_acc, g_cls_acc = \
                    self.train_generator_step(noise, sampled_labels, sampled_labels_onehot, valid_smooth_gen_batch)

                # Package results in same format as before for compatibility
                g_loss = [
                    float(g_total.numpy()),
                    float(g_val_loss.numpy()),
                    float(g_cls_loss.numpy()),
                    float(g_val_acc.numpy()),
                    float(g_cls_acc.numpy())
                ]
                epoch_g_losses.append(g_loss[0])

                # ─── Progress Reporting ───
                if step % max(1, actual_steps // 10) == 0:
                    print(f"Step {step}/{actual_steps} - D loss: {avg_d_loss:.4f}, G loss: {g_loss[0]:.4f}")

            # --- End Of Steps Loop --- #

            # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            # ┃                       EPOCH SUMMARY                                 ┃
            # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

            # ─── Collect Epoch Metrics ───
            avg_epoch_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
            avg_epoch_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)

            # ─── Generator Metrics from Last Step ───
            g_metrics = {
                "Total Loss": f"{g_loss[0]:.4f}",
                "Validity Loss": f"{g_loss[1]:.4f}",  # This is Discriminator_loss
                "Class Loss": f"{g_loss[2]:.4f}",  # This is Discriminator_1_loss
                "Generator Fooling Rate": f"{(1 - g_loss[3]) * 100:.2f}%",  # Inverse of Discriminator_binary_accuracy
                "Class Categorical Accuracy": f"{g_loss[4] * 100:.2f}%"  # Discriminator_1_categorical_accuracy
            }

            # ─── Log Epoch Metrics ───
            self.logger.info(f"Epoch {epoch + 1} Summary:")
            self.logger.info(f"Discriminator Average Loss: {avg_epoch_d_loss:.4f}")
            self.logger.info(f"Generator Loss: {avg_epoch_g_loss:.4f}")
            self.logger.info(f"Generator Fooling Rate: {(1 - g_loss[3]) * 100:.2f}%")
            self.logger.info(f"Generator Class Accuracy: {g_loss[4] * 100:.2f}%")

            # ─── DIAGNOSTIC: Check discriminator on training data ───
            if epoch == 0:  # Only on first epoch
                self.logger.info("=== DIAGNOSTIC: Training Data Check ===")
                train_sample = X_train[:100]
                # Convert to numpy array if it's a DataFrame
                if hasattr(train_sample, 'values'):
                    train_sample = train_sample.values
                train_sample = tf.cast(train_sample, tf.float32)

                train_val_pred, train_cls_pred = self.discriminator(train_sample, training=False)
                self.logger.info(f"Training data validity - Mean: {tf.reduce_mean(train_val_pred):.4f}, Min: {tf.reduce_min(train_val_pred):.4f}, Max: {tf.reduce_max(train_val_pred):.4f}")
                self.logger.info(f"Training data - Mean: {tf.reduce_mean(train_sample):.4f}, Std: {tf.math.reduce_std(train_sample):.4f}")
                self.logger.info(f"Training class predictions (first 5): {train_cls_pred[:5].numpy()}")

                # Check pre-activation values to diagnose saturation
                # The validity Dense layer has sigmoid built in, so we need to check its input
                from tensorflow.keras.models import Model

                # Find the validity layer
                validity_layer = None
                for layer in self.discriminator.layers:
                    if 'validity' in layer.name.lower():
                        validity_layer = layer
                        break

                if validity_layer is not None:
                    # Get the input tensor to the validity layer (this is before sigmoid)
                    validity_input_layer = validity_layer.input

                    # Create a model that outputs the pre-sigmoid values
                    pre_sigmoid_model = Model(
                        inputs=self.discriminator.input,
                        outputs=validity_input_layer
                    )

                    # Get the activations going into the validity Dense layer
                    pre_validity_activations = pre_sigmoid_model(train_sample, training=False)
                    self.logger.info(f"PRE-VALIDITY DENSE activations - Mean: {tf.reduce_mean(pre_validity_activations):.4f}, Std: {tf.math.reduce_std(pre_validity_activations):.4f}")

                    # Now manually compute what the output should be without sigmoid
                    # We need to get the weights and bias of the validity layer
                    weights, bias = validity_layer.get_weights()
                    pre_sigmoid = tf.matmul(pre_validity_activations, weights) + bias
                    self.logger.info(f"PRE-SIGMOID logits - Mean: {tf.reduce_mean(pre_sigmoid):.4f}, Min: {tf.reduce_min(pre_sigmoid):.4f}, Max: {tf.reduce_max(pre_sigmoid):.4f}")

                    # Reference values
                    self.logger.info(f"Reference: sigmoid(0) = 0.5, sigmoid(-3) = {tf.nn.sigmoid(-3.0):.4f}, sigmoid(-5) = {tf.nn.sigmoid(-5.0):.4f}")

                    # Check if saturated
                    mean_logit = tf.reduce_mean(pre_sigmoid)
                    if mean_logit < -3.0:
                        self.logger.warning(f"WARNING: Pre-sigmoid logits are very negative (mean={mean_logit:.4f}) - sigmoid is saturated!")
                    elif mean_logit > 3.0:
                        self.logger.warning(f"WARNING: Pre-sigmoid logits are very positive (mean={mean_logit:.4f}) - sigmoid is saturated!")
                else:
                    self.logger.warning("Could not find validity layer for pre-activation diagnostics")

            # ─── Store Metrics History ───
            d_metrics_history.append(avg_epoch_d_loss)
            g_metrics_history.append(avg_epoch_g_loss)

            # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            # ┃                  ADAPTIVE RATIO ADJUSTMENT                          ┃
            # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

            # ─── Calculate Performance Ratio ───
            d_g_loss_ratio = avg_epoch_d_loss / avg_epoch_g_loss

            # ─── Adaptive Ratio Adjustment (every 5 epochs) ───
            if epoch > 0 and epoch % 5 == 0:  # Adjust every 5 epochs
                if d_g_loss_ratio < 0.5:  # Discriminator getting too good
                    d_to_g_ratio = max(1, d_to_g_ratio - 1)
                    self.logger.info(f"Adjusting d_to_g_ratio down to {d_to_g_ratio}:1")
                elif d_g_loss_ratio > 2.0:  # Discriminator struggling
                    d_to_g_ratio += 1
                    self.logger.info(f"Adjusting d_to_g_ratio up to {d_to_g_ratio}:1")

            # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            # ┃                      EPOCH VALIDATION                               ┃
            # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
            self.logger.info(f"=== Epoch {epoch + 1} Validation ===")

            # ─── GAN Validation ───
            d_val_loss, d_val_metrics = self.validation_disc()
            g_val_loss, g_val_metrics = self.validation_gen()

            # ─── Probabilistic Fusion Validation ───
            self.logger.info("=== Probabilistic Fusion Validation on Real Data ===")
            fusion_results, fusion_metrics = self.validate_with_probabilistic_fusion(self.x_val, self.y_val)
            self.logger.info(f"Probabilistic Fusion Accuracy: {fusion_metrics['accuracy'] * 100:.2f}%")
            self.logger.info(f"Predicted Class Distribution: {fusion_metrics['predicted_class_distribution']}")

            # ─── NIDS Validation ───
            nids_val_metrics = None
            if self.nids is not None:
                nids_custom_loss, nids_val_metrics = self.validation_NIDS()
                self.logger.info(f"Validation NIDS Custom Loss: {nids_custom_loss:.4f}")

            # ─── Log Epoch Metrics ───
            self.log_epoch_metrics(epoch, d_val_metrics, g_val_metrics, nids_val_metrics, fusion_metrics)

        # --- End Of Epochs Loop--- #

        # ═══════════════════════════════════════════════════════════════════════
        # TRAINING COMPLETION
        # ═══════════════════════════════════════════════════════════════════════
        # Return the training history for analysis
        return {
            "discriminator_loss": d_metrics_history,
            "generator_loss": g_metrics_history
        }

#########################################################################
#                          VALIDATION METHODS                           #
#########################################################################
    def validation_disc(self):
        """
        Evaluate the discriminator on the validation set.
        Separates benign and attack samples for more detailed metrics.
        """
        # ═══════════════════════════════════════════════════════════════════════
        # VALIDATION DATA PREPARATION
        # ═══════════════════════════════════════════════════════════════════════
        val_valid_labels = np.ones((len(self.x_val), 1))

        # ─── Ensure One-Hot Encoding ───
        if self.y_val.ndim == 1 or self.y_val.shape[1] != self.num_classes:
            y_val_onehot = tf.one_hot(self.y_val, depth=self.num_classes)
        else:
            y_val_onehot = self.y_val

        # ═══════════════════════════════════════════════════════════════════════
        # SEPARATE BENIGN AND ATTACK SAMPLES
        # ═══════════════════════════════════════════════════════════════════════
        if self.y_val.ndim > 1:
            val_labels_idx = tf.argmax(self.y_val, axis=1)
        else:
            val_labels_idx = self.y_val

        benign_indices = tf.where(tf.equal(val_labels_idx, 0))
        attack_indices = tf.where(tf.equal(val_labels_idx, 1))

        # ─── Create Separated Validation Sets ───
        x_val_benign = tf.gather(self.x_val, benign_indices[:, 0])
        y_val_benign_onehot = tf.gather(y_val_onehot, benign_indices[:, 0])
        x_val_attack = tf.gather(self.x_val, attack_indices[:, 0])
        y_val_attack_onehot = tf.gather(y_val_onehot, attack_indices[:, 0])

        # ─── Fix Shape Issues ───
        if len(x_val_benign.shape) > 2:
            x_val_benign = tf.reshape(x_val_benign, (x_val_benign.shape[0], -1))
        if len(x_val_attack.shape) > 2:
            x_val_attack = tf.reshape(x_val_attack, (x_val_attack.shape[0], -1))

        # ═══════════════════════════════════════════════════════════════════════
        # GENERATE FAKE VALIDATION DATA
        # ═══════════════════════════════════════════════════════════════════════
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        fake_labels = self.generate_balanced_fake_labels(len(self.x_val))
        fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)
        fake_valid_labels = np.zeros((len(self.x_val), 1))
        # Use direct call instead of predict() for better performance
        generated_data = self.generator([noise, fake_labels], training=False)

        # ═══════════════════════════════════════════════════════════════════════
        # DIAGNOSTIC: Check discriminator predictions on validation data
        # ═══════════════════════════════════════════════════════════════════════
        sample_benign = x_val_benign[:100]
        sample_attack = x_val_attack[:100]

        benign_val_pred, benign_cls_pred = self.discriminator(sample_benign, training=False)
        attack_val_pred, attack_cls_pred = self.discriminator(sample_attack, training=False)

        self.logger.info("=== DIAGNOSTIC: Discriminator Predictions ===")
        self.logger.info(f"Benign validity - Mean: {tf.reduce_mean(benign_val_pred):.4f}, Min: {tf.reduce_min(benign_val_pred):.4f}, Max: {tf.reduce_max(benign_val_pred):.4f}")
        self.logger.info(f"Attack validity - Mean: {tf.reduce_mean(attack_val_pred):.4f}, Min: {tf.reduce_min(attack_val_pred):.4f}, Max: {tf.reduce_max(attack_val_pred):.4f}")
        self.logger.info(f"Benign class predictions (first 5): {benign_cls_pred[:5].numpy()}")
        self.logger.info(f"Attack class predictions (first 5): {attack_cls_pred[:5].numpy()}")

        # Check data statistics
        self.logger.info(f"Benign data - Mean: {tf.reduce_mean(sample_benign):.4f}, Std: {tf.math.reduce_std(tf.cast(sample_benign, tf.float32)):.4f}")
        self.logger.info(f"Attack data - Mean: {tf.reduce_mean(sample_attack):.4f}, Std: {tf.math.reduce_std(tf.cast(sample_attack, tf.float32)):.4f}")

        # ═══════════════════════════════════════════════════════════════════════
        # EVALUATE ON EACH DATA TYPE
        # ═══════════════════════════════════════════════════════════════════════
        # ─── Benign Evaluation ───
        benign_valid_labels = np.ones((len(x_val_benign), 1))
        d_total_benign, d_val_loss_benign, d_cls_loss_benign, d_val_acc_benign, d_cls_acc_benign = \
            self.evaluate_discriminator(x_val_benign, y_val_benign_onehot, benign_valid_labels)

        d_loss_benign = [
            float(d_total_benign.numpy()),
            float(d_val_loss_benign.numpy()),
            float(d_cls_loss_benign.numpy()),
            float(d_val_acc_benign.numpy()),
            float(d_cls_acc_benign.numpy())
        ]

        # ─── Attack Evaluation ───
        attack_valid_labels = np.ones((len(x_val_attack), 1))
        d_total_attack, d_val_loss_attack, d_cls_loss_attack, d_val_acc_attack, d_cls_acc_attack = \
            self.evaluate_discriminator(x_val_attack, y_val_attack_onehot, attack_valid_labels)

        d_loss_attack = [
            float(d_total_attack.numpy()),
            float(d_val_loss_attack.numpy()),
            float(d_cls_loss_attack.numpy()),
            float(d_val_acc_attack.numpy()),
            float(d_cls_acc_attack.numpy())
        ]

        # ─── Fake Data Evaluation ───
        d_total_fake, d_val_loss_fake, d_cls_loss_fake, d_val_acc_fake, d_cls_acc_fake = \
            self.evaluate_discriminator(generated_data, fake_labels_onehot, fake_valid_labels)

        d_loss_fake = [
            float(d_total_fake.numpy()),
            float(d_val_loss_fake.numpy()),
            float(d_cls_loss_fake.numpy()),
            float(d_val_acc_fake.numpy()),
            float(d_cls_acc_fake.numpy())
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # APPLY WEIGHTED LOSS CALCULATION
        # ═══════════════════════════════════════════════════════════════════════
        # ─── Log Data Distribution ───
        benign_ratio = len(x_val_benign) / (len(x_val_benign) + len(x_val_attack))
        attack_ratio = len(x_val_attack) / (len(x_val_benign) + len(x_val_attack))
        self.logger.info(f"Validation data distribution: Benign {benign_ratio:.2f}, Attack {attack_ratio:.2f}")

        # ─── Calculate Weighted Loss ───
        total_loss, metrics = self.calculate_weighted_loss(
            d_loss_benign,
            d_loss_attack,
            d_loss_fake,
            attack_weight=0.7,
            benign_weight=0.3,
            validity_weight=0.4,
            class_weight=0.6
        )

        # ─── Additional Validation Logging ───
        self.logger.info("Validation Discriminator Evaluation:")
        self.logger.info(f"Weighted Total Loss: {total_loss:.4f}")

        return total_loss, metrics

    def validation_gen(self):
        """
        Evaluate the generator using a validation batch.
        The generator is evaluated by its ability to "fool" the discriminator.
        """
        # ═══════════════════════════════════════════════════════════════════════
        # GENERATOR VALIDATION DATA PREPARATION
        # ═══════════════════════════════════════════════════════════════════════
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        sampled_labels = self.generate_balanced_fake_labels(len(self.x_val))
        sampled_labels_onehot = tf.one_hot(sampled_labels, depth=self.num_classes)
        valid_labels = np.ones((len(self.x_val), 1))

        # ═══════════════════════════════════════════════════════════════════════
        # GENERATOR EVALUATION
        # ═══════════════════════════════════════════════════════════════════════
        g_total, g_val_loss, g_cls_loss, g_val_acc, g_cls_acc = \
            self.evaluate_generator(noise, sampled_labels, sampled_labels_onehot, valid_labels)

        g_loss = [
            float(g_total.numpy()),
            float(g_val_loss.numpy()),
            float(g_cls_loss.numpy()),
            float(g_val_acc.numpy()),
            float(g_cls_acc.numpy())
        ]

        # ─── Log Detailed Metrics ───
        self.logger.info("Validation Generator Evaluation:")
        self.logger.info(
            f"Total Loss: {g_loss[0]:.4f}, Validity Loss: {g_loss[1]:.4f}, Class Loss: {g_loss[2]:.4f}")
        self.logger.info(
            f"Generator Fooling Rate: {(1 - g_loss[3]) * 100:.2f}%")
        self.logger.info(
            f"Class Categorical Accuracy: {g_loss[4] * 100:.2f}%")

        # ─── Create Metrics Dictionary ───
        g_metrics = {
            "Total Loss": f"{g_loss[0]:.4f}",
            "Validity Loss": f"{g_loss[1]:.4f}",
            "Class Loss": f"{g_loss[2]:.4f}",
            "Generator Fooling Rate": f"{(1 - g_loss[3]) * 100:.2f}%",
            "Class Categorical Accuracy": f"{g_loss[4] * 100:.2f}%"
        }
        return g_loss[0], g_metrics

    def validation_NIDS(self):
        """
        Evaluate the NIDS model on validation data augmented with generated fake samples.
        Real data is labeled as 1 (benign) and fake/generated data as 0 (attack).
        Prints detailed metrics including a classification report and returns the custom
        NIDS loss along with a metrics dictionary.
        """
        if self.nids is None:
            print("NIDS model is not defined.")
            return None

        # ═══════════════════════════════════════════════════════════════════════
        # PREPARE REAL AND FAKE VALIDATION DATA
        # ═══════════════════════════════════════════════════════════════════════
        # ─── Real Data Preparation ───
        X_real = self.x_val
        y_real = np.ones((len(self.x_val),), dtype="int32")  # Real samples labeled 1

        # ─── Fake Data Generation ───
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        fake_labels = self.generate_balanced_fake_labels(len(self.x_val))
        # Use direct call instead of predict() for better performance
        generated_samples = self.generator([noise, fake_labels], training=False)
        X_fake = generated_samples
        y_fake = np.zeros((len(self.x_val),), dtype="int32")  # Fake samples labeled 0

        # ═══════════════════════════════════════════════════════════════════════
        # COMPUTE CUSTOM NIDS LOSS
        # ═══════════════════════════════════════════════════════════════════════
        real_output = self.nids.predict(X_real)
        fake_output = self.nids.predict(X_fake)
        custom_nids_loss = self.nids_loss(real_output, fake_output)

        # ═══════════════════════════════════════════════════════════════════════
        # EVALUATE ON COMBINED DATASET
        # ═══════════════════════════════════════════════════════════════════════
        X_combined = np.vstack([X_real, X_fake])
        y_combined = np.hstack([y_real, y_fake])
        nids_eval = self.nids.evaluate(X_combined, y_combined, verbose=0)
        # Expected order: [loss, accuracy, precision, recall, auc, logcosh]

        # ═══════════════════════════════════════════════════════════════════════
        # COMPUTE ADDITIONAL METRICS
        # ═══════════════════════════════════════════════════════════════════════
        y_pred_probs = self.nids.predict(X_combined)
        y_pred = (y_pred_probs > 0.5).astype("int32")
        f1 = f1_score(y_combined, y_pred)
        class_report = classification_report(
            y_combined, y_pred, target_names=["Attack (Fake)", "Benign (Real)"]
        )

        # ─── Log NIDS Validation Results ───
        self.logger.info("Validation NIDS Evaluation with Augmented Data:")
        self.logger.info(f"Custom NIDS Loss (Real vs Fake): {custom_nids_loss:.4f}")
        self.logger.info(f"Overall NIDS Loss: {nids_eval[0]:.4f}, Accuracy: {nids_eval[1]:.4f}, "
                         f"Precision: {nids_eval[2]:.4f}, Recall: {nids_eval[3]:.4f}, "
                         f"AUC: {nids_eval[4]:.4f}, LogCosh: {nids_eval[5]:.4f}")
        self.logger.info("Classification Report:")
        self.logger.info(class_report)
        self.logger.info(f"F1 Score: {f1:.4f}")

        # ─── Create Metrics Dictionary ───
        metrics = {
            "Custom NIDS Loss": f"{custom_nids_loss:.4f}",
            "Loss": f"{nids_eval[0]:.4f}",
            "Accuracy": f"{nids_eval[1]:.4f}",
            "Precision": f"{nids_eval[2]:.4f}",
            "Recall": f"{nids_eval[3]:.4f}",
            "AUC": f"{nids_eval[4]:.4f}",
            "LogCosh": f"{nids_eval[5]:.4f}",
            "F1 Score": f"{f1:.4f}"
        }
        return custom_nids_loss, metrics

    #########################################################################
    #                          EVALUATION METHODS                          #
    #########################################################################
    def evaluate(self, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            X_test = self.x_test
            y_test = self.y_test

        # ═══════════════════════════════════════════════════════════════════════
        # DISCRIMINATOR EVALUATION WITH CLASS SEPARATION
        # ═══════════════════════════════════════════════════════════════════════
        self.logger.info("-- Evaluating Discriminator --")

        # ─── Separate Test Samples by Class ───
        if y_test.ndim > 1:
            test_labels_idx = tf.argmax(y_test, axis=1)
        else:
            test_labels_idx = y_test

        benign_indices = tf.where(tf.equal(test_labels_idx, 0))
        attack_indices = tf.where(tf.equal(test_labels_idx, 1))

        # ─── Create Class-Specific Test Sets ───
        x_test_benign = tf.gather(X_test, benign_indices[:, 0]) if len(benign_indices) > 0 else tf.zeros(
            (0, X_test.shape[1]))
        y_test_benign = tf.gather(y_test, benign_indices[:, 0]) if len(benign_indices) > 0 else tf.zeros(
            (0, y_test.shape[1] if y_test.ndim > 1 else 0))
        x_test_attack = tf.gather(X_test, attack_indices[:, 0]) if len(attack_indices) > 0 else tf.zeros(
            (0, X_test.shape[1]))
        y_test_attack = tf.gather(y_test, attack_indices[:, 0]) if len(attack_indices) > 0 else tf.zeros(
            (0, y_test.shape[1] if y_test.ndim > 1 else 0))

        # ─── Ensure One-Hot Encoding ───
        if y_test.ndim == 1 or y_test.shape[1] != self.num_classes:
            if len(benign_indices) > 0:
                y_test_benign_onehot = tf.one_hot(tf.cast(y_test_benign, tf.int32), depth=self.num_classes)
            else:
                y_test_benign_onehot = tf.zeros((0, self.num_classes))

            if len(attack_indices) > 0:
                y_test_attack_onehot = tf.one_hot(tf.cast(y_test_attack, tf.int32), depth=self.num_classes)
            else:
                y_test_attack_onehot = tf.zeros((0, self.num_classes))
        else:
            y_test_benign_onehot = y_test_benign
            y_test_attack_onehot = y_test_attack

        # ─── Log Test Data Distribution ───
        benign_count = len(x_test_benign)
        attack_count = len(x_test_attack)
        total_count = benign_count + attack_count
        benign_ratio = benign_count / total_count if total_count > 0 else 0
        attack_ratio = attack_count / total_count if total_count > 0 else 0

        self.logger.info(f"Test Data Distribution: {benign_count} Benign ({benign_ratio:.2f}), "
                         f"{attack_count} Attack ({attack_ratio:.2f})")

        # ─── Generate Fake Test Data ───
        fake_count = min(benign_count, attack_count) * 2  # Generate a balanced number of fake samples
        if fake_count > 0:
            noise = tf.random.normal((fake_count, self.latent_dim))
            fake_labels = self.generate_balanced_fake_labels(fake_count)
            fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)
            fake_valid_labels = np.zeros((fake_count, 1))
            # Use direct call instead of predict() for better performance
            generated_data = self.generator([noise, fake_labels], training=False)
        else:
            self.logger.warning("No test samples available to generate fake data")
            generated_data = tf.zeros((0, X_test.shape[1]))
            fake_labels_onehot = tf.zeros((0, self.num_classes))
            fake_valid_labels = np.zeros((0, 1))

        # ═══════════════════════════════════════════════════════════════════════
        # EVALUATE DISCRIMINATOR ON EACH DATA TYPE
        # ═══════════════════════════════════════════════════════════════════════
        d_loss_benign = None
        d_loss_attack = None
        d_loss_fake = None

        # ▼ Benign Data Evaluation ▼
        if benign_count > 0:
            benign_valid_labels = np.ones((benign_count, 1))
            d_total_benign, d_val_loss_benign, d_cls_loss_benign, d_val_acc_benign, d_cls_acc_benign = \
                self.evaluate_discriminator(x_test_benign, y_test_benign_onehot, benign_valid_labels)

            d_loss_benign = [
                float(d_total_benign.numpy()),
                float(d_val_loss_benign.numpy()),
                float(d_cls_loss_benign.numpy()),
                float(d_val_acc_benign.numpy()),
                float(d_cls_acc_benign.numpy())
            ]

            self.logger.info(
                f"Benign Test -> Total Loss: {d_loss_benign[0]:.4f}, "
                f"Validity Loss: {d_loss_benign[1]:.4f}, "
                f"Class Loss: {d_loss_benign[2]:.4f}, "
                f"Validity Accuracy: {d_loss_benign[3] * 100:.2f}%, "
                f"Class Accuracy: {d_loss_benign[4] * 100:.2f}%"
            )

        # ▼ Attack Data Evaluation ▼
        if attack_count > 0:
            attack_valid_labels = np.ones((attack_count, 1))
            d_total_attack, d_val_loss_attack, d_cls_loss_attack, d_val_acc_attack, d_cls_acc_attack = \
                self.evaluate_discriminator(x_test_attack, y_test_attack_onehot, attack_valid_labels)

            d_loss_attack = [
                float(d_total_attack.numpy()),
                float(d_val_loss_attack.numpy()),
                float(d_cls_loss_attack.numpy()),
                float(d_val_acc_attack.numpy()),
                float(d_cls_acc_attack.numpy())
            ]

            self.logger.info(
                f"Attack Test -> Total Loss: {d_loss_attack[0]:.4f}, "
                f"Validity Loss: {d_loss_attack[1]:.4f}, "
                f"Class Loss: {d_loss_attack[2]:.4f}, "
                f"Validity Accuracy: {d_loss_attack[3] * 100:.2f}%, "
                f"Class Accuracy: {d_loss_attack[4] * 100:.2f}%"
            )

        # ▼ Fake Data Evaluation ▼
        if fake_count > 0:
            d_total_fake, d_val_loss_fake, d_cls_loss_fake, d_val_acc_fake, d_cls_acc_fake = \
                self.evaluate_discriminator(generated_data, fake_labels_onehot, fake_valid_labels)

            d_loss_fake = [
                float(d_total_fake.numpy()),
                float(d_val_loss_fake.numpy()),
                float(d_cls_loss_fake.numpy()),
                float(d_val_acc_fake.numpy()),
                float(d_cls_acc_fake.numpy())
            ]
            self.logger.info(
                f"Fake Test -> Total Loss: {d_loss_fake[0]:.4f}, "
                f"Validity Loss: {d_loss_fake[1]:.4f}, "
                f"Class Loss: {d_loss_fake[2]:.4f}, "
                f"Validity Accuracy: {d_loss_fake[3] * 100:.2f}%, "
                f"Class Accuracy: {d_loss_fake[4] * 100:.2f}%"
            )

        # ─── Apply Weighted Loss Calculation ───
        if d_loss_benign is not None and d_loss_attack is not None and d_loss_fake is not None:
            weighted_loss, d_eval_metrics = self.calculate_weighted_loss(
                d_loss_benign,
                d_loss_attack,
                d_loss_fake,
                attack_weight=0.7,
                benign_weight=0.3,
                validity_weight=0.4,
                class_weight=0.6
            )
            self.logger.info(f"Weighted Total Discriminator Loss: {weighted_loss:.4f}")
        else:
            # ─── Fallback to Simple Average ───
            available_losses = []
            if d_loss_benign is not None:
                available_losses.append(d_loss_benign[0])
            if d_loss_attack is not None:
                available_losses.append(d_loss_attack[0])
            if d_loss_fake is not None:
                available_losses.append(d_loss_fake[0])

            if available_losses:
                avg_loss = sum(available_losses) / len(available_losses)
                self.logger.info(f"Average Discriminator Loss: {avg_loss:.4f}")
            else:
                self.logger.warning("No loss data available for discriminator")
                avg_loss = 0.0

            # ─── Create Basic Metrics Dictionary ───
            d_eval_metrics = {
                "Loss": f"{avg_loss:.4f}",
                "Note": "Limited metrics available due to insufficient test data",
            }
            # Add available metrics
            if d_loss_benign is not None:
                d_eval_metrics.update({
                    "Benign Total Loss": f"{d_loss_benign[0]:.4f}",
                    "Benign Validity Acc": f"{d_loss_benign[3] * 100:.2f}%",
                    "Benign Class Acc": f"{d_loss_benign[4] * 100:.2f}%"
                })
            if d_loss_attack is not None:
                d_eval_metrics.update({
                    "Attack Total Loss": f"{d_loss_attack[0]:.4f}",
                    "Attack Validity Acc": f"{d_loss_attack[3] * 100:.2f}%",
                    "Attack Class Acc": f"{d_loss_attack[4] * 100:.2f}%"
                })
            if d_loss_fake is not None:
                d_eval_metrics.update({
                    "Fake Total Loss": f"{d_loss_fake[0]:.4f}",
                    "Fake Validity Acc": f"{d_loss_fake[3] * 100:.2f}%",
                    "Fake Class Acc": f"{d_loss_fake[4] * 100:.2f}%"
                })

        # ═══════════════════════════════════════════════════════════════════════
        # GENERATOR EVALUATION
        # ═══════════════════════════════════════════════════════════════════════
        self.logger.info("-- Evaluating Generator --")

        # ─── Prepare Generator Test Data ───
        noise = tf.random.normal((len(y_test), self.latent_dim))
        sampled_labels = self.generate_balanced_fake_labels(len(y_test))
        sampled_labels_onehot = tf.one_hot(sampled_labels, depth=self.num_classes)
        valid_labels = np.ones((len(y_test), 1))

        # ─── Run Generator Evaluation ───
        g_total, g_val_loss, g_cls_loss, g_val_acc, g_cls_acc = \
            self.evaluate_generator(noise, sampled_labels, sampled_labels_onehot, valid_labels)

        # ─── Extract Generator Metrics ───
        g_loss_total = float(g_total.numpy())
        g_loss_validity = float(g_val_loss.numpy())
        g_loss_class = float(g_cls_loss.numpy())
        g_validity_bin_acc = float(g_val_acc.numpy())
        g_class_cat_acc = float(g_cls_acc.numpy())

        # ─── Create Generator Metrics Dictionary ───
        g_eval_metrics = {
            "Loss": f"{g_loss_total:.4f}",
            "Validity Loss": f"{g_loss_validity:.4f}",
            "Class Loss": f"{g_loss_class:.4f}",
            "Generator Fooling Rate": f"{(1 - g_validity_bin_acc) * 100:.2f}%",
            "Class Categorical Accuracy": f"{g_class_cat_acc * 100:.2f}%"
        }

        # ─── Log Generator Results ───
        self.logger.info(
            f"Generator Total Loss: {g_loss_total:.4f} | Validity Loss: {g_loss_validity:.4f} | Class Loss: {g_loss_class:.4f}"
        )
        self.logger.info(
            f"Generator Fooling Rate: {(1 - g_validity_bin_acc) * 100:.2f}%"
        )
        self.logger.info(
            f"Class Categorical Accuracy: {g_class_cat_acc * 100:.2f}%"
        )

        # ═══════════════════════════════════════════════════════════════════════
        # NIDS EVALUATION
        # ═══════════════════════════════════════════════════════════════════════
        nids_eval_metrics = None
        if self.nids is not None:
            self.logger.info("-- Evaluating NIDS --")

            # ─── Prepare NIDS Test Data ───
            X_real = X_test
            y_real = np.ones((len(X_test),), dtype="int32")

            # ─── Generate Fake Test Data for NIDS ───
            noise = tf.random.normal((len(X_test), self.latent_dim))
            fake_labels = self.generate_balanced_fake_labels(len(X_test))
            # Use direct call instead of predict() for better performance
            generated_samples = self.generator([noise, fake_labels], training=False)
            X_fake = generated_samples
            y_fake = np.zeros((len(X_test),), dtype="int32")

            # ─── Compute Custom NIDS Loss ───
            real_output = self.nids.predict(X_real)
            fake_output = self.nids.predict(X_fake)
            custom_nids_loss = self.nids_loss(real_output, fake_output)

            # ─── Combine Data and Evaluate ───
            X_combined = np.vstack([X_real, X_fake])
            y_combined = np.hstack([y_real, y_fake])
            nids_eval_results = self.nids.evaluate(X_combined, y_combined, verbose=0)
            # Expected order: [loss, accuracy, precision, recall, auc, logcosh]

            # ─── Compute Additional NIDS Metrics ───
            y_pred_probs = self.nids.predict(X_combined)
            y_pred = (y_pred_probs > 0.5).astype("int32")
            f1 = f1_score(y_combined, y_pred)
            class_report = classification_report(
                y_combined, y_pred, target_names=["Attack (Fake)", "Benign (Real)"]
            )

            # ─── Create NIDS Metrics Dictionary ───
            nids_eval_metrics = {
                "Custom NIDS Loss": f"{custom_nids_loss:.4f}",
                "Loss": f"{nids_eval_results[0]:.4f}",
                "Accuracy": f"{nids_eval_results[1]:.4f}",
                "Precision": f"{nids_eval_results[2]:.4f}",
                "Recall": f"{nids_eval_results[3]:.4f}",
                "AUC": f"{nids_eval_results[4]:.4f}",
                "LogCosh": f"{nids_eval_results[5]:.4f}",
                "F1 Score": f"{f1:.4f}"
            }

            # ─── Log NIDS Results ───
            self.logger.info(f"NIDS Custom Loss: {custom_nids_loss:.4f}")
            self.logger.info(
                f"NIDS Evaluation -> Loss: {nids_eval_results[0]:.4f}, Accuracy: {nids_eval_results[1]:.4f}, "
                f"Precision: {nids_eval_results[2]:.4f}, Recall: {nids_eval_results[3]:.4f}, "
                f"AUC: {nids_eval_results[4]:.4f}, LogCosh: {nids_eval_results[5]:.4f}")
            self.logger.info("NIDS Classification Report:")
            self.logger.info(class_report)

        # ═══════════════════════════════════════════════════════════════════════
        # PROBABILISTIC FUSION EVALUATION
        # ═══════════════════════════════════════════════════════════════════════
        self.logger.info("-- Evaluating Probabilistic Fusion --")
        fusion_results, fusion_metrics = self.validate_with_probabilistic_fusion(X_test, y_test)
        self.logger.info(f"Probabilistic Fusion Accuracy: {fusion_metrics['accuracy'] * 100:.2f}%")
        self.logger.info(f"Predicted Class Distribution: {fusion_metrics['predicted_class_distribution']}")

        # ═══════════════════════════════════════════════════════════════════════
        # LOG OVERALL EVALUATION METRICS
        # ═══════════════════════════════════════════════════════════════════════
        self.log_evaluation_metrics(d_eval_metrics, g_eval_metrics, nids_eval_metrics, fusion_metrics)

#########################################################################
#                         LOSS CALCULATION METHODS                     #
#########################################################################

    def calculate_weighted_loss(self, d_loss_benign, d_loss_attack, d_loss_fake,
                                attack_weight=0.7, benign_weight=0.3,
                                validity_weight=0.4, class_weight=0.6):
        """
        Calculate weighted discriminator loss combining benign, attack, and fake samples.
        """
        # ═══════════════════════════════════════════════════════════════════════
        # LOSS COMPONENT EXTRACTION
        # ═══════════════════════════════════════════════════════════════════════
        # ─── Benign Sample Components ───
        d_loss_benign_validity = d_loss_benign[1]
        d_loss_benign_class = d_loss_benign[2]
        d_benign_valid_acc = d_loss_benign[3]
        d_benign_class_acc = d_loss_benign[4]

        # ─── Attack Sample Components ───
        d_loss_attack_validity = d_loss_attack[1]
        d_loss_attack_class = d_loss_attack[2]
        d_attack_valid_acc = d_loss_attack[3]
        d_attack_class_acc = d_loss_attack[4]

        # ─── Fake Sample Components ───
        d_loss_fake_validity = d_loss_fake[1]
        d_loss_fake_class = d_loss_fake[2]
        d_fake_valid_acc = d_loss_fake[3]
        d_fake_class_acc = d_loss_fake[4]

        # ═══════════════════════════════════════════════════════════════════════
        # WEIGHTED LOSS CALCULATIONS
        # ═══════════════════════════════════════════════════════════════════════
        # ─── Weighted Validity Loss ───
        d_loss_validity_real = (benign_weight * d_loss_benign_validity) + (attack_weight * d_loss_attack_validity)
        d_loss_validity = 0.5 * (d_loss_validity_real + d_loss_fake_validity)

        # ─── Weighted Class Loss ───
        d_loss_class_real = (benign_weight * d_loss_benign_class) + (attack_weight * d_loss_attack_class)
        d_loss_class = 0.5 * (d_loss_class_real + d_loss_fake_class)

        # ─── Combined Loss with Task Weights ───
        d_loss = (validity_weight * d_loss_validity) + (class_weight * d_loss_class)

        # ═══════════════════════════════════════════════════════════════════════
        # METRICS CALCULATION FOR LOGGING
        # ═══════════════════════════════════════════════════════════════════════
        # ─── Total Losses for Each Sample Type ───
        d_loss_benign_total = benign_weight * (d_loss_benign[0])
        d_loss_attack_total = attack_weight * (d_loss_attack[0])
        d_loss_fake_total = 0.5 * (d_loss_fake[0])
        d_loss_total = d_loss_benign_total + d_loss_attack_total + d_loss_fake_total

        # ─── Weighted Accuracies ───
        d_valid_acc_real = (benign_weight * d_benign_valid_acc + attack_weight * d_attack_valid_acc)
        d_class_acc_real = (benign_weight * d_benign_class_acc + attack_weight * d_attack_class_acc)

        # ─── Create Metrics Dictionary ───
        d_metrics = {
            "Total Loss": f"{d_loss_total:.4f}",
            "Benign Loss": f"{d_loss_benign[0]:.4f}",
            "Attack Loss": f"{d_loss_attack[0]:.4f}",
            "Fake Loss": f"{d_loss_fake[0]:.4f}",
            "Validity Loss": f"{d_loss_validity:.4f}",
            "Class Loss": f"{d_loss_class:.4f}",
            "Benign Validity Acc": f"{d_benign_valid_acc * 100:.2f}%",
            "Attack Validity Acc": f"{d_attack_valid_acc * 100:.2f}%",
            "Fake Validity Acc": f"{d_fake_valid_acc * 100:.2f}%",
            "Benign Class Acc": f"{d_benign_class_acc * 100:.2f}%",
            "Attack Class Acc": f"{d_attack_class_acc * 100:.2f}%",
            "Fake Class Acc": f"{d_fake_class_acc * 100:.2f}%"
        }

        return d_loss, d_metrics

    def calculate_jsd_loss(self, real_outputs, fake_outputs):
        """
        Calculate Jensen-Shannon Divergence Loss between real and fake sample distributions.
        This can be used as an alternative or supplement to binary cross entropy for
        measuring discriminator performance.
        """
        # ─── Average Probabilities ───
        p_real = tf.reduce_mean(real_outputs, axis=0)
        p_fake = tf.reduce_mean(fake_outputs, axis=0)

        # ─── Calculate Mixtures ───
        p_mixture = 0.5 * (p_real + p_fake)

        # ─── Calculate JS Divergence ───
        kl_real_mix = tf.reduce_sum(p_real * tf.math.log(p_real / p_mixture + 1e-10))
        kl_fake_mix = tf.reduce_sum(p_fake * tf.math.log(p_fake / p_mixture + 1e-10))

        # ─── JS Divergence ───
        jsd = 0.5 * (kl_real_mix + kl_fake_mix)

        return jsd

    def nids_loss(self, real_output, fake_output):
        """
        Compute the NIDS loss on real and fake samples.
        For real samples, the target is 1 (benign), and for fake samples, 0 (attack).
        Returns a scalar loss value.
        """
        # ─── Define Labels ───
        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)

        # ─── Define Loss Function ───
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # ─── Calculate Outputs ───
        real_loss = bce(real_labels, real_output)
        fake_loss = bce(fake_labels, fake_output)

        # ─── Sum Total Loss ───
        total_loss = real_loss + fake_loss
        return total_loss.numpy()

#########################################################################
#                    PROBABILISTIC FUSION METHODS                      #
#########################################################################
    def probabilistic_fusion(self, input_data):
        """
        Apply probabilistic fusion to combine validity and class predictions.
        Returns combined probabilities for all four possible outcomes.
        """
        # ═══════════════════════════════════════════════════════════════════════
        # GET DISCRIMINATOR PREDICTIONS
        # ═══════════════════════════════════════════════════════════════════════
        validity_scores, class_predictions = self.discriminator.predict(input_data)

        total_samples = len(input_data)
        results = []

        # ═══════════════════════════════════════════════════════════════════════
        # CALCULATE JOINT PROBABILITIES FOR EACH SAMPLE
        # ═══════════════════════════════════════════════════════════════════════
        for i in range(total_samples):
            # ─── Validity Probabilities ───
            p_valid = validity_scores[i][0]  # Probability of being valid/real
            p_invalid = 1 - p_valid  # Probability of being invalid/fake

            # ─── Class Probabilities ───
            p_benign = class_predictions[i][0]  # Probability of being benign
            p_attack = class_predictions[i][1]  # Probability of being attack

            # ─── Calculate Joint Probabilities ───
            p_valid_benign = p_valid * p_benign
            p_valid_attack = p_valid * p_attack
            p_invalid_benign = p_invalid * p_benign
            p_invalid_attack = p_invalid * p_attack

            # ─── Store All Probabilities ───
            probabilities = {
                "valid_benign": p_valid_benign,
                "valid_attack": p_valid_attack,
                "invalid_benign": p_invalid_benign,
                "invalid_attack": p_invalid_attack
            }

            # ─── Find Most Likely Outcome ───
            most_likely = max(probabilities, key=probabilities.get)

            # ─── Create Result Dictionary ───
            result = {
                "classification": most_likely,
                "probabilities": probabilities
            }

            results.append(result)

        return results

    def validate_with_probabilistic_fusion(self, validation_data, validation_labels=None):
        """
        Evaluate model using probabilistic fusion and calculate metrics if labels are available.
        """
        # ═══════════════════════════════════════════════════════════════════════
        # APPLY PROBABILISTIC FUSION
        # ═══════════════════════════════════════════════════════════════════════
        fusion_results = self.probabilistic_fusion(validation_data)

        # ─── Extract Classifications ───
        classifications = [result["classification"] for result in fusion_results]

        # ─── Count Class Distribution ───
        predicted_class_distribution = Counter(classifications)
        self.logger.info(f"Predicted Class Distribution: {dict(predicted_class_distribution)}")

        # ═══════════════════════════════════════════════════════════════════════
        # CALCULATE ACCURACY IF LABELS AVAILABLE
        # ═══════════════════════════════════════════════════════════════════════
        if validation_labels is not None:
            correct_predictions = 0
            correct_classifications = []
            true_classifications = []

            for i, result in enumerate(fusion_results):
                # ─── Get True Label ───
                if isinstance(validation_labels, np.ndarray) and validation_labels.ndim > 1:
                    true_class_idx = np.argmax(validation_labels[i])
                else:
                    true_class_idx = validation_labels[i]

                true_class = "benign" if true_class_idx == 0 else "attack"

                # NOTE: For validation data (which is real), expected validity is "valid"
                true_validity = "valid"

                # ─── Construct True Combined Label ───
                true_combined = f"{true_validity}_{true_class}"
                true_classifications.append(true_combined)

                # ─── Check Prediction Match ───
                if result["classification"] == true_combined:
                    correct_predictions += 1
                    correct_classifications.append(result["classification"])

            # ─── Calculate Distributions ───
            correct_class_distribution = Counter(correct_classifications)
            true_class_distribution = Counter(true_classifications)

            self.logger.info(f"True Class Distribution: {dict(true_class_distribution)}")

            # ─── Calculate Final Accuracy ───
            accuracy = correct_predictions / len(validation_data)
            self.logger.info(f"Accuracy: {accuracy:.4f}")

            # ─── Create Metrics Dictionary ───
            metrics = {
                "accuracy": accuracy,
                "total_samples": len(validation_data),
                "correct_predictions": correct_predictions,
                "predicted_class_distribution": dict(predicted_class_distribution),
                "correct_class_distribution": dict(correct_class_distribution),
                "true_class_distribution": dict(true_class_distribution)
            }

            return classifications, metrics

        return classifications, {"predicted_class_distribution": dict(predicted_class_distribution)}

    def analyze_fusion_results(self, fusion_results):
        """Analyze the distribution of probabilities from fusion results"""
        # ═══════════════════════════════════════════════════════════════════════
        # EXTRACT PROBABILITIES BY CATEGORY
        # ═══════════════════════════════════════════════════════════════════════
        valid_benign_probs = [r["probabilities"]["valid_benign"] for r in fusion_results]
        valid_attack_probs = [r["probabilities"]["valid_attack"] for r in fusion_results]
        invalid_benign_probs = [r["probabilities"]["invalid_benign"] for r in fusion_results]
        invalid_attack_probs = [r["probabilities"]["invalid_attack"] for r in fusion_results]

        # ═══════════════════════════════════════════════════════════════════════
        # CALCULATE SUMMARY STATISTICS
        # ═══════════════════════════════════════════════════════════════════════
        categories = ["Valid Benign", "Valid Attack", "Invalid Benign", "Invalid Attack"]
        all_probs = [valid_benign_probs, valid_attack_probs, invalid_benign_probs, invalid_attack_probs]

        for cat, probs in zip(categories, all_probs):
            self.logger.info(
                f"{cat}: Mean={np.mean(probs):.4f}, Median={np.median(probs):.4f}, Max={np.max(probs):.4f}")

#########################################################################
#                           MODEL SAVING METHODS                       #
#########################################################################
    def save(self, save_name):
        import os
        # Calculate absolute path to ModelArchive
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..', '..', '..', '..'))
        model_archive_path = os.path.join(project_root, 'ModelArchive')
        
        # Create ModelArchive directory if it doesn't exist
        os.makedirs(model_archive_path, exist_ok=True)
        
        # Save each submodel separately using absolute paths
        self.generator.save(os.path.join(model_archive_path, f"generator_local_ACGAN_{save_name}.h5"))
        self.discriminator.save(os.path.join(model_archive_path, f"discriminator_local_ACGAN_{save_name}.h5"))

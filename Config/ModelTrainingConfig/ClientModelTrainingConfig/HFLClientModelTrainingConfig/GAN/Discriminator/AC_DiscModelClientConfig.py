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
from collections import Counter

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
#                                               FEDERATED DISCRIMINATOR CLIENT                              #
################################################################################################################

class ACDiscriminatorClient(fl.client.NumPyClient):
    def __init__(self, discriminator, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 num_classes, input_dim, epochs, steps_per_epoch, learning_rate,
                 log_file="training.log", use_class_labels=True):
        # -- models
        self.discriminator = discriminator

        # -- I/O Specs for models
        self.batch_size = BATCH_SIZE
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.use_class_labels = use_class_labels  # Whether to use class labels in training

        # -- training duration
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        # -- Data
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        # -- Early Stopping Configuration
        self.early_stopping_patience = 5
        self.min_delta = 0.001  # Minimum improvement to consider as progress

        print(log_file)
        # -- Setup Logging
        self.setup_logger(log_file)

        # -- Optimizers and Learning Rate Scheduling
        lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=0.98, staircase=True)
        self.disc_optimizer = Adam(learning_rate=lr_schedule_disc, beta_1=0.5, beta_2=0.999)

        # -- Loss Functions Setup (for custom training loops)
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        # -- Metrics Setup (for custom training loops)
        self.d_binary_accuracy = tf.keras.metrics.BinaryAccuracy(name='d_binary_accuracy')
        self.d_categorical_accuracy = tf.keras.metrics.CategoricalAccuracy(name='d_categorical_accuracy')

        # -- Model Compilations based on whether we use class labels
        # NOTE: Compilation is kept for backward compatibility, but custom training loops are preferred
        if self.use_class_labels:
            self.discriminator.compile(
                loss={'validity': 'binary_crossentropy', 'class': 'categorical_crossentropy'},
                optimizer=self.disc_optimizer,
                metrics={
                    'validity': ['binary_accuracy'],
                    'class': ['categorical_accuracy']
                }
            )
        else:
            # If not using class labels, we only care about validity output
            self.discriminator.compile(
                loss={'validity': 'binary_crossentropy'},
                optimizer=self.disc_optimizer,
                metrics={'validity': ['binary_accuracy']}
            )

#########################################################################
#                           LOGGING FUNCTIONS                          #
#########################################################################
    def setup_logger(self, log_file):
        """Set up a logger that records both to a file and to the console."""
        self.logger = logging.getLogger("ACDiscriminatorClient")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def log_model_settings(self):
        """Logs model summary and hyperparameters."""
        self.logger.info("=== Model Settings ===")
        self.logger.info("Discriminator Model Summary:")
        disc_summary = []
        self.discriminator.summary(print_fn=lambda x: disc_summary.append(x))
        for line in disc_summary:
            self.logger.info(line)

        self.logger.info("--- Hyperparameters ---")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Number of Classes: {self.num_classes}")
        self.logger.info(f"Input Dimension: {self.input_dim}")
        self.logger.info(f"Epochs: {self.epochs}")
        self.logger.info(f"Steps per Epoch: {self.steps_per_epoch}")
        self.logger.info(f"Learning Rate (Discriminator): {self.disc_optimizer.learning_rate}")
        self.logger.info(f"Using Class Labels: {self.use_class_labels}")
        self.logger.info("=" * 50)

    def log_epoch_metrics(self, epoch, d_metrics, fusion_metrics=None):
        """Logs a formatted summary of the metrics for the current epoch."""
        self.logger.info(f"=== Epoch {epoch} Metrics Summary ===")
        self.logger.info("Discriminator Metrics:")
        for key, value in d_metrics.items():
            self.logger.info(f"  {key}: {value}")
        if fusion_metrics is not None:
            self.logger.info("Probabilistic Fusion Metrics:")
            for key, value in fusion_metrics.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    def log_evaluation_metrics(self, d_eval, fusion_eval=None):
        """Logs a formatted summary of evaluation metrics."""
        self.logger.info("=== Evaluation Metrics Summary ===")
        self.logger.info("Discriminator Evaluation:")
        for key, value in d_eval.items():
            self.logger.info(f"  {key}: {value}")
        if fusion_eval is not None:
            self.logger.info("Probabilistic Fusion Evaluation:")
            for key, value in fusion_eval.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

# ═══════════════════════════════════════════════════════════════════════
# MODEL ACCESS METHODS
# ═══════════════════════════════════════════════════════════════════════
    def get_parameters(self, config):
        # Return the discriminator's weights
        return self.discriminator.get_weights()

    #########################################################################
    # Helper method for TRAINING PROCESS #
    #########################################################################
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
        # Get discriminator predictions
        if self.use_class_labels:
            validity_scores, class_predictions = self.discriminator.predict(input_data)
        else:
            validity_scores = self.discriminator.predict(input_data)
            # If no class labels, create dummy class predictions (all benign)
            class_predictions = np.ones((len(input_data), 2))
            class_predictions[:, 0] = 1.0  # All benign
            class_predictions[:, 1] = 0.0  # No attack

        total_samples = len(input_data)
        results = []

        for i in range(total_samples):
            # Validity probabilities: P(valid) and P(invalid)
            p_valid = validity_scores[i][0]  # Probability of being valid/real
            p_invalid = 1 - p_valid  # Probability of being invalid/fake

            # Class probabilities: 2 classes (benign=0, attack=1)
            p_benign = class_predictions[i][0]  # Probability of being benign
            p_attack = class_predictions[i][1]  # Probability of being attack

            # Calculate joint probabilities for all combinations
            p_valid_benign = p_valid * p_benign
            p_valid_attack = p_valid * p_attack
            p_invalid_benign = p_invalid * p_benign
            p_invalid_attack = p_invalid * p_attack

            # Store all probabilities in a dictionary
            probabilities = {
                "valid_benign": p_valid_benign,
                "valid_attack": p_valid_attack,
                "invalid_benign": p_invalid_benign,
                "invalid_attack": p_invalid_attack
            }

            # Find the most likely outcome
            most_likely = max(probabilities, key=probabilities.get)

            # For analysis, add the actual probabilities alongside the classification
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
        fusion_results = self.probabilistic_fusion(validation_data)

        # Extract classifications
        classifications = [result["classification"] for result in fusion_results]

        # Count occurrences of each class
        predicted_class_distribution = Counter(classifications)
        self.logger.info(f"Predicted Class Distribution: {dict(predicted_class_distribution)}")

        # If we have ground truth labels, calculate accuracy
        if validation_labels is not None:
            correct_predictions = 0
            correct_classifications = []
            true_classifications = []

            for i, result in enumerate(fusion_results):
                # Get the true label (assuming 0=benign, 1=attack)
                if isinstance(validation_labels, np.ndarray) and validation_labels.ndim > 1:
                    true_class_idx = np.argmax(validation_labels[i])
                else:
                    true_class_idx = validation_labels[i]

                true_class = "benign" if true_class_idx == 0 else "attack"

                # For validation data (which is real), expected validity is "valid"
                true_validity = "valid"  # Since validation data is real data

                # Construct the true combined label
                true_combined = f"{true_validity}_{true_class}"

                # Add to true classifications list
                true_classifications.append(true_combined)

                # Check if prediction matches
                if result["classification"] == true_combined:
                    correct_predictions += 1
                    correct_classifications.append(result["classification"])

            # Count distribution of correctly classified samples
            correct_class_distribution = Counter(correct_classifications)

            # Count distribution of true classes
            true_class_distribution = Counter(true_classifications)
            self.logger.info(f"True Class Distribution: {dict(true_class_distribution)}")

            accuracy = correct_predictions / len(validation_data)
            self.logger.info(f"Accuracy: {accuracy:.4f}")

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
        # Extract probabilities for each category
        valid_benign_probs = [r["probabilities"]["valid_benign"] for r in fusion_results]
        valid_attack_probs = [r["probabilities"]["valid_attack"] for r in fusion_results]
        invalid_benign_probs = [r["probabilities"]["invalid_benign"] for r in fusion_results]
        invalid_attack_probs = [r["probabilities"]["invalid_attack"] for r in fusion_results]

        # Calculate summary statistics
        categories = ["Valid Benign", "Valid Attack", "Invalid Benign", "Invalid Attack"]
        all_probs = [valid_benign_probs, valid_attack_probs, invalid_benign_probs, invalid_attack_probs]

        for cat, probs in zip(categories, all_probs):
            self.logger.info(
                f"{cat}: Mean={np.mean(probs):.4f}, Median={np.median(probs):.4f}, Max={np.max(probs):.4f}")

        # You could add additional visualizations or analysis here

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
            # Tuning history: 0.5x (too weak) → 0.2x → 0.1x (too strong) → 0.15x (WORKING - showed 11% acc by epoch 3)
            # Backup options if needed: 0.3x/5.0x or 0.2x/6.0x
            total_loss = (0.15 * validity_loss) + class_loss

        # Calculate gradients and update weights
        gradients = tape.gradient(total_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        # Calculate accuracies
        validity_acc = self.d_binary_accuracy(real_validity_labels, validity_pred)
        class_acc = self.d_categorical_accuracy(real_labels, class_pred)

        return total_loss, validity_loss, class_loss, validity_acc, class_acc

    @tf.function
    def train_discriminator_step_validity_only(self, real_data, real_validity_labels):
        """
        Custom training step for discriminator on real data (validity only, no class labels).

        Args:
            real_data: Real input features
            real_validity_labels: Validity labels (1 for real)

        Returns:
            Tuple of (total_loss, validity_acc)
        """
        # Convert inputs to float32 for type consistency
        real_data = tf.cast(real_data, tf.float32)
        real_validity_labels = tf.cast(real_validity_labels, tf.float32)

        with tf.GradientTape() as tape:
            # Forward pass with training=True
            validity_pred = self.discriminator(real_data, training=True)

            # Calculate loss
            validity_loss = self.binary_crossentropy(real_validity_labels, validity_pred)

            total_loss = validity_loss

        # Calculate gradients and update weights
        gradients = tape.gradient(total_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        # Calculate accuracy
        validity_acc = self.d_binary_accuracy(real_validity_labels, validity_pred)

        return total_loss, validity_acc

    # -- Custom Evaluation Helper -- #
    @tf.function
    def evaluate_discriminator(self, data, labels, validity_labels):
        """
        Evaluate discriminator without updating weights.

        Args:
            data: Input features
            labels: One-hot encoded class labels (or None if not using class labels)
            validity_labels: Validity labels

        Returns:
            Tuple of (total_loss, validity_loss, class_loss, validity_acc, class_acc)
            Note: class_loss and class_acc will be 0 if not using class labels
        """
        # Convert inputs to float32 to ensure type consistency
        data = tf.cast(data, tf.float32)
        validity_labels = tf.cast(validity_labels, tf.float32)

        if self.use_class_labels:
            labels = tf.cast(labels, tf.float32)

            # Forward pass with training=False for evaluation
            validity_pred, class_pred = self.discriminator(data, training=False)

            # Calculate losses
            validity_loss = tf.keras.losses.binary_crossentropy(validity_labels, validity_pred)
            validity_loss = tf.reduce_mean(validity_loss)

            class_loss = tf.keras.losses.categorical_crossentropy(labels, class_pred)
            class_loss = tf.reduce_mean(class_loss)

            total_loss = validity_loss + class_loss

            # Calculate accuracies
            validity_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.round(validity_pred), validity_labels), tf.float32)
            )
            class_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(class_pred, axis=1), tf.argmax(labels, axis=1)), tf.float32)
            )
        else:
            # Only validity output
            validity_pred = self.discriminator(data, training=False)

            # Calculate validity loss
            validity_loss = tf.keras.losses.binary_crossentropy(validity_labels, validity_pred)
            validity_loss = tf.reduce_mean(validity_loss)

            total_loss = validity_loss
            class_loss = tf.constant(0.0)

            # Calculate validity accuracy
            validity_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.round(validity_pred), validity_labels), tf.float32)
            )
            class_acc = tf.constant(0.0)

        return total_loss, validity_loss, class_loss, validity_acc, class_acc

    #########################################################################
    #                            TRAINING PROCESS                          #
    #########################################################################
    def fit(self, parameters, config):
        """
        Train the discriminator using class-separated training approach.
        For FL discriminator clients, trains on real data only (no generator available).
        However, we still separate benign and attack classes for better training.

        Parameters:
        -----------
        parameters : list
            Model weights from the FL server
        config : dict
            Configuration from FL server
        """
        # ═══════════════════════════════════════════════════════════════════════
        # FL SERVER WEIGHT INITIALIZATION
        # ═══════════════════════════════════════════════════════════════════════
        self.discriminator.set_weights(parameters)

        # ═══════════════════════════════════════════════════════════════════════
        # ENSURE DISCRIMINATOR IS TRAINABLE
        # ═══════════════════════════════════════════════════════════════════════
        self.discriminator.trainable = True
        for layer in self.discriminator.layers:
            layer.trainable = True

        # ═══════════════════════════════════════════════════════════════════════
        # RE-COMPILE DISCRIMINATOR
        # ═══════════════════════════════════════════════════════════════════════
        if self.use_class_labels:
            self.discriminator.compile(
                loss={'validity': 'binary_crossentropy', 'class': 'categorical_crossentropy'},
                optimizer=self.disc_optimizer,
                metrics={
                    'validity': ['binary_accuracy'],
                    'class': ['categorical_accuracy']
                }
            )
        else:
            self.discriminator.compile(
                loss={'validity': 'binary_crossentropy'},
                optimizer=self.disc_optimizer,
                metrics={'validity': ['binary_accuracy']}
            )

        # ═══════════════════════════════════════════════════════════════════════
        # TRAINING DATA PREPARATION
        # ═══════════════════════════════════════════════════════════════════════
        X_train = self.x_train
        y_train = self.y_train

        # Log model settings at the start
        self.log_model_settings()

        # ═══════════════════════════════════════════════════════════════════════
        # CLASS-SPECIFIC DATA SEPARATION (Only if using class labels)
        # ═══════════════════════════════════════════════════════════════════════
        if self.use_class_labels:
            benign_indices = tf.where(tf.equal(tf.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train, 0))
            attack_indices = tf.where(tf.equal(tf.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train, 1))

            self.logger.info(f"Training data: {len(benign_indices)} benign, {len(attack_indices)} attack samples")
            use_class_separation = len(benign_indices) >= self.batch_size and len(attack_indices) >= self.batch_size
        else:
            use_class_separation = False

        # ═══════════════════════════════════════════════════════════════════════
        # LABEL SMOOTHING CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════════
        valid_smoothing_factor = 0.15
        self.logger.info(f"Using valid label smoothing with factor: {valid_smoothing_factor}")
        self.logger.info("Training discriminator using REAL DATA ONLY (Federated)")

        # ═══════════════════════════════════════════════════════════════════════
        # METRICS TRACKING INITIALIZATION
        # ═══════════════════════════════════════════════════════════════════════
        d_metrics_history = []

        # ═══════════════════════════════════════════════════════════════════════
        # EARLY STOPPING TRACKING
        # ═══════════════════════════════════════════════════════════════════════
        best_fusion_accuracy = 0.0
        best_epoch = 0
        patience_counter = 0
        best_weights = None

        self.logger.info(f"Early stopping enabled with patience={self.early_stopping_patience}, min_delta={self.min_delta}")

        # ═══════════════════════════════════════════════════════════════════════
        # MAIN TRAINING LOOP
        # ═══════════════════════════════════════════════════════════════════════
        for epoch in range(self.epochs):
            print(f'\n=== Epoch {epoch + 1}/{self.epochs} ===\n')
            self.logger.info(f'=== Epoch {epoch + 1}/{self.epochs} ===')

            # ───────────────────────────────────────────────────────────────────
            # EPOCH INITIALIZATION
            # ───────────────────────────────────────────────────────────────────
            epoch_loss = 0
            epoch_validity_loss = 0
            epoch_class_loss = 0
            epoch_validity_acc = 0
            epoch_class_acc = 0

            # Determine steps per epoch
            actual_steps = min(self.steps_per_epoch, len(X_train) // self.batch_size)

            # ───────────────────────────────────────────────────────────────────
            # STEP-BY-STEP TRAINING
            # ───────────────────────────────────────────────────────────────────
            for step in range(actual_steps):
                if use_class_separation and self.use_class_labels:
                    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                    # ┃           CLASS-SEPARATED TRAINING                       ┃
                    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

                    # ▼ BATCH 1: Train on Benign Data ▼
                    benign_idx = tf.random.shuffle(benign_indices)[:self.batch_size]
                    benign_data = tf.gather(X_train, benign_idx)
                    benign_labels = tf.gather(y_train, benign_idx)

                    # Fix shape issues
                    if len(benign_data.shape) > 2:
                        benign_data = tf.reshape(benign_data, (benign_data.shape[0], -1))

                    # Ensure one-hot encoding
                    if len(benign_labels.shape) == 1:
                        benign_labels_onehot = tf.one_hot(tf.cast(benign_labels, tf.int32), depth=self.num_classes)
                    else:
                        benign_labels_onehot = benign_labels

                    if len(benign_labels_onehot.shape) > 2:
                        benign_labels_onehot = tf.reshape(benign_labels_onehot, (benign_labels_onehot.shape[0], self.num_classes))

                    # Create validity labels with smoothing
                    valid_smooth_benign = tf.ones((benign_data.shape[0], 1)) * (1 - valid_smoothing_factor)

                    # Train on benign data using custom training step
                    d_total_benign, d_val_loss_benign, d_cls_loss_benign, d_val_acc_benign, d_cls_acc_benign = \
                        self.train_discriminator_step(benign_data, benign_labels_onehot, valid_smooth_benign)

                    # Package losses for compatibility with existing metrics tracking
                    d_loss_benign = [
                        float(d_total_benign.numpy()),
                        float(d_val_loss_benign.numpy()),
                        float(d_cls_loss_benign.numpy()),
                        float(d_val_acc_benign.numpy()),
                        float(d_cls_acc_benign.numpy())
                    ]

                    # ▼ BATCH 2: Train on Attack Data ▼
                    attack_idx = tf.random.shuffle(attack_indices)[:self.batch_size]
                    attack_data = tf.gather(X_train, attack_idx)
                    attack_labels = tf.gather(y_train, attack_idx)

                    # Fix shape issues
                    if len(attack_data.shape) > 2:
                        attack_data = tf.reshape(attack_data, (attack_data.shape[0], -1))

                    # Ensure one-hot encoding
                    if len(attack_labels.shape) == 1:
                        attack_labels_onehot = tf.one_hot(tf.cast(attack_labels, tf.int32), depth=self.num_classes)
                    else:
                        attack_labels_onehot = attack_labels

                    if len(attack_labels_onehot.shape) > 2:
                        attack_labels_onehot = tf.reshape(attack_labels_onehot, (attack_labels_onehot.shape[0], self.num_classes))

                    # Create validity labels with smoothing
                    valid_smooth_attack = tf.ones((attack_data.shape[0], 1)) * (1 - valid_smoothing_factor)

                    # Train on attack data using custom training step
                    d_total_attack, d_val_loss_attack, d_cls_loss_attack, d_val_acc_attack, d_cls_acc_attack = \
                        self.train_discriminator_step(attack_data, attack_labels_onehot, valid_smooth_attack)

                    # Package losses for compatibility with existing metrics tracking
                    d_loss_attack = [
                        float(d_total_attack.numpy()),
                        float(d_val_loss_attack.numpy()),
                        float(d_cls_loss_attack.numpy()),
                        float(d_val_acc_attack.numpy()),
                        float(d_cls_acc_attack.numpy())
                    ]

                    # Accumulate metrics (average of benign and attack)
                    epoch_loss += (d_loss_benign[0] + d_loss_attack[0]) / 2
                    epoch_validity_loss += (d_loss_benign[1] + d_loss_attack[1]) / 2
                    epoch_class_loss += (d_loss_benign[2] + d_loss_attack[2]) / 2
                    epoch_validity_acc += (d_loss_benign[3] + d_loss_attack[3]) / 2
                    epoch_class_acc += (d_loss_benign[4] + d_loss_attack[4]) / 2

                else:
                    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                    # ┃           STANDARD TRAINING (NO CLASS SEPARATION)        ┃
                    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

                    # Sample a batch of real data
                    idx = tf.random.shuffle(tf.range(len(X_train)))[:self.batch_size]
                    real_data = tf.gather(X_train, idx)
                    real_labels = tf.gather(y_train, idx)

                    # Prepare training data based on whether we use class labels
                    if self.use_class_labels:
                        # Ensure labels are one-hot encoded
                        if len(real_labels.shape) == 1:
                            real_labels_onehot = tf.one_hot(tf.cast(real_labels, tf.int32), depth=self.num_classes)
                        else:
                            real_labels_onehot = real_labels

                        # Create validity labels with smoothing
                        valid_smooth = tf.ones((self.batch_size, 1)) * (1 - valid_smoothing_factor)

                        # Train discriminator on real data using custom training step
                        d_total_real, d_val_loss_real, d_cls_loss_real, d_val_acc_real, d_cls_acc_real = \
                            self.train_discriminator_step(real_data, real_labels_onehot, valid_smooth)

                        # Package losses for compatibility with existing metrics tracking
                        d_loss_real = [
                            float(d_total_real.numpy()),
                            float(d_val_loss_real.numpy()),
                            float(d_cls_loss_real.numpy()),
                            float(d_val_acc_real.numpy()),
                            float(d_cls_acc_real.numpy())
                        ]

                        # Accumulate metrics
                        epoch_loss += d_loss_real[0]
                        epoch_validity_loss += d_loss_real[1]
                        epoch_class_loss += d_loss_real[2]
                        epoch_validity_acc += d_loss_real[3]
                        epoch_class_acc += d_loss_real[4]
                    else:
                        # Train with only validity labels using custom training step
                        valid_smooth = tf.ones((self.batch_size, 1)) * (1 - valid_smoothing_factor)

                        d_total_real, d_val_acc_real = \
                            self.train_discriminator_step_validity_only(real_data, valid_smooth)

                        # Package losses for compatibility with existing metrics tracking
                        d_loss_real = [
                            float(d_total_real.numpy()),
                            float(d_val_acc_real.numpy())
                        ]

                        # Accumulate metrics
                        epoch_loss += d_loss_real[0]
                        epoch_validity_loss += d_loss_real[0]
                        epoch_validity_acc += d_loss_real[1]

                # Print progress every few steps
                if step % max(1, actual_steps // 10) == 0:
                    print(f"Step {step}/{actual_steps} - D loss: {epoch_loss / (step + 1):.4f}")

            # ───────────────────────────────────────────────────────────────────
            # EPOCH SUMMARY
            # ───────────────────────────────────────────────────────────────────
            step_count = actual_steps
            avg_loss = epoch_loss / step_count
            avg_validity_loss = epoch_validity_loss / step_count
            avg_validity_acc = epoch_validity_acc / step_count

            # Log training metrics
            self.logger.info("Training Discriminator (REAL DATA ONLY)")

            if self.use_class_labels:
                avg_class_loss = epoch_class_loss / step_count
                avg_class_acc = epoch_class_acc / step_count

                self.logger.info(
                    f"Discriminator Loss: {avg_loss:.4f} | Validity Loss: {avg_validity_loss:.4f} | Class Loss: {avg_class_loss:.4f}")
                self.logger.info(
                    f"Validity Binary Accuracy: {avg_validity_acc * 100:.2f}%")
                self.logger.info(
                    f"Class Categorical Accuracy: {avg_class_acc * 100:.2f}%")

                # Collect discriminator metrics with class information
                d_metrics = {
                    "Total Loss": f"{avg_loss:.4f}",
                    "Validity Loss": f"{avg_validity_loss:.4f}",
                    "Class Loss": f"{avg_class_loss:.4f}",
                    "Validity Binary Accuracy": f"{avg_validity_acc * 100:.2f}%",
                    "Class Categorical Accuracy": f"{avg_class_acc * 100:.2f}%"
                }
            else:
                self.logger.info(
                    f"Discriminator Loss: {avg_loss:.4f} (Validity Loss)")
                self.logger.info(
                    f"Validity Binary Accuracy: {avg_validity_acc * 100:.2f}%")

                # Collect discriminator metrics without class information
                d_metrics = {
                    "Total Loss": f"{avg_loss:.4f}",
                    "Validity Loss": f"{avg_validity_loss:.4f}",
                    "Validity Binary Accuracy": f"{avg_validity_acc * 100:.2f}%"
                }

            # Store metrics history
            d_metrics_history.append(avg_loss)

            # --------------------------
            # Validation every epoch
            # --------------------------
            self.logger.info(f"=== Epoch {epoch + 1} Validation ===")

            d_val_loss, d_val_metrics = self.validation_disc()

            # -- Probabilistic Fusion Validation -- #
            self.logger.info("=== Probabilistic Fusion Validation on Real Data ===")
            fusion_results, fusion_metrics = self.validate_with_probabilistic_fusion(self.x_val, self.y_val)
            self.logger.info(f"Probabilistic Fusion Accuracy: {fusion_metrics['accuracy'] * 100:.2f}%")

            # Log distribution of classifications
            self.logger.info(f"Predicted Class Distribution: {fusion_metrics['predicted_class_distribution']}")

            # Log the metrics for this epoch
            self.log_epoch_metrics(epoch, d_val_metrics, fusion_metrics)
            self.logger.info(
                f"Epoch {epoch + 1}: D Loss: {avg_loss:.4f}, D Validity Acc: {avg_validity_acc * 100:.2f}%")

            # --------------------------
            # Early Stopping Check
            # --------------------------
            current_fusion_accuracy = fusion_metrics['accuracy']

            # Check if current accuracy is better than best (with min_delta threshold)
            if current_fusion_accuracy > best_fusion_accuracy + self.min_delta:
                # Improvement detected
                best_fusion_accuracy = current_fusion_accuracy
                best_epoch = epoch
                patience_counter = 0

                # Save best model weights (copy to avoid reference issues)
                best_weights = [w.copy() for w in self.discriminator.get_weights()]

                self.logger.info(f"✓ New best model! Fusion accuracy improved to {best_fusion_accuracy * 100:.2f}% at epoch {best_epoch}")
                self.logger.info(f"  Model weights saved. Patience counter reset to 0/{self.early_stopping_patience}")
            else:
                # No improvement
                patience_counter += 1
                self.logger.info(f"⚠ No improvement in fusion accuracy (best: {best_fusion_accuracy * 100:.2f}% at epoch {best_epoch})")
                self.logger.info(f"  Patience counter: {patience_counter}/{self.early_stopping_patience}")

                # Check if we should stop
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    self.logger.info(f"Best fusion accuracy: {best_fusion_accuracy * 100:.2f}% at epoch {best_epoch}")
                    self.logger.info(f"Restoring best model weights from epoch {best_epoch}...")

                    # Restore best weights
                    if best_weights is not None:
                        self.discriminator.set_weights(best_weights)
                        self.logger.info("✓ Best model weights restored successfully")

                    # Log training completion with early stopping
                    self.logger.info("=" * 50)
                    self.logger.info("TRAINING COMPLETED (EARLY STOPPED)")
                    self.logger.info("=" * 50)
                    self.logger.info(f"Best probabilistic fusion accuracy: {best_fusion_accuracy * 100:.2f}%")
                    self.logger.info(f"Best model from epoch: {best_epoch}")
                    self.logger.info(f"Total epochs trained: {epoch + 1}")
                    self.logger.info(f"Training stopped early due to no improvement for {self.early_stopping_patience} epochs")
                    self.logger.info("=" * 50 + "\n")

                    # Return early
                    return self.discriminator.get_weights(), len(self.x_train), {
                        "best_fusion_accuracy": best_fusion_accuracy,
                        "best_epoch": best_epoch,
                        "total_epochs_trained": epoch + 1,
                        "early_stopped": True
                    }

        # Training completed all epochs
        self.logger.info("=" * 50)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 50)
        self.logger.info(f"Best probabilistic fusion accuracy: {best_fusion_accuracy * 100:.2f}%")
        self.logger.info(f"Best model from epoch: {best_epoch}")
        self.logger.info(f"Total epochs trained: {self.epochs}")
        self.logger.info("Training completed all epochs")
        self.logger.info("=" * 50 + "\n")

        # Restore best weights if we have them
        if best_weights is not None:
            self.discriminator.set_weights(best_weights)
            self.logger.info(f"✓ Best model weights from epoch {best_epoch} restored")

        return self.discriminator.get_weights(), len(self.x_train), {
            "best_fusion_accuracy": best_fusion_accuracy,
            "best_epoch": best_epoch,
            "total_epochs_trained": self.epochs,
            "early_stopped": False
        }

    #########################################################################
    #                          VALIDATION METHODS                           #
    #########################################################################
    def validation_disc(self):
        """
        Evaluate the discriminator on the validation set using real data.
        Uses custom evaluation helper for consistency with training approach.
        For federated learning, we focus on real data validation (no generator available).
        Returns the average total loss and a metrics dictionary.
        """
        # ═══════════════════════════════════════════════════════════════════════
        # VALIDATION DATA PREPARATION
        # ═══════════════════════════════════════════════════════════════════════
        val_valid_labels = np.ones((len(self.x_val), 1))

        # ═══════════════════════════════════════════════════════════════════════
        # EVALUATE ON REAL VALIDATION DATA
        # ═══════════════════════════════════════════════════════════════════════
        if self.use_class_labels:
            # Ensure y_val is one-hot encoded if needed
            if self.y_val.ndim == 1 or self.y_val.shape[1] != self.num_classes:
                y_val_onehot = tf.one_hot(self.y_val, depth=self.num_classes)
            else:
                y_val_onehot = self.y_val

            # Use custom evaluation helper
            d_total, d_val_loss, d_cls_loss, d_val_acc, d_cls_acc = \
                self.evaluate_discriminator(self.x_val, y_val_onehot, val_valid_labels)

            # Convert to float for logging
            d_loss_real = [
                float(d_total.numpy()),
                float(d_val_loss.numpy()),
                float(d_cls_loss.numpy()),
                float(d_val_acc.numpy()),
                float(d_cls_acc.numpy())
            ]

            # ─── Log Metrics ───
            self.logger.info("Validation Discriminator Evaluation (Real Data Only):")
            self.logger.info(
                f"Real Data -> Total Loss: {d_loss_real[0]:.4f}, "
                f"Validity Loss: {d_loss_real[1]:.4f}, "
                f"Class Loss: {d_loss_real[2]:.4f}, "
                f"Validity Binary Accuracy: {d_loss_real[3] * 100:.2f}%, "
                f"Class Categorical Accuracy: {d_loss_real[4] * 100:.2f}%"
            )

            # ─── Create Metrics Dictionary ───
            metrics = {
                "Real Total Loss": f"{d_loss_real[0]:.4f}",
                "Real Validity Loss": f"{d_loss_real[1]:.4f}",
                "Real Class Loss": f"{d_loss_real[2]:.4f}",
                "Real Validity Binary Accuracy": f"{d_loss_real[3] * 100:.2f}%",
                "Real Class Categorical Accuracy": f"{d_loss_real[4] * 100:.2f}%"
            }

            avg_total_loss = d_loss_real[0]

        else:
            # Without class labels, pass dummy labels (zeros)
            dummy_labels = tf.zeros((len(self.x_val), 2))  # Dummy for non-class-label mode

            # Use custom evaluation helper
            d_total, d_val_loss, d_cls_loss, d_val_acc, d_cls_acc = \
                self.evaluate_discriminator(self.x_val, dummy_labels, val_valid_labels)

            # Convert to float for logging
            d_loss_real = [
                float(d_total.numpy()),
                float(d_val_acc.numpy())
            ]

            # ─── Log Metrics ───
            self.logger.info("Validation Discriminator Evaluation (Real Data Only):")
            self.logger.info(
                f"Real Data -> Loss: {d_loss_real[0]:.4f}, "
                f"Binary Accuracy: {d_loss_real[1] * 100:.2f}%"
            )

            # ─── Create Metrics Dictionary ───
            metrics = {
                "Real Loss": f"{d_loss_real[0]:.4f}",
                "Real Binary Accuracy": f"{d_loss_real[1] * 100:.2f}%"
            }

            avg_total_loss = d_loss_real[0]

        return avg_total_loss, metrics

#########################################################################
#                          EVALUATION METHODS                          #
#########################################################################
    def evaluate(self, parameters, config):
        """
        Evaluate the discriminator on test data using custom evaluation helper.
        For FL discriminator clients, evaluates on real data only (no generator available).

        Parameters:
        -----------
        parameters : list
            Model weights from the FL server
        config : dict
            Configuration from FL server
        """
        # ═══════════════════════════════════════════════════════════════════════
        # FL SERVER WEIGHT INITIALIZATION
        # ═══════════════════════════════════════════════════════════════════════
        self.discriminator.set_weights(parameters)

        # ═══════════════════════════════════════════════════════════════════════
        # TEST DATA PREPARATION
        # ═══════════════════════════════════════════════════════════════════════
        X_test = self.x_test
        y_test = self.y_test

        self.logger.info("-- Evaluating Discriminator --")

        # ═══════════════════════════════════════════════════════════════════════
        # DISCRIMINATOR EVALUATION
        # ═══════════════════════════════════════════════════════════════════════
        test_valid_labels = np.ones((len(X_test), 1))

        if self.use_class_labels:
            # ─── Ensure One-Hot Encoding ───
            if y_test.ndim == 1 or y_test.shape[1] != self.num_classes:
                y_test_onehot = tf.one_hot(y_test, depth=self.num_classes)
            else:
                y_test_onehot = y_test

            # ─── Use Custom Evaluation Helper ───
            d_total, d_val_loss, d_cls_loss, d_val_acc, d_cls_acc = \
                self.evaluate_discriminator(X_test, y_test_onehot, test_valid_labels)

            # ─── Extract Metrics ───
            d_loss_total = float(d_total.numpy())
            d_loss_validity = float(d_val_loss.numpy())
            d_loss_class = float(d_cls_loss.numpy())
            d_validity_bin_acc = float(d_val_acc.numpy())
            d_class_cat_acc = float(d_cls_acc.numpy())

            # ─── Create Metrics Dictionary ───
            d_eval_metrics = {
                "Loss": f"{d_loss_total:.4f}",
                "Validity Loss": f"{d_loss_validity:.4f}",
                "Class Loss": f"{d_loss_class:.4f}",
                "Validity Binary Accuracy": f"{d_validity_bin_acc * 100:.2f}%",
                "Class Categorical Accuracy": f"{d_class_cat_acc * 100:.2f}%"
            }

            # ─── Log Results ───
            self.logger.info(
                f"Discriminator Total Loss: {d_loss_total:.4f} | Validity Loss: {d_loss_validity:.4f} | Class Loss: {d_loss_class:.4f}"
            )
            self.logger.info(
                f"Validity Binary Accuracy: {d_validity_bin_acc * 100:.2f}%"
            )
            self.logger.info(
                f"Class Categorical Accuracy: {d_class_cat_acc * 100:.2f}%"
            )

        else:
            # ─── Without Class Labels ───
            dummy_labels = tf.zeros((len(X_test), 2))  # Dummy for non-class-label mode

            # ─── Use Custom Evaluation Helper ───
            d_total, d_val_loss, d_cls_loss, d_val_acc, d_cls_acc = \
                self.evaluate_discriminator(X_test, dummy_labels, test_valid_labels)

            # ─── Extract Metrics ───
            d_loss_total = float(d_total.numpy())
            d_validity_bin_acc = float(d_val_acc.numpy())

            # ─── Create Metrics Dictionary ───
            d_eval_metrics = {
                "Loss": f"{d_loss_total:.4f}",
                "Validity Binary Accuracy": f"{d_validity_bin_acc * 100:.2f}%"
            }

            # ─── Log Results ───
            self.logger.info(
                f"Discriminator Loss: {d_loss_total:.4f}"
            )
            self.logger.info(
                f"Validity Binary Accuracy: {d_validity_bin_acc * 100:.2f}%"
            )

            d_class_cat_acc = 0.0  # Set default for return value

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
        self.log_evaluation_metrics(d_eval_metrics, fusion_metrics)

        # ═══════════════════════════════════════════════════════════════════════
        # FL RETURN
        # ═══════════════════════════════════════════════════════════════════════
        return float(d_loss_total), len(self.x_test), {
            "accuracy": float(d_validity_bin_acc) if not self.use_class_labels else float(d_class_cat_acc)}

#########################################################################
#                           MODEL SAVING METHODS                       #
#########################################################################
    def save(self, save_name):
        # Save each submodel separately
        self.discriminator.save(f"../../../../../../ModelArchive/discriminator_fed_ACGAN_{save_name}.h5")

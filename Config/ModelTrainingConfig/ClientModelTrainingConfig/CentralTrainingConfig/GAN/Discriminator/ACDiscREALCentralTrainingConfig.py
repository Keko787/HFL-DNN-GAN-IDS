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
#                                               FL-GAN TRAINING Setup                                         #
################################################################################################################

class CentralACDiscREAL:
    def __init__(self, discriminator, generator, nids, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 noise_dim, latent_dim, num_classes, input_dim, epochs, steps_per_epoch, learning_rate,
                 log_file="training.log", use_class_labels=True):
        # -- models
        self.generator = generator
        self.discriminator = discriminator
        self.nids = nids

        # -- I/O Specs for models
        self.batch_size = BATCH_SIZE
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.use_class_labels = use_class_labels  # Whether to use class labels in training

        # -- training duration
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        # -- Data
        # Features
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        # Categorical Labels
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        # -- Setup Logging
        self.setup_logger(log_file)

        # -- Optimizers
        # LR decay
        lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.98, staircase=True)

        # Compile optimizer
        self.disc_optimizer = Adam(learning_rate=lr_schedule_disc, beta_1=0.5, beta_2=0.999)

        print("Discriminator Output:", self.discriminator.output_names)

        # -- Model Compilations based on whether we use class labels
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

    # -- logging Functions -- #
    def setup_logger(self, log_file):
        """Set up a logger that records both to a file and to the console."""
        self.logger = logging.getLogger("CentralACDiscREAL")
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

        # Avoid adding duplicate handlers if logger already has them.
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def log_model_settings(self):
        """Logs model names, structures, and hyperparameters."""
        self.logger.info("=== Model Settings ===")

        self.logger.info("Generator Model Summary:")
        generator_summary = []
        self.generator.summary(print_fn=lambda x: generator_summary.append(x))
        for line in generator_summary:
            self.logger.info(line)

        self.logger.info("Discriminator Model Summary:")
        discriminator_summary = []
        self.discriminator.summary(print_fn=lambda x: discriminator_summary.append(x))
        for line in discriminator_summary:
            self.logger.info(line)

        if self.nids is not None:
            self.logger.info("NIDS Model Summary:")
            nids_summary = []
            self.nids.summary(print_fn=lambda x: nids_summary.append(x))
            for line in nids_summary:
                self.logger.info(line)
        else:
            self.logger.info("NIDS Model is not defined.")

        self.logger.info("--- Hyperparameters ---")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Noise Dimension: {self.noise_dim}")
        self.logger.info(f"Latent Dimension: {self.latent_dim}")
        self.logger.info(f"Number of Classes: {self.num_classes}")
        self.logger.info(f"Input Dimension: {self.input_dim}")
        self.logger.info(f"Epochs: {self.epochs}")
        self.logger.info(f"Steps per Epoch: {self.steps_per_epoch}")
        self.logger.info(f"Learning Rate (Discriminator): {self.disc_optimizer.learning_rate}")
        self.logger.info(f"Using Class Labels: {self.use_class_labels}")
        self.logger.info("=" * 50)

    def log_epoch_metrics(self, epoch, d_metrics, nids_metrics=None, fusion_metrics=None):
        """Logs a formatted summary of the metrics for this epoch."""
        self.logger.info(f"=== Epoch {epoch} Metrics Summary ===")
        self.logger.info("Discriminator Metrics:")
        for key, value in d_metrics.items():
            self.logger.info(f"  {key}: {value}")
        if nids_metrics is not None:
            self.logger.info("NIDS Metrics:")
            for key, value in nids_metrics.items():
                self.logger.info(f"  {key}: {value}")
        if fusion_metrics is not None:
            self.logger.info("Probabilistic Fusion Metrics:")
            for key, value in fusion_metrics.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    def log_evaluation_metrics(self, d_eval, nids_eval=None, fusion_eval=None):
        """Logs a formatted summary of evaluation metrics."""
        self.logger.info("=== Evaluation Metrics Summary ===")
        self.logger.info("Discriminator Evaluation:")
        for key, value in d_eval.items():
            self.logger.info(f"  {key}: {value}")
        if nids_eval is not None:
            self.logger.info("NIDS Evaluation:")
            for key, value in nids_eval.items():
                self.logger.info(f"  {key}: {value}")
        if fusion_eval is not None:
            self.logger.info("Probabilistic Fusion Evaluation:")
            for key, value in fusion_eval.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    # -- Train -- #
    def fit(self, X_train=None, y_train=None):
        """
        Train the discriminator using REAL DATA ONLY.

        Parameters:
        -----------
        X_train : array-like, optional
            Training features. If None, uses self.x_train.
        y_train : array-like, optional
            Training labels. If None, uses self.y_train.
        """
        if X_train is None or y_train is None:
            X_train = self.x_train
            y_train = self.y_train

        # Log model settings at the start
        self.log_model_settings()

        # -- Apply label smoothing -- #
        # Create smoothed labels for real data (only training with real data)
        valid_smoothing_factor = 0.08
        valid_smooth = tf.ones((self.batch_size, 1)) * (1 - valid_smoothing_factor)

        self.logger.info(f"Using valid label smoothing with factor: {valid_smoothing_factor}")
        self.logger.info("Training discriminator using REAL DATA ONLY")

        # -- Initialize metrics tracking -- #
        d_metrics_history = []

        # -- Training Loop -- #
        for epoch in range(self.epochs):
            print("Discriminator Metrics:", self.discriminator.metrics_names)

            print(f'\n=== Epoch {epoch + 1}/{self.epochs} ===\n')
            self.logger.info(f'=== Epoch {epoch + 1}/{self.epochs} ===')

            # --------------------------
            # Train Discriminator (REAL DATA ONLY)
            # --------------------------

            # Determine how many steps per epoch based on batch size
            actual_steps = min(self.steps_per_epoch, len(X_train) // self.batch_size)

            # Track metrics across all steps
            epoch_loss = 0
            epoch_validity_loss = 0
            epoch_class_loss = 0
            epoch_validity_acc = 0
            epoch_class_acc = 0

            for step in range(actual_steps):
                # Sample a batch of real data
                idx = tf.random.shuffle(tf.range(len(X_train)))[:self.batch_size]
                real_data = tf.gather(X_train, idx)
                real_labels = tf.gather(y_train, idx)

                # Prepare the training data based on whether we use class labels
                if self.use_class_labels:
                    # Ensure labels are one-hot encoded
                    if len(real_labels.shape) == 1:
                        real_labels_onehot = tf.one_hot(tf.cast(real_labels, tf.int32), depth=self.num_classes)
                    else:
                        real_labels_onehot = real_labels

                    # Train discriminator on real data with both validity and class labels
                    d_loss_real = self.discriminator.train_on_batch(real_data, [valid_smooth, real_labels_onehot])
                else:
                    # Train discriminator on real data with only validity labels
                    d_loss_real = self.discriminator.train_on_batch(real_data, valid_smooth)

                # Accumulate metrics based on whether we use class labels
                if self.use_class_labels:
                    # With class labels, we have more metrics to track
                    epoch_loss += d_loss_real[0]
                    epoch_validity_loss += d_loss_real[1]
                    epoch_class_loss += d_loss_real[2]
                    epoch_validity_acc += d_loss_real[3]
                    epoch_class_acc += d_loss_real[4]  # Index 4 for categorical accuracy
                else:
                    # Without class labels, we have fewer metrics
                    epoch_loss += d_loss_real[0]
                    epoch_validity_loss += d_loss_real[0]  # Total loss is validity loss
                    epoch_validity_acc += d_loss_real[1]  # Index 1 for binary accuracy

                # Print progress every few steps
                if step % max(1, actual_steps // 10) == 0:
                    print(f"Step {step}/{actual_steps} - D loss: {d_loss_real[0]:.4f}")

            # Calculate average metrics for the epoch
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

            # Validate discriminator on both real and fake data
            d_val_loss, d_val_metrics = self.validation_disc()

            # -- Probabilistic Fusion Validation -- #
            self.logger.info("=== Probabilistic Fusion Validation on Real Data ===")
            fusion_results, fusion_metrics = self.validate_with_probabilistic_fusion(self.x_val, self.y_val)
            self.logger.info(f"Probabilistic Fusion Accuracy: {fusion_metrics['accuracy'] * 100:.2f}%")

            # Log distribution of classifications
            self.logger.info(f"Predicted Class Distribution: {fusion_metrics['predicted_class_distribution']}")

            # -- NIDS Validation -- #
            nids_val_metrics = None
            if self.nids is not None:
                nids_custom_loss, nids_val_metrics = self.validation_NIDS()
                self.logger.info(f"Validation NIDS Custom Loss: {nids_custom_loss:.4f}")

            # Log the metrics for this epoch
            self.log_epoch_metrics(epoch, d_val_metrics, nids_val_metrics, fusion_metrics)
            self.logger.info(
                f"Epoch {epoch + 1}: D Loss: {avg_loss:.4f}, D Validity Acc: {avg_validity_acc * 100:.2f}%")

        # Return the training history for analysis
        return {
            "discriminator_loss": d_metrics_history
        }

    def nids_loss(self, real_output, fake_output):
        """
        Compute the NIDS loss on real and fake samples.
        For real samples, the target is 1 (benign), and for fake samples, 0 (attack).
        Returns a scalar loss value.
        """
        # define labels
        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)

        # define loss function
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # calculate outputs
        real_loss = bce(real_labels, real_output)
        fake_loss = bce(fake_labels, fake_output)

        # sum up total loss
        total_loss = real_loss + fake_loss
        return total_loss.numpy()

    # -- Probabilistic Fusion Methods -- #
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

    # -- Validate -- #
    def validation_disc(self):
        """
        Evaluate the discriminator on the validation set.
        First, evaluate on real data (with labels = 1) and then on fake data (labels = 0).
        Prints and returns the average total loss along with a metrics dictionary.
        """
        # --- Evaluate on real validation data ---
        val_valid_labels = np.ones((len(self.x_val), 1))

        # Different evaluation approach based on whether we use class labels
        if self.use_class_labels:
            # Ensure y_val is one-hot encoded if needed
            if self.y_val.ndim == 1 or self.y_val.shape[1] != self.num_classes:
                y_val_onehot = tf.one_hot(self.y_val, depth=self.num_classes)
            else:
                y_val_onehot = self.y_val

            d_loss_real = self.discriminator.evaluate(
                self.x_val, [val_valid_labels, y_val_onehot], verbose=0
            )
        else:
            # Only validate with validity labels
            d_loss_real = self.discriminator.evaluate(
                self.x_val, val_valid_labels, verbose=0
            )

        # --- Evaluate on generated (fake) data ---
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        fake_labels = tf.random.uniform(
            (len(self.x_val),), minval=0, maxval=self.num_classes, dtype=tf.int32
        )
        fake_valid_labels = np.zeros((len(self.x_val), 1))

        # Generate fake data using the generator
        generated_data = self.generator.predict([noise, fake_labels])

        # Evaluate with different targets based on whether we use class labels
        if self.use_class_labels:
            fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)
            d_loss_fake = self.discriminator.evaluate(
                generated_data, [fake_valid_labels, fake_labels_onehot], verbose=0
            )
        else:
            d_loss_fake = self.discriminator.evaluate(
                generated_data, fake_valid_labels, verbose=0
            )

        # --- Compute average loss ---
        avg_total_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

        self.logger.info("Validation Discriminator Evaluation:")

        # Log metrics based on whether we use class labels
        if self.use_class_labels:
            # Log for real data with class metrics
            self.logger.info(
                f"Real Data -> Total Loss: {d_loss_real[0]:.4f}, "
                f"Validity Loss: {d_loss_real[1]:.4f}, "
                f"Class Loss: {d_loss_real[2]:.4f}, "
                f"Validity Binary Accuracy: {d_loss_real[3] * 100:.2f}%, "
                f"Class Categorical Accuracy: {d_loss_real[4] * 100:.2f}%"
            )

            # Log for fake data with class metrics
            self.logger.info(
                f"Fake Data -> Total Loss: {d_loss_fake[0]:.4f}, "
                f"Validity Loss: {d_loss_fake[1]:.4f}, "
                f"Class Loss: {d_loss_fake[2]:.4f}, "
                f"Validity Binary Accuracy: {d_loss_fake[3] * 100:.2f}%, "
                f"Class Categorical Accuracy: {d_loss_fake[4] * 100:.2f}%"
            )
        else:
            # Log for real data without class metrics
            self.logger.info(
                f"Real Data -> Loss: {d_loss_real[0]:.4f}, "
                f"Binary Accuracy: {d_loss_real[1] * 100:.2f}%"
            )

            # Log for fake data without class metrics
            self.logger.info(
                f"Fake Data -> Loss: {d_loss_fake[0]:.4f}, "
                f"Binary Accuracy: {d_loss_fake[1] * 100:.2f}%"
            )

        self.logger.info(f"Average Discriminator Loss: {avg_total_loss:.4f}")

        # Create metrics dictionary based on whether we use class labels
        if self.use_class_labels:
            metrics = {
                "Real Total Loss": f"{d_loss_real[0]:.4f}",
                "Real Validity Loss": f"{d_loss_real[1]:.4f}",
                "Real Class Loss": f"{d_loss_real[2]:.4f}",
                "Real Validity Binary Accuracy": f"{d_loss_real[3] * 100:.2f}%",
                "Real Class Categorical Accuracy": f"{d_loss_real[4] * 100:.2f}%",
                "Fake Total Loss": f"{d_loss_fake[0]:.4f}",
                "Fake Validity Loss": f"{d_loss_fake[1]:.4f}",
                "Fake Class Loss": f"{d_loss_fake[2]:.4f}",
                "Fake Validity Binary Accuracy": f"{d_loss_fake[3] * 100:.2f}%",
                "Fake Class Categorical Accuracy": f"{d_loss_fake[4] * 100:.2f}%",
                "Average Total Loss": f"{avg_total_loss:.4f}"
            }
        else:
            metrics = {
                "Real Loss": f"{d_loss_real[0]:.4f}",
                "Real Binary Accuracy": f"{d_loss_real[1] * 100:.2f}%",
                "Fake Loss": f"{d_loss_fake[0]:.4f}",
                "Fake Binary Accuracy": f"{d_loss_fake[1] * 100:.2f}%",
                "Average Total Loss": f"{avg_total_loss:.4f}"
            }
        return avg_total_loss, metrics

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

        # --- Prepare Real Data ---
        X_real = self.x_val
        y_real = np.ones((len(self.x_val),), dtype="int32")  # Real samples labeled 1

        # --- Generate Fake Data ---
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        fake_labels = tf.random.uniform(
            (len(self.x_val),), minval=0, maxval=self.num_classes, dtype=tf.int32
        )
        generated_samples = self.generator.predict([noise, fake_labels])
        X_fake = generated_samples  # Use directly without scaling
        y_fake = np.zeros((len(self.x_val),), dtype="int32")  # Fake samples labeled 0

        # --- Compute custom NIDS loss ---
        real_output = self.nids.predict(X_real)
        fake_output = self.nids.predict(X_fake)
        custom_nids_loss = self.nids_loss(real_output, fake_output)

        # --- Evaluate on the Combined Dataset ---
        X_combined = np.vstack([X_real, X_fake])
        y_combined = np.hstack([y_real, y_fake])
        nids_eval = self.nids.evaluate(X_combined, y_combined, verbose=0)
        # Expected order: [loss, accuracy, precision, recall, auc, logcosh]

        # --- Compute Additional Metrics ---
        y_pred_probs = self.nids.predict(X_combined)
        y_pred = (y_pred_probs > 0.5).astype("int32")
        f1 = f1_score(y_combined, y_pred)
        class_report = classification_report(
            y_combined, y_pred, target_names=["Attack (Fake)", "Benign (Real)"]
        )

        self.logger.info("Validation NIDS Evaluation with Augmented Data:")
        self.logger.info(f"Custom NIDS Loss (Real vs Fake): {custom_nids_loss:.4f}")
        self.logger.info(f"Overall NIDS Loss: {nids_eval[0]:.4f}, Accuracy: {nids_eval[1]:.4f}, "
                         f"Precision: {nids_eval[2]:.4f}, Recall: {nids_eval[3]:.4f}, "
                         f"AUC: {nids_eval[4]:.4f}, LogCosh: {nids_eval[5]:.4f}")
        self.logger.info("Classification Report:")
        self.logger.info(class_report)
        self.logger.info(f"F1 Score: {f1:.4f}")

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

    # -- Evaluate -- #
    def evaluate(self, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            X_test = self.x_test
            y_test = self.y_test

        # --------------------------
        # Test Discriminator
        # --------------------------
        self.logger.info("-- Evaluating Discriminator --")

        # Evaluate on real test data based on whether we use class labels
        if self.use_class_labels:
            # Convert y_test to one-hot if needed
            if y_test.ndim == 1 or y_test.shape[1] != self.num_classes:
                y_test_onehot = tf.one_hot(y_test, depth=self.num_classes)
            else:
                y_test_onehot = y_test

            results = self.discriminator.evaluate(X_test, [tf.ones((len(y_test), 1)), y_test_onehot], verbose=0)

            # Using indices from the evaluate output with class labels:
            d_loss_total = results[0]
            d_loss_validity = results[1]
            d_loss_class = results[2]
            d_validity_bin_acc = results[3]
            d_class_cat_acc = results[4]

            d_eval_metrics = {
                "Loss on Real Data": f"{d_loss_total:.4f}",
                "Validity Loss": f"{d_loss_validity:.4f}",
                "Class Loss": f"{d_loss_class:.4f}",
                "Validity Binary Accuracy": f"{d_validity_bin_acc * 100:.2f}%",
                "Class Categorical Accuracy": f"{d_class_cat_acc * 100:.2f}%"
            }

            self.logger.info(
                f"Discriminator on Real Data -> Total Loss: {d_loss_total:.4f} | Validity Loss: {d_loss_validity:.4f} | Class Loss: {d_loss_class:.4f}"
            )
            self.logger.info(
                f"Validity Binary Accuracy: {d_validity_bin_acc * 100:.2f}%"
            )
            self.logger.info(
                f"Class Categorical Accuracy: {d_class_cat_acc * 100:.2f}%"
            )
        else:
            # Evaluate with only validity labels
            results = self.discriminator.evaluate(X_test, tf.ones((len(y_test), 1)), verbose=0)

            # Using indices from the evaluate output without class labels:
            d_loss_total = results[0]
            d_validity_bin_acc = results[1]

            d_eval_metrics = {
                "Loss on Real Data": f"{d_loss_total:.4f}",
                "Validity Binary Accuracy": f"{d_validity_bin_acc * 100:.2f}%"
            }

            self.logger.info(
                f"Discriminator on Real Data -> Loss: {d_loss_total:.4f}"
            )
            self.logger.info(
                f"Validity Binary Accuracy: {d_validity_bin_acc * 100:.2f}%"
            )

        # Also evaluate on fake data to assess overall performance
        noise = tf.random.normal((len(y_test), self.latent_dim))
        fake_labels = tf.random.uniform((len(y_test),), minval=0, maxval=self.num_classes, dtype=tf.int32)
        fake_valid_labels = np.zeros((len(y_test), 1))
        generated_data = self.generator.predict([noise, fake_labels])

        # Evaluate fake data based on whether we use class labels
        if self.use_class_labels:
            fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)
            fake_results = self.discriminator.evaluate(generated_data, [fake_valid_labels, fake_labels_onehot],
                                                       verbose=0)

            # Add fake data evaluation to metrics with class information
            d_fake_loss_total = fake_results[0]
            d_fake_validity_bin_acc = fake_results[3]
            d_fake_class_cat_acc = fake_results[4]

            self.logger.info(
                f"Discriminator on Fake Data -> Total Loss: {d_fake_loss_total:.4f}, "
                f"Validity Binary Accuracy: {d_fake_validity_bin_acc * 100:.2f}%, "
                f"Class Categorical Accuracy: {d_fake_class_cat_acc * 100:.2f}%"
            )

            # Add fake data metrics to evaluation metrics
            d_eval_metrics["Loss on Fake Data"] = f"{d_fake_loss_total:.4f}"
            d_eval_metrics["Fake Validity Binary Accuracy"] = f"{d_fake_validity_bin_acc * 100:.2f}%"
            d_eval_metrics["Fake Class Categorical Accuracy"] = f"{d_fake_class_cat_acc * 100:.2f}%"
        else:
            fake_results = self.discriminator.evaluate(generated_data, fake_valid_labels, verbose=0)

            # Add fake data evaluation to metrics without class information
            d_fake_loss_total = fake_results[0]
            d_fake_validity_bin_acc = fake_results[1]

            self.logger.info(
                f"Discriminator on Fake Data -> Loss: {d_fake_loss_total:.4f}, "
                f"Binary Accuracy: {d_fake_validity_bin_acc * 100:.2f}%"
            )

            # Add fake data metrics to evaluation metrics
            d_eval_metrics["Loss on Fake Data"] = f"{d_fake_loss_total:.4f}"
            d_eval_metrics["Fake Binary Accuracy"] = f"{d_fake_validity_bin_acc * 100:.2f}%"

        # --------------------------
        # Test NIDS
        # --------------------------
        nids_eval_metrics = None
        if self.nids is not None:
            self.logger.info("-- Evaluating NIDS --")
            # Prepare real test data (labeled as benign, 1)
            X_real = X_test
            y_real = np.ones((len(X_test),), dtype="int32")

            # Generate fake test data (labeled as attack, 0)
            noise = tf.random.normal((len(X_test), self.latent_dim))
            fake_labels = tf.random.uniform((len(X_test),), minval=0, maxval=self.num_classes, dtype=tf.int32)
            generated_samples = self.generator.predict([noise, fake_labels])
            X_fake = generated_samples
            y_fake = np.zeros((len(X_test),), dtype="int32")

            # Compute custom NIDS loss on real and fake outputs
            real_output = self.nids.predict(X_real)
            fake_output = self.nids.predict(X_fake)
            custom_nids_loss = self.nids_loss(real_output, fake_output)

            # Combine real and fake data for evaluation
            X_combined = np.vstack([X_real, X_fake])
            y_combined = np.hstack([y_real, y_fake])
            nids_eval_results = self.nids.evaluate(X_combined, y_combined, verbose=0)
            # Expected order: [loss, accuracy, precision, recall, auc, logcosh]

            # Compute additional metrics
            y_pred_probs = self.nids.predict(X_combined)
            y_pred = (y_pred_probs > 0.5).astype("int32")
            f1 = f1_score(y_combined, y_pred)
            class_report = classification_report(
                y_combined, y_pred, target_names=["Attack (Fake)", "Benign (Real)"]
            )

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
            self.logger.info(f"NIDS Custom Loss: {custom_nids_loss:.4f}")
            self.logger.info(
                f"NIDS Evaluation -> Loss: {nids_eval_results[0]:.4f}, Accuracy: {nids_eval_results[1]:.4f}, "
                f"Precision: {nids_eval_results[2]:.4f}, Recall: {nids_eval_results[3]:.4f}, "
                f"AUC: {nids_eval_results[4]:.4f}, LogCosh: {nids_eval_results[5]:.4f}")
            self.logger.info("NIDS Classification Report:")
            self.logger.info(class_report)

        # --------------------------
        # Probabilistic Fusion Evaluation
        # --------------------------
        self.logger.info("-- Evaluating Probabilistic Fusion --")
        fusion_results, fusion_metrics = self.validate_with_probabilistic_fusion(X_test, y_test)
        self.logger.info(f"Probabilistic Fusion Accuracy: {fusion_metrics['accuracy'] * 100:.2f}%")
        self.logger.info(f"Predicted Class Distribution: {fusion_metrics['predicted_class_distribution']}")

        # Log the overall evaluation metrics
        self.log_evaluation_metrics(d_eval_metrics, nids_eval_metrics, fusion_metrics)

    def save(self, save_name):
        # Save each submodel separately
        self.discriminator.save(f"../../../../../ModelArchive/discriminator_local_ACGAN_{save_name}.h5")
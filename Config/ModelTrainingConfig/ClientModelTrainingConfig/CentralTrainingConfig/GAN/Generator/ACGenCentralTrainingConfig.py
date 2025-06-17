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

class CentralACGenerator:
    def __init__(self, discriminator, generator, nids, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 noise_dim, latent_dim, num_classes, input_dim, epochs, steps_per_epoch, log_file="training.log"):
        #-- models
        self.generator = generator
        self.discriminator = discriminator
        self.nids = nids

        #-- I/O Specs for models
        self.batch_size = BATCH_SIZE
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_dim = input_dim

        #-- training duration
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        #-- Data
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

        #-- Optimizers
        # LR decay
        lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0002, decay_steps=10000, decay_rate=0.98, staircase=True)

        lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.98, staircase=True)

        # Compile optimizer
        self.gen_optimizer = Adam(learning_rate=lr_schedule_gen, beta_1=0.5, beta_2=0.999)
        self.disc_optimizer = Adam(learning_rate=lr_schedule_disc, beta_1=0.5, beta_2=0.999)

        print("Discriminator Output:", self.discriminator.output_names)

        #-- Model Compilations
        # # Compile Discriminator separately (before freezing)
        # self.discriminator.compile(
        #     loss={'validity': 'binary_crossentropy', 'class': 'categorical_crossentropy'},
        #     optimizer=self.disc_optimizer,
        #     metrics={
        #         'validity': ['accuracy', 'binary_accuracy', 'AUC'],
        #         'class': ['accuracy', 'categorical_accuracy']
        #     }
        # )

        # Freeze Discriminator only for AC-GAN training
        self.discriminator.trainable = False

        # Define AC-GAN (Generator + Frozen Discriminator)
        # I/O
        noise_input = tf.keras.Input(shape=(latent_dim,))
        label_input = tf.keras.Input(shape=(1,), dtype='int32')
        generated_data = self.generator([noise_input, label_input])
        validity, class_pred = self.discriminator(generated_data)
        # Compile Combined Model
        self.ACGAN = Model([noise_input, label_input], [validity, class_pred])

        print("ACGAN Output:", self.ACGAN.output_names)

        self.ACGAN.compile(
            loss={'Discriminator': 'binary_crossentropy', 'Discriminator_1': 'categorical_crossentropy'},
            optimizer=self.gen_optimizer,
            metrics={
                'Discriminator': ['binary_accuracy'],
                'Discriminator_1': ['categorical_accuracy']
            }
        )

    # Saving function for ACGAN
    def setACGAN(self):
        return self.ACGAN

    # -- logging Functions -- #


    def setup_logger(self, log_file):
        """Set up a logger that records both to a file and to the console."""
        self.logger = logging.getLogger("CentralACGan")
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
        self.logger.info(f"Learning Rate (Generator): {self.gen_optimizer.learning_rate}")
        self.logger.info("=" * 50)

    def log_epoch_metrics(self, epoch, d_metrics, g_metrics, nids_metrics=None):
        """Logs a formatted summary of the metrics for this epoch."""
        self.logger.info(f"=== Epoch {epoch} Metrics Summary ===")
        self.logger.info("Generator Metrics:")
        for key, value in g_metrics.items():
            self.logger.info(f"  {key}: {value}")
        if nids_metrics is not None:
            self.logger.info("NIDS Metrics:")
            for key, value in nids_metrics.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    def log_evaluation_metrics(self, d_eval, g_eval, nids_eval=None):
        """Logs a formatted summary of evaluation metrics."""
        self.logger.info("=== Evaluation Metrics Summary ===")
        self.logger.info("Generator Evaluation:")
        for key, value in g_eval.items():
            self.logger.info(f"  {key}: {value}")
        if nids_eval is not None:
            self.logger.info("NIDS Evaluation:")
            for key, value in nids_eval.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    # -- Train -- #
    def fit(self, X_train=None, y_train=None):
        if X_train is None or y_train is None:
            X_train = self.x_train
            y_train = self.y_train

        # Log model settings at the start
        self.log_model_settings()

        # For generator training, we use a slightly different smoothing
        # to keep the generator from becoming too confident
        gen_smoothing_factor = 0.1
        valid_smooth_gen = tf.ones((self.batch_size, 1)) * (1 - gen_smoothing_factor)  # Slightly less than 1.0

        for epoch in range(self.epochs):
            print("Discriminator Metrics:",self.discriminator.metrics_names)
            print("ACGAN Metrics:",self.ACGAN.metrics_names)

            print(f'\n=== Epoch {epoch}/{self.epochs} ===\n')
            self.logger.info(f'=== Epoch {epoch}/{self.epochs} ===')

            # --------------------------
            # Train Generator (AC-GAN)
            # --------------------------

            # Generate noise and label inputs for ACGAN
            noise = tf.random.normal((self.batch_size, self.latent_dim))
            sampled_labels = tf.random.uniform((self.batch_size,), minval=0, maxval=self.num_classes,
                                               dtype=tf.int32)
            sampled_labels_onehot = tf.one_hot(sampled_labels, depth=self.num_classes)

            # Train ACGAN with sampled noise data
            g_loss = self.ACGAN.train_on_batch([noise, sampled_labels], [valid_smooth_gen, sampled_labels_onehot])

            # Collect generator metrics
            g_metrics = {
                "Total Loss": f"{g_loss[0]:.4f}",
                "Validity Loss": f"{g_loss[1]:.4f}",  # This is Discriminator_loss
                "Class Loss": f"{g_loss[2]:.4f}",  # This is Discriminator_1_loss
                "Validity Binary Accuracy": f"{g_loss[3] * 100:.2f}%",  # Discriminator_binary_accuracy
                "Class Categorical Accuracy": f"{g_loss[4] * 100:.2f}%"  # Discriminator_1_categorical_accuracy
            }
            self.logger.info("Training Generator with ACGAN FLOW")
            self.logger.info(
                f"AC-GAN Generator Total Loss: {g_loss[0]:.4f} | Validity Loss: {g_loss[1]:.4f} | Class Loss: {g_loss[2]:.4f}")
            self.logger.info(
                f"Validity Binary Accuracy: {g_loss[3] * 100:.2f}%")
            self.logger.info(
                f"Class Categorical Accuracy: {g_loss[4] * 100:.2f}%")

            # --------------------------
            # Validation every 1 epochs
            # --------------------------
            if epoch % 1 == 0:
                self.logger.info(f"=== Epoch {epoch} Validation ===")
                d_val_loss, d_val_metrics = self.validation_disc()
                g_val_loss, g_val_metrics = self.validation_gen()
                nids_val_metrics = None
                if self.nids is not None:
                    nids_custom_loss, nids_val_metrics = self.validation_NIDS()
                    self.logger.info(f"Validation NIDS Custom Loss: {nids_custom_loss:.4f}")

                # Log the metrics for this epoch using our new logging method
                self.log_epoch_metrics(epoch, d_val_metrics, g_val_metrics, nids_val_metrics)
                self.logger.info(
                    f"Epoch {epoch}: ACGAN Loss: {g_loss[0]:.4f}")

        # -- Loss Calculation -- #

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

    # -- Validate -- #
    def validation_disc(self):
        """
        Evaluate the discriminator on the validation set.
        First, evaluate on real data (with labels = 1) and then on fake data (labels = 0).
        Prints and returns the average total loss along with a metrics dictionary.
        """
        # --- Evaluate on real validation data ---
        val_valid_labels = np.ones((len(self.x_val), 1))

        # Ensure y_val is one-hot encoded if needed
        if self.y_val.ndim == 1 or self.y_val.shape[1] != self.num_classes:
            y_val_onehot = tf.one_hot(self.y_val, depth=self.num_classes)
        else:
            y_val_onehot = self.y_val

        d_loss_real = self.discriminator.evaluate(
            self.x_val, [val_valid_labels, y_val_onehot], verbose=0
        )

        # --- Evaluate on generated (fake) data ---
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        fake_labels = tf.random.uniform(
            (len(self.x_val),), minval=0, maxval=self.num_classes, dtype=tf.int32
        )
        fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)
        fake_valid_labels = np.zeros((len(self.x_val), 1))
        generated_data = self.generator.predict([noise, fake_labels])
        d_loss_fake = self.discriminator.evaluate(
            generated_data, [fake_valid_labels, fake_labels_onehot], verbose=0
        )

        # --- Compute average loss ---
        avg_total_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

        self.logger.info("Validation Discriminator Evaluation:")
        # Log for real data: using all relevant indices
        self.logger.info(
            f"Real Data -> Total Loss: {d_loss_real[0]:.4f}, "
            f"Validity Loss: {d_loss_real[1]:.4f}, "
            f"Class Loss: {d_loss_real[2]:.4f}, "
            f"Validity Binary Accuracy: {d_loss_real[3] * 100:.2f}%, "
            f"Class Categorical Accuracy: {d_loss_real[4] * 100:.2f}%"
        )
        print("-----------")
        self.logger.info(
            f"Fake Data -> Total Loss: {d_loss_fake[0]:.4f}, "
            f"Validity Loss: {d_loss_fake[1]:.4f}, "
            f"Class Loss: {d_loss_fake[2]:.4f}, "
            f"Validity Binary Accuracy: {d_loss_fake[3] * 100:.2f}%, "
            f"Class Categorical Accuracy: {d_loss_fake[4] * 100:.2f}%"
        )
        self.logger.info(f"Average Discriminator Loss: {avg_total_loss:.4f}")

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
        return avg_total_loss, metrics

    def validation_gen(self):
        """
        Evaluate the generator (via the AC-GAN) using a validation batch.
        The generator is evaluated by its ability to “fool” the discriminator.
        Prints and returns the total generator loss along with key metrics.
        """
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        sampled_labels = tf.random.uniform(
            (len(self.x_val),), minval=0, maxval=self.num_classes, dtype=tf.int32
        )
        sampled_labels_onehot = tf.one_hot(sampled_labels, depth=self.num_classes)
        valid_labels = np.ones((len(self.x_val), 1))

        g_loss = self.ACGAN.evaluate(
            [noise, sampled_labels],
            [valid_labels, sampled_labels_onehot],
            verbose=1
        )

        # Log detailed metrics
        self.logger.info("Validation Generator (AC-GAN) Evaluation:")
        self.logger.info(
            f"Total Loss: {g_loss[0]:.4f}, Validity Loss: {g_loss[1]:.4f}, Class Loss: {g_loss[2]:.4f}")
        self.logger.info(
            f"Validity Binary Accuracy: {g_loss[3] * 100:.2f}%")
        self.logger.info(
            f"Class Categorical Accuracy: {g_loss[4] * 100:.2f}%")

        # Create a metrics dictionary with the full set of metrics
        g_metrics = {
            "Total Loss": f"{g_loss[0]:.4f}",
            "Validity Loss": f"{g_loss[1]:.4f}",
            "Class Loss": f"{g_loss[2]:.4f}",
            "Validity Binary Accuracy": f"{g_loss[3] * 100:.2f}%",
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

        # --- Prepare Real Data ---
        X_real = self.x_val
        y_real = np.ones((len(self.x_val),), dtype="int32")  # Real samples labeled 1

        # --- Generate Fake Data ---
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        fake_labels = tf.random.uniform(
            (len(self.x_val),), minval=0, maxval=self.num_classes, dtype=tf.int32
        )
        generated_samples = self.generator.predict([noise, fake_labels])
        X_fake = generated_samples
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
        # run the model
        results = self.discriminator.evaluate(X_test, [tf.ones((len(y_test), 1)), y_test], verbose=0)
        # Using the updated ordering:
        d_loss_total = results[0]
        d_loss_validity = results[1]
        d_loss_class = results[2]
        d_validity_bin_acc = results[3]
        d_class_cat_acc = results[4]

        d_eval_metrics = {
            "Loss": f"{d_loss_total:.4f}",
            "Validity Loss": f"{d_loss_validity:.4f}",
            "Class Loss": f"{d_loss_class:.4f}",
            "Validity Binary Accuracy": f"{d_validity_bin_acc * 100:.2f}%",
            "Class Categorical Accuracy": f"{d_class_cat_acc * 100:.2f}%"
        }
        self.logger.info(
            f"Discriminator Total Loss: {d_loss_total:.4f} | Validity Loss: {d_loss_validity:.4f} | Class Loss: {d_loss_class:.4f}"
        )
        self.logger.info(
            f"Validity Binary Accuracy: {d_validity_bin_acc * 100:.2f}%"
        )
        self.logger.info(
            f"Class Categorical Accuracy: {d_class_cat_acc * 100:.2f}%"
        )

        # --------------------------
        # Test Generator (ACGAN)
        # --------------------------
        self.logger.info("-- Evaluating Generator --")

        # get the noise samples
        noise = tf.random.normal((len(y_test), self.latent_dim))
        sampled_labels = tf.random.uniform((len(y_test),), minval=0, maxval=self.num_classes, dtype=tf.int32)

        # run the model
        g_loss = self.ACGAN.evaluate([noise, sampled_labels],
                                     [tf.ones((len(y_test), 1)),
                                      tf.one_hot(sampled_labels, depth=self.num_classes)],
                                     verbose=1)

        # Using the updated ordering for ACGAN:
        g_loss_total = g_loss[0]
        g_loss_validity = g_loss[1]
        g_loss_class = g_loss[2]
        g_validity_bin_acc = g_loss[3]
        g_class_cat_acc = g_loss[4]

        g_eval_metrics = {
            "Loss": f"{g_loss_total:.4f}",
            "Validity Loss": f"{g_loss_validity:.4f}",
            "Class Loss": f"{g_loss_class:.4f}",
            "Validity Binary Accuracy": f"{g_validity_bin_acc * 100:.2f}%",
            "Class Categorical Accuracy": f"{g_class_cat_acc * 100:.2f}%"
        }
        self.logger.info(
            f"Generator Total Loss: {g_loss_total:.4f} | Validity Loss: {g_loss_validity:.4f} | Class Loss: {g_loss_class:.4f}"
        )
        self.logger.info(
            f"Validity Binary Accuracy: {g_validity_bin_acc * 100:.2f}%"
        )
        self.logger.info(
            f"Class Categorical Accuracy: {g_class_cat_acc * 100:.2f}%"
        )

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

        # Log the overall evaluation metrics using our logging function
        self.log_evaluation_metrics(d_eval_metrics, g_eval_metrics, nids_eval_metrics)

    def save(self, save_name):
        # Save each submodel separately
        self.generator.save(f"../../../../../ModelArchive/generator_local_ACGAN_{save_name}.h5")

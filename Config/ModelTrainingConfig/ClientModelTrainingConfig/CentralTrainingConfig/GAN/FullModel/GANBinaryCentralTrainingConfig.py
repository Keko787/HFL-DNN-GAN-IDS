#########################################################
#    Imports / Env setup                                #
#########################################################

import os
import random
import time
from datetime import datetime
import argparse
import logging


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

def generate_and_save_network_traffic(model, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image.png')
    plt.show()


class CentralBinaryGan:
    def __init__(self, model, nids, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 noise_dim, epochs, steps_per_epoch, learning_rate):

        self.gan = model
        self.nids = nids

        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(self.BATCH_SIZE)
        self.x_val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.BATCH_SIZE)

        lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0002, decay_steps=10000, decay_rate=0.98, staircase=True)

        lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.98, staircase=True)

        self.gen_optimizer = Adam(learning_rate=lr_schedule_gen, beta_1=0.5, beta_2=0.999)
        self.disc_optimizer = Adam(learning_rate=lr_schedule_disc, beta_1=0.5, beta_2=0.999)

        self.generator = self.gan.layers[0]
        self.discriminator = self.gan.layers[1]

        self.disc_accuracy = tf.keras.metrics.BinaryAccuracy(name='disc_accuracy')
        self.disc_precision = tf.keras.metrics.Precision(name='disc_precision')
        self.disc_recall = tf.keras.metrics.Recall(name='disc_recall')

        self.gen_accuracy = tf.keras.metrics.BinaryAccuracy(name='gen_accuracy')
        self.gen_precision = tf.keras.metrics.Precision(name='gen_precision')

        # --- Logger Setup ---
    def setup_logger(self, log_file):
        """Set up a logger that records both to a file and to the console."""
        self.logger = logging.getLogger("Central Gan")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler.
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # Console handler.
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # Avoid duplicate handlers.
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    #-- Metrics--#
    def log_metrics(self, step, disc_loss, gen_loss):
        print(f"Step {step}, D Loss: {disc_loss.numpy():.4f}, G Loss: {gen_loss.numpy():.4f}")
        print(f"Discriminator Metrics -- Accuracy: {self.disc_accuracy.result().numpy():.4f}, "
              f"Precision: {self.disc_precision.result().numpy():.4f}, "
              f"Recall: {self.disc_recall.result().numpy():.4f}")
        print(f"Generator Metrics -- Accuracy: {self.gen_accuracy.result().numpy():.4f}, "
              f"Precision: {self.gen_precision.result().numpy():.4f}")

    def update_metrics(self, real_output=None, fake_output=None):
        if real_output is not None:
            # Update discriminator metrics: real samples are labeled 0, fake samples are labeled 1
            real_labels = tf.zeros_like(real_output)
            fake_labels = tf.ones_like(fake_output)
            all_labels = tf.concat([real_labels, fake_labels], axis=0)
            all_predictions = tf.concat([real_output, fake_output], axis=0)
            self.disc_accuracy.update_state(all_labels, all_predictions)
            self.disc_precision.update_state(all_labels, all_predictions)
            self.disc_recall.update_state(all_labels, all_predictions)

        if fake_output is not None:
            # Update generator metrics: generator's goal is to have fake outputs classified as 0
            target_gen = tf.zeros_like(fake_output)
            self.gen_accuracy.update_state(target_gen, fake_output)
            self.gen_precision.update_state(target_gen, fake_output)

    def reset_metrics(self):
        # Reset discriminator metrics
        self.disc_accuracy.reset_states()
        self.disc_precision.reset_states()
        self.disc_recall.reset_states()
        # Reset generator metrics
        self.gen_accuracy.reset_states()
        self.gen_precision.reset_states()

    #-- loss calculation --#
    def discriminator_loss(self, real_output, fake_output):
        # Create binary labels: 0 for real, 1 for fake
        real_labels = tf.zeros_like(real_output)
        fake_labels = tf.ones_like(fake_output)

        # Compute binary cross-entropy loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_loss = bce(real_labels, real_output)
        fake_loss = bce(fake_labels, fake_output)

        return real_loss + fake_loss

    def generator_loss(self, fake_output):
        # Generator tries to make fake samples be classified as real (0)
        fake_labels = tf.zeros_like(fake_output)  # Label 0 for fake samples
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return bce(fake_labels, fake_output)

    def nids_loss(self, real_output, fake_output):
        """
        Compute loss for the NIDS model based on real and fake samples.

        real_output: Predictions from NIDS for real validation data.
        fake_output: Predictions from NIDS for generated (fake) data.

        Returns:
            Total NIDS loss (real + fake loss).
        """

        # Binary labels: 0 for real samples, 1 for fake samples
        real_labels = tf.zeros_like(real_output)
        fake_labels = tf.ones_like(fake_output)

        # Compute binary cross-entropy loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_loss = bce(real_labels, real_output)  # Loss for real samples
        fake_loss = bce(fake_labels, fake_output)  # Loss for fake samples

        # Optional: Print per-batch losses
        print(f'Real Loss = {real_loss.numpy()}, Fake Loss = {fake_loss.numpy()}')

        return real_loss + fake_loss

    # -- Train -- #
    def fit(self):
        for epoch in range(self.epochs):
            for step, (real_data, real_labels) in enumerate(self.x_train_ds.take(self.steps_per_epoch)):
                # generate noise for generator to use.
                real_batch_size = tf.shape(real_data)[0]  # Ensure real batch size
                noise = tf.random.normal([real_batch_size, self.noise_dim])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    # Generate samples
                    generated_samples = self.generator(noise, training=True)

                    # Discriminator predictions
                    real_output = self.discriminator(real_data, training=True)
                    fake_output = self.discriminator(generated_samples, training=True)

                    # Compute losses
                    disc_loss = self.discriminator_loss(real_output, fake_output)
                    gen_loss = self.generator_loss(fake_output)

                # Compute gradients and apply updates
                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                # Apply gradient clipping
                gradients_of_generator, _ = tf.clip_by_global_norm(gradients_of_generator, 5.0)
                gradients_of_discriminator, _ = tf.clip_by_global_norm(gradients_of_discriminator, 5.0)

                self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
                self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                # Update Metrics
                # After computing real_output and fake_output
                self.update_metrics(real_output, fake_output)

                if step % 100 == 0:
                    self.log_metrics(step, disc_loss, gen_loss)

            # reset Training Metrics
            self.reset_metrics()

            # Validation after each epoch
            val_disc_loss = self.evaluate_validation_disc()
            print(f'Epoch {epoch + 1}, Validation D Loss: {val_disc_loss}')

            if self.nids is not None:
                val_nids_loss = self.evaluate_validation_NIDS()
                print(f'Epoch {epoch + 1}, Validation NIDS Loss: {val_nids_loss}')

    # -- Validate -- #
    def evaluate_validation_disc(self):
        # Reset metrics before starting the validation loop.
        self.reset_metrics()

        total_disc_loss = 0.0
        num_batches = 0

        for step, (real_data, real_labels) in enumerate(self.x_val_ds):
            # Generate fake samples using the generator
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = self.generator(noise, training=False)

            # Pass real and fake data through the discriminator
            real_output = self.discriminator(real_data, training=False)
            fake_output = self.discriminator(generated_samples, training=False)

            # Compute the discriminator loss using the real and fake outputs
            disc_loss = self.discriminator_loss(real_output, fake_output)
            total_disc_loss += disc_loss.numpy()
            num_batches += 1

            # Update metrics for this validation batch.
            self.update_metrics(real_output, fake_output)

        # Average discriminator loss over all validation batches
        avg_disc_loss = total_disc_loss / num_batches

        # Retrieve the metric results.
        val_accuracy = self.disc_accuracy.result().numpy()
        val_precision = self.disc_precision.result().numpy()
        val_recall = self.disc_recall.result().numpy()
        print(
            f"Validation Discriminator Metrics - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        # Return both the average loss and the metrics dictionary.
        return avg_disc_loss, {"accuracy": val_accuracy, "precision": val_precision, "recall": val_recall}

    def evaluate_validation_gen(self):
        # Reset generator metrics before validation.
        self.gen_accuracy.reset_states()
        self.gen_precision.reset_states()

        total_gen_loss = 0.0
        num_batches = 0

        # For generator validation, you might loop over several batches.
        for step in range(len(self.x_val_ds)):
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = self.generator(noise, training=False)
            fake_output = self.discriminator(generated_samples, training=False)

            # Compute generator loss
            gen_loss = self.generator_loss(fake_output)
            total_gen_loss += gen_loss.numpy()
            num_batches += 1

            # Update only the generator metrics.
            self.update_metrics(fake_output=fake_output)

        avg_gen_loss = total_gen_loss / num_batches
        val_gen_accuracy = self.gen_accuracy.result().numpy()
        val_gen_precision = self.gen_precision.result().numpy()
        print(f"Validation Generator Metrics - Accuracy: {val_gen_accuracy:.4f}, Precision: {val_gen_precision:.4f}")
        return avg_gen_loss, {"accuracy": val_gen_accuracy, "precision": val_gen_precision}

    def evaluate_validation_NIDS(self):
        total_nids_loss = 0.0
        num_batches = 0

        y_true_list = []
        y_pred_list = []

        all_real_data = []
        all_fake_data = []
        all_real_labels = []

        for step, (real_data, real_labels) in enumerate(self.x_val_ds):
            # Generate fake samples (same batch size as real)
            noise = tf.random.normal([real_data.shape[0], self.noise_dim])  # Match batch size
            generated_samples = self.generator(noise, training=False)

            # Get predictions from the NIDS model
            real_output = self.nids(real_data, training=False)
            fake_output = self.nids(generated_samples, training=False)

            # Compute NIDS loss using the custom function
            batch_nids_loss = self.nids_loss(real_output, fake_output).numpy()
            total_nids_loss += batch_nids_loss
            num_batches += 1

            # Convert real_output to binary predictions (threshold = 0.5)
            real_pred = (real_output.numpy() > 0.5).astype("int32")
            real_labels = real_labels.numpy().astype("int32")

            # Store real predictions and labels
            y_true_list.extend(real_labels)
            y_pred_list.extend(real_pred)

            # Store real & fake data for final evaluation
            all_real_data.append(real_data.numpy())
            all_fake_data.append(generated_samples.numpy())
            all_real_labels.append(real_labels)

            # Optional: Print per-batch losses
            print(f'Batch {step + 1}: NIDS Loss = {batch_nids_loss}')

        # Compute average loss
        avg_nids_loss = total_nids_loss / num_batches if num_batches > 0 else 0.0

        # Convert stored data into NumPy arrays
        X_real = np.vstack(all_real_data)
        X_fake = np.vstack(all_fake_data)
        y_real = np.hstack(all_real_labels)

        # Create fake labels (0 for generated samples)
        y_fake = np.zeros(X_fake.shape[0], dtype="int32")

        # Merge real and fake data for final evaluation
        X_combined = np.vstack((X_real, X_fake))  # Merge real and fake features
        y_combined = np.hstack((y_real, y_fake))  # Merge real and fake labels

        # Compute final evaluation with both real & fake data
        loss, accuracy, precision, recall, auc, logcosh = self.nids.evaluate(X_combined, y_combined, verbose=0)

        # Compare evaluation with only real data
        loss_real, accuracy_real, precision_real, recall_real, auc_real, logcosh_real = self.nids.evaluate(self.x_val, self.y_val, verbose=0)

        # Get final predictions
        y_pred_probs = self.nids.predict(X_combined)
        y_pred = (y_pred_probs > 0.5).astype("int32")

        # Compute additional metrics
        f1 = f1_score(y_combined, y_pred)
        class_report = classification_report(y_combined, y_pred, target_names=["Attack (Fake)", "Benign (Real)"])

        print("\n===== Validation Classification Report =====")
        print(class_report)
        print(f"\nValidation NIDS with Augmented Data F1 Score: {f1:.4f}")
        print("\n===== Validation Augmented Data  =====")
        print(f"Validation NIDS with Augmented Data Loss: {loss:.4f}")
        print(f"Validation NIDS with Augmented Data Accuracy: {accuracy:.4f}")
        print(f"Validation NIDS with Augmented Data Precision: {precision:.4f}")
        print(f"Validation NIDS with Augmented Data Recall: {recall:.4f}")
        print(f"Validation NIDS with Augmented Data AUC: {auc:.4f}")
        print(f"Validation NIDS with Augmented Data LogCosh: {logcosh:.4f}")
        print("\n===== Validation Real Data  =====")
        print(f"Validation NIDS with Real Data Loss: {loss_real:.4f}")
        print(f"Validation NIDS with Real Data Accuracy: {accuracy_real:.4f}")
        print(f"Validation NIDS with Real Data Precision: {precision_real:.4f}")
        print(f"Validation NIDS with Real Data Recall: {recall_real:.4f}")
        print(f"Validation NIDS with Real Data AUC: {auc_real:.4f}")
        print(f"Validation NIDS with Real Data LogCosh: {logcosh_real:.4f}")

        return avg_nids_loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc,
                               "F1-Score": f1}

    # -- Evaluate -- #
    def evaluate(self):
        total_disc_loss = 0.0
        total_gen_loss = 0.0
        num_batches = 0

        for step, (test_data_batch, test_labels_batch) in enumerate(self.x_test_ds):
            # Generate fake samples
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = self.generator(noise, training=False)

            # Discriminator predictions
            real_output = self.discriminator(test_data_batch, training=False)  # Real test data
            fake_output = self.discriminator(generated_samples, training=False)  # Generated samples

            # Binary cross-entropy loss for discriminator
            disc_loss = self.discriminator_loss(real_output, fake_output)
            total_disc_loss += disc_loss.numpy()

            # Binary cross-entropy loss for generator
            gen_loss = self.generator_loss(fake_output)
            total_gen_loss += gen_loss.numpy()

            num_batches += 1

        # Average losses over all test batches
        avg_disc_loss = total_disc_loss / num_batches
        avg_gen_loss = total_gen_loss / num_batches

        print(f"Final Evaluation Discriminator Loss: {avg_disc_loss}")
        print(f"Final Evaluation Generator Loss: {avg_gen_loss}")

        # Return average discriminator loss, number of test samples, and an empty dictionary (optional outputs)
        return avg_disc_loss, len(self.x_test_ds), {}

    def save(self, save_name):
        self.gan.save(f"../../../../../ModelArchive/local_GAN_{save_name}.h5")

        # Assuming `model` is the GAN model created with Sequential([generator, discriminator])
        generator = self.gan.layers[0]
        discriminator = self.gan.layers[1]

        # Save each submodel separately
        generator.save(f"../../../../../ModelArchive/generator_local_GAN_{save_name}.h5")
        discriminator.save(f"../../../../../ModelArchive/discriminator_local_GAN_{save_name}.h5")

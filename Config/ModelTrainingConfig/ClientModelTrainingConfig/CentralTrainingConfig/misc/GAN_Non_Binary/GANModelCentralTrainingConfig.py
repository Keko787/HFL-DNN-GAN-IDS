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


class CentralGan:
    def __init__(self, model, nids, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 noise_dim, epochs, steps_per_epoch, learning_rate):
        self.model = model
        self.nids = nids

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val  # Add validation data
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate

        self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices(self.x_test).batch(self.BATCH_SIZE)

        self.gen_optimizer = Adam(self.learning_rate)
        self.disc_optimizer = Adam(self.learning_rate)

    def discriminator_loss(self, real_normal_output, real_intrusive_output, fake_output):
        # Create labels matching the shape of the output logits
        real_normal_labels = tf.ones((tf.shape(real_normal_output)[0],), dtype=tf.int32)  # Label 1 for normal
        real_intrusive_labels = tf.zeros((tf.shape(real_intrusive_output)[0],), dtype=tf.int32)  # Label 0 for intrusive
        fake_labels = tf.fill([tf.shape(fake_output)[0]], 2)  # Label 2 for fake traffic

        # Calculate sparse categorical cross-entropy loss for each group separately
        real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(real_normal_labels, real_normal_output)
        real_intrusive_loss = tf.keras.losses.sparse_categorical_crossentropy(real_intrusive_labels,
                                                                              real_intrusive_output)
        fake_loss = tf.keras.losses.sparse_categorical_crossentropy(fake_labels, fake_output)

        # Compute the mean for each loss group independently
        mean_real_normal_loss = tf.reduce_mean(real_normal_loss)
        mean_real_intrusive_loss = tf.reduce_mean(real_intrusive_loss)
        mean_fake_loss = tf.reduce_mean(fake_loss)

        # Total loss as the average of mean losses for each group
        total_loss = (mean_real_normal_loss + mean_real_intrusive_loss + mean_fake_loss) / 3
        return total_loss

    def generator_loss(self, fake_output):
        # Generator aims to fool the discriminator by making fake samples appear as class 0 (normal)
        fake_labels = tf.zeros((tf.shape(fake_output)[0],), dtype=tf.int32)  # Shape (batch_size,)
        return tf.keras.losses.sparse_categorical_crossentropy(fake_labels, fake_output)

    def evaluate_validation_disc(self):
        generator = self.model.layers[0]
        discriminator = self.model.layers[1]

        # Create a dataset for validation data
        val_data = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val)).batch(self.BATCH_SIZE)

        total_disc_loss = 0.0
        num_batches = 0

        for step, (val_data_batch, val_labels_batch) in enumerate(val_data):
            # Generate fake samples
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = generator(noise, training=False)

            # Separate validation data into normal and intrusive
            normal_mask = tf.equal(val_labels_batch, 1)
            intrusive_mask = tf.equal(val_labels_batch, 0)

            normal_data = tf.boolean_mask(val_data_batch, normal_mask)
            intrusive_data = tf.boolean_mask(val_data_batch, intrusive_mask)

            # Pass real and fake data through the discriminator
            real_normal_output = discriminator(normal_data, training=False)
            real_intrusive_output = discriminator(intrusive_data, training=False)
            fake_output = discriminator(generated_samples, training=False)

            # Compute discriminator loss
            disc_loss = self.discriminator_loss(real_normal_output, real_intrusive_output, fake_output)
            total_disc_loss += disc_loss.numpy()
            num_batches += 1

        # Average discriminator loss across batches
        avg_disc_loss = total_disc_loss / num_batches
        return avg_disc_loss

    def evaluate_validation_gen(self):
        generator = self.model.layers[0]
        discriminator = self.model.layers[1]

        # Create a dataset for validation data
        val_data = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val)).batch(self.BATCH_SIZE)

        total_gen_loss = 0.0
        num_batches = 0

        for step in val_data:
            # Generate fake samples
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = generator(noise, training=False)

            # Fake data through the discriminator
            fake_output = discriminator(generated_samples, training=False)

            # Compute generator loss
            gen_loss = self.generator_loss(fake_output)
            total_gen_loss += gen_loss.numpy()
            num_batches += 1

        # Average generator loss across batches
        avg_gen_loss = total_gen_loss / num_batches
        return avg_gen_loss

    def evaluate_validation_NIDS(self):
        generator = self.model.layers[0]

        # Create a dataset for validation data
        val_data = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val)).batch(self.BATCH_SIZE)

        total_nids_loss = 0.0
        total_gen_loss = 0.0
        num_batches = 0

        for step, (val_data_batch, val_labels_batch) in enumerate(val_data):
            # Generate fake samples
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = generator(noise, training=False)

            # Separate validation data into normal and intrusive
            normal_mask = tf.equal(val_labels_batch, 1)
            intrusive_mask = tf.equal(val_labels_batch, 0)

            normal_data = tf.boolean_mask(val_data_batch, normal_mask)
            intrusive_data = tf.boolean_mask(val_data_batch, intrusive_mask)

            # Pass real and fake data through the NIDS model
            real_normal_output = self.nids(normal_data, training=False)
            real_intrusive_output = self.nids(intrusive_data, training=False)
            fake_output = self.nids(generated_samples, training=False)

            # Define target labels
            real_normal_labels = tf.ones((real_normal_output.shape[0],), dtype=tf.float32)
            real_intrusive_labels = tf.zeros((real_intrusive_output.shape[0],), dtype=tf.float32)
            fake_labels = tf.ones((fake_output.shape[0],), dtype=tf.float32)

            # Compute losses
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            real_normal_loss = bce(real_normal_labels, real_normal_output[:, 0])
            real_intrusive_loss = bce(real_intrusive_labels, real_intrusive_output[:, 1])
            fake_loss = bce(fake_labels, fake_output[:, 0])

            # Combine losses
            nids_loss = real_normal_loss + real_intrusive_loss
            total_nids_loss += nids_loss.numpy()
            total_gen_loss += fake_loss.numpy()
            num_batches += 1

        # Average losses across batches
        avg_nids_loss = total_nids_loss / num_batches
        avg_gen_loss = total_gen_loss / num_batches

        print(f"Validation GEN-NIDS Loss: {avg_gen_loss}")
        return avg_nids_loss

    def fit(self):
        generator = self.model.layers[0]
        discriminator = self.model.layers[1]

        # Create a TensorFlow dataset that includes both features and labels
        train_data = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(self.BATCH_SIZE)

        for epoch in range(self.epochs):
            for step, (real_data, real_labels) in enumerate(train_data.take(self.steps_per_epoch)):
                # Create masks for normal and intrusive traffic based on labels
                normal_mask = tf.equal(real_labels, 1)  # Assuming label 1 for normal
                intrusive_mask = tf.equal(real_labels, 0)  # Assuming label 0 for intrusive

                # Filter data based on these masks
                normal_data = tf.boolean_mask(real_data, normal_mask)
                intrusive_data = tf.boolean_mask(real_data, intrusive_mask)
                # Generate fake data
                noise = tf.random.normal([tf.shape(real_data)[0], self.noise_dim])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

                    # synthetic outputs generated by generator
                    generated_samples = generator(noise, training=True)

                    # Discriminator outputs
                    real_normal_output = discriminator(normal_data, training=True)
                    real_intrusive_output = discriminator(intrusive_data, training=True)
                    fake_output = discriminator(generated_samples, training=True)

                    # Calculate losses of both models
                    disc_loss = self.discriminator_loss(real_normal_output, real_intrusive_output, fake_output)
                    gen_loss = self.generator_loss(fake_output)

                # Apply gradients to both generator and discriminator (Training the models)
                # calculating gradiants of loss respect weights
                # (chain rule loss of model respect to outputs product of output respect to weights)
                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

                # Applying new parameters given from gradients
                self.gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

                if step % 100 == 0:
                    print(f'Epoch {epoch + 1}, Step {step}, D Loss: {disc_loss.numpy()}, G Loss: {gen_loss.numpy()}')

            # After each epoch, evaluate on the validation set
            val_disc_loss = self.evaluate_validation_disc()
            # val_gen_loss = self.evaluate_validation_gen()

            print(f'Epoch {epoch + 1}, Validation D Loss: {val_disc_loss}')

            if self.nids is not None:
                val_nids_loss = self.evaluate_validation_NIDS()
                print(f'Epoch {epoch + 1}, Validation NIDS Loss: {val_nids_loss}')

            # Return parameters for both generator and discriminator
            return len(self.x_train), {}

    def evaluate(self):
        generator = self.model.layers[0]
        discriminator = self.model.layers[1]

        # Create a TensorFlow dataset for testing
        test_data = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(self.BATCH_SIZE)

        total_disc_loss = 0.0
        total_gen_loss = 0.0
        num_batches = 0

        for step, (test_data_batch, test_labels_batch) in enumerate(test_data):
            # generate noise for generator to use.
            real_batch_size = tf.shape(test_data_batch)[0]  # Ensure real batch size
            noise = tf.random.normal([real_batch_size, self.noise_dim])

            # generate fake samples
            generated_samples = generator(noise, training=False)

            # Separate test data into normal and intrusive using boolean masking
            normal_mask = tf.equal(test_labels_batch, 1)  # Assuming label 1 for normal
            intrusive_mask = tf.equal(test_labels_batch, 0)  # Assuming label 0 for intrusive

            # Apply masks to create separate datasets
            normal_data = tf.boolean_mask(test_data_batch, normal_mask)
            intrusive_data = tf.boolean_mask(test_data_batch, intrusive_mask)

            print(f"Batch {step + 1}:")
            print(f"Normal data shape: {normal_data.shape}")
            print(f"Intrusive data shape: {intrusive_data.shape}")
            print(f"Generated samples shape: {generated_samples.shape}")

            # Discriminator predictions
            real_normal_output = discriminator(normal_data, training=False)
            real_intrusive_output = discriminator(intrusive_data, training=False)
            fake_output = discriminator(generated_samples, training=False)

            # Discriminator loss
            disc_loss = self.discriminator_loss(real_normal_output, real_intrusive_output, fake_output)
            total_disc_loss += disc_loss.numpy()

            # Generator loss
            gen_loss = self.generator_loss(fake_output)
            total_gen_loss += gen_loss.numpy()

            num_batches += 1

        # Average losses over all test batches
        avg_disc_loss = total_disc_loss / num_batches
        avg_gen_loss = total_gen_loss / num_batches

        print(f"Final Evaluation Discriminator Loss: {avg_disc_loss}")
        print(f"Final Evaluation Generator Loss: {avg_gen_loss}")

        # Return average discriminator loss, number of test samples, and an empty dictionary (optional outputs)
        return avg_disc_loss, len(self.x_test), {}


#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
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
#                                       Generator Training                                     #
################################################################################################################

# --- Class to handle generator training ---#
class GeneratorClient(fl.client.NumPyClient):
    def __init__(self, generator, discriminator, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE, noise_dim,
                 epochs, steps_per_epoch):
        self.generator = generator
        self.discriminator = discriminator  # Discriminator is fixed during generator training

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

        self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices(self.x_test).batch(self.BATCH_SIZE)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.gen_accuracy = tf.keras.metrics.BinaryAccuracy(name='gen_accuracy')
        self.gen_precision = tf.keras.metrics.Precision(name='gen_precision')

        # -- Metrics--#

    def log_metrics(self, step, gen_loss):
        print(f"Step {step}, G Loss: {gen_loss.numpy():.4f}")
        print(f"Generator Metrics -- Accuracy: {self.gen_accuracy.result().numpy():.4f}, "
              f"Precision: {self.gen_precision.result().numpy():.4f}")

    def update_metrics(self, fake_output=None):
        if fake_output is not None:
            # Update generator metrics: generator's goal is to have fake outputs classified as 0
            target_gen = tf.zeros_like(fake_output)
            self.gen_accuracy.update_state(target_gen, fake_output)
            self.gen_precision.update_state(target_gen, fake_output)

    def reset_metrics(self):
        # Reset generator metrics
        self.gen_accuracy.reset_states()
        self.gen_precision.reset_states()

    #-- Loss Functions
    # loss based on generating fake data that get misclassified as real by discriminator
    def generator_loss(self, fake_output):
        # Generator aims to fool the discriminator by making fake samples appear as class 0 (normal)
        fake_labels = tf.zeros((tf.shape(fake_output)[0],), dtype=tf.int32)  # Shape (batch_size,)
        return tf.keras.losses.sparse_categorical_crossentropy(fake_labels, fake_output)

    #-- Federation Helper Functions
    def get_parameters(self, config):
        return self.generator.get_weights()

    #-- Training
    def fit(self, parameters, config):
        self.generator.set_weights(parameters)

        for epoch in range(self.epochs):
            for step in range(self.steps_per_epoch):
                # generate noise for generator to use.
                real_batch_size = tf.shape(self.x_train)[0]  # Ensure real batch size
                noise = tf.random.normal([real_batch_size, self.noise_dim])

                # captures operations for the generator’s forward pass, computing gradients based on how well it generated fake samples that fooled the discriminator. These gradients then update the generator’s weights.
                # using tape to track trainable variables during generation and loss calculations
                with tf.GradientTape() as tape:
                    # Generate fake samples
                    generated_data = self.generator(noise, training=True)

                    # Discriminator output for fake data classifications
                    fake_output = self.discriminator(generated_data, training=False)

                    # Loss for generator to fool the discriminator based on its classifications
                    loss = self.generator_loss(fake_output)

                # calculate the gradient based on the loss respect to the weights of the model
                gradients = tape.gradient(loss, self.generator.trainable_variables)

                # Update the model based on the gradient of the loss respect to the weights of the model
                self.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

                # Update Metrics
                # After computing real_output and fake_output
                self.update_metrics(fake_output)

                if step % 100 == 0:
                    self.log_metrics(step, loss)

                # reset Training Metrics
            self.reset_metrics()

            # After each epoch, evaluate on the validation set
            val_gen_loss = self.evaluate_validation()
            print(f'Epoch {epoch + 1}, Validation G Loss: {val_gen_loss}')

        return self.get_parameters(config={}), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.generator.set_weights(parameters)
        total_loss = 0.0
        num_batches = 0

        for _ in self.x_test_ds:
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = self.generator(noise, training=False)
            fake_output = self.discriminator(generated_samples, training=False)
            batch_loss = self.generator_loss(fake_output)

            # If batch_loss is not scalar, reduce it to a single value
            if isinstance(batch_loss, tf.Tensor) and batch_loss.shape.rank > 0:
                batch_loss = tf.reduce_mean(batch_loss)

            total_loss += batch_loss
            num_batches += 1

        # Calculate the average loss over all batches
        average_loss = total_loss / num_batches
        return float(average_loss.numpy()), len(self.x_test), {}

    # Function to evaluate the generator on validation data
    def evaluate_validation(self):
        # Generate fake samples using the generator
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)

        # Pass the generated samples through the discriminator
        fake_output = self.discriminator(generated_samples, training=False)

        # Compute the generator loss (how well it fools the discriminator)
        gen_loss = self.generator_loss(fake_output)

        # Aggregate the generator loss to a scalar if it is an array
        if isinstance(gen_loss, tf.Tensor) and gen_loss.shape.rank > 0:
            gen_loss = tf.reduce_mean(gen_loss)  # or tf.reduce_sum() based on your requirements

        return float(gen_loss.numpy())

    def save(self, save_name):
        self.generator.save(f"../../../../../ModelArchive/generator_fed_GAN_{save_name}.h5")

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


class CentralBinaryGan:
    def __init__(self, model, nids, x_train, x_val, y_train, y_val, x_test, y_test,
                 BATCH_SIZE, noise_dim, epochs, steps_per_epoch,
                 gen_learning_rate=0.0002, disc_learning_rate=0.0001):
        self.model = model
        self.nids = nids
        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.x_train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(self.BATCH_SIZE)
        self.x_val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.BATCH_SIZE)

        lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0002, decay_steps=10000, decay_rate=0.98, staircase=True)

        lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.98, staircase=True)

        self.gen_optimizer = Adam(learning_rate=lr_schedule_gen, beta_1=0.5, beta_2=0.999)
        self.disc_optimizer = Adam(learning_rate=lr_schedule_disc, beta_1=0.5, beta_2=0.999)

        self.generator = self.model.layers[0]
        self.discriminator = self.model.layers[1]

    # loss functions
    def discriminator_loss(self, real_output, fake_output):
        real_labels = tf.random.uniform(shape=tf.shape(real_output), minval=0.8, maxval=1.0)  # Label smoothing
        fake_labels = tf.zeros_like(fake_output)

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_loss = bce(real_labels, real_output)
        fake_loss = bce(fake_labels, fake_output)

        return real_loss + fake_loss

    def generator_loss(self, fake_output):
        fake_labels = tf.random.uniform(shape=tf.shape(fake_output), minval=0.9, maxval=1.0)  # Encourage better fooling
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return bce(fake_labels, fake_output)

    # Training
    def fit(self):
        for epoch in range(self.epochs):
            for step, (real_data, _) in enumerate(self.x_train_ds.take(self.steps_per_epoch)):
                noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_samples = self.generator(noise, training=True)
                    real_output = self.discriminator(real_data, training=True)
                    fake_output = self.discriminator(generated_samples, training=True)

                    disc_loss = self.discriminator_loss(real_output, fake_output)
                    gen_loss = self.generator_loss(fake_output)

                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                # Apply gradient clipping
                gradients_of_generator, _ = tf.clip_by_global_norm(gradients_of_generator, 5.0)
                gradients_of_discriminator, _ = tf.clip_by_global_norm(gradients_of_discriminator, 5.0)

                self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
                self.disc_optimizer.apply_gradients(
                    zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                if step % 100 == 0:
                    print(f"Epoch {epoch + 1}, Step {step}, D Loss: {disc_loss.numpy()}, G Loss: {gen_loss.numpy()}")

            val_disc_loss = self.evaluate_validation_disc()
            print(f"Epoch {epoch + 1}, Validation D Loss: {val_disc_loss}")

            if self.nids is not None:
                val_nids_loss = self.evaluate_validation_NIDS()
                print(f'Epoch {epoch + 1}, Validation NIDS Loss: {val_nids_loss}')

    # validation
    def evaluate_validation_disc(self):
        total_disc_loss = 0.0
        num_batches = 0

        for step, (real_data, _) in enumerate(self.x_val_ds):
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = self.generator(noise, training=False)

            real_output = self.discriminator(real_data, training=False)
            fake_output = self.discriminator(generated_samples, training=False)

            disc_loss = self.discriminator_loss(real_output, fake_output)
            total_disc_loss += disc_loss.numpy()

            num_batches += 1
        return total_disc_loss / num_batches

    def evaluate_validation_gen(self):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)
        fake_output = self.discriminator(generated_samples, training=False)
        gen_loss = self.generator_loss(fake_output)
        return float(gen_loss.numpy())

    def evaluate_validation_NIDS(self):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)

        real_output = self.nids(self.x_val, training=False)
        fake_output = self.nids(generated_samples, training=False)

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_loss = bce(tf.ones_like(real_output), real_output)
        fake_loss = bce(tf.zeros_like(fake_output), fake_output)

        nids_loss = real_loss
        gen_loss = fake_loss

        print(f'Validation GEN-NIDS Loss: {gen_loss}')

        return float(nids_loss.numpy())

    # evaluation
    def evaluate(self):
        total_disc_loss = 0.0
        total_gen_loss = 0.0
        num_batches = 0

        for step, (test_data_batch, _) in enumerate(self.x_test_ds):
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = self.generator(noise, training=False)

            real_output = self.discriminator(test_data_batch, training=False)
            fake_output = self.discriminator(generated_samples, training=False)

            disc_loss = self.discriminator_loss(real_output, fake_output)
            gen_loss = self.generator_loss(fake_output)

            total_disc_loss += disc_loss.numpy()
            total_gen_loss += gen_loss.numpy()

            num_batches += 1

        avg_disc_loss = total_disc_loss / num_batches
        avg_gen_loss = total_gen_loss / num_batches

        print(f"Final Evaluation Discriminator Loss: {avg_disc_loss}")
        print(f"Final Evaluation Generator Loss: {avg_gen_loss}")

        return avg_disc_loss, len(self.x_test_ds), {}

    def save_model(self):
        self.model.save("GAN_Binary.keras", save_format="keras")
        print("Model saved successfully.")

    # def evaluate_legacy(self):
    #     generator = self.model.layers[0]
    #     discriminator = self.model.layers[1]
    #
    #     # Create a TensorFlow dataset for testing
    #     test_data = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(self.BATCH_SIZE)
    #
    #     total_disc_loss = 0.0
    #     total_gen_loss = 0.0
    #     num_batches = 0
    #
    #     for step, (test_data_batch, test_labels_batch) in enumerate(test_data):
    #         # Generate fake samples
    #         noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
    #         generated_samples = generator(noise, training=False)
    #
    #         # Binary masks for separating normal and intrusive test data
    #         normal_mask = tf.equal(test_labels_batch, 1)  # Label 1 for normal
    #         intrusive_mask = tf.equal(test_labels_batch, 0)  # Label 0 for intrusive
    #
    #         # Apply masks to create separate datasets
    #         normal_data = tf.boolean_mask(test_data_batch, normal_mask)
    #         intrusive_data = tf.boolean_mask(test_data_batch, intrusive_mask)
    #
    #         print(f"Batch {step + 1}:")
    #         print(f"Normal data shape: {normal_data.shape}")
    #         print(f"Intrusive data shape: {intrusive_data.shape}")
    #         print(f"Generated samples shape: {generated_samples.shape}")
    #
    #         # Discriminator predictions
    #         real_output = discriminator(test_data_batch, training=False)  # Real test data
    #         fake_output = discriminator(generated_samples, training=False)  # Generated samples
    #
    #         # Binary cross-entropy loss for discriminator
    #         disc_loss = self.discriminator_loss(real_output, fake_output)
    #         total_disc_loss += disc_loss.numpy()
    #
    #         # Binary cross-entropy loss for generator
    #         gen_loss = self.generator_loss(fake_output)
    #         total_gen_loss += gen_loss.numpy()
    #
    #         num_batches += 1
    #
    #     # Average losses over all test batches
    #     avg_disc_loss = total_disc_loss / num_batches
    #     avg_gen_loss = total_gen_loss / num_batches
    #
    #     print(f"Final Evaluation Discriminator Loss: {avg_disc_loss}")
    #     print(f"Final Evaluation Generator Loss: {avg_gen_loss}")
    #
    #     # Return average discriminator loss, number of test samples, and an empty dictionary (optional outputs)
    #     return avg_disc_loss, len(self.x_test), {}





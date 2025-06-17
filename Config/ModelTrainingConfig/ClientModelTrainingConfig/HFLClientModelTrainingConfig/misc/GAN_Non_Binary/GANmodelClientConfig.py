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


class GanClient(fl.client.NumPyClient):
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

    def get_parameters(self, config):
        # Combine generator and discriminator weights into a single list
        return self.model.get_weights()

    def evaluate_validation_disc(self):
        generator = self.model.layers[0]
        discriminator = self.model.layers[1]

        # Generate fake samples using the generator
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = generator(noise, training=False)

        # Separate validation data into normal and intrusive using boolean masking
        normal_mask = tf.equal(self.y_val, 1)  # Assuming label 1 for normal
        intrusive_mask = tf.equal(self.y_val, 0)  # Assuming label 0 for intrusive

        # Apply masks to create separate datasets
        normal_data = tf.boolean_mask(self.x_val, normal_mask)
        intrusive_data = tf.boolean_mask(self.x_val, intrusive_mask)

        # Pass real and fake data through the discriminator
        real_normal_output = discriminator(normal_data, training=False)
        real_intrusive_output = discriminator(intrusive_data, training=False)
        fake_output = discriminator(generated_samples, training=False)

        # Compute the discriminator loss using the real and fake outputs
        disc_loss = self.discriminator_loss(real_normal_output, real_intrusive_output, fake_output)

        return float(disc_loss.numpy())

    def evaluate_validation_gen(self):
        generator = self.model.layers[0]
        discriminator = self.model.layers[1]

        # Generate fake samples using the generator
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = generator(noise, training=False)

        # fake data through the discriminator
        fake_output = discriminator(generated_samples, training=False)

        # Compute the generator loss: How well does the generator fool the discriminator
        gen_loss = self.generator_loss(fake_output)

        return float(gen_loss.numpy())

    def evaluate_validation_NIDS(self):
        generator = self.model.layers[0]

        # Generate fake samples using the generator
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = generator(noise, training=False)

        # Separate validation data into normal and intrusive using boolean masking
        normal_mask = tf.equal(self.y_val, 1)  # Assuming label 1 for normal
        intrusive_mask = tf.equal(self.y_val, 0)  # Assuming label 0 for intrusive

        # Apply masks to create separate datasets
        normal_data = tf.boolean_mask(self.x_val, normal_mask)
        intrusive_data = tf.boolean_mask(self.x_val, intrusive_mask)

        # Step 3: Pass real and fake data through the NIDS model to get binary classification probabilities
        real_normal_output = self.nids(normal_data, training=False)  # Expected [P(normal), P(intrusive)]
        real_intrusive_output = self.nids(intrusive_data, training=False)  # Expected [P(normal), P(intrusive)]
        fake_output = self.nids(generated_samples, training=False)  # Expected [P(normal), P(intrusive)]

        # Step 5: Define target labels for binary cross-entropy loss
        # Real normal traffic should have high probability in the normal class
        real_normal_labels = tf.ones((real_normal_output.shape[0],),
                                     dtype=tf.int32)  # Shape matches the number of samples
        # Real intrusive traffic should have high probability in the intrusive class
        real_intrusive_labels = tf.zeros((real_intrusive_output.shape[0],),
                                         dtype=tf.int32)  # Shape matches the number of samples
        # Fake traffic (generated samples) should be labeled as normal (to fool the NIDS)
        fake_labels = tf.ones((fake_output.shape[0],), dtype=tf.int32)  # Shape matches the number of generated samples

        # Step 6: Calculate binary cross-entropy loss for each category
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_normal_loss = bce(real_normal_labels, real_normal_output[:, 0])  # Compare with "normal" class probability
        real_intrusive_loss = bce(real_intrusive_labels,
                                  real_intrusive_output[:, 1])  # Compare with "intrusive" class probability
        fake_loss = bce(fake_labels, fake_output[:, 0])  # Compare with "normal" class probability for generated samples

        # Step 7: Combine the losses
        nids_loss = real_normal_loss + real_intrusive_loss
        gen_loss = fake_loss  # Generator loss to fool the NIDS

        print(f'Validation GEN-NIDS Loss: {gen_loss}')

        return float(nids_loss.numpy())

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
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

                # generate noise for generator to use.
                real_batch_size = tf.shape(real_data)[0]  # Ensure real batch size
                noise = tf.random.normal([real_batch_size, self.noise_dim])

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
            return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        generator = self.model.layers[0]
        discriminator = self.model.layers[1]

        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = generator(noise, training=False)

        # Separate test data into normal and intrusive using boolean masking
        normal_mask = tf.equal(self.y_test, 1)  # Assuming label 1 for normal
        intrusive_mask = tf.equal(self.y_test, 0)  # Assuming label 0 for intrusive

        # Apply masks to create separate datasets
        normal_data = tf.boolean_mask(self.x_test, normal_mask)
        intrusive_data = tf.boolean_mask(self.x_test, intrusive_mask)

        print(normal_data.shape)
        print(intrusive_data.shape)
        print(generated_samples.shape)

        real_normal_output = discriminator(normal_data, training=False)
        real_intrusive_output = discriminator(intrusive_data, training=False)
        fake_output = discriminator(generated_samples, training=False)

        disc_loss = self.discriminator_loss(real_normal_output, real_intrusive_output, fake_output)

        # Compute the generator loss: How well does the generator fool the discriminator
        gen_loss = self.generator_loss(fake_output)

        print(f'Evaluation D Loss: {disc_loss}, Evaluation G Loss: {gen_loss}')

        return float(disc_loss.numpy()), len(self.x_test), {}

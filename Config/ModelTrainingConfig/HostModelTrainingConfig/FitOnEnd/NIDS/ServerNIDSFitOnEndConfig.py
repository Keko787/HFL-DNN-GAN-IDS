# Custom FedAvg strategy with server-side model training and saving
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
import tensorflow_privacy as tfp
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
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


class NIDSFitOnEndStrategy(fl.server.strategy.FedAvg):
    def __init__(self, discriminator, generator, nids, dataset_used, node, earlyStopEnabled, lrSchedRedEnabled, modelCheckpointEnabled, DP_enabled,
                 X_train_data, y_train_data,X_test_data, y_test_data, X_val_data, y_val_data, l2_norm_clip,
                 noise_multiplier, num_microbatches, batch_size, epochs, steps_per_epoch, learning_rate, synth_portion, latent_dim, num_classes,
                 metric_to_monitor_es, es_patience,restor_best_w, metric_to_monitor_l2lr, l2lr_patience,
                 save_best_only, metric_to_monitor_mc, checkpoint_mode, save_name, serverLoad, **kwargs):

        super().__init__(**kwargs)
        self.ACdiscriminator = discriminator
        self.ACgenerator = generator
        self.nids = nids

        self.data_used = dataset_used
        self.node = node

        # flags
        self.DP_enabled = DP_enabled
        self.earlyStopEnabled = earlyStopEnabled
        self.lrSchedRedEnabled = lrSchedRedEnabled
        self.modelCheckpointEnabled = modelCheckpointEnabled

        # data
        self.X_train_data = X_train_data
        self.y_train_data = y_train_data
        self.X_test_data = X_test_data
        self.y_test_data = y_test_data
        self.X_val_data = X_val_data
        self.y_val_data = y_val_data

        # hyperparams
        # basic
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        # ACGAN
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.synth_portion = synth_portion
        # dp
        self.num_microbatches = num_microbatches
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier

        # callback params
        self.callbackFunctions = []
        # early stop
        self.metric_to_monitor_es = metric_to_monitor_es
        self.es_patience = es_patience
        self.restor_best_w = restor_best_w
        # lr schedule
        self.metric_to_monitor_l2lr = metric_to_monitor_l2lr
        self.l2lr_factor = l2lr_patience
        self.l2lr_patience = es_patience
        # model checkpoint
        self.save_best_only = save_best_only
        self.metric_to_monitor_mc = metric_to_monitor_mc
        self.checkpoint_mode = checkpoint_mode

        # saving
        self.save_name = save_name
        self.serverLoad = serverLoad
        self.file_name = f"../../../../ModelArchive/NIDS_AT_{self.save_name}"

        # counters
        self.roundCount = 0
        self.evaluateCount = 0

        # init callback functions based on inputs
        # 1. early stop
        if self.earlyStopEnabled:
            early_stopping = EarlyStopping(monitor=self.metric_to_monitor_es, patience=self.es_patience,
                                           restore_best_weights=self.restor_best_w)

            self.callbackFunctions.append(early_stopping)
        # 2. lr sched
        if self.lrSchedRedEnabled:
            lr_scheduler = ReduceLROnPlateau(monitor=self.metric_to_monitor_l2lr, factor=self.l2lr_factor,
                                             patience=self.l2lr_patience)

            self.callbackFunctions.append(lr_scheduler)
        # 3. model checkpoint
        if self.modelCheckpointEnabled:
            model_checkpoint = ModelCheckpoint(f'best_model_{self.save_name}.h5', save_best_only=self.save_best_only,
                                               monitor=self.metric_to_monitor_mc, mode=self.checkpoint_mode)

            # add to callback functions list being added during fitting
            self.callbackFunctions.append(model_checkpoint)

    def initialize_parameters(self, client_manager):
        """Send pre-trained model weights to clients on the first round, if serverLoad is enabled."""
        if self.serverLoad:
            return ndarrays_to_parameters(self.nids.get_weights())
        return None

    def generate_synthetic_data(self, num_samples):
        """Generates synthetic data using the AC-GAN generator."""
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        fake_labels = np.random.randint(0, self.num_classes, num_samples)
        return self.ACgenerator.predict([noise, fake_labels]), fake_labels

    def generate_balanced_synthetic_data(self, num_samples):
        """
        Generates balanced synthetic data using the AC-GAN generator with equal distribution
        of classes 0 and 1.

        Parameters:
        - num_samples: Total number of samples to generate

        Returns:
        - X_synthetic: Generated synthetic data
        - y_synthetic: Corresponding class labels
        """
        # Calculate samples per class (ensure we get the same number for each class)
        samples_per_class = num_samples // self.num_classes

        # Prepare containers for our data
        synthetic_data_list = []
        labels_list = []

        # Generate equal samples for each class
        for class_idx in range(self.num_classes):
            # Generate noise
            noise = np.random.normal(0, 1, (samples_per_class, self.latent_dim))

            # Create labels array for this class
            fake_labels = np.full(samples_per_class, class_idx)

            # Generate data for this class
            class_data = self.ACgenerator.predict([noise, fake_labels])

            # Store the generated data and labels
            synthetic_data_list.append(class_data)
            labels_list.append(fake_labels)

        # Combine all the generated data
        X_synthetic = np.vstack(synthetic_data_list)
        y_synthetic = np.hstack(labels_list)

        # Shuffle to avoid having blocks of the same class
        shuffle_idx = np.random.permutation(len(X_synthetic))
        X_synthetic = X_synthetic[shuffle_idx]
        y_synthetic = y_synthetic[shuffle_idx]

        return X_synthetic, y_synthetic

    def aggregate_fit(self, server_round, results, failures):
        """Aggregates client updates, fine-tunes, and sends weights back."""
        self.roundCount += 1
        print(f"Round: {self.roundCount}\n")
        start_time = time.time()

        #-- Set the model with global weights, Bring in the parameters for the global model --#
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving global model after round {server_round}...")
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters[0])
            if len(aggregated_weights) == len(self.nids.get_weights()):
                self.nids.set_weights(aggregated_weights)
        # EoF Set global weights

        #-- ReCompile Model --#
        # Differential Privacy (optional)
        if self.DP_enabled:
            print("\nApplying Differential Privacy Optimizer...\n")
            dp_optimizer = tfp.DPKerasAdamOptimizer(
                l2_norm_clip=self.l2_norm_clip,
                noise_multiplier=self.noise_multiplier,
                num_microbatches=self.num_microbatches,
                learning_rate=self.learning_rate
            )
            self.nids.compile(optimizer=dp_optimizer,
                              loss='binary_crossentropy',
                              metrics=['accuracy', Precision(), Recall(), AUC(), LogCosh()])

        # make basic compile (default)
        else:
            print("\nUsing Standard Adam Optimizer...\n")
            optimizer = Adam(learning_rate=self.learning_rate)
            self.nids.compile(optimizer=optimizer,
                              loss='binary_crossentropy',
                              metrics=['accuracy', Precision(), Recall(), AUC(), LogCosh()])
        # EoF Recompile

        #-- Generate Synthetic Data for Augmentation --#
        # If generator exists (optional), generate a selected portion of data to augment the training dataset
        if self.ACgenerator is not None:
            # rename file_name for saved model after adv training
            self.file_name = f"NIDS_ATSS_{self.save_name}"

            # gather the synthetic samples with balanced classes
            print("\nGenerating Balanced Synthetic Data for Augmentation...\n")
            num_synthetic_samples = int(self.synth_portion * len(self.X_train_data))
            X_synthetic, y_synthetic = self.generate_balanced_synthetic_data(num_synthetic_samples)

            # Log the distribution of synthetic labels
            unique_labels, counts = np.unique(y_synthetic, return_counts=True)
            print(f"Synthetic data class distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"  Class {label}: {count} samples ({count / len(y_synthetic) * 100:.1f}%)")

            # Concatenate with real training data
            self.X_train_data = np.vstack((self.X_train_data, X_synthetic))
            self.y_train_data = np.hstack((self.y_train_data, y_synthetic))

            # Log the distribution of the combined dataset
            unique_labels, counts = np.unique(self.y_train_data, return_counts=True)
            print(f"Combined dataset class distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"  Class {label}: {count} samples ({count / len(self.y_train_data) * 100:.1f}%)")
        # EoF Synthetic Augmentation

        #-- Train NIDS Model on (with or without Augmented) Dataset--#
        print(f"\nTraining NIDS Model on Augmented Dataset for {self.epochs} epochs...\n")
        history = self.nids.fit(self.X_train_data, self.y_train_data,
                                validation_data=(self.X_val_data, self.y_val_data),
                                epochs=self.epochs, batch_size=self.batch_size,
                                steps_per_epoch=self.steps_per_epoch,
                                verbose=1)
        # EoF Training.

        # Save the fine-tuned model
        self.nids.save(self.file_name)
        print(f"Model fine-tuned and saved after round {server_round}.")

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Debugging: Print the shape of the loss
        loss_tensor = history.history['loss']
        val_loss_tensor = history.history['val_loss']
        # print(f"Loss tensor shape: {tf.shape(loss_tensor)}")
        # print(f"Validation Loss tensor shape: {tf.shape(val_loss_tensor)}")

        # Save training metrics
        self.recordTraining("training_log.txt", history, elapsed_time, self.roundCount, val_loss_tensor)

        return self.nids.get_weights(), {}

    #########################################################
    #    Metric Saving Functions                           #
    #########################################################

    def recordTraining(self, name, history, elapsed_time, roundCount, val_loss):
        with open(name, 'a') as f:
            f.write(f"Node|{self.node}| Round: {roundCount}\n")
            f.write(f"Training Time Elapsed: {elapsed_time} seconds\n")
            for epoch in range(self.epochs):
                f.write(f"Epoch {epoch + 1}/{self.epochs}\n")
                for metric, values in history.history.items():
                    # Debug: print the length of values list and the current epoch
                    print(f"Metric: {metric}, Values Length: {len(values)}, Epoch: {epoch}")
                    if epoch < len(values):
                        f.write(f"{metric}: {values[epoch]}\n")
                    else:
                        print(f"Skipping metric {metric} for epoch {epoch} due to out-of-range error.")
                if epoch < len(val_loss):
                    f.write(f"Validation Loss: {val_loss[epoch]}\n")
                else:
                    print(f"Skipping Validation Loss for epoch {epoch} due to out-of-range error.")
                f.write("\n")

    def recordEvaluation(self, name, elapsed_time, evaluateCount, loss, accuracy, precision, recall, auc, logcosh):
        with open(name, 'a') as f:
            f.write(f"Node|{self.node}| Round: {evaluateCount}\n")
            f.write(f"Evaluation Time Elapsed: {elapsed_time} seconds\n")
            f.write(f"Loss: {loss}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"AUC: {auc}\n")
            f.write(f"LogCosh: {logcosh}\n")
            f.write("\n")

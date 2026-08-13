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
################################################################################################################
#                                       NIDS Training                                       #
################################################################################################################

#########################################################
    #    Adversarial Training Function                   #
##########################################################

    # Function to generate adversarial examples using FGSM
def create_adversarial_example(model, x, y, epsilon=0.01):
    # Ensure x is a tensor and has the correct shape (batch_size, input_dim)
    # print("Original x shape:", x.shape)
    # print("Original y shape:", y.shape)

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = tf.expand_dims(x, axis=0)  # Adding batch dimension
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    y = tf.expand_dims(y, axis=0)  # Adding batch dimension to match prediction shape

    # print("Expanded x shape:", x.shape)
    # print("Expanded y shape:", y.shape)

    # Create a gradient tape context to record operations for automatic differentiation
    with tf.GradientTape() as tape:
        tape.watch(x)  # Adds the tensor x to the list of watched tensors, allowing its gradients to be computed
        prediction = model(x)  # Passes x through the model to get predictions
        y = tf.reshape(y, prediction.shape)  # Reshape y to match the shape of prediction
        # print("Reshaped y shape:", y.shape)
        loss = tf.keras.losses.binary_crossentropy(y,
                                                   prediction)  # Computes the binary crossentropy loss between true labels y and predictions

    # Computes the gradient of the loss with respect to the input x
    gradient = tape.gradient(loss, x)

    # Creates the perturbation using the sign of the gradient and scales it by epsilon
    perturbation = epsilon * tf.sign(gradient)

    # Adds the perturbation to the original input to create the adversarial example
    adversarial_example = x + perturbation
    adversarial_example = tf.clip_by_value(adversarial_example, 0, 1)  # Ensure values are within valid range
    adversarial_example = tf.squeeze(adversarial_example, axis=0)  # Removing the batch dimension

    return adversarial_example


#########################################################
#    Federated Learning Setup                           #
#########################################################
class FlNidsClient(fl.client.NumPyClient):

    def __init__(self, model_used, dataset_used, node, adversarialTrainingEnabled, earlyStopEnabled, DP_enabled,
                 lrSchedRedEnabled, modelCheckpointEnabled, X_train_data, y_train_data, X_test_data, y_test_data,
                 X_val_data, y_val_data, l2_norm_clip, noise_multiplier, num_microbatches, batch_size, epochs,
                 steps_per_epoch, learning_rate, adv_portion, metric_to_monitor_es, es_patience, restor_best_w,
                 metric_to_monitor_l2lr, l2lr_patience, save_best_only, metric_to_monitor_mc, checkpoint_mode,
                 evaluationLog, trainingLog, modelname = "nids"):

        # ---         Variable init              --- #
        # model
        self.model = model_used

        # type of model
        self.data_used = dataset_used
        self.model_name = modelname
        self.node = node

        # flags
        self.adversarialTrainingEnabled = adversarialTrainingEnabled
        self.earlyStopEnabled = earlyStopEnabled
        self.DP_enabled = DP_enabled
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
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        # dp
        self.num_microbatches = num_microbatches
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        # adversarial
        self.adv_portion = adv_portion

        # callback params
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

        # logs
        self.evaluationLog = evaluationLog
        self.trainingLog = trainingLog

        # counters
        self.roundCount = 0
        self.evaluateCount = 0

        # ---         Differential Privacy Engine Model Compile              --- #

        if self.DP_enabled:
            print("\nIncluding DP into optimizer...\n")

            # Making Custom Optimizer Component with Differential Privacy
            dp_optimizer = tfp.DPKerasAdamOptimizer(
                l2_norm_clip=self.l2_norm_clip,
                noise_multiplier=self.noise_multiplier,
                num_microbatches=self.num_microbatches,
                learning_rate=self.learning_rate
            )

            # compile model with custom dp optimizer
            self.model.compile(optimizer=dp_optimizer,
                               loss=tf.keras.losses.binary_crossentropy,
                               metrics=['accuracy', Precision(), Recall(), AUC(), LogCosh()])

        # ---              Normal Model Compile                        --- #

        else:
            print("\nDefault optimizer...\n")

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            self.model.compile(optimizer=optimizer,
                               loss=tf.keras.losses.binary_crossentropy,
                               metrics=['accuracy', Precision(), Recall(), AUC(), LogCosh()]
                               )

            print("\nModel Compiled...\n")

        # ---                   Callback components                   --- #

        # init main call back functions list
        self.callbackFunctions = []

        # init callback functions based on inputs

        if self.earlyStopEnabled:
            early_stopping = EarlyStopping(monitor=self.metric_to_monitor_es, patience=self.es_patience,
                                           restore_best_weights=self.restor_best_w)

            self.callbackFunctions.append(early_stopping)

        if self.lrSchedRedEnabled:
            lr_scheduler = ReduceLROnPlateau(monitor=self.metric_to_monitor_l2lr, factor=self.l2lr_factor,
                                             patience=self.l2lr_patience)

            self.callbackFunctions.append(lr_scheduler)

        if self.modelCheckpointEnabled:
            model_checkpoint = ModelCheckpoint(f'best_model_{self.model_name}.h5', save_best_only=self.save_best_only,
                                               monitor=self.metric_to_monitor_mc, mode=self.checkpoint_mode)

            # add to callback functions list being added during fitting
            self.callbackFunctions.append(model_checkpoint)

        # ---                   Model Analysis                   --- #

        self.model.summary()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # increment round count
        self.roundCount += 1

        # debug print
        print("Round:", self.roundCount, "\n")

        # Record start time
        start_time = time.time()

        self.model.set_weights(parameters)

        if self.adversarialTrainingEnabled:

            total_examples = len(self.X_train_data)
            print_every = max(total_examples // 10000, 1)  # Print progress every 0.1%

            # Define proportion of data to use for adversarial training (e.g., 10%)
            adv_proportion = self.adv_portion

            num_adv_examples = int(total_examples * adv_proportion)
            print("# of adversarial examples", num_adv_examples)
            adv_indices = random.sample(range(total_examples), num_adv_examples)

            adv_examples = []
            for idx, (x, y) in enumerate(zip(self.X_train_data.to_numpy(), self.y_train_data.to_numpy())):
                if idx in adv_indices:
                    adv_example = create_adversarial_example(self.model, x, y)
                    adv_examples.append(adv_example)
                else:
                    adv_examples.append(x)

                if (idx + 1) % print_every == 0 or (idx + 1) == total_examples:
                    print(f"Progress: {(idx + 1) / total_examples * 100:.2f}%")

            adv_X_train_data = np.array(adv_examples)

            adv_X_train_data = pd.DataFrame(adv_X_train_data, columns=self.X_train_data.columns)
            combined_X_train_data = pd.concat([self.X_train_data, adv_X_train_data])
            combined_y_train_data = pd.concat([self.y_train_data, self.y_train_data])

            history = self.model.fit(combined_X_train_data, combined_y_train_data,
                                     validation_data=(self.X_val_data, self.y_val_data),
                                     epochs=self.epochs, batch_size=self.batch_size,
                                     steps_per_epoch=self.steps_per_epoch,
                                     callbacks=self.callbackFunctions)
        else:
            # Train Model
            history = self.model.fit(self.X_train_data, self.y_train_data,
                                     validation_data=(self.X_val_data, self.y_val_data),
                                     epochs=self.epochs, batch_size=self.batch_size,
                                     steps_per_epoch=self.steps_per_epoch,
                                     callbacks=self.callbackFunctions)

        # Record end time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Debugging: Print the shape of the loss
        loss_tensor = history.history['loss']
        val_loss_tensor = history.history['val_loss']
        # print(f"Loss tensor shape: {tf.shape(loss_tensor)}")
        # print(f"Validation Loss tensor shape: {tf.shape(val_loss_tensor)}")

        # Save metrics to file
        logName = self.trainingLog
        #logName = f'training_metrics_{dataset_used}_optimized_{l2_norm_clip}_{noise_multiplier}.txt'
        self.recordTraining(logName, history, elapsed_time, self.roundCount, val_loss_tensor)

        return self.model.get_weights(), len(self.X_train_data), {}

    def evaluate(self, parameters, config):
        # increment evaluate count
        self.evaluateCount += 1

        # debug print
        print("Evaluate Round:", self.evaluateCount, "\n")

        # Record start time
        start_time = time.time()

        # set the weights given from server
        self.model.set_weights(parameters)

        # Test the model
        loss, accuracy, precision, recall, auc, logcosh = self.model.evaluate(self.X_test_data, self.y_test_data)

        # Record end time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Save metrics to file
        logName1 = self.evaluationLog
        #logName = f'evaluation_metrics_{dataset_used}_optimized_{l2_norm_clip}_{noise_multiplier}.txt'
        self.recordEvaluation(logName1, elapsed_time, self.evaluateCount, loss, accuracy, precision, recall, auc, logcosh)

        return loss, len(self.X_test_data), {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc,
                                             "LogCosh": logcosh}

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
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"AUC: {auc}\n")
            f.write(f"LogCosh: {logcosh}\n")
            f.write("\n")


def recordConfig(name, dataset_used, DP_enabled, adversarialTrainingEnabled, regularizationEnabled, input_dim, epochs,
                 batch_size, steps_per_epoch, betas, learning_rate, l2_norm_clip, noise_multiplier, num_microbatches,
                 adv_portion, l2_alpha, model):
    with open(name, 'a') as f:
        f.write(f"Dataset Used: {dataset_used}\n")
        f.write(
            f"Defenses Enabled: DP - {DP_enabled}, Adversarial Training - {adversarialTrainingEnabled}, Regularization - {regularizationEnabled}\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"Input Dim (Feature Size): {input_dim}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Steps per epoch: {steps_per_epoch}\n")
        f.write(f"Betas: {betas}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        if DP_enabled:
            f.write(f"L2 Norm Clip: {l2_norm_clip}\n")
            f.write(f"Noise Multiplier: {noise_multiplier}\n")
            f.write(f"MicroBatches: {num_microbatches}\n")
        if adversarialTrainingEnabled:
            f.write(f"Adversarial Sample %: {adv_portion * 100}%\n")
        if regularizationEnabled:
            f.write(f"L2 Alpha: {l2_alpha}\n")
        f.write(f"Model Layer Structure:\n")
        for layer in model.layers:
            f.write(
                f"Layer: {layer.name}, Type: {layer.__class__.__name__}, Output Shape: {layer.output_shape}, Params: {layer.count_params()}\n")
        f.write("\n")

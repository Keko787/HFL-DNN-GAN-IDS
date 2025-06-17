#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
import os
import random
import time
from datetime import datetime
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from sklearn.metrics import classification_report, f1_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow_privacy as tfp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CentralNidsClient:

    def __init__(self, model_used, dataset_used, node, earlyStopEnabled, DP_enabled,
                 lrSchedRedEnabled, modelCheckpointEnabled, X_train_data, y_train_data, X_test_data, y_test_data,
                 X_val_data, y_val_data, l2_norm_clip, noise_multiplier, num_microbatches, batch_size, epochs,
                 steps_per_epoch, learning_rate, metric_to_monitor_es, es_patience, restor_best_w,
                 metric_to_monitor_l2lr, l2lr_patience, save_best_only, metric_to_monitor_mc, checkpoint_mode,
                 evaluationLog, trainingLog, modelname = "nids"):

        # ---         Variable init              --- #

        # model
        self.model = model_used

        # type of model
        self.data_used = dataset_used
        self.node = node
        self.model_name = modelname

        # flags
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

    def fit(self):

        # -- Training model -- #

        # Record start time
        start_time = time.time()

        # Train Model (TRAINING LOOP HERE)
        history = self.model.fit(self.X_train_data, self.y_train_data,
                                 validation_data=(self.X_val_data, self.y_val_data),
                                 epochs=self.epochs, batch_size=self.batch_size,
                                 steps_per_epoch=self.steps_per_epoch,
                                 callbacks=self.callbackFunctions)

        # Record end time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # -- Recording and Debug -- #

        # Debugging: Print the shape of the loss
        loss_tensor = history.history['loss']
        val_loss_tensor = history.history['val_loss']
        # print(f"Loss tensor shape: {tf.shape(loss_tensor)}")
        # print(f"Validation Loss tensor shape: {tf.shape(val_loss_tensor)}")

        # Save metrics to file
        logName = self.trainingLog
        self.recordTraining(logName, history, elapsed_time, self.roundCount, val_loss_tensor)

    def evaluate(self):

        # -- Testing model -- #

        # Record start time
        start_time = time.time()

        # Get model predictions
        y_pred_probs = self.model.predict(self.X_test_data)
        y_pred = (y_pred_probs > 0.5).astype("int32")  # Convert probabilities to binary labels

        # Test the model
        loss, accuracy, precision, recall, auc, logcosh = self.model.evaluate(self.X_test_data, self.y_test_data)

        # Record end time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # -- Recording and Debug -- #

        # Compute F1-Score
        f1 = f1_score(self.y_test_data, y_pred)

        # Compute class-wise precision & recall
        class_report = classification_report(self.y_test_data, y_pred, target_names=["Benign", "Attack"])

        # Save metrics to file
        logName1 = self.evaluationLog
        #logName = f'evaluation_metrics_{dataset_used}_optimized_{l2_norm_clip}_{noise_multiplier}.txt'
        self.recordEvaluation(logName1, elapsed_time, self.evaluateCount, loss, accuracy, precision, recall, auc, logcosh)

        print("\n===== Classification Report =====")
        print(class_report)
        print(f"F1 Score: {f1:.4f}")

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

    def save(self, save_name):
        self.model.save(f"../../../../ModelArchive/local_NIDS_{save_name}.h5")


def recordConfig(name, dataset_used, DP_enabled, regularizationEnabled, input_dim, epochs,
                 batch_size, steps_per_epoch, betas, learning_rate, l2_norm_clip, noise_multiplier, num_microbatches
                 , l2_alpha, model):
    with open(name, 'a') as f:
        f.write(f"Dataset Used: {dataset_used}\n")
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
        if regularizationEnabled:
            f.write(f"L2 Alpha: {l2_alpha}\n")
        f.write(f"Model Layer Structure:\n")
        for layer in model.layers:
            f.write(
                f"Layer: {layer.name}, Type: {layer.__class__.__name__}, Output Shape: {layer.output_shape}, Params: {layer.count_params()}\n")
        f.write("\n")

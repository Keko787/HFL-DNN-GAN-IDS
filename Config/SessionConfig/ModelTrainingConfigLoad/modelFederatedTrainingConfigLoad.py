#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
import os
import random
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import expand_dims

sys.path.append(os.path.abspath('../../..'))

# TensorFlow & Flower
if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']

import flwr as fl
import tensorflow as tf
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# other plugins
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from ModelTrainingConfig.ClientModelTrainingConfig.HFLClientModelTrainingConfig.NIDS.NIDSModelClientConfig import FlNidsClient
from ModelTrainingConfig.ClientModelTrainingConfig.HFLClientModelTrainingConfig.GAN.FullModel.GANBinaryModelClientConfig import GanBinaryClient
from ModelTrainingConfig.ClientModelTrainingConfig.HFLClientModelTrainingConfig.GAN.Discriminator.DiscBinaryModelClientConfig import BinaryDiscriminatorClient
from ModelTrainingConfig.ClientModelTrainingConfig.HFLClientModelTrainingConfig.GAN.Generator.GenModelClientConfig import GeneratorClient
from ModelTrainingConfig.ClientModelTrainingConfig.HFLClientModelTrainingConfig.GAN.FullModel.WGANBinaryClientTrainingConfig import BinaryWGanClient
from ModelTrainingConfig.ClientModelTrainingConfig.HFLClientModelTrainingConfig.GAN.Discriminator.WGAN_DiscriminatorBinaryClientTrainingConfig import BinaryWDiscriminatorClient
from ModelTrainingConfig.ClientModelTrainingConfig.HFLClientModelTrainingConfig.GAN.Generator.WGAN_GeneratorBinaryClientTrainingConfig import BinaryWGeneratorClient
from ModelTrainingConfig.ClientModelTrainingConfig.HFLClientModelTrainingConfig.GAN.FullModel.ACGANClientTrainingConfig import ACGanClient
from ModelTrainingConfig.ClientModelTrainingConfig.HFLClientModelTrainingConfig.GAN.Discriminator.AC_DiscModelClientConfig import ACDiscriminatorClient

################################################################################################################
#                      Federation TRAINING CONFIG CLIENT CLASS OBJECT LOADER                                      #
################################################################################################################

def modelFederatedTrainingConfigLoad(nids, discriminator, generator, GAN, dataset_used, model_type, train_type,
                                   earlyStopEnabled, DP_enabled, lrSchedRedEnabled, modelCheckpointEnabled, X_train_data,
                                   X_val_data, y_train_data, y_val_data, X_test_data, y_test_data, node, BATCH_SIZE,
                                   epochs, noise_dim, steps_per_epoch, input_dim, num_classes, latent_dim, betas,
                                   learning_rate, l2_alpha, l2_norm_clip, noise_multiplier, num_microbatches,
                                   metric_to_monitor_es, es_patience, restor_best_w, metric_to_monitor_l2lr,
                                   l2lr_patience, save_best_only, metric_to_monitor_mc, checkpoint_mode, evaluationLog,
                                   trainingLog):

    client = None

    if model_type == 'NIDS':
        client = FlNidsClient(nids, dataset_used, node, earlyStopEnabled, DP_enabled,
                                   lrSchedRedEnabled, modelCheckpointEnabled, X_train_data, y_train_data, X_test_data,
                                   y_test_data, X_val_data, y_val_data, l2_norm_clip, noise_multiplier,
                                   num_microbatches,
                                   BATCH_SIZE, epochs, steps_per_epoch, learning_rate, metric_to_monitor_es,
                                   es_patience, restor_best_w, metric_to_monitor_l2lr, l2lr_patience, save_best_only,
                                   metric_to_monitor_mc, checkpoint_mode, evaluationLog, trainingLog)

    elif model_type == 'GAN':
        if train_type == "Both":
            client = GanBinaryClient(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                  y_test_data, BATCH_SIZE,
                                  noise_dim, epochs, steps_per_epoch, learning_rate)
        elif train_type == "Generator":
            client = GeneratorClient(generator, discriminator, X_train_data, X_val_data, y_train_data, y_val_data,
                                      X_test_data, y_test_data, BATCH_SIZE,
                                      noise_dim, epochs, steps_per_epoch)

        elif train_type == "Discriminator":
            client = BinaryDiscriminatorClient(discriminator, generator, X_train_data, X_val_data, y_train_data, y_val_data,
                                          X_test_data, y_test_data, BATCH_SIZE, noise_dim, epochs, steps_per_epoch)


    elif model_type == 'WGAN-GP':
        if train_type == "Both":
            client = BinaryWGanClient(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                   y_test_data, BATCH_SIZE,
                                   noise_dim, epochs, steps_per_epoch, learning_rate)
        elif train_type == "Generator":
            client = BinaryWGeneratorClient(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                  y_test_data, BATCH_SIZE,
                 noise_dim, epochs, steps_per_epoch, learning_rate)
        elif train_type == "Discriminator":
            client = BinaryWDiscriminatorClient(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                  y_test_data, BATCH_SIZE,
                                                noise_dim, epochs, steps_per_epoch, learning_rate)

    elif model_type == 'AC-GAN':
        if train_type == "Both":
            client = ACGanClient(GAN, nids, X_train_data, X_val_data, y_train_data,
                              y_val_data, X_test_data, y_test_data, BATCH_SIZE,
                              noise_dim, latent_dim, num_classes, input_dim, epochs, steps_per_epoch, learning_rate)
        elif train_type == "Generator":
            client = None
        elif train_type == "Discriminator":
            client = ACDiscriminatorClient(discriminator=discriminator,
                                           x_train=X_train_data,
                                           x_val=X_val_data,
                                           y_train=y_train_data,
                                           y_val=y_val_data,
                                           x_test=X_test_data,
                                           y_test=y_test_data,
                                           BATCH_SIZE=BATCH_SIZE,
                                           num_classes=num_classes,
                                           input_dim=input_dim,
                                           epochs=epochs,
                                           steps_per_epoch=steps_per_epoch,
                                           learning_rate=learning_rate,
                                           log_file=trainingLog,
                                           use_class_labels=True)
    return client

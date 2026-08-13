#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
import os
import random
from datetime import datetime
import argparse
sys.path.append(os.path.abspath('../..'))
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import expand_dims
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# import math
# import glob
# from tqdm import tqdm
# import seaborn as sns
# import pickle
# import joblib

from centralTrainingConfig.discriminatorCentralTrainingConfig import CentralDiscriminator
from centralTrainingConfig.generatorModelCentralTrainingConfig import CentralGenerator
from centralTrainingConfig.nidsModelCentralTrainingConfig import CentralNidsClient, recordConfig
from centralTrainingConfig.GANBinaryCentralTrainingConfig import CentralBinaryGan
from centralTrainingConfig.WGANBinaryCentralTrainingConfig import CentralBinaryWGan
from centralTrainingConfig.ACGANCentralTrainingConfig import CentralACGan

def modelServerFedStratConfigLoad(nids, discriminator, generator, GAN, dataset_used, model_type, train_type,
                                   earlyStopEnabled, DP_enabled, lrSchedRedEnabled, modelCheckpointEnabled, X_train_data,
                                   X_val_data, y_train_data, y_val_data, X_test_data, y_test_data, node, BATCH_SIZE,
                                   epochs, noise_dim, steps_per_epoch, input_dim, num_classes, latent_dim, betas,
                                   learning_rate, l2_alpha, l2_norm_clip, noise_multiplier, num_microbatches,
                                   metric_to_monitor_es, es_patience, restor_best_w, metric_to_monitor_l2lr,
                                   l2lr_patience, save_best_only, metric_to_monitor_mc, checkpoint_mode, evaluationLog,
                                   trainingLog):

    client = None

    if model_type == 'NIDS':
        client = CentralNidsClient(nids, dataset_used, node, earlyStopEnabled, DP_enabled,
                                   lrSchedRedEnabled, modelCheckpointEnabled, X_train_data, y_train_data, X_test_data,
                                   y_test_data, X_val_data, y_val_data, l2_norm_clip, noise_multiplier,
                                   num_microbatches,
                                   BATCH_SIZE, epochs, steps_per_epoch, learning_rate, metric_to_monitor_es,
                                   es_patience, restor_best_w, metric_to_monitor_l2lr, l2lr_patience, save_best_only,
                                   metric_to_monitor_mc, checkpoint_mode, evaluationLog, trainingLog)

    elif model_type == 'GAN':
        if train_type == "Both":
            client = CentralBinaryGan(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                  y_test_data, BATCH_SIZE,
                                  noise_dim, epochs, steps_per_epoch, learning_rate)

        elif train_type == "Discriminator":
            client = CentralDiscriminator(discriminator, generator, X_train_data, X_val_data, y_train_data, y_val_data,
                                          X_test_data, y_test_data, BATCH_SIZE, noise_dim, epochs, steps_per_epoch)


    elif model_type == 'WGAN-GP':
        if train_type == "Both":
            client = CentralBinaryWGan(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                   y_test_data, BATCH_SIZE,
                                   noise_dim, epochs, steps_per_epoch, learning_rate)

        elif train_type == "Discriminator":
            client = None

    elif model_type == 'AC-GAN':
        if train_type == "Both":
            client = CentralACGan(discriminator, generator, nids, X_train_data, X_val_data, y_train_data,
                              y_val_data, X_test_data, y_test_data, BATCH_SIZE,
                              noise_dim, latent_dim, num_classes, input_dim, epochs, steps_per_epoch, learning_rate)
        elif train_type == "Generator":
            client = None
        elif train_type == "Discriminator":
            client = None

    return client

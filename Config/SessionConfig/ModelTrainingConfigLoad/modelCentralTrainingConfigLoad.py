#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
import os
import random
from datetime import datetime
import numpy as np
from numpy import expand_dims
import pandas as pd
import matplotlib.pyplot as plt

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

from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.NIDS.nidsModelCentralTrainingConfig import CentralNidsClient, recordConfig
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Generator.generatorModelCentralTrainingConfig import CentralGenerator
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Discriminator.discriminatorBinaryCentralTrainingConfig import CentralBinaryDiscriminator
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.FullModel.GANBinaryCentralTrainingConfig import CentralBinaryGan
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.FullModel.WGANBinaryCentralTrainingConfig import CentralBinaryWGan
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Generator.WGenBinaryCentralTrainingConfig import CentralBinaryWGen
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Discriminator.WDiscBinaryCentralTrainingConfig import CentralBinaryWDisc
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.FullModel.ACGANCentralTrainingConfig import CentralACGan
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Generator.ACGenCentralTrainingConfig import CentralACGenerator
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Discriminator.ACDiscREALCentralTrainingConfig import CentralACDiscREAL
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Discriminator.ACDiscCentralTrainingConfig import CentralACDisc

# from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.FullModel.CANGANCentralTrainingConfig import CANGAN

################################################################################################################
#                      CENTRAL/Local TRAINING CONFIG CLIENT CLASS OBJECT LOADER                                #
################################################################################################################

def modelCentralTrainingConfigLoad(nids, discriminator, generator, GAN, dataset_used, model_type, train_type,
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

    # if models are adv nids
    elif model_type == "NIDS-IOT-Binary":
        client = None

    elif model_type == "NIDS-IOT-Multiclass":
        client = None

    elif model_type == "NIDS-IOT-Multiclass-Dynamic":
        client = None

    # if model are the GAN types
    elif model_type == 'GAN':
        if train_type == "Both":
            client = CentralBinaryGan(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                  y_test_data, BATCH_SIZE,
                                  noise_dim, epochs, steps_per_epoch, learning_rate)
        elif train_type == "Generator":
            client = CentralGenerator(generator, discriminator, X_train_data, X_val_data, y_train_data, y_val_data,
                                      X_test_data, y_test_data, BATCH_SIZE,
                                      noise_dim, epochs, steps_per_epoch)

        elif train_type == "Discriminator":
            client = CentralBinaryDiscriminator(discriminator, generator, X_train_data, X_val_data, y_train_data, y_val_data,
                                          X_test_data, y_test_data, BATCH_SIZE, noise_dim, epochs, steps_per_epoch)


    elif model_type == 'WGAN-GP':
        if train_type == "Both":
            client = CentralBinaryWGan(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                   y_test_data, BATCH_SIZE,
                                   noise_dim, epochs, steps_per_epoch, learning_rate)
        elif train_type == "Generator":
            client = CentralBinaryWGen(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                   y_test_data, BATCH_SIZE,
                                   noise_dim, epochs, steps_per_epoch, learning_rate)
        elif train_type == "Discriminator":
            client = CentralBinaryWDisc(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                   y_test_data, BATCH_SIZE,
                                   noise_dim, epochs, steps_per_epoch, learning_rate)

    elif model_type == 'AC-GAN':
        if train_type == "Both":
            client = CentralACGan(discriminator, generator, nids, X_train_data, X_val_data, y_train_data,
                              y_val_data, X_test_data, y_test_data, BATCH_SIZE,
                              noise_dim, latent_dim, num_classes, input_dim, epochs, steps_per_epoch, learning_rate)
        elif train_type == "Generator":
            client = CentralACGenerator(discriminator, generator, nids, X_train_data, X_val_data, y_train_data,
                                        y_val_data, X_test_data, y_test_data, BATCH_SIZE, noise_dim, latent_dim,
                                        num_classes, input_dim, epochs, steps_per_epoch)
        elif train_type == "Discriminator":
            client = CentralACDiscREAL(discriminator=discriminator,
                                       generator=generator,
                                       nids=nids,
                                       x_train=X_train_data,
                                       x_val=X_val_data,
                                       y_train=y_train_data,
                                       y_val=y_val_data,
                                       x_test=X_test_data,
                                       y_test=y_test_data,
                                       BATCH_SIZE=BATCH_SIZE,
                                       noise_dim=noise_dim,
                                       latent_dim=latent_dim,
                                       num_classes=num_classes,
                                       input_dim=input_dim,
                                       epochs=epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       learning_rate=learning_rate,
                                       use_class_labels=True,
                                       log_file=trainingLog)
            # optionally use discriminator that uses centralized training with fake data as well
            # client = CentralACDisc(discriminator=discriminator,
            #                       generator=generator,
            #                       nids=nids,
            #                       x_train=X_train_data,
            #                       x_val=X_val_data,
            #                       y_train=y_train_data,
            #                       y_val=y_val_data,
            #                       x_test=X_test_data,
            #                       y_test=y_test_data,
            #                       BATCH_SIZE=BATCH_SIZE,
            #                       noise_dim=noise_dim,
            #                       latent_dim=latent_dim,
            #                       num_classes=num_classes,
            #                       input_dim=input_dim,
            #                       epochs=epochs,
            #                       steps_per_epoch=steps_per_epoch,
            #                       learning_rate=learning_rate,
            #                       log_file=trainingLog)

        # elif model_type == 'CANGAN':
        #     if train_type == "Both":
        #         # client = CentralCANGAN(discriminator, generator, nids, X_train_data, X_val_data, y_train_data,
        #         #                       y_val_data, X_test_data, y_test_data, BATCH_SIZE,
        #         #                       noise_dim, latent_dim, num_classes, input_dim, epochs, steps_per_epoch,
        #         #                       learning_rate)
            # elif train_type == "Generator":
            #
            # elif train_type == "Discriminator":


    return client

#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
import os
import random
from datetime import datetime
import argparse
sys.path.append(os.path.abspath('../../..'))
# TensorFlow & Flower
if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']
import flwr as fl
import tensorflow as tf
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow_addons.layers import SpectralNormalization
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
from Config.modelStructures.NIDS.NIDS_Struct import create_CICIOT_Model, create_IOTBOTNET_Model, cnn_lstm_gru_model_multiclass, cnn_lstm_gru_model_binary, cnn_lstm_gru_model_multiclass_dynamic, create_optimized_NIDS_model
from Config.modelStructures.GAN.discriminatorStruct import create_discriminator_binary, create_discriminator_binary_optimized, create_discriminator_binary, build_AC_discriminator, create_discriminator
from Config.modelStructures.GAN.generatorStruct import create_generator, create_generator_optimized, build_AC_generator
from Config.modelStructures.GAN.ganStruct import create_model, load_GAN_model, create_model_binary, create_model_binary_optimized, create_model_W_binary, load_and_merge_ACmodels, create_model_AC


def modelCreateLoad(modelType, train_type, pretrainedNids, pretrainedGan, pretrainedGenerator, pretrainedDiscriminator,
                    dataset_used, input_dim, noise_dim, regularizationEnabled, DP_enabled, l2_alpha,
                    latent_dim, num_classes):
    nids = None
    discriminator = None
    generator = None
    GAN = None

    if pretrainedNids:
        print(f"Loading pretrained NIDS from {pretrainedNids}")
        with tf.keras.utils.custom_object_scope({'LogCosh': LogCosh}):
            nids = tf.keras.models.load_model(pretrainedNids)

    if modelType == 'NIDS':
        # Optionally load the pretrained nids model
        if pretrainedNids:
            print(f"Loading pretrained NIDS from {pretrainedNids}")
            with tf.keras.utils.custom_object_scope({'LogCosh': LogCosh}):
                nids = tf.keras.models.load_model(pretrainedNids)
        else:
            if dataset_used == "CICIOT":
                print("No pretrained discriminator provided. Creating a new mdoel.")

                nids = create_CICIOT_Model(input_dim, regularizationEnabled, DP_enabled, l2_alpha)
                # nids = create_optimized_NIDS_model(input_dim)
            elif dataset_used == "IOTBOTNET":
                print("No pretrained discriminator provided. Creating a new model.")

                nids = create_IOTBOTNET_Model(input_dim, regularizationEnabled, l2_alpha)

    elif modelType == 'NIDS-IOT-Binary':
        if pretrainedNids:
            print(f"Loading pretrained NIDS from {pretrainedNids}")
            nids = tf.keras.models.load_model(pretrainedNids)
        else:
            nids = cnn_lstm_gru_model_binary(input_dim)

    elif modelType == 'NIDS-IOT-Multiclass':
        if pretrainedNids:
            print(f"Loading pretrained NIDS from {pretrainedNids}")
            nids = tf.keras.models.load_model(pretrainedNids)
        else:
            nids = cnn_lstm_gru_model_multiclass(input_dim)

    elif modelType == 'NIDS-IOT-Multiclass-Dynamic':
        if pretrainedNids:
            print(f"Loading pretrained NIDS from {pretrainedNids}")
            nids = tf.keras.models.load_model(pretrainedNids)
        else:
            nids = cnn_lstm_gru_model_multiclass_dynamic(input_dim, num_classes)


    elif modelType == 'GAN':
        if train_type == 'Both':
            if pretrainedGan:
                print(f"Loading pretrained GAN Model from {pretrainedGan}")
                GAN = tf.keras.models.load_model(pretrainedGan)

            elif pretrainedGenerator and not pretrainedDiscriminator:

                print(f"Pretrained Generator provided from {pretrainedGenerator}. Creating a new Discriminator model.")
                generator = tf.keras.models.load_model(pretrainedGenerator)

                discriminator = create_discriminator_binary(input_dim)

                GAN = load_GAN_model(generator, discriminator)

            elif pretrainedDiscriminator and not pretrainedGenerator:
                print(
                    f"Pretrained Discriminator provided from {pretrainedDiscriminator}. Creating a new Generator model.")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)

                generator = create_generator(input_dim, noise_dim)

                GAN = load_GAN_model(generator, discriminator)

            elif pretrainedDiscriminator and pretrainedGenerator:
                print(
                    f"Pretrained Generator and Discriminator provided from {pretrainedGenerator} , {pretrainedDiscriminator}")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)
                generator = tf.keras.models.load_model(pretrainedGenerator)

                GAN = load_GAN_model(generator, discriminator)

            else:
                print("No pretrained GAN provided. Creating a new GAN model.")
                GAN = create_model_binary(input_dim, noise_dim)

        elif train_type == 'Generator':
            # Load or create the discriminator model
            if pretrainedDiscriminator:
                print(f"Loading pretrained discriminator from {pretrainedDiscriminator}")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)
            else:
                print("No pretrained discriminator provided. Creating a new discriminator model.")
                discriminator = create_discriminator_binary(input_dim)

            # Load or create the generator model
            if pretrainedGenerator:
                print(f"Loading pretrained generator from {pretrainedGenerator}")
                generator = tf.keras.models.load_model(pretrainedGenerator)
            else:
                print("No pretrained generator provided. Creating a new generator.")
                generator = create_generator(input_dim, noise_dim)

        elif train_type == 'Discriminator':
            if pretrainedDiscriminator:
                print(f"Loading pretrained discriminator from {pretrainedDiscriminator}")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)
            else:
                print("No pretrained discriminator provided. Creating a new discriminator.")
                discriminator = create_discriminator(input_dim)

            # Load or create the generator model
            if pretrainedGenerator:
                print(f"Loading pretrained generator from {pretrainedGenerator}")
                generator = tf.keras.models.load_model(pretrainedGenerator)
            else:
                print("No pretrained generator provided. Creating a new generator.")
                generator = create_generator(input_dim, noise_dim)

    elif modelType == 'WGAN-GP':
        if train_type == 'Both':
            # Load or create the discriminator, generator, or whole gan model
            if pretrainedGan:
                print(f"Loading pretrained GAN Model from {pretrainedGan}")
                with tf.keras.utils.custom_object_scope({"SpectralNormalization": SpectralNormalization}):
                    GAN = tf.keras.models.load_model(pretrainedGan)

            elif pretrainedGenerator and not pretrainedDiscriminator:
                print(f"Pretrained Generator provided from {pretrainedGenerator}. Creating a new Discriminator model.")
                generator = tf.keras.models.load_model(pretrainedGenerator)

                discriminator = create_discriminator_binary(input_dim)

                GAN = load_GAN_model(generator, discriminator)

            elif pretrainedDiscriminator and not pretrainedGenerator:
                print(
                    f"Pretrained Discriminator provided from {pretrainedDiscriminator}. Creating a new Generator model.")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)

                generator = create_generator(input_dim, noise_dim)

                GAN = load_GAN_model(generator, discriminator)

            elif pretrainedDiscriminator and pretrainedGenerator:
                print(
                    f"Pretrained Generator and Discriminator provided from {pretrainedGenerator} , {pretrainedDiscriminator}")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)
                generator = tf.keras.models.load_model(pretrainedGenerator)

                GAN = load_GAN_model(generator, discriminator)

            else:
                print("No pretrained GAN provided. Creating a new GAN model.")
                GAN = create_model_W_binary(input_dim, noise_dim)

        elif train_type == 'Generator':
            # Load or create the discriminator model
            if pretrainedDiscriminator:
                print(f"Loading pretrained discriminator from {pretrainedDiscriminator}")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)
            else:
                print("No pretrained discriminator provided. Creating a new discriminator model.")
                discriminator = create_discriminator_binary(input_dim)

            # Load or create the generator model
            if pretrainedGenerator:
                print(f"Loading pretrained generator from {pretrainedGenerator}")
                generator = tf.keras.models.load_model(pretrainedGenerator)
            else:
                print("No pretrained generator provided. Creating a new generator.")
                generator = create_generator(input_dim, noise_dim)

        elif train_type == 'Discriminator':
            if pretrainedDiscriminator:
                print(f"Loading pretrained discriminator from {pretrainedDiscriminator}")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)
            else:
                print("No pretrained discriminator provided. Creating a new discriminator.")
                discriminator = create_discriminator(input_dim)

            # Load or create the generator model
            if pretrainedGenerator:
                print(f"Loading pretrained generator from {pretrainedGenerator}")
                generator = tf.keras.models.load_model(pretrainedGenerator)
            else:
                print("No pretrained generator provided. Creating a new generator.")
                generator = create_generator(input_dim, noise_dim)

    elif modelType == 'AC-GAN':
        if train_type == 'Both':
            if pretrainedDiscriminator and pretrainedGenerator:
                print(
                    f"Pretrained Generator and Discriminator provided from {pretrainedGenerator} , {pretrainedDiscriminator}")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)

                generator = tf.keras.models.load_model(pretrainedGenerator)

                GAN = load_and_merge_ACmodels(pretrainedGenerator, pretrainedDiscriminator, latent_dim, num_classes, input_dim)

            elif pretrainedGenerator and not pretrainedDiscriminator:
                print(f"Pretrained Generator provided from {pretrainedGenerator}. Creating a new Discriminator model.")
                generator = tf.keras.models.load_model(pretrainedGenerator)

                discriminator = build_AC_discriminator(input_dim, num_classes)

                GAN = load_and_merge_ACmodels(pretrainedGenerator, pretrainedDiscriminator, latent_dim, num_classes, input_dim)

            elif pretrainedDiscriminator and not pretrainedGenerator:
                print(
                    f"Pretrained Discriminator provided from {pretrainedDiscriminator}. Creating a new Generator model.")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)

                generator = build_AC_generator(latent_dim, num_classes, input_dim)

                GAN = load_and_merge_ACmodels(pretrainedGenerator, pretrainedDiscriminator, latent_dim, num_classes, input_dim)

            else:
                print("No pretrained ACGAN provided. Creating a new ACGAN model.")
                generator = build_AC_generator(latent_dim, num_classes, input_dim)

                discriminator = build_AC_discriminator(input_dim, num_classes)

                GAN = load_and_merge_ACmodels(pretrainedGenerator, pretrainedDiscriminator, latent_dim, num_classes, input_dim)

        elif train_type == 'Generator':
            if pretrainedDiscriminator and pretrainedGenerator:
                print(
                    f"Pretrained Generator and Discriminator provided from {pretrainedGenerator} , {pretrainedDiscriminator}")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)

                generator = tf.keras.models.load_model(pretrainedGenerator)

            elif pretrainedGenerator and not pretrainedDiscriminator:
                print(f"Pretrained Generator provided from {pretrainedGenerator}. Creating a new Discriminator model.")
                generator = tf.keras.models.load_model(pretrainedGenerator)

                discriminator = build_AC_discriminator(input_dim, num_classes)

            elif pretrainedDiscriminator and not pretrainedGenerator:
                print(
                    f"Pretrained Discriminator provided from {pretrainedDiscriminator}. Creating a new Generator model.")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)

                generator = build_AC_generator(latent_dim, num_classes, input_dim)

            else:
                print("No pretrained ACGAN provided. Creating a new ACGAN model.")
                generator = build_AC_generator(latent_dim, num_classes, input_dim)

                discriminator = build_AC_discriminator(input_dim, num_classes)
        elif train_type == 'Discriminator':
            if pretrainedDiscriminator and pretrainedGenerator:
                print(
                    f"Pretrained Generator and Discriminator provided from {pretrainedGenerator} , {pretrainedDiscriminator}")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)

                generator = tf.keras.models.load_model(pretrainedGenerator)

            elif pretrainedGenerator and not pretrainedDiscriminator:
                print(f"Pretrained Generator provided from {pretrainedGenerator}. Creating a new Discriminator model.")
                generator = tf.keras.models.load_model(pretrainedGenerator)

                discriminator = build_AC_discriminator(input_dim, num_classes)

            elif pretrainedDiscriminator and not pretrainedGenerator:
                print(
                    f"Pretrained Discriminator provided from {pretrainedDiscriminator}. Creating a new Generator model.")
                discriminator = tf.keras.models.load_model(pretrainedDiscriminator)

                generator = build_AC_generator(latent_dim, num_classes, input_dim)

            else:
                print("No pretrained ACGAN provided. Creating a new ACGAN model.")
                generator = build_AC_generator(latent_dim, num_classes, input_dim)

                discriminator = build_AC_discriminator(input_dim, num_classes)


    return nids, discriminator, generator, GAN

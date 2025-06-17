#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
import os
sys.path.append(os.path.abspath('../../..'))
import random
import time
from datetime import datetime
import argparse

from modelStructures.GAN.discriminatorStruct import create_discriminator, create_discriminator_binary_optimized, create_discriminator_binary, create_discriminator_binary_optimized_spectral, create_W_discriminator_binary_optimized, build_AC_discriminator
from modelStructures.GAN.generatorStruct import create_generator, create_generator_optimized, create_W_generator, build_AC_generator

if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']

import flwr as fl

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LSTM, Conv1D, MaxPooling1D, GRU, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GAN
def create_model(input_dim, noise_dim):
    model = Sequential()

    model.add(create_generator(input_dim, noise_dim))
    model.add(create_discriminator(input_dim))

    return model

# GAN Binary
def create_model_binary_optimized(input_dim, noise_dim):
    model = Sequential()

    model.add(create_generator_optimized(input_dim, noise_dim))
    model.add(create_discriminator_binary_optimized(input_dim))

    return model

def create_model_binary(input_dim, noise_dim):
    model = Sequential()

    model.add(create_generator(input_dim, noise_dim))
    model.add(create_discriminator_binary(input_dim))

    return model

# ACGAN
def create_model_AC(latent_dim, num_classes, input_dim):
    # Define inputs
    noise_input = Input(shape=(latent_dim,), name="noise_input")
    label_input = Input(shape=(1,), dtype='int32', name="label_input")

    # Generate data
    generated_data = build_AC_generator(latent_dim, num_classes, input_dim)([noise_input, label_input])
    # Pass generated data to discriminator
    validity, class_output = build_AC_discriminator(input_dim, num_classes)(generated_data)

    # Create merged model
    merged_model = Model(inputs=[noise_input, label_input],
                         outputs=[validity, class_output],
                         name="ACGAN")
    return merged_model

# WGAN Binary
def create_model_W_binary(input_dim, noise_dim):
    model = Sequential()

    model.add(create_W_generator(input_dim, noise_dim))
    model.add(create_W_discriminator_binary_optimized(input_dim))

    return model


# model Loading
def load_GAN_model(generator, discriminator):
    model = Sequential([generator, discriminator])

    return model


from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model


def load_and_merge_ACmodels_V0(generator_path, discriminator_path, latent_dim, num_classes, input_dim):

    pretrained_generator = None
    pretrained_discriminator = None

    # Load pre-trained models from disk
    if generator_path is not None or discriminator_path is not None:

        pretrained_generator = load_model(generator_path)
        pretrained_discriminator = load_model(discriminator_path)

    # Define inputs for the merged model
    noise_input = Input(shape=(latent_dim,), name="noise_input")
    label_input = Input(shape=(1,), dtype='int32', name="label_input")

    if generator_path is not None:
        # Generate data using the pre-trained generator
        generated_data = pretrained_generator([noise_input, label_input])
    else:
        generated_data = build_AC_generator(latent_dim, num_classes, input_dim)([noise_input, label_input])

    if discriminator_path is not None:
        # Pass the generated data to the pre-trained discriminator
        validity, class_output = pretrained_discriminator(generated_data)
    else:
        validity, class_output = build_AC_discriminator(input_dim, num_classes)(generated_data)

    # Create and return the merged AC-GAN model
    merged_model = Model(
        inputs=[noise_input, label_input],
        outputs=[validity, class_output],
        name="ACGAN"
    )
    return merged_model


def load_and_merge_ACmodels(generator_path, discriminator_path, latent_dim, num_classes, input_dim):
    noise_input = Input(shape=(latent_dim,), name="noise_input")
    label_input = Input(shape=(1,), dtype='int32', name="label_input")

    # Build or load the generator.
    if generator_path is not None:
        gen_model = load_model(generator_path)
    else:
        gen_model = build_AC_generator(latent_dim, num_classes, input_dim)

    generated_data = gen_model([noise_input, label_input])
    generated_data = tf.identity(generated_data, name="ACGenerator")

    # Build or load the discriminator.
    if discriminator_path is not None:
        disc_model = load_model(discriminator_path)
    else:
        disc_model = build_AC_discriminator(input_dim, num_classes)

    validity, class_output = disc_model(generated_data)

    merged_model = Model(
        inputs=[noise_input, label_input],
        outputs=[validity, class_output],
        name="ACGAN"
    )

    # Store the generator and discriminator as attributes.
    merged_model.generator = gen_model
    merged_model.discriminator = disc_model
    return merged_model


# Submodel
def split_GAN_model(model):
    # Assuming `self.model` is the GAN model created with Sequential([generator, discriminator])
    generator = model.layers[0]
    discriminator = model.layers[1]

    return generator, discriminator

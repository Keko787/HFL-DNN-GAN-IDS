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


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model=None, model_save_path="global_model.h5", **kwargs):
        super().__init__(**kwargs)
        self.model_save_path = model_save_path
        self.model = model

    def aggregate_fit(self, server_round, results, failures):
        """Aggregates client results and saves the global model."""
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving global model after round {server_round}...")

            # Convert Parameters object to numpy arrays
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters[0])

            if len(aggregated_weights) == len(self.model.get_weights()):
                self.model.set_weights(aggregated_weights)
                self.model.save(self.model_save_path)
                print(f"Model saved at: {self.model_save_path}")
            else:
                print(
                    f"Warning: Weight mismatch. Expected {len(self.model.get_weights())} but got {len(aggregated_weights)}.")

        return aggregated_parameters




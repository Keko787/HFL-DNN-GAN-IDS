#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
import os
from datetime import datetime
import argparse
sys.path.append(os.path.abspath('../../..'))
# TensorFlow & Flower
if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']
import flwr as fl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# other plugins
# import math
# import glob
# from tqdm import tqdm
# import seaborn as sns
# import pickle
# import joblib

from Config.SessionConfig.datasetLoadProcess import datasetLoadProcess
from Config.SessionConfig.hyperparameterLoading import hyperparameterLoading
from Config.SessionConfig.modelCreateLoad import modelCreateLoad
from Config.SessionConfig.ModelTrainingConfigLoad.modelCentralTrainingConfigLoad import modelCentralTrainingConfigLoad
from Config.SessionConfig.ModelTrainingConfigLoad.modelFederatedTrainingConfigLoad import modelFederatedTrainingConfigLoad

################################################################################################################
#                                                   Execute                                                   #
################################################################################################################

def main():
    print("\n ////////////////////////////// \n")
    print("Centralized Training:", "\n")

    # Generate a static timestamp at the start of the script
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 1. Argument Parsing --- #
    parser = argparse.ArgumentParser(description='Select dataset, model selection, and to enable DP respectively')

    # -- Dataset Settings -- #
    parser.add_argument('--dataset', type=str, choices=["CICIOT", "IOTBOTNET", "IOT"], default="CICIOT",
                        help='Datasets to use: CICIOT, IOTBOTNET, IOT (different from IOTBOTNET)')
    parser.add_argument('--dataset_processing', type=str, choices=["Default", "MM[-1,-1]", "AC-GAN, IOT", "IOT-MinMax"],
                        default="Default",
                        help='Datasets to use: Default, MM[-1,1], AC-GAN, IOT')

    # -- Training / Model Parameters -- #
    parser.add_argument('--trainingArea', type=str, choices=["Central", "Federated"], default="Central",
                        help='Please select Central, Federated as the place to train the model')
    parser.add_argument("--host", type=str, default="1", help="Fixed Server node number 1-4, or type a custom ip address")
    parser.add_argument('--serverBased', action='store_true',
                        help='Only load the model structure and get the weights from the server')

    parser.add_argument('--model_type', type=str, choices=["NIDS", "NIDS-IOT-Binary", "NIDS-IOT-Multiclass", "NIDS-IOT-Multiclass-Dynamic", "GAN", "WGAN-GP", "AC-GAN"],
                        help='Please select NIDS, NIDS-IOT-Binary, NIDS-IOT-Multiclass, NIDS-IOT-Multiclass-Dynamic, GAN, WGAN-GP, or AC-GAN as the model type to train')

    parser.add_argument('--model_training', type=str, choices=["NIDS", "Generator", "Discriminator", "Both"], default="Both",
                        help='Please select NIDS, Generator, Discriminator, Both as the sub-model type to train')

    # Optional
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model")

    # -- Loading Models Optional -- +
    parser.add_argument('--pretrained_GAN', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)
    parser.add_argument('--pretrained_generator', type=str, help="Path to pretrained generator model (optional)",
                        default=None)
    parser.add_argument('--pretrained_discriminator', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)
    parser.add_argument('--pretrained_nids', type=str,
                        help="Path to pretrained nids model (optional)", default=None)

    # -- Saving Models Optional -- +
    parser.add_argument('--save_name', type=str,
                        help="name of model files you save as", default=f"{timestamp}")

    # init variables to handle arguments
    args = parser.parse_args()

    # argument variables
    dataset_used = args.dataset
    dataset_processing = args.dataset_processing

    # Model Spec
    host = args.host
    server_based = args.serverBased
    TrainingArea = args.trainingArea
    model_type = args.model_type
    train_type = args.model_training
    if model_type == "AC-GAN":
        dataset_processing = "AC-GAN"

    if model_type == "NIDS" or model_type == "NIDS-IOT-Binary" or model_type == "NIDS-IOT-Multiclass" or model_type == "NIDS-IOT-Multiclass-Dynamic":
        train_type = "NIDS"

    if model_type == "NIDS-IOT-Binary" or model_type == "NIDS-IOT-Multiclass" or model_type == "NIDS-IOT-Multiclass-Dynamic":
        dataset_used = "IOT"
    if dataset_processing == "IOT" or dataset_processing == "IOT-MinMax":
        dataset_used = "IOT"


    # Training / Hyper Param
    epochs = args.epochs
    regularizationEnabled = True
    DP_enabled = None
    earlyStopEnabled = None
    lrSchedRedEnabled = None
    modelCheckpointEnabled = None

    # Whole GAN model
    pretrainedGan = args.pretrained_GAN

    # Individual GAN Submodels
    pretrainedGenerator = args.pretrained_generator
    pretrainedDiscriminator = args.pretrained_discriminator

    # Optional NIDS
    pretrainedNids = args.pretrained_nids

    # Save/Record Param
    save_name = args.save_name
    evaluationLog = timestamp
    trainingLog = timestamp
    node = 1

    # -- Display selected arguments --#
    print("|MAIN CONFIG|", "\n")
    # main experiment config
    print("Selected DATASET:", dataset_used, "\n")
    print("Selected Preprocessing:", dataset_processing, "\n")
    print("Selected Model Type:", model_type, "\n")
    print("Selected Model Training:", train_type, "\n")
    print("Selected Epochs:", epochs, "\n")
    print("Loaded GAN/GAN-Variant:", pretrainedGan, "\n")
    print("Loaded Generator Model:", pretrainedGenerator, "\n")
    print("Loaded Discriminator Model:", pretrainedDiscriminator, "\n")
    print("Loaded NIDS Model:", pretrainedNids, "\n")
    print("Save Name for the models in Trained in this session:", save_name, "\n")

    # --- 2 Load & Preprocess Data ---#
    X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = datasetLoadProcess(dataset_used,
                                                                                                      dataset_processing)

    # --- 3 Model Hyperparameter & Training Parameters ---#
    (BATCH_SIZE, noise_dim, steps_per_epoch, input_dim, num_classes, latent_dim, betas, learning_rate, l2_alpha,
     l2_norm_clip, noise_multiplier, num_microbatches, metric_to_monitor_es, es_patience, restor_best_w,
     metric_to_monitor_l2lr, l2lr_patience, save_best_only,
     metric_to_monitor_mc, checkpoint_mode) = hyperparameterLoading(model_type, X_train_data,
                                                                    regularizationEnabled, DP_enabled, earlyStopEnabled,
                                                                    lrSchedRedEnabled, modelCheckpointEnabled)

    # --- 4 Model Loading & Creation ---#
    nids, discriminator, generator, GAN = modelCreateLoad(model_type, train_type, pretrainedNids, pretrainedGan,
                                                          pretrainedGenerator, pretrainedDiscriminator, dataset_used,
                                                          input_dim, noise_dim, regularizationEnabled, DP_enabled,
                                                          l2_alpha, latent_dim, num_classes)
    # --- 5A Load Training Config ---#
    if TrainingArea == "Federated":
        if server_based is True:  # Receive the global model weights initially to train with
            client = modelFederatedTrainingConfigLoad(nids, discriminator, generator, GAN, dataset_used, model_type, train_type,
                                                      earlyStopEnabled, DP_enabled, lrSchedRedEnabled, modelCheckpointEnabled,
                                                      X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data,
                                                      node, BATCH_SIZE, epochs, noise_dim, steps_per_epoch, input_dim, num_classes,
                                                      latent_dim, betas, learning_rate, l2_alpha, l2_norm_clip, noise_multiplier,
                                                      num_microbatches, metric_to_monitor_es, es_patience,
                                                      restor_best_w, metric_to_monitor_l2lr, l2lr_patience, save_best_only,
                                                      metric_to_monitor_mc, checkpoint_mode, evaluationLog, trainingLog)

        else:  # Use a pretrained model or receive model from peers.
            client = modelFederatedTrainingConfigLoad(nids, discriminator, generator, GAN, dataset_used, model_type, train_type,
                                                      earlyStopEnabled, DP_enabled, lrSchedRedEnabled, modelCheckpointEnabled,
                                                      X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data,
                                                      node, BATCH_SIZE, epochs, noise_dim, steps_per_epoch, input_dim, num_classes,
                                                      latent_dim, betas, learning_rate, l2_alpha, l2_norm_clip, noise_multiplier,
                                                      num_microbatches, metric_to_monitor_es, es_patience,
                                                      restor_best_w, metric_to_monitor_l2lr, l2lr_patience, save_best_only,
                                                      metric_to_monitor_mc, checkpoint_mode, evaluationLog, trainingLog)

        # -- Federated TRAINING -- #
        if host == "4":
            server_address = "192.168.129.8:8080"
        elif host == "3":
            server_address = "192.168.129.7:8080"
        elif host == "2":
            server_address = "192.168.129.6:8080"
        elif host == "1":
            server_address = "192.168.129.3:8080"
        else:  # custom address
            server_address = f"{host}:8080"

        print("Server Address: ", server_address)

        # --- 6/7A Train & Evaluate Model ---#
        fl.client.start_client(server_address=server_address, client=client.to_client())

        client.save(save_name)
        # -- EOF Federated TRAINING -- #

        # --- 5B Load Training Config ---#
    else:
        client = modelCentralTrainingConfigLoad(nids, discriminator, generator, GAN, dataset_used, model_type, train_type,
                                                earlyStopEnabled, DP_enabled, lrSchedRedEnabled, modelCheckpointEnabled,
                                                X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data,
                                                node, BATCH_SIZE, epochs, noise_dim, steps_per_epoch, input_dim, num_classes,
                                                latent_dim, betas, learning_rate, l2_alpha, l2_norm_clip, noise_multiplier,
                                                num_microbatches, metric_to_monitor_es, es_patience,
                                                restor_best_w, metric_to_monitor_l2lr, l2lr_patience, save_best_only,
                                                metric_to_monitor_mc, checkpoint_mode, evaluationLog, trainingLog)

        # --- 6A Centrally Train Model ---#
        client.fit()

        # --- 7A Centrally Evaluate Model ---#
        client.evaluate()

    # --- 8 Locally Save Model After Training ---#
        client.save(save_name)


if __name__ == "__main__":
    main()

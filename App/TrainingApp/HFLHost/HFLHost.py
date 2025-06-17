#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
import os
from datetime import datetime
import argparse
sys.path.append(os.path.abspath('..'))
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

from SessionConfig.datasetLoadProcess import datasetLoadProcess
from SessionConfig.hyperparameterLoading import hyperparameterLoading
from SessionConfig.modelCreateLoad import modelCreateLoad

# Load and Saving Configs
from ModelTrainingConfig.HostModelTrainingConfig.ModelManagement.ServerSaveOnlyConfig import SaveModelFedAvg
from ModelTrainingConfig.HostModelTrainingConfig.ModelManagement.ServerLoadOnlyConfig import LoadModelFedAvg
from ModelTrainingConfig.HostModelTrainingConfig.ModelManagement.ServerLoadNSaveConfig import LoadSaveModelFedAvg

# Fit on End configs
from ModelTrainingConfig.HostModelTrainingConfig.FitOnEnd.ServerNIDSFitOnEndConfig import NIDSFitOnEndStrategy
from ModelTrainingConfig.HostModelTrainingConfig.FitOnEnd.ServerDiscBinaryFitOnEndConfig import DiscriminatorSyntheticStrategy
from ModelTrainingConfig.HostModelTrainingConfig.FitOnEnd.ServerWDiscFitOnEndConfig import WDiscriminatorSyntheticStrategy
from ModelTrainingConfig.HostModelTrainingConfig.FitOnEnd.ServerACDiscFitOnEndConfig import ACDiscriminatorSyntheticStrategy

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
                        default="Default", help='Datasets to use: Default, MM[-1,1], AC-GAN, IOT')

    # -- Server Hosting Modes -- #
    parser.add_argument('--serverLoad', action='store_true',
                        help='Only load the model structure and get the weights from the server')
    parser.add_argument('--serverSave', action='store_true',
                        help='Only load the model structure and get the weights from the server')
    parser.add_argument('--fitOnEnd', action='store_true',
                        help='Only load the model structure and get the weights from the server')

    # -- Training / Model Parameters -- #
    parser.add_argument('--model_type', type=str,
                        choices=["NIDS", "NIDS-IOT-Binary", "NIDS-IOT-Multiclass", "NIDS-IOT-Multiclass-Dynamic", "GAN",
                                 "WGAN-GP", "AC-GAN"], help='Please select NIDS, NIDS-IOT-Binary, NIDS-IOT-Multiclass, NIDS-IOT-Multiclass-Dynamic, GAN, WGAN-GP, or AC-GAN as the model type to train')

    parser.add_argument('--model_training', type=str, choices=["NIDS", "Discriminator", "Both"],
                        default="Both",
                        help='Please select NIDS, Generator, Discriminator, Both as the sub-model type to train')

    # Optional
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model")

    parser.add_argument("--rounds", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], default=1,
                        help="Rounds of training 1-10")

    parser.add_argument("--synth_portion", type=float, choices=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6], default=0,
                       help="Percentage of Synthetic data compared to training data size to be augmented to that dataset, 0-0.6")

    parser.add_argument("--min_clients", type=int, choices=[1, 2, 3, 4, 5, 6], default=2,
                        help="Minimum number of clients required for training")

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
    fitOnEnd = args.fitOnEnd
    serverSave = args.serverSave
    serverLoad = args.serverLoad
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
    synth_portion = args.synth_portion
    regularizationEnabled = True
    DP_enabled = None
    earlyStopEnabled = None
    lrSchedRedEnabled = None
    modelCheckpointEnabled = None

    roundInput = args.rounds
    minClients = args.min_clients

    # Whole GAN model
    pretrainedGan = args.pretrained_GAN

    # Individual GAN Submodels
    pretrainedGenerator = args.pretrained_generator
    pretrainedDiscriminator = args.pretrained_discriminator

    # Optional NIDS
    pretrainedNids = args.pretrained_nids

    # Save/Record Param
    save_name_input = args.save_name
    evaluationLog = timestamp
    trainingLog = timestamp
    node = 1

    # Make Save name for model based on Arguments
    save_name = f""
    if fitOnEnd is True:
        save_name = f"fitOnEnd_{dataset_used}_{dataset_processing}_{model_type}_{train_type}_{save_name_input}.h5"
    # if base strategies
    else:
        save_name = f"{model_type}_{train_type}_{save_name_input}.h5"

    # -- Display selected arguments --#
    print("|MAIN CONFIG|", "\n")
    # main experiment config
    if fitOnEnd is True:
        print("Selected DATASET:", dataset_used, "\n")
        print("Selected Preprocessing:", dataset_processing, "\n")
        print("Selected Model Type:", model_type, "\n")
        print("Selected Model Training:", train_type, "\n")
        print("Selected Epochs:", epochs, "\n")
        print("Loaded GAN/GAN-Variant:", dataset_used, "\n")
    print("Loaded Generator Model:", pretrainedGenerator, "\n")
    print("Loaded Discriminator Model:", pretrainedDiscriminator, "\n")
    print("Loaded NIDS Model:", pretrainedNids, "\n")
    print("Save Name for the models in Trained in this session:", save_name, "\n")

    # -- 1B determine whether to do default federation hosting or do a custom strategy --#
    if serverLoad is False and serverSave is False and fitOnEnd is False:
        # --- Default, No Loading, No Saving ---#
        fl.server.start_server(
            config=fl.server.ServerConfig(num_rounds=roundInput),
            strategy=fl.server.strategy.FedAvg(
                min_fit_clients=minClients,
                min_evaluate_clients=minClients,
                min_available_clients=minClients
            )
        )

    # if the user wants to either load, save, or fit a model
    else:
        # --- 2 Load & Preprocess Data ---#
        X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = datasetLoadProcess(dataset_used,
                                                                                                          dataset_processing)

        # --- 3 Model Hyperparameter & Training Parameters ---#
        (BATCH_SIZE, noise_dim, steps_per_epoch, input_dim, num_classes, latent_dim, betas, learning_rate, l2_alpha,
         l2_norm_clip, noise_multiplier, num_microbatches, metric_to_monitor_es, es_patience, restor_best_w,
         metric_to_monitor_l2lr, l2lr_patience, save_best_only,
         metric_to_monitor_mc, checkpoint_mode) = hyperparameterLoading(model_type, X_train_data,
                                                                        regularizationEnabled, DP_enabled,
                                                                        earlyStopEnabled,
                                                                        lrSchedRedEnabled, modelCheckpointEnabled)

        # --- 4 Model Loading & Creation ---#
        nids, discriminator, generator, GAN = modelCreateLoad(model_type, train_type, pretrainedNids, pretrainedGan,
                                                              pretrainedGenerator, pretrainedDiscriminator,
                                                              dataset_used,
                                                              input_dim, noise_dim, regularizationEnabled, DP_enabled,
                                                              l2_alpha, latent_dim, num_classes)
        # -- 5. run server based on selected config -- #
        # selet model for base hosting config
        if train_type == "GAN":
            model = GAN
        elif train_type == "Discriminator":
            model = discriminator
        else:
            model = nids

        # Non Fit on end Strats
        if fitOnEnd is False:
            # Server Load only Config
            if serverLoad is True and serverSave is False:
                # --- Load Model ---#
                fl.server.start_server(
                    config=fl.server.ServerConfig(num_rounds=roundInput),
                    strategy=LoadModelFedAvg(
                        model=model,
                        min_fit_clients=2,
                        min_evaluate_clients=2,
                        min_available_clients=2
                    )
                )

            # Server saving only
            elif serverLoad is False and serverSave is True:
                # --- Save Model ---#
                fl.server.start_server(
                    config=fl.server.ServerConfig(num_rounds=roundInput),
                    strategy=SaveModelFedAvg(
                        model=model,
                        model_save_path=save_name,
                        min_fit_clients=2,
                        min_evaluate_clients=2,
                        min_available_clients=2
                    )
                )

            # Server Load and Save model
            elif serverLoad is True and serverSave is True:
                # --- Load and Save Model ---#
                fl.server.start_server(
                    config=fl.server.ServerConfig(num_rounds=roundInput),
                    strategy=LoadSaveModelFedAvg(
                        model=model,
                        model_save_path=save_name,
                        min_fit_clients=2,
                        min_evaluate_clients=2,
                        min_available_clients=2
                    )
                )

        # If the server is fitting the global model at end of federation round
        else:
            # NIDS fit on end advanced synthetic training
            if train_type == "NIDS":
                fl.server.start_server(
                    config=fl.server.ServerConfig(num_rounds=roundInput),
                    strategy=NIDSFitOnEndStrategy(
                        discriminator=discriminator,  # Pre-trained or newly created discriminator
                        generator=generator,  # Pre-trained or newly created generator
                        nids=nids,
                        dataset_used=dataset_used,  # or "IOTBOTNET" depending on dataset
                        node=node,  # Server node identifier
                        earlyStopEnabled=earlyStopEnabled,  # Enable early stopping
                        lrSchedRedEnabled=lrSchedRedEnabled,
                        modelCheckpointEnabled=modelCheckpointEnabled,
                        DP_enabled=DP_enabled,  # Differential privacy enabled/disabled
                        X_train_data=X_train_data, y_train_data=y_train_data,  # Server-side training data
                        X_test_data=X_test_data, y_test_data=y_test_data,  # Test data
                        X_val_data=X_val_data, y_val_data=y_val_data,  # Validation data
                        l2_norm_clip=l2_norm_clip,  # L2 norm clipping for DP
                        noise_multiplier=noise_multiplier,  # Noise multiplier for DP
                        num_microbatches=num_microbatches,  # Microbatches for DP
                        batch_size=BATCH_SIZE,  # Training batch size
                        epochs=epochs,  # Number of fine-tuning epochs
                        steps_per_epoch=steps_per_epoch,  # Training steps per epoch
                        learning_rate=learning_rate,  # Optimizer learning rate
                        synth_portion=synth_portion,  # Portion of synthetic data used
                        latent_dim=latent_dim,
                        num_classes=num_classes,
                        metric_to_monitor_es=metric_to_monitor_es,  # Early stopping monitor metric
                        es_patience=es_patience,  # Early stopping patience
                        restor_best_w=restor_best_w,  # Restore best weights on early stopping
                        metric_to_monitor_l2lr=metric_to_monitor_l2lr,  # Learning rate schedule monitor
                        l2lr_patience=l2lr_patience,  # Learning rate schedule patience
                        save_best_only=save_best_only,  # Save best model only
                        metric_to_monitor_mc=metric_to_monitor_mc,  # Model checkpoint monitor metric
                        checkpoint_mode=checkpoint_mode,  # Save best model based on max value of metric
                        save_name=save_name,
                        serverLoad=serverLoad,
                    )
                )

            # fit on end (also save) discriminator from GAN models
            else:
                # Discriminator advanced global synthetic training
                if model_type == "GAN":
                    fl.server.start_server(
                        config=fl.server.ServerConfig(num_rounds=roundInput),
                        strategy=DiscriminatorSyntheticStrategy(
                            gan=GAN,
                            generator=generator,
                            discriminator=discriminator,
                            x_train=X_train_data,
                            x_val=X_val_data,
                            y_train=y_train_data,
                            y_val=y_val_data,
                            x_test=X_test_data,
                            y_test=y_test_data,
                            BATCH_SIZE=BATCH_SIZE,
                            noise_dim=noise_dim,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            dataset_used=dataset_used,
                            input_dim=input_dim
                            )
                    )

                # WGAN-GP Discriminator advanced global synthetic training
                elif model_type == "WGAN-GP":
                    fl.server.start_server(
                        config=fl.server.ServerConfig(num_rounds=roundInput),
                        strategy=WDiscriminatorSyntheticStrategy(
                            gan=GAN,
                            nids=nids,
                            x_train=X_train_data,
                            x_val=X_val_data,
                            y_train=y_train_data,
                            y_val=y_val_data,
                            x_test=X_test_data,
                            y_test=y_test_data,
                            BATCH_SIZE=BATCH_SIZE,
                            noise_dim=noise_dim,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            dataset_used=dataset_used,
                            input_dim=input_dim,
                        )
                    )

                # AC Discriminator advanced global synthetic training
                elif model_type == "AC-GAN":
                    fl.server.start_server(
                        config=fl.server.ServerConfig(num_rounds=roundInput),
                        strategy=ACDiscriminatorSyntheticStrategy(
                            GAN=GAN,
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
                            log_file="training.log")
                    )


if __name__ == "__main__":
    main()

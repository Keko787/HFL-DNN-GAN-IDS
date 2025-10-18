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
from Config.SessionConfig.ArgumentConfigLoad import parse_HFL_Host_args, display_HFL_host_opening_message
from Config.SessionConfig.ModelTrainingConfigLoad.HFLStrategyTrainingConfigLoad import _run_standard_federation_strategies, _run_fit_on_end_strategies


################################################################################################################
#                                                   Execute                                                   #
################################################################################################################

def main():
    # Generate a static timestamp at the start of the script
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 1. Parse Arguments and Display Opening Message --- #
    args = parse_HFL_Host_args()
    display_HFL_host_opening_message(args, args.timestamp)

    # --- 2. Extract variables from args for compatibility with existing code --- #
    dataset_used = args.dataset
    dataset_processing = args.dataset_processing

    # Model Spec
    fitOnEnd = args.fitOnEnd
    serverSave = args.serverSave
    serverLoad = args.serverLoad
    model_type = args.model_type
    train_type = args.model_training

    # Training / Hyper Param
    epochs = args.epochs
    synth_portion = args.synth_portion
    regularizationEnabled = args.regularizationEnabled
    DP_enabled = args.DP_enabled
    earlyStopEnabled = args.earlyStopEnabled
    lrSchedRedEnabled = args.lrSchedRedEnabled
    modelCheckpointEnabled = args.modelCheckpointEnabled

    roundInput = args.rounds
    minClients = args.min_clients

    # Pretrained models
    pretrainedGan = args.pretrained_GAN
    pretrainedGenerator = args.pretrained_generator
    pretrainedDiscriminator = args.pretrained_discriminator
    pretrainedNids = args.pretrained_nids

    # Save/Record Param
    save_name_input = args.save_name
    save_name = args.full_save_name  # Use the pre-computed save name
    evaluationLog = args.evaluationLog
    trainingLog = args.trainingLog
    node = args.node


    # --- 3. Determine federation strategy --- #
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
        print("🔄 Loading data and initializing models...")
        # --- 4. Load & Preprocess Data ---#
        X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = datasetLoadProcess(dataset_used=args.dataset,
                                                                                                          dataset_preprocessing=args.dataset_processing,
                                                                                                          ciciot_train_sample_size=args.ciciot_train_sample_size,
                                                                                                          ciciot_test_sample_size=args.ciciot_test_sample_size,
                                                                                                          ciciot_training_dataset_size=args.ciciot_training_dataset_size,
                                                                                                          ciciot_testing_dataset_size=args.ciciot_testing_dataset_size,
                                                                                                          ciciot_attack_eval_samples_ratio=args.ciciot_attack_eval_samples_ratio,
                                                                                                          ciciot_random_seed=args.ciciot_random_seed)

        # --- 5. Model Hyperparameter & Training Parameters ---#
        (BATCH_SIZE, noise_dim, steps_per_epoch, input_dim, num_classes, latent_dim, betas, learning_rate, l2_alpha,
         l2_norm_clip, noise_multiplier, num_microbatches, metric_to_monitor_es, es_patience, restor_best_w,
         metric_to_monitor_l2lr, l2lr_patience, save_best_only,
         metric_to_monitor_mc, checkpoint_mode) = hyperparameterLoading(model_type, X_train_data,
                                                                        regularizationEnabled, DP_enabled,
                                                                        earlyStopEnabled,
                                                                        lrSchedRedEnabled, modelCheckpointEnabled)

        # --- 6. Model Loading & Creation ---#
        nids, discriminator, generator, GAN = modelCreateLoad(model_type, train_type, pretrainedNids, pretrainedGan,
                                                              pretrainedGenerator, pretrainedDiscriminator,
                                                              dataset_used,
                                                              input_dim, noise_dim, regularizationEnabled, DP_enabled,
                                                              l2_alpha, latent_dim, num_classes)
        # --- 7. Select model for base hosting config --- #
        # selet model for base hosting config
        if train_type == "GAN":
            model = GAN
        elif train_type == "Discriminator":
            model = discriminator
        else:
            model = nids

            # --- 8. Run server based on selected config --- #
            if not fitOnEnd:
                _run_standard_federation_strategies(
                    serverLoad, serverSave, roundInput, minClients, model, save_name
                )
            else:
                _run_fit_on_end_strategies(
                    train_type, model_type, roundInput, args,
                    discriminator, generator, nids, GAN,
                    X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data,
                    BATCH_SIZE, noise_dim, steps_per_epoch, input_dim, num_classes, latent_dim,
                    epochs, learning_rate, synth_portion, l2_norm_clip, noise_multiplier,
                    num_microbatches, metric_to_monitor_es, es_patience, restor_best_w,
                    metric_to_monitor_l2lr, l2lr_patience, save_best_only, metric_to_monitor_mc,
                    checkpoint_mode, save_name, serverLoad, dataset_used, earlyStopEnabled,
                    lrSchedRedEnabled, modelCheckpointEnabled, DP_enabled, node
                )


if __name__ == "__main__":
    main()

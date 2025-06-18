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
from Config.SessionConfig.ArgumentConfigLoad import parse_training_client_args, display_training_client_opening_message
from Config.SessionConfig.ModelTrainingConfigLoad.modelCentralTrainingConfigLoad import modelCentralTrainingConfigLoad
from Config.SessionConfig.ModelTrainingConfigLoad.modelFederatedTrainingConfigLoad import modelFederatedTrainingConfigLoad


################################################################################################################
#                                                   Main                                                  #
################################################################################################################
def main():
    # Generate a static timestamp at the start of the script
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 1. Parse Arguments (LIGHTWEIGHT!) --- #
    args = parse_training_client_args()

    # -- Display selected arguments --#
    display_training_client_opening_message(args, timestamp)

    # --- 2 Load & Preprocess Data ---#
    print("📊 Loading and preprocessing dataset...")
    X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = datasetLoadProcess( args.dataset, args.dataset_processing)

    print("✅ Dataset loaded successfully!")
    print(f"   • Training samples: {len(X_train_data)}")
    print(f"   • Validation samples: {len(X_val_data)}")
    print(f"   • Test samples: {len(X_test_data)}")
    print()

    # --- 3 Model Hyperparameter & Training Parameters ---#
    print("⚙️  Loading hyperparameters and training configuration...")
    (BATCH_SIZE, noise_dim, steps_per_epoch, input_dim, num_classes, latent_dim, betas, learning_rate, l2_alpha,
     l2_norm_clip, noise_multiplier, num_microbatches, metric_to_monitor_es, es_patience, restor_best_w,
     metric_to_monitor_l2lr, l2lr_patience, save_best_only,
     metric_to_monitor_mc, checkpoint_mode) = hyperparameterLoading(args.model_type, X_train_data,
                                                                    args.regularizationEnabled, args.DP_enabled,
                                                                    args.earlyStopEnabled, args.lrSchedRedEnabled,
                                                                    args.modelCheckpointEnabled)

    print("✅ Hyperparameters configured!")
    print(f"   • Batch Size: {BATCH_SIZE}")
    print(f"   • Steps per Epoch: {steps_per_epoch}")
    print(f"   • Input Dimension: {input_dim}")
    if num_classes:
        print(f"   • Number of Classes: {num_classes}")
    print()

    # --- 4 Model Loading & Creation ---#
    print("🧠 Creating and/or loading models...")
    nids, discriminator, generator, GAN = modelCreateLoad(args.model_type, args.model_training, args.pretrained_nids,
                                                          args.pretrained_GAN, args.pretrained_generator,
                                                          args.pretrained_discriminator, args.dataset,
                                                          input_dim, noise_dim, args.regularizationEnabled,
                                                          args.DP_enabled, l2_alpha, latent_dim, num_classes)

    print("✅ Models initialized successfully!")
    print()

    # --- 5A Load Training Config ---#
    if args.trainingArea == "Federated":
        # -- Loading Federated Training Client -- #
        client = modelFederatedTrainingConfigLoad(nids, discriminator, generator, GAN, args.dataset, args.model_type,
                                                  args.model_training, args.earlyStopEnabled, args.DP_enabled,
                                                  args.lrSchedRedEnabled, args.modelCheckpointEnabled,
                                                  X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                                  y_test_data,
                                                  args.node, BATCH_SIZE, args.epochs, noise_dim, steps_per_epoch,
                                                  input_dim,
                                                  num_classes, latent_dim, betas, learning_rate, l2_alpha, l2_norm_clip,
                                                  noise_multiplier, num_microbatches, metric_to_monitor_es, es_patience,
                                                  restor_best_w, metric_to_monitor_l2lr, l2lr_patience, save_best_only,
                                                  metric_to_monitor_mc, checkpoint_mode, args.evaluationLog,
                                                  args.trainingLog)
        # -- Selecting Host -- #
        # Custom Address
        if args.custom_host is not None:
            server_address = f"{args.custom_host}:8080"
            print(f"✓ Using custom host: {server_address}")
        # Preset Default Nodes
        else:
            if args.host == "4":
                server_address = "192.168.129.8:8080"
            elif args.host == "3":
                server_address = "192.168.129.7:8080"
            elif args.host == "2":
                server_address = "192.168.129.6:8080"
            elif args.host == "1":
                server_address = "192.168.129.3:8080"
            else:  # custom address failsafe
                server_address = f"{args.host}:8080"
            print(f"✓ Using server: {server_address}")

        print("Server Address: ", server_address)

        # --- 6/7A Train & Evaluate Model ---#
        fl.client.start_client(server_address=server_address, client=client.to_client())

        client.save(args.save_name)
        # -- EOF Federated TRAINING -- #

        # --- 5B Load Training Config ---#
    else:
        client = modelCentralTrainingConfigLoad(nids, discriminator, generator, GAN, args.dataset, args.model_type,
                                                args.model_training, args.earlyStopEnabled, args.DP_enabled,
                                                args.lrSchedRedEnabled, args.modelCheckpointEnabled,
                                                X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                                y_test_data,
                                                args.node, BATCH_SIZE, args.epochs, noise_dim, steps_per_epoch,
                                                input_dim,
                                                num_classes, latent_dim, betas, learning_rate, l2_alpha, l2_norm_clip,
                                                noise_multiplier, num_microbatches, metric_to_monitor_es, es_patience,
                                                restor_best_w, metric_to_monitor_l2lr, l2lr_patience, save_best_only,
                                                metric_to_monitor_mc, checkpoint_mode, args.evaluationLog,
                                                args.trainingLog)

        # --- 6A Centrally Train Model ---#
        client.fit()

        # --- 7A Centrally Evaluate Model ---#
        client.evaluate()

    # --- 8 Locally Save Model After Training ---#
        client.save(args.save_name)


################################################################################################################
#                                                   Execute                                                   #
################################################################################################################
if __name__ == "__main__":
    main()

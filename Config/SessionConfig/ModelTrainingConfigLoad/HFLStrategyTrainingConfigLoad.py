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


# Load and Saving Configs
from Config.ModelTrainingConfig.HostModelTrainingConfig.ModelManagement.ServerSaveOnlyConfig import SaveModelFedAvg
from Config.ModelTrainingConfig.HostModelTrainingConfig.ModelManagement.ServerLoadOnlyConfig import LoadModelFedAvg
from Config.ModelTrainingConfig.HostModelTrainingConfig.ModelManagement.ServerLoadNSaveConfig import LoadSaveModelFedAvg

# Fit on End configs
from Config.ModelTrainingConfig.HostModelTrainingConfig.FitOnEnd.NIDS.ServerNIDSFitOnEndConfig import NIDSFitOnEndStrategy
from Config.ModelTrainingConfig.HostModelTrainingConfig.FitOnEnd.Discriminator.ServerDiscBinaryFitOnEndConfig import DiscriminatorSyntheticStrategy
from Config.ModelTrainingConfig.HostModelTrainingConfig.FitOnEnd.Discriminator.ServerWDiscFitOnEndConfig import WDiscriminatorSyntheticStrategy
from Config.ModelTrainingConfig.HostModelTrainingConfig.FitOnEnd.Discriminator.ServerACDiscBothFitOnEndConfig import ACDiscriminatorSyntheticStrategy


def _run_standard_federation_strategies(serverLoad, serverSave, roundInput, minClients, model, save_name):
    """Handle standard federation strategies without fit-on-end."""

    if serverLoad and not serverSave:
        print("📥 Starting Load-Only Federation Strategy...")
        fl.server.start_server(
            config=fl.server.ServerConfig(num_rounds=roundInput),
            strategy=LoadModelFedAvg(
                model=model,
                min_fit_clients=minClients,
                min_evaluate_clients=minClients,
                min_available_clients=minClients
            )
        )

    elif not serverLoad and serverSave:
        print("💾 Starting Save-Only Federation Strategy...")
        fl.server.start_server(
            config=fl.server.ServerConfig(num_rounds=roundInput),
            strategy=SaveModelFedAvg(
                model=model,
                model_save_path=save_name,
                min_fit_clients=minClients,
                min_evaluate_clients=minClients,
                min_available_clients=minClients
            )
        )

    elif serverLoad and serverSave:
        print("🔄 Starting Load-and-Save Federation Strategy...")
        fl.server.start_server(
            config=fl.server.ServerConfig(num_rounds=roundInput),
            strategy=LoadSaveModelFedAvg(
                model=model,
                model_save_path=save_name,
                min_fit_clients=minClients,
                min_evaluate_clients=minClients,
                min_available_clients=minClients
            )
        )


def _run_fit_on_end_strategies(train_type, model_type, roundInput, args,
                               discriminator, generator, nids, GAN,
                               X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data,
                               BATCH_SIZE, noise_dim, steps_per_epoch, input_dim, num_classes, latent_dim,
                               epochs, learning_rate, synth_portion, l2_norm_clip, noise_multiplier,
                               num_microbatches, metric_to_monitor_es, es_patience, restor_best_w,
                               metric_to_monitor_l2lr, l2lr_patience, save_best_only, metric_to_monitor_mc,
                               checkpoint_mode, save_name, serverLoad, dataset_used, earlyStopEnabled,
                               lrSchedRedEnabled, modelCheckpointEnabled, DP_enabled, node):
    """Handle fit-on-end federation strategies."""

    if train_type == "NIDS":
        print("🎯 Starting NIDS Fit-on-End Strategy...")
        fl.server.start_server(
            config=fl.server.ServerConfig(num_rounds=roundInput),
            strategy=NIDSFitOnEndStrategy(
                discriminator=discriminator,
                generator=generator,
                nids=nids,
                dataset_used=dataset_used,
                node=node,
                earlyStopEnabled=earlyStopEnabled,
                lrSchedRedEnabled=lrSchedRedEnabled,
                modelCheckpointEnabled=modelCheckpointEnabled,
                DP_enabled=DP_enabled,
                X_train_data=X_train_data, y_train_data=y_train_data,
                X_test_data=X_test_data, y_test_data=y_test_data,
                X_val_data=X_val_data, y_val_data=y_val_data,
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                batch_size=BATCH_SIZE,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                learning_rate=learning_rate,
                synth_portion=synth_portion,
                latent_dim=latent_dim,
                num_classes=num_classes,
                metric_to_monitor_es=metric_to_monitor_es,
                es_patience=es_patience,
                restor_best_w=restor_best_w,
                metric_to_monitor_l2lr=metric_to_monitor_l2lr,
                l2lr_patience=l2lr_patience,
                save_best_only=save_best_only,
                metric_to_monitor_mc=metric_to_monitor_mc,
                checkpoint_mode=checkpoint_mode,
                save_name=save_name,
                serverLoad=serverLoad,
            )
        )

    elif model_type == "GAN":
        print("🎯 Starting GAN Discriminator Fit-on-End Strategy...")
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

    elif model_type == "WGAN-GP":
        print("🎯 Starting WGAN-GP Discriminator Fit-on-End Strategy...")
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

    elif model_type == "AC-GAN":
        print("🎯 Starting AC-GAN Discriminator Fit-on-End Strategy...")
        fl.server.start_server(
            config=fl.server.ServerConfig(num_rounds=roundInput),
            strategy=ACDiscriminatorSyntheticStrategy(
                discriminator=discriminator,
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
                log_file="training.log"
            )
        )
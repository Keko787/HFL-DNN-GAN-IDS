import argparse
from datetime import datetime


################################################################################################################
#                                                   Training Client Script Arguments                           #
################################################################################################################
def parse_training_client_args():
    """Parse and process training arguments. Returns processed arguments."""
    # ═══════════════════════════════════════════════════════════════════════
    # Initiate Preset Variables
    # ═══════════════════════════════════════════════════════════════════════

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ═══════════════════════════════════════════════════════════════════════
    # Parsing Arguments
    # ═══════════════════════════════════════════════════════════════════════
    # ───  Initiate Parser ───
    parser = argparse.ArgumentParser(description='Select dataset, model selection, and to enable DP respectively')

    # ───  Dataset Settings ───
    parser.add_argument('--dataset', type=str, choices=["CICIOT", "IOTBOTNET", "IOT"], default="CICIOT",
                        help='Datasets to use: CICIOT, IOTBOTNET, IOT (different from IOTBOTNET)')

    parser.add_argument('--dataset_processing', type=str, choices=["Default", "MM[-1,-1]", "AC-GAN, IOT", "IOT-MinMax"],
                        default="Default", help='Datasets to use: Default, MM[-1,1], AC-GAN, IOT')

    # ─── Federation Settings ───
    parser.add_argument('--trainingArea', type=str, choices=["Central", "Federated"], default="Central",
                        help='Please select Central, Federated as the place to train the model')

    parser.add_argument("--host-default", type=str, default="1",
                        help="Fixed Server node number 1-4")

    parser.add_argument('--custom-host', type=str,
                            help='Custom IP address or hostname')

    parser.add_argument('--serverBased', action='store_true',
                        help='Only load the model structure and get the weights from the server')

    # ─── Model Training Settings ───
    parser.add_argument('--model_type', type=str,
                        choices=["NIDS", "NIDS-IOT-Binary", "NIDS-IOT-Multiclass", "NIDS-IOT-Multiclass-Dynamic", "GAN",
                                 "WGAN-GP", "AC-GAN"],
                        help='Please select NIDS, NIDS-IOT-Binary, NIDS-IOT-Multiclass, NIDS-IOT-Multiclass-Dynamic, GAN, WGAN-GP, or AC-GAN as the model type to train')

    parser.add_argument('--model_training', type=str, choices=["NIDS", "Generator", "Discriminator", "Both"],
                        default="Both",
                        help='Please select NIDS, Generator, Discriminator, Both as the sub-model type to train')

    # ─── Model Training Session Settings ───
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model")

    # ─── Loading Models (Optional) ───
    parser.add_argument('--pretrained_GAN', type=str, help="Path to pretrained discriminator model (optional)",
                        default=None)

    parser.add_argument('--pretrained_generator', type=str, help="Path to pretrained generator model (optional)",
                        default=None)

    parser.add_argument('--pretrained_discriminator', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)

    parser.add_argument('--pretrained_nids', type=str, help="Path to pretrained nids model (optional)", default=None)

    # ─── Saving Models ───
    parser.add_argument('--save_name', type=str, help="name of model files you save as", default=f"{timestamp}")

    args = parser.parse_args()

    # ═══════════════════════════════════════════════════════════════════════
    # Processing Variables
    # ═══════════════════════════════════════════════════════════════════════

    # ─── Apply conditional logic directly to args ───
    # Allows user not to input Dataset processing for acgan
    if args.model_type == "AC-GAN":
        args.dataset_processing = "AC-GAN"

    # Puts all NIDS models under as the NIDS training scheme for consistency
    if args.model_type in ["NIDS", "NIDS-IOT-Binary", "NIDS-IOT-Multiclass", "NIDS-IOT-Multiclass-Dynamic"]:
        args.model_training = "NIDS"

    # All IoT models get the iot dataset without explicit input
    if args.model_type in ["NIDS-IOT-Binary", "NIDS-IOT-Multiclass", "NIDS-IOT-Multiclass-Dynamic"]:
        args.dataset = "IOT"

    # Ditto for Dataset Processing
    if args.dataset_processing in ["IOT", "IOT-MinMax"]:
        args.dataset = "IOT"

    # ─── Add computed fields ───
    args.timestamp = timestamp
    args.regularizationEnabled = True
    args.DP_enabled = None
    args.earlyStopEnabled = None
    args.lrSchedRedEnabled = None
    args.modelCheckpointEnabled = None
    args.evaluationLog = timestamp
    args.trainingLog = timestamp
    args.node = 1

    return args
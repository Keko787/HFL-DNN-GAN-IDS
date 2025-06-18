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

    # ───  Initiate Arguments ───
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


def display_training_client_opening_message(args, timestamp):
    """
    Display an enhanced opening message for the Training Client
    """
    print("=" * 80)
    print("🚀 MACHINE LEARNING TRAINING CLIENT")
    print("=" * 80)
    print(f"📅 Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🆔 Session ID: {timestamp}")
    print("-" * 80)

    # Training Mode Section
    training_mode = "🌐 FEDERATED" if args.trainingArea == "Federated" else "🏠 CENTRALIZED"
    print(f"⚙️  Training Mode: {training_mode}")

    # Dataset & Model Information
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔄 Preprocessing: {args.dataset_processing}")
    print(f"🧠 Model Type: {args.model_type}")
    print(f"🎯 Training Method: {args.model_training}")
    print(f"🔢 Epochs: {args.epochs}")

    # Pre-trained Models Section
    if any([args.pretrained_GAN, args.pretrained_generator, args.pretrained_discriminator, args.pretrained_nids]):
        print("-" * 40)
        print("📥 PRE-TRAINED MODELS:")
        if args.pretrained_GAN:
            print(f"   • GAN Model: {args.pretrained_GAN}")
        if args.pretrained_generator:
            print(f"   • Generator: {args.pretrained_generator}")
        if args.pretrained_discriminator:
            print(f"   • Discriminator: {args.pretrained_discriminator}")
        if args.pretrained_nids:
            print(f"   • NIDS Model: {args.pretrained_nids}")

    # Save Configuration
    if args.save_name:
        print("-" * 40)
        print(f"💾 Output Model Name: {args.save_name}")

    # Federated Training Specific Info
    if args.trainingArea == "Federated":
        print("-" * 40)
        print("🌐 FEDERATED LEARNING CONFIG:")
        if args.custom_host:
            print(f"   • Custom Server: {args.custom_host}:8080")
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
            print(f"   • Server Address: {server_address}")
        print(f"   • Node ID: {args.node}")

    print("=" * 80)
    print("🔄 Initializing training pipeline...")
    print()


################################################################################################################
#                                                   HFL Host Script Arguments                                  #
################################################################################################################
def parse_HFL_Host_args():
    """Parse and process HFL Host server arguments. Returns processed arguments."""
    # ═══════════════════════════════════════════════════════════════════════
    # Initiate Preset Variables
    # ═══════════════════════════════════════════════════════════════════════

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ═══════════════════════════════════════════════════════════════════════
    # Parsing Arguments
    # ═══════════════════════════════════════════════════════════════════════
    # ───  Initiate Parser ───
    parser = argparse.ArgumentParser(description='Hierarchical Federated Learning Host Server Configuration')

    # ───  Dataset Settings ───
    parser.add_argument('--dataset', type=str, choices=["CICIOT", "IOTBOTNET", "IOT"], default="CICIOT",
                        help='Datasets to use: CICIOT, IOTBOTNET, IOT (different from IOTBOTNET)')

    parser.add_argument('--dataset_processing', type=str, choices=["Default", "MM[-1,-1]", "AC-GAN, IOT", "IOT-MinMax"],
                        default="Default", help='Dataset preprocessing: Default, MM[-1,1], AC-GAN, IOT')

    # ─── Server Hosting Modes ───
    parser.add_argument('--serverLoad', action='store_true',
                        help='Enable server-side model loading functionality')

    parser.add_argument('--serverSave', action='store_true',
                        help='Enable server-side model saving functionality')

    parser.add_argument('--fitOnEnd', action='store_true',
                        help='Enable fit-on-end advanced training strategies')

    # ─── Model Configuration ───
    parser.add_argument('--model_type', type=str,
                        choices=["NIDS", "NIDS-IOT-Binary", "NIDS-IOT-Multiclass", "NIDS-IOT-Multiclass-Dynamic", "GAN",
                                 "WGAN-GP", "AC-GAN"],
                        help='Model architecture: NIDS variants, GAN variants')

    parser.add_argument('--model_training', type=str, choices=["NIDS", "Discriminator", "GAN"],
                        default="GAN",
                        help='Training focus: NIDS, Discriminator, or Both components')

    # ─── Training Session Parameters ───
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per round")

    parser.add_argument("--rounds", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], default=1,
                        help="Number of federated learning rounds (1-10)")

    parser.add_argument("--synth_portion", type=float, choices=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6], default=0,
                        help="Synthetic data augmentation ratio (0-0.6)")

    parser.add_argument("--min_clients", type=int, choices=[1, 2, 3, 4, 5, 6], default=2,
                        help="Minimum number of clients required for federated training")

    # ─── Pre-trained Models (Optional) ───
    parser.add_argument('--pretrained_GAN', type=str, default=None,
                        help="Path to pre-trained GAN model (optional)")

    parser.add_argument('--pretrained_generator', type=str, default=None,
                        help="Path to pre-trained generator model (optional)")

    parser.add_argument('--pretrained_discriminator', type=str, default=None,
                        help="Path to pre-trained discriminator model (optional)")

    parser.add_argument('--pretrained_nids', type=str, default=None,
                        help="Path to pre-trained NIDS model (optional)")

    # ─── Model Saving Configuration ───
    parser.add_argument('--save_name', type=str, default=f"{timestamp}",
                        help="Base name for saved model files")

    # ───  Initiate Arguments ───
    args = parser.parse_args()

    # ═══════════════════════════════════════════════════════════════════════
    # Processing Variables & Conditional Logic
    # ═══════════════════════════════════════════════════════════════════════

    # ─── Apply conditional logic directly to args ───
    # Auto-configure dataset processing for AC-GAN
    if args.model_type == "AC-GAN":
        args.dataset_processing = "AC-GAN"

    # Configure training type for NIDS models
    if args.model_type in ["NIDS", "NIDS-IOT-Binary", "NIDS-IOT-Multiclass", "NIDS-IOT-Multiclass-Dynamic"]:
        args.model_training = "NIDS"

    # Auto-select IOT dataset for IOT-specific models
    if args.model_type in ["NIDS-IOT-Binary", "NIDS-IOT-Multiclass", "NIDS-IOT-Multiclass-Dynamic"]:
        args.dataset = "IOT"

    # Dataset selection based on processing type
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

    # ─── Generate dynamic save name ───
    if args.fitOnEnd:
        args.full_save_name = f"fitOnEnd_{args.dataset}_{args.dataset_processing}_{args.model_type}_{args.model_training}_{args.save_name}.h5"
    else:
        args.full_save_name = f"{args.model_type}_{args.model_training}_{args.save_name}.h5"

    return args


def display_HFL_host_opening_message(args, timestamp):
    """
    Display an enhanced opening message for the HFL Host Server
    """
    print("=" * 80)
    print("🌐 HIERARCHICAL FEDERATED LEARNING HOST SERVER")
    print("=" * 80)
    print(f"📅 Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🆔 Session ID: {timestamp}")
    print("-" * 80)

    # Server Mode Configuration
    server_modes = []
    if args.serverLoad:
        server_modes.append("📥 MODEL LOADING")
    if args.serverSave:
        server_modes.append("💾 MODEL SAVING")
    if args.fitOnEnd:
        server_modes.append("🎯 FIT-ON-END TRAINING")

    if server_modes:
        print("🔧 Server Modes: " + " | ".join(server_modes))
    else:
        print("🔧 Server Mode: 🎯 STANDARD FEDERATED AVERAGING")

    # Dataset & Model Configuration
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔄 Preprocessing: {args.dataset_processing}")
    print(f"🧠 Model Type: {args.model_type}")
    print(f"🎯 Training Focus: {args.model_training}")

    # Training Parameters
    print("-" * 40)
    print("⚙️  TRAINING CONFIGURATION:")
    print(f"   • Federated Rounds: {args.rounds}")
    print(f"   • Epochs per Round: {args.epochs}")
    print(f"   • Minimum Clients: {args.min_clients}")
    if args.synth_portion > 0:
        print(f"   • Synthetic Data Ratio: {args.synth_portion:.1%}")

    # Pre-trained Models Section
    if any([args.pretrained_GAN, args.pretrained_generator, args.pretrained_discriminator, args.pretrained_nids]):
        print("-" * 40)
        print("📥 PRE-TRAINED MODELS:")
        if args.pretrained_GAN:
            print(f"   • GAN Model: {args.pretrained_GAN}")
        if args.pretrained_generator:
            print(f"   • Generator: {args.pretrained_generator}")
        if args.pretrained_discriminator:
            print(f"   • Discriminator: {args.pretrained_discriminator}")
        if args.pretrained_nids:
            print(f"   • NIDS Model: {args.pretrained_nids}")

    # Model Saving Configuration
    if args.serverSave or args.fitOnEnd:
        print("-" * 40)
        print(f"💾 Model Output: {args.full_save_name}")

    # Advanced Strategy Information
    if args.fitOnEnd:
        print("-" * 40)
        print("🎯 ADVANCED STRATEGY DETAILS:")
        if args.model_training == "NIDS":
            print("   • Strategy: NIDS Fit-on-End with Synthetic Data Augmentation")
        elif args.model_type == "GAN":
            print("   • Strategy: Discriminator Synthetic Training")
        elif args.model_type == "WGAN-GP":
            print("   • Strategy: WGAN-GP Discriminator Advanced Training")
        elif args.model_type == "AC-GAN":
            print("   • Strategy: AC-GAN Discriminator Multi-class Training")

        if args.synth_portion > 0:
            print(f"   • Synthetic augmentation will enhance training with {args.synth_portion:.1%} additional data")

    print("=" * 80)
    print("🚀 Starting Federated Learning Server...")
    print("⏳ Waiting for client connections...")
    print()

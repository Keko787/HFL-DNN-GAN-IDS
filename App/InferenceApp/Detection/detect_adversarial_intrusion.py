import sys
import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import logging

sys.path.append(os.path.abspath('..'))

# Import dataset loading and preprocessing
from datasetHandling.loadCiciotOptimized import loadCICIOT
from datasetHandling.iotbotnetDatasetLoad import loadIOTBOTNET
from datasetHandling.datasetPreprocess import preprocess_dataset
from detectIntrusions import predict_and_act

# Import GAN structure for synthetic data detection
from centralTrainingConfig.GANBinaryCentralTrainingConfig import CentralBinaryGan
from modelStructures.discriminatorStruct import create_discriminator_binary, create_discriminator_binary_optimized, create_discriminator_binary
from modelStructures.generatorStruct import create_generator, create_generator_optimized
from modelStructures.ganStruct import create_model, load_GAN_model, create_model_binary, create_model_binary_optimized

# Logging setup
logging.basicConfig(filename="misc/detection_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")


def discriminator_loss(real_output, fake_output):
    # Create binary labels: 1 for real, 0 for fake
    real_labels = tf.ones_like(real_output)
    fake_labels = tf.zeros_like(fake_output)

    # Compute binary cross-entropy loss
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    real_loss = bce(real_labels, real_output)
    fake_loss = bce(fake_labels, fake_output)

    return real_loss + fake_loss


def generator_loss(fake_output):
    # Generator tries to make fake samples be classified as real (1)
    fake_labels = tf.ones_like(fake_output)  # Label 1 for fake samples
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return bce(fake_labels, fake_output)


def load_model(pretrained_model):
    """ Loads the pre-trained NIDS model. """
    try:
        if pretrained_model:
            print(f"📥 Loading pretrained model from {pretrained_model}...")
            with tf.keras.utils.custom_object_scope({'LogCosh': tf.keras.losses.LogCosh}):
                model = tf.keras.models.load_model(pretrained_model)
            print("✅ Model successfully loaded!")
            return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        sys.exit(1)


def generate_synthetic_data(generator, num_samples, noise_dim):
    """ Generate synthetic network traffic using a trained GAN generator. """
    print(f"🎭 Generating {num_samples} synthetic network traffic samples...")
    noise = tf.random.normal([num_samples, noise_dim])
    synthetic_data = generator(noise, training=False)
    return synthetic_data.numpy()


# -- Evaluate -- #
def detect_adversarial_or_synthetic(batch_size=256, noise_dim=100, generator=None, discriminator=None, x_test=None, y_test=None):
    """
    Evaluates the discriminator and generator on real and generated samples.

    Returns:
        avg_disc_loss: The average discriminator loss over all test batches.
        total_samples: The total number of evaluated samples.
        metrics_dict: A dictionary containing relevant evaluation metrics.
    """
    print("\n🎭 Running Evaluation on Test Data...\n")

    # Convert test data into TensorFlow dataset
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # Initialize lists for storing losses
    disc_losses = []
    gen_losses = []

    # Initialize lists for storing losses
    disc_losses = []
    gen_losses = []

    # Track classification results
    real_correct = 0
    fake_correct = 0
    fake_misclassified = 0
    real_misclassified = 0

    # Get total real and fake samples correctly
    total_real_samples = len(x_test)
    total_fake_samples = 0  # Will be updated dynamically

    # Process test data in batches
    for step, (test_data_batch, test_labels_batch) in enumerate(test_ds):
        # Generate fake samples
        noise = tf.random.normal([batch_size, noise_dim])
        generated_samples = generator(noise, training=False)

        # Get discriminator predictions for real and fake data
        real_output = discriminator.predict(test_data_batch)
        fake_output = discriminator.predict(generated_samples)

        # Compute losses
        disc_loss = discriminator_loss(real_output, fake_output)
        gen_loss = generator_loss(fake_output)

        # Store losses
        disc_losses.append(disc_loss.numpy())
        gen_losses.append(gen_loss.numpy())

        # Convert probabilities to binary classification (Fix)
        real_pred = (real_output.flatten() > 0.5).astype("int32")
        fake_pred = (fake_output.flatten() > 0.5).astype("int32")

        # Update total fake samples only once
        if step == 0:
            total_fake_samples += len(fake_pred)

        # Logging for synthetic vs real classification
        real_correct += np.sum(real_pred == 1)  # Correct real classifications
        fake_correct += np.sum(fake_pred == 0)  # Correct fake classifications
        fake_misclassified += np.sum(fake_pred == 1)  # Fake classified as real
        real_misclassified += np.sum(real_pred == 0)  # Fake classified as real

        # Debugging output for batch processing
        print(f"🔍 Batch {step + 1}:")
        print(f"    Real batch shape: {test_data_batch.shape}, Fake batch shape: {generated_samples.shape}")
        print(f"    Disc Loss: {disc_loss.numpy()}, Gen Loss: {gen_loss.numpy()}")

        # Compute average losses
    avg_disc_loss = np.mean(disc_losses)
    avg_gen_loss = np.mean(gen_losses)

    # Compute total number of samples
    total_samples = total_real_samples + total_fake_samples

    # Print summary of samples before classification results
    print("\n📊 Total Sample Summary:")
    print(f"🔹 Total Real Samples: {total_real_samples}")
    print(f"🔹 Total Fake Samples: {total_fake_samples}")
    print(f"🔹 Total Samples Evaluated: {total_samples}\n")

    # Logging Classification Performance
    logging.info("\n🚨 Synthetic Data Detection Report 🚨")
    logging.info(f"✔️ Correctly Identified Real Samples: {real_correct} / {total_real_samples}")
    logging.info(f"❌ Real Samples Misclassified as Fake: {real_misclassified} / {total_real_samples}")
    logging.info(f"✔️ Correctly Identified Fake Samples: {fake_correct} / {total_fake_samples}")
    logging.info(f"❌ Fake Samples Misclassified as Real: {fake_misclassified} / {total_fake_samples}")
    logging.info(f"📊 Total Samples Evaluated: {total_samples}")

    print("\n📊 Final Evaluation Results:")
    print(f"✅ Average Discriminator Loss: {avg_disc_loss}")
    print(f"✅ Average Generator Loss: {avg_gen_loss}")
    print(f"✔️ Correctly Identified Real Samples: {real_correct} / {total_real_samples}")
    print(f"❌ Real Samples Misclassified as Fake: {real_misclassified} / {total_real_samples}")
    print(f"✔️ Correctly Identified Fake Samples: {fake_correct} / {total_fake_samples}")
    print(f"❌ Fake Samples Misclassified as Real: {fake_misclassified} / {total_fake_samples}")
    print(f"📊 Total Samples Evaluated: {total_samples}\n")

    # Return metrics
    return avg_disc_loss, len(x_test), {"generator_loss": avg_gen_loss}


def detect_intrusions(nids_model, X_real, y_real, threshold=0.5):
    """
    Runs the NIDS model on filtered real data to classify it as benign or an attack.

    Args:
        nids_model: The trained NIDS model.
        X_real: Only real network traffic data (filtered by the discriminator).
        y_real: Ground truth labels corresponding to real data.
        threshold: Probability threshold for intrusion classification.

    Returns:
        None (Prints and logs results)
    """
    print("\n🔍 Running NIDS Intrusion Detection...")
    y_pred_probs = nids_model.predict(X_real)
    y_pred = (y_pred_probs < threshold).astype("int32")

    # Evaluate NIDS Performance
    nids_report = classification_report(y_real, y_pred, target_names=["Attack", "Benign"])
    print("\n📊 NIDS Model Performance:\n", nids_report)
    logging.info("\nNIDS Performance:\n" + nids_report)

    # Confusion Matrix
    cm = confusion_matrix(y_real, y_pred)
    print("\n📊 Confusion Matrix (NIDS Detection):")
    print(cm)
    logging.info(f"Confusion Matrix: \n{cm}")


def main():
    print("\n============================")
    print("🔍 Adversarial & Synthetic Data Detection")
    print("============================\n")

    # Argument Parsing
    parser = argparse.ArgumentParser(description='Detect adversarial and synthetic traffic.')
    parser.add_argument('--dataset', type=str, choices=["CICIOT", "IOTBOTNET"], default="CICIOT",
                        help='Dataset to use: CICIOT or IOTBOTNET')
    parser.add_argument('--pretrained_generator', type=str, help="Path to pretrained generator model (optional)",
                        default=None)
    parser.add_argument('--pretrained_discriminator', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)
    parser.add_argument('--pretrained_nids', type=str,
                        help="Path to pretrained nids model (optional)", default=None)
    parser.add_argument('--num_synthetic', type=int, default=1000,
                        help="Number of synthetic samples to generate.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for classifying intrusions.")


    args = parser.parse_args()
    pretrainedGan = None
    pretrainedGenerator = args.pretrained_generator
    pretrainedDiscriminator = args.pretrained_discriminator
    pretrainedNids = args.pretrained_nids

    # Load Dataset
    print(f"📡 Loading {args.dataset} dataset...")
    if args.dataset == "CICIOT":
        train_data, test_data, irrelevant_features = loadCICIOT()
    elif args.dataset == "IOTBOTNET":
        train_data, test_data, irrelevant_features = loadIOTBOTNET()

    # Preprocess Dataset
    print("🔄 Preprocessing dataset...")
    X_train, X_val, y_train, y_val, X_test, y_test = preprocess_dataset(
        args.dataset, train_data, test_data, None, None, irrelevant_features, None
    )

    # --- Load or Create model ----#

    # --- Model setup --- #

    # --- Hyperparameters ---#
    BATCH_SIZE = 256
    noise_dim = 100

    input_dim = X_train.shape[1]

    # Load or create the discriminator, generator, or whole gan model
    if pretrainedGan:
        print(f"Loading pretrained GAN Model from {pretrainedGan}")
        model = tf.keras.models.load_model(pretrainedGan)

    elif pretrainedGenerator and not pretrainedDiscriminator:

        print(f"Pretrained Generator provided from {pretrainedGenerator}. Creating a new Discriminator model.")
        generator = tf.keras.models.load_model(args.pretrained_generator)

        discriminator = create_discriminator_binary(input_dim)

        model = load_GAN_model(generator, discriminator)

    elif pretrainedDiscriminator and not pretrainedGenerator:
        print(f"Pretrained Discriminator provided from {pretrainedDiscriminator}. Creating a new Generator model.")
        discriminator = tf.keras.models.load_model(args.pretrained_discriminator)

        generator = create_generator(input_dim, noise_dim)

        model = load_GAN_model(generator, discriminator)

    elif pretrainedDiscriminator and pretrainedGenerator:
        print(f"Pretrained Generator and Discriminator provided from {pretrainedGenerator} , {pretrainedDiscriminator}")
        discriminator_model = tf.keras.models.load_model(args.pretrained_discriminator)
        generator_model = tf.keras.models.load_model(args.pretrained_generator)

        model = load_GAN_model(generator_model, discriminator_model)

    else:
        print("No pretrained GAN provided. Creating a new GAN model.")
        model = create_model_binary(input_dim, noise_dim)

    # load NIDS model
    print(f"Loading pretrained NIDS from {pretrainedNids}")
    nids = load_model(pretrainedNids)

    # Split the gan model into its submodels
    generator = model.layers[0]
    discriminator = model.layers[1]

    detect_adversarial_or_synthetic(BATCH_SIZE, noise_dim, generator, discriminator, X_test,
                                    y_test)

    # Predict and Take Action
    predict_and_act(nids, X_test, y_test, threshold=args.threshold)


if __name__ == "__main__":
    main()

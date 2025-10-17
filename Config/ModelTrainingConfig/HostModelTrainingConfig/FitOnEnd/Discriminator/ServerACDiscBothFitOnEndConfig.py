from datetime import datetime
import flwr as fl
import argparse
import tensorflow as tf
import logging
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.metrics import AUC, Precision, Recall
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, classification_report


# Custom FedAvg strategy with server-side model training and saving
class ACDiscriminatorSyntheticStrategy(fl.server.strategy.FedAvg):
    def __init__(self, GAN, nids, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 noise_dim, latent_dim, num_classes, input_dim, epochs, steps_per_epoch, learning_rate,
                 log_file="training.log", **kwargs):
        super().__init__(**kwargs)
        # -- models
        self.GAN = GAN
        # Reconstruct the generator model from the merged model:
        self.generator = self.GAN.generator  # directly use the stored generator
        self.discriminator = self.GAN.discriminator  # directly use the stored discriminator

        self.nids = nids

        # -- I/O Specs for models
        self.batch_size = BATCH_SIZE
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_dim = input_dim

        # -- training duration
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        # -- Data
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        # -- Early Stopping Configuration
        self.early_stopping_patience = 5
        self.min_delta = 0.001  # Minimum improvement to consider as progress

        # -- Setup Logging
        self.setup_logger(log_file)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # -- Optimizers
        # LR decay
        lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0002, decay_steps=10000, decay_rate=0.98, staircase=True)

        lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.98, staircase=True)

        # Init optimizer
        self.gen_optimizer = Adam(learning_rate=lr_schedule_gen, beta_1=0.5, beta_2=0.999)
        self.disc_optimizer = Adam(learning_rate=lr_schedule_disc, beta_1=0.5, beta_2=0.999)

        # -- Loss Functions Setup (for custom training loops)
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        # -- Metrics Setup (for custom training loops)
        self.d_binary_accuracy = tf.keras.metrics.BinaryAccuracy(name='d_binary_accuracy')
        self.d_categorical_accuracy = tf.keras.metrics.CategoricalAccuracy(name='d_categorical_accuracy')
        self.g_binary_accuracy = tf.keras.metrics.BinaryAccuracy(name='g_binary_accuracy')
        self.g_categorical_accuracy = tf.keras.metrics.CategoricalAccuracy(name='g_categorical_accuracy')

        # NOTE: No model compilation needed - using custom training loops only
        print("Discriminator Output:", self.discriminator.output_names)

    # -- logging Functions -- #
    def setup_logger(self, log_file):
        """Set up a logger that records both to a file and to the console."""
        self.logger = logging.getLogger("CentralACGan")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # Avoid adding duplicate handlers if logger already has them.
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def log_model_settings(self):
        """Logs model names, structures, and hyperparameters."""
        self.logger.info("=== Model Settings ===")

        self.logger.info("Generator Model Summary:")
        generator_summary = []
        self.generator.summary(print_fn=lambda x: generator_summary.append(x))
        for line in generator_summary:
            self.logger.info(line)

        self.logger.info("Discriminator Model Summary:")
        discriminator_summary = []
        self.discriminator.summary(print_fn=lambda x: discriminator_summary.append(x))
        for line in discriminator_summary:
            self.logger.info(line)

        if self.nids is not None:
            self.logger.info("NIDS Model Summary:")
            nids_summary = []
            self.nids.summary(print_fn=lambda x: nids_summary.append(x))
            for line in nids_summary:
                self.logger.info(line)
        else:
            self.logger.info("NIDS Model is not defined.")

        self.logger.info("--- Hyperparameters ---")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Noise Dimension: {self.noise_dim}")
        self.logger.info(f"Latent Dimension: {self.latent_dim}")
        self.logger.info(f"Number of Classes: {self.num_classes}")
        self.logger.info(f"Input Dimension: {self.input_dim}")
        self.logger.info(f"Epochs: {self.epochs}")
        self.logger.info(f"Steps per Epoch: {self.steps_per_epoch}")
        self.logger.info(f"Learning Rate (Generator): {self.gen_optimizer.learning_rate}")
        self.logger.info(f"Learning Rate (Discriminator): {self.disc_optimizer.learning_rate}")
        self.logger.info("=" * 50)

    def log_epoch_metrics(self, epoch, d_metrics, g_metrics, nids_metrics=None):
        """Logs a formatted summary of the metrics for this epoch."""
        self.logger.info(f"=== Epoch {epoch} Metrics Summary ===")
        self.logger.info("Discriminator Metrics:")
        for key, value in d_metrics.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("Generator Metrics:")
        for key, value in g_metrics.items():
            self.logger.info(f"  {key}: {value}")
        if nids_metrics is not None:
            self.logger.info("NIDS Metrics:")
            for key, value in nids_metrics.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    def log_evaluation_metrics(self, d_eval, g_eval=None, nids_eval=None):
        """Logs a formatted summary of evaluation metrics."""
        self.logger.info("=== Evaluation Metrics Summary ===")
        self.logger.info("Discriminator Evaluation:")
        for key, value in d_eval.items():
            self.logger.info(f"  {key}: {value}")
        if g_eval is not None:
            self.logger.info("Generator Evaluation:")
            for key, value in g_eval.items():
                self.logger.info(f"  {key}: {value}")
        if nids_eval is not None:
            self.logger.info("NIDS Evaluation:")
            for key, value in nids_eval.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    #########################################################################
    # Helper method for TRAINING PROCESS to balanced fake label generation  #
    #########################################################################
    #########################################################################
    #                      fake label generator HELPER                     #
    #########################################################################
    def generate_balanced_fake_labels(self, total_samples):
        """
        Generate balanced fake labels ensuring equal distribution of classes.

        Parameters:
        -----------
        total_samples : int
            Total number of fake labels to generate

        Returns:
        --------
        tf.Tensor
            Balanced and shuffled fake labels
        """
        half_samples = total_samples // 2
        remaining_samples = total_samples - half_samples

        # Create balanced labels
        fake_labels_0 = tf.zeros(half_samples, dtype=tf.int32)  # Benign class
        fake_labels_1 = tf.ones(remaining_samples, dtype=tf.int32)  # Attack class

        # Concatenate and shuffle
        fake_labels = tf.concat([fake_labels_0, fake_labels_1], axis=0)
        fake_labels = tf.random.shuffle(fake_labels)

        return fake_labels

        #########################################################################
        #                      BATCH DATA PROCESSING HELPER                     #
        #########################################################################

    def process_batch_data(self, data, labels, valid_smoothing_factor):
        """
        Process batch data and labels to ensure correct shapes and encoding.

        Args:
            data: Input feature data
            labels: Corresponding labels
            valid_smoothing_factor: Label smoothing factor for validity labels

        Returns:
            Tuple of (processed_data, processed_labels, validity_labels)
        """
        # • Fix shape issues - ensure 2D data
        if len(data.shape) > 2:
            data = tf.reshape(data, (data.shape[0], -1))

        # • Ensure one-hot encoding
        if len(labels.shape) == 1:
            labels_onehot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
        else:
            labels_onehot = labels

        # • Ensure correct shape for labels
        if len(labels_onehot.shape) > 2:
            labels_onehot = tf.reshape(labels_onehot, (labels_onehot.shape[0], self.num_classes))

        # • Create validity labels with smoothing
        validity_labels = tf.ones((data.shape[0], 1)) * (1 - valid_smoothing_factor)

        return data, labels_onehot, validity_labels

    #########################################################################
    #                         LOSS CALCULATION METHODS                     #
    #########################################################################
    def nids_loss(self, real_output, fake_output):
        """
        Compute the NIDS loss on real and fake samples.
        For real samples, the target is 1 (benign), and for fake samples, 0 (attack).
        Returns a scalar loss value.
        """
        # define labels
        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)

        # define loss function
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # calculate outputs
        real_loss = bce(real_labels, real_output)
        fake_loss = bce(fake_labels, fake_output)

        # sum up total loss
        total_loss = real_loss + fake_loss
        return total_loss.numpy()

    #########################################################################
    #                         LOSS CALCULATION METHODS                     #
    #########################################################################

    def calculate_weighted_loss(self, d_loss_benign, d_loss_attack, d_loss_fake,
                                attack_weight=0.7, benign_weight=0.3,
                                validity_weight=0.4, class_weight=0.6):
        """
        Calculate weighted discriminator loss combining benign, attack, and fake samples.
        """
        # ═══════════════════════════════════════════════════════════════════════
        # LOSS COMPONENT EXTRACTION
        # ═══════════════════════════════════════════════════════════════════════
        # ─── Benign Sample Components ───
        d_loss_benign_validity = d_loss_benign[1]
        d_loss_benign_class = d_loss_benign[2]
        d_benign_valid_acc = d_loss_benign[3]
        d_benign_class_acc = d_loss_benign[4]

        # ─── Attack Sample Components ───
        d_loss_attack_validity = d_loss_attack[1]
        d_loss_attack_class = d_loss_attack[2]
        d_attack_valid_acc = d_loss_attack[3]
        d_attack_class_acc = d_loss_attack[4]

        # ─── Fake Sample Components ───
        d_loss_fake_validity = d_loss_fake[1]
        d_loss_fake_class = d_loss_fake[2]
        d_fake_valid_acc = d_loss_fake[3]
        d_fake_class_acc = d_loss_fake[4]

        # ═══════════════════════════════════════════════════════════════════════
        # WEIGHTED LOSS CALCULATIONS
        # ═══════════════════════════════════════════════════════════════════════
        # ─── Weighted Validity Loss ───
        d_loss_validity_real = (benign_weight * d_loss_benign_validity) + (attack_weight * d_loss_attack_validity)
        d_loss_validity = 0.5 * (d_loss_validity_real + d_loss_fake_validity)

        # ─── Weighted Class Loss ───
        d_loss_class_real = (benign_weight * d_loss_benign_class) + (attack_weight * d_loss_attack_class)
        d_loss_class = 0.5 * (d_loss_class_real + d_loss_fake_class)

        # ─── Combined Loss with Task Weights ───
        d_loss = (validity_weight * d_loss_validity) + (class_weight * d_loss_class)

        # ═══════════════════════════════════════════════════════════════════════
        # METRICS CALCULATION FOR LOGGING
        # ═══════════════════════════════════════════════════════════════════════
        # ─── Total Losses for Each Sample Type ───
        d_loss_benign_total = benign_weight * (d_loss_benign[0])
        d_loss_attack_total = attack_weight * (d_loss_attack[0])
        d_loss_fake_total = 0.5 * (d_loss_fake[0])
        d_loss_total = d_loss_benign_total + d_loss_attack_total + d_loss_fake_total

        # ─── Weighted Accuracies ───
        d_valid_acc_real = (benign_weight * d_benign_valid_acc + attack_weight * d_attack_valid_acc)
        d_class_acc_real = (benign_weight * d_benign_class_acc + attack_weight * d_attack_class_acc)

        # ─── Create Metrics Dictionary ───
        d_metrics = {
            "Total Loss": f"{d_loss_total:.4f}",
            "Benign Loss": f"{d_loss_benign[0]:.4f}",
            "Attack Loss": f"{d_loss_attack[0]:.4f}",
            "Fake Loss": f"{d_loss_fake[0]:.4f}",
            "Validity Loss": f"{d_loss_validity:.4f}",
            "Class Loss": f"{d_loss_class:.4f}",
            "Benign Validity Acc": f"{d_benign_valid_acc * 100:.2f}%",
            "Attack Validity Acc": f"{d_attack_valid_acc * 100:.2f}%",
            "Fake Validity Acc": f"{d_fake_valid_acc * 100:.2f}%",
            "Benign Class Acc": f"{d_benign_class_acc * 100:.2f}%",
            "Attack Class Acc": f"{d_attack_class_acc * 100:.2f}%",
            "Fake Class Acc": f"{d_fake_class_acc * 100:.2f}%"
        }

        return d_loss, d_metrics

    def calculate_jsd_loss(self, real_outputs, fake_outputs):
        """
        Calculate Jensen-Shannon Divergence Loss between real and fake sample distributions.
        This can be used as an alternative or supplement to binary cross entropy for
        measuring discriminator performance.
        """
        # ─── Average Probabilities ───
        p_real = tf.reduce_mean(real_outputs, axis=0)
        p_fake = tf.reduce_mean(fake_outputs, axis=0)

        # ─── Calculate Mixtures ───
        p_mixture = 0.5 * (p_real + p_fake)

        # ─── Calculate JS Divergence ───
        kl_real_mix = tf.reduce_sum(p_real * tf.math.log(p_real / p_mixture + 1e-10))
        kl_fake_mix = tf.reduce_sum(p_fake * tf.math.log(p_fake / p_mixture + 1e-10))

        # ─── JS Divergence ───
        jsd = 0.5 * (kl_real_mix + kl_fake_mix)

        return jsd

    def nids_loss(self, real_output, fake_output):
        """
        Compute the NIDS loss on real and fake samples.
        For real samples, the target is 1 (benign), and for fake samples, 0 (attack).
        Returns a scalar loss value.
        """
        # ─── Define Labels ───
        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)

        # ─── Define Loss Function ───
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # ─── Calculate Outputs ───
        real_loss = bce(real_labels, real_output)
        fake_loss = bce(fake_labels, fake_output)

        # ─── Sum Total Loss ───
        total_loss = real_loss + fake_loss
        return total_loss.numpy()

    #########################################################################
    #                    PROBABILISTIC FUSION METHODS                      #
    #########################################################################
    def probabilistic_fusion(self, input_data):
        """
        Apply probabilistic fusion to combine validity and class predictions.
        Returns combined probabilities for all four possible outcomes.
        """
        # Get discriminator predictions
        validity_scores, class_predictions = self.discriminator.predict(input_data)

        total_samples = len(input_data)
        results = []

        for i in range(total_samples):
            # Validity probabilities: P(valid) and P(invalid)
            p_valid = validity_scores[i][0]  # Probability of being valid/real
            p_invalid = 1 - p_valid  # Probability of being invalid/fake

            # Class probabilities: 2 classes (benign=0, attack=1)
            p_benign = class_predictions[i][0]  # Probability of being benign
            p_attack = class_predictions[i][1]  # Probability of being attack

            # Calculate joint probabilities for all combinations
            p_valid_benign = p_valid * p_benign
            p_valid_attack = p_valid * p_attack
            p_invalid_benign = p_invalid * p_benign
            p_invalid_attack = p_invalid * p_attack

            # Store all probabilities in a dictionary
            probabilities = {
                "valid_benign": p_valid_benign,
                "valid_attack": p_valid_attack,
                "invalid_benign": p_invalid_benign,
                "invalid_attack": p_invalid_attack
            }

            # Find the most likely outcome
            most_likely = max(probabilities, key=probabilities.get)

            # For analysis, add the actual probabilities alongside the classification
            result = {
                "classification": most_likely,
                "probabilities": probabilities
            }

            results.append(result)

        return results

    def validate_with_probabilistic_fusion(self, validation_data, validation_labels=None):
        """
        Evaluate model using probabilistic fusion and calculate metrics if labels are available.
        """
        fusion_results = self.probabilistic_fusion(validation_data)

        # Extract classifications
        classifications = [result["classification"] for result in fusion_results]

        # Count occurrences of each class
        predicted_class_distribution = Counter(classifications)
        self.logger.info(f"Predicted Class Distribution: {dict(predicted_class_distribution)}")

        # If we have ground truth labels, calculate accuracy
        if validation_labels is not None:
            correct_predictions = 0
            correct_classifications = []
            true_classifications = []

            for i, result in enumerate(fusion_results):
                # Get the true label (assuming 0=benign, 1=attack)
                if isinstance(validation_labels, np.ndarray) and validation_labels.ndim > 1:
                    true_class_idx = np.argmax(validation_labels[i])
                else:
                    true_class_idx = validation_labels[i]

                true_class = "benign" if true_class_idx == 0 else "attack"

                # For validation data (which is real), expected validity is "valid"
                true_validity = "valid"  # Since validation data is real data

                # Construct the true combined label
                true_combined = f"{true_validity}_{true_class}"

                # Add to true classifications list
                true_classifications.append(true_combined)

                # Check if prediction matches
                if result["classification"] == true_combined:
                    correct_predictions += 1
                    correct_classifications.append(result["classification"])

            # Count distribution of correctly classified samples
            correct_class_distribution = Counter(correct_classifications)

            # Count distribution of true classes
            true_class_distribution = Counter(true_classifications)
            self.logger.info(f"True Class Distribution: {dict(true_class_distribution)}")

            accuracy = correct_predictions / len(validation_data)
            self.logger.info(f"Accuracy: {accuracy:.4f}")

            metrics = {
                "accuracy": accuracy,
                "total_samples": len(validation_data),
                "correct_predictions": correct_predictions,
                "predicted_class_distribution": dict(predicted_class_distribution),
                "correct_class_distribution": dict(correct_class_distribution),
                "true_class_distribution": dict(true_class_distribution)
            }

            return classifications, metrics

        return classifications, {"predicted_class_distribution": dict(predicted_class_distribution)}

    def analyze_fusion_results(self, fusion_results):
        """Analyze the distribution of probabilities from fusion results"""
        # Extract probabilities for each category
        valid_benign_probs = [r["probabilities"]["valid_benign"] for r in fusion_results]
        valid_attack_probs = [r["probabilities"]["valid_attack"] for r in fusion_results]
        invalid_benign_probs = [r["probabilities"]["invalid_benign"] for r in fusion_results]
        invalid_attack_probs = [r["probabilities"]["invalid_attack"] for r in fusion_results]

        # Calculate summary statistics
        categories = ["Valid Benign", "Valid Attack", "Invalid Benign", "Invalid Attack"]
        all_probs = [valid_benign_probs, valid_attack_probs, invalid_benign_probs, invalid_attack_probs]

        for cat, probs in zip(categories, all_probs):
            self.logger.info(
                f"{cat}: Mean={np.mean(probs):.4f}, Median={np.median(probs):.4f}, Max={np.max(probs):.4f}")

        # You could add additional visualizations or analysis here

#########################################################################
#                   CUSTOM TRAINING STEP METHODS                        #
#########################################################################
    @tf.function
    def train_discriminator_step(self, real_data, real_labels, real_validity_labels):
        """
        Custom training step for discriminator on real data.

        Args:
            real_data: Real input features
            real_labels: One-hot encoded class labels
            real_validity_labels: Validity labels (1 for real)

        Returns:
            Tuple of (total_loss, validity_loss, class_loss, validity_acc, class_acc)
        """
        # Convert inputs to float32 for type consistency
        real_data = tf.cast(real_data, tf.float32)
        real_labels = tf.cast(real_labels, tf.float32)
        real_validity_labels = tf.cast(real_validity_labels, tf.float32)

        with tf.GradientTape() as tape:
            # Forward pass with training=True
            validity_pred, class_pred = self.discriminator(real_data, training=True)

            # Calculate losses
            validity_loss = tf.keras.losses.binary_crossentropy(real_validity_labels, validity_pred)
            validity_loss = tf.reduce_mean(validity_loss)
            class_loss = tf.keras.losses.categorical_crossentropy(real_labels, class_pred)
            class_loss = tf.reduce_mean(class_loss)

            # Reduce validity loss weight for real data to balance gradients
            total_loss = (0.15 * validity_loss) + class_loss

        # Calculate gradients and update weights
        gradients = tape.gradient(total_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        # Calculate accuracies
        validity_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.round(validity_pred), real_validity_labels), tf.float32)
        )
        class_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(class_pred, axis=1), tf.argmax(real_labels, axis=1)), tf.float32)
        )

        return total_loss, validity_loss, class_loss, validity_acc, class_acc

    @tf.function
    def train_discriminator_on_fake_step(self, fake_data, fake_labels, fake_validity_labels):
        """
        Custom training step for discriminator on fake/generated data.

        Args:
            fake_data: Generated input features
            fake_labels: One-hot encoded class labels for generated data
            fake_validity_labels: Validity labels (0 for fake)

        Returns:
            Tuple of (total_loss, validity_loss, class_loss, validity_acc, class_acc)
        """
        # Convert inputs to float32 for type consistency
        fake_data = tf.cast(fake_data, tf.float32)
        fake_labels = tf.cast(fake_labels, tf.float32)
        fake_validity_labels = tf.cast(fake_validity_labels, tf.float32)

        with tf.GradientTape() as tape:
            # Forward pass with training=True
            validity_pred, class_pred = self.discriminator(fake_data, training=True)

            # Calculate losses
            validity_loss = tf.keras.losses.binary_crossentropy(fake_validity_labels, validity_pred)
            validity_loss = tf.reduce_mean(validity_loss)
            class_loss = tf.keras.losses.categorical_crossentropy(fake_labels, class_pred)
            class_loss = tf.reduce_mean(class_loss)

            # Increase validity loss weight for fake data to balance with real data
            total_loss = (7.0 * validity_loss) + class_loss

        # Calculate gradients and update weights
        gradients = tape.gradient(total_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        # Calculate accuracies
        validity_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.round(validity_pred), fake_validity_labels), tf.float32)
        )
        class_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(class_pred, axis=1), tf.argmax(fake_labels, axis=1)), tf.float32)
        )

        return total_loss, validity_loss, class_loss, validity_acc, class_acc


    # -- Custom Evaluation Helper -- #
    @tf.function
    def evaluate_discriminator(self, data, labels, validity_labels):
        """
        Evaluate discriminator without updating weights.

        Args:
            data: Input features
            labels: One-hot encoded class labels
            validity_labels: Validity labels

        Returns:
            Tuple of (total_loss, validity_loss, class_loss, validity_acc, class_acc)
        """
        # Convert inputs to float32 to ensure type consistency
        data = tf.cast(data, tf.float32)
        labels = tf.cast(labels, tf.float32)
        validity_labels = tf.cast(validity_labels, tf.float32)

        # Forward pass with training=False for evaluation
        validity_pred, class_pred = self.discriminator(data, training=False)

        # Calculate losses
        validity_loss = tf.keras.losses.binary_crossentropy(validity_labels, validity_pred)
        validity_loss = tf.reduce_mean(validity_loss)

        class_loss = tf.keras.losses.categorical_crossentropy(labels, class_pred)
        class_loss = tf.reduce_mean(class_loss)

        total_loss = validity_loss + class_loss

        # Calculate accuracies
        validity_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.round(validity_pred), validity_labels), tf.float32)
        )
        class_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(class_pred, axis=1), tf.argmax(labels, axis=1)), tf.float32)
        )

        return total_loss, validity_loss, class_loss, validity_acc, class_acc

#########################################################################
#                            TRAINING PROCESS                          #
#########################################################################
    def aggregate_fit(self, server_round, results, failures):
        # -- Set the model with global weights, Bring in the parameters for the global model --#
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving global model after round {server_round}...")
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters[0])
            if len(aggregated_weights) == len(self.discriminator.get_weights()):
                self.discriminator.set_weights(aggregated_weights)
        # EoF Set global weights
        # save model before synthetic contextualization
        model_save_path = "../../../../../ModelArchive/discriminator_GLOBAL_B4Fit_ACGAN.h5"
        self.discriminator.save(model_save_path)
        print(f"Model saved at: {model_save_path}")

        # -- make sure discriminator is trainable for individual training -- #
        self.discriminator.trainable = True
        # Ensure all layers within discriminator are trainable
        for layer in self.discriminator.layers:
            layer.trainable = True

        # -- Set the Data --#
        X_train = np.array(self.x_train)
        y_train = np.array(self.y_train)

        print("Xtrain Data shape:", X_train.shape)

        # Separate benign (0) and attack (1) samples for class-balanced training
        benign_indices = np.where(y_train == 0)[0]
        attack_indices = np.where(y_train == 1)[0]

        self.logger.info(f"Dataset composition - Benign: {len(benign_indices)}, Attack: {len(attack_indices)}")

        # Calculate actual steps per epoch based on smaller class
        actual_steps_per_epoch = min(
            self.steps_per_epoch,
            min(len(benign_indices), len(attack_indices)) // self.batch_size
        )

        # Log model settings at the start
        self.log_model_settings()
        self.logger.info(f"Using {actual_steps_per_epoch} steps per epoch for class-balanced training")

        # -- Label smoothing factors -- #
        valid_smoothing_factor = 0.15
        fake_smoothing_factor = 0.1
        gen_smoothing_factor = 0.1

        self.logger.info(f"Using valid label smoothing with factor: {valid_smoothing_factor}")
        self.logger.info(f"Using fake label smoothing with factor: {fake_smoothing_factor}")

        # -- Early Stopping Tracking
        best_fusion_accuracy = 0.0
        best_epoch = 0
        patience_counter = 0
        best_discriminator_weights = None
        best_generator_weights = None

        self.logger.info(f"Early stopping enabled with patience={self.early_stopping_patience}, min_delta={self.min_delta}")

        # -- Training loop --#
        for epoch in range(self.epochs):
            print(f'\n=== Epoch {epoch}/{self.epochs} ===\n')
            self.logger.info(f'=== Epoch {epoch}/{self.epochs} ===')

            # Shuffle indices at the start of each epoch
            shuffled_benign_indices = np.random.permutation(benign_indices)
            shuffled_attack_indices = np.random.permutation(attack_indices)

            # Initialize offsets for this epoch
            benign_offset = 0
            attack_offset = 0

            # Accumulate metrics for epoch-level reporting
            epoch_d_losses = []

            # Iterate through steps
            for step in range(actual_steps_per_epoch):
                # --------------------------
                # Sample Benign Batch
                # --------------------------
                benign_start = benign_offset
                benign_end = min(benign_start + self.batch_size, len(benign_indices))
                benign_idx = shuffled_benign_indices[benign_start:benign_end]
                benign_batch_data = tf.gather(X_train, benign_idx)
                benign_batch_labels = tf.gather(y_train, benign_idx)
                benign_batch_size = benign_end - benign_start
                benign_offset = benign_end

                # --------------------------
                # Sample Attack Batch
                # --------------------------
                attack_start = attack_offset
                attack_end = min(attack_start + self.batch_size, len(attack_indices))
                attack_idx = shuffled_attack_indices[attack_start:attack_end]
                attack_batch_data = tf.gather(X_train, attack_idx)
                attack_batch_labels = tf.gather(y_train, attack_idx)
                attack_batch_size = attack_end - attack_start
                attack_offset = attack_end

                # --------------------------
                # Train Discriminator on Real Data
                # --------------------------

                # Ensure labels are one-hot encoded
                if len(benign_batch_labels.shape) == 1:
                    benign_labels_onehot = tf.one_hot(tf.cast(benign_batch_labels, tf.int32), depth=self.num_classes)
                else:
                    benign_labels_onehot = benign_batch_labels

                if len(attack_batch_labels.shape) == 1:
                    attack_labels_onehot = tf.one_hot(tf.cast(attack_batch_labels, tf.int32), depth=self.num_classes)
                else:
                    attack_labels_onehot = attack_batch_labels

                # Create validity labels with smoothing
                valid_smooth_benign = tf.ones((benign_batch_size, 1)) * (1 - valid_smoothing_factor)
                valid_smooth_attack = tf.ones((attack_batch_size, 1)) * (1 - valid_smoothing_factor)

                # Train on benign batch
                d_total_benign, d_val_loss_benign, d_cls_loss_benign, d_val_acc_benign, d_cls_acc_benign = \
                    self.train_discriminator_step(benign_batch_data, benign_labels_onehot, valid_smooth_benign)

                # Train on attack batch
                d_total_attack, d_val_loss_attack, d_cls_loss_attack, d_val_acc_attack, d_cls_acc_attack = \
                    self.train_discriminator_step(attack_batch_data, attack_labels_onehot, valid_smooth_attack)

                # --------------------------
                # Train Discriminator on Fake Data
                # --------------------------

                # Determine fake batch size (use max of benign/attack batch size)
                fake_batch_size = max(benign_batch_size, attack_batch_size)

                # Generate fake data
                noise = tf.random.normal((fake_batch_size, self.latent_dim))
                fake_labels = tf.random.uniform((fake_batch_size,), minval=0, maxval=self.num_classes, dtype=tf.int32)
                fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)
                generated_data = self.generator.predict([noise, fake_labels], verbose=0)

                fake_smooth = tf.zeros((fake_batch_size, 1)) + fake_smoothing_factor

                # Train on fake batch
                d_total_fake, d_val_loss_fake, d_cls_loss_fake, d_val_acc_fake, d_cls_acc_fake = \
                    self.train_discriminator_on_fake_step(generated_data, fake_labels_onehot, fake_smooth)

                # --------------------------
                # Average Discriminator Metrics
                # --------------------------
                d_total_loss = (float(d_total_benign.numpy()) + float(d_total_attack.numpy()) + float(d_total_fake.numpy())) / 3.0
                d_validity_loss = (float(d_val_loss_benign.numpy()) + float(d_val_loss_attack.numpy()) + float(d_val_loss_fake.numpy())) / 3.0
                d_class_loss = (float(d_cls_loss_benign.numpy()) + float(d_cls_loss_attack.numpy()) + float(d_cls_loss_fake.numpy())) / 3.0
                d_validity_acc = (float(d_val_acc_benign.numpy()) + float(d_val_acc_attack.numpy()) + float(d_val_acc_fake.numpy())) / 3.0
                d_class_acc = (float(d_cls_acc_benign.numpy()) + float(d_cls_acc_attack.numpy()) + float(d_cls_acc_fake.numpy())) / 3.0

                # Store metrics for epoch-level reporting
                epoch_d_losses.append(d_total_loss)

                # Log every 10 steps to avoid overwhelming the logs
                if step % 10 == 0:
                    self.logger.info(
                        f"Step {step}/{actual_steps_per_epoch} - "
                        f"D Loss: {d_total_loss:.4f} (Benign: {float(d_total_benign.numpy()):.4f}, "
                        f"Attack: {float(d_total_attack.numpy()):.4f}, Fake: {float(d_total_fake.numpy()):.4f})"
                    )

            # Collect final epoch metrics (averaged across all steps)
            avg_d_total_loss = np.mean(epoch_d_losses)

            d_metrics = {
                "Total Loss": f"{d_total_loss:.4f}",
                "Validity Loss": f"{d_validity_loss:.4f}",
                "Class Loss": f"{d_class_loss:.4f}",
                "Validity Accuracy": f"{d_validity_acc * 100:.2f}%",
                "Class Accuracy": f"{d_class_acc * 100:.2f}%"
            }

            self.logger.info(f"=== Epoch {epoch} Training Summary ===")
            self.logger.info(f"Average Discriminator Loss: {avg_d_total_loss:.4f}")
            self.logger.info(f"Final Step - D Loss: {d_total_loss:.4f}")

            # --------------------------
            # Validation every 1 epochs
            # --------------------------
            if epoch % 1 == 0:
                self.logger.info(f"=== Epoch {epoch} Validation ===")
                d_val_loss, d_val_metrics = self.validation_disc()

                # -- Probabilistic Fusion Validation -- #
                self.logger.info("=== Probabilistic Fusion Validation ===")
                fusion_results, fusion_metrics = self.validate_with_probabilistic_fusion(self.x_val, self.y_val)
                self.logger.info(f"Probabilistic Fusion Accuracy: {fusion_metrics['accuracy'] * 100:.2f}%")

                # Log distribution of classifications
                self.logger.info(f"Predicted Class Distribution: {fusion_metrics['predicted_class_distribution']}")
                self.logger.info(f"Correct Class Distribution: {fusion_metrics['correct_class_distribution']}")
                self.logger.info(f"True Class Distribution: {fusion_metrics['true_class_distribution']}")

                nids_val_metrics = None
                if self.nids is not None:
                    nids_custom_loss, nids_val_metrics = self.validation_NIDS()
                    self.logger.info(f"Validation NIDS Custom Loss: {nids_custom_loss:.4f}")

                # Log the metrics for this epoch using our new logging method
                # Note: Only discriminator metrics since we're not training the generator
                self.logger.info(f"=== Epoch {epoch} Validation Metrics ===")
                self.logger.info("Discriminator Metrics:")
                for key, value in d_val_metrics.items():
                    self.logger.info(f"  {key}: {value}")
                if nids_val_metrics is not None:
                    self.logger.info("NIDS Metrics:")
                    for key, value in nids_val_metrics.items():
                        self.logger.info(f"  {key}: {value}")
                self.logger.info("=" * 50)

                # --------------------------
                # Early Stopping Check
                # --------------------------
                current_fusion_accuracy = fusion_metrics['accuracy']

                # Check if current accuracy is better than best (with min_delta threshold)
                if current_fusion_accuracy > best_fusion_accuracy + self.min_delta:
                    # Improvement detected
                    best_fusion_accuracy = current_fusion_accuracy
                    best_epoch = epoch
                    patience_counter = 0

                    # Save best model weights (copy to avoid reference issues)
                    best_discriminator_weights = [w.copy() for w in self.discriminator.get_weights()]
                    best_generator_weights = [w.copy() for w in self.generator.get_weights()]

                    self.logger.info(f"✓ New best model! Fusion accuracy improved to {best_fusion_accuracy * 100:.2f}% at epoch {best_epoch}")
                    self.logger.info(f"  Model weights saved. Patience counter reset to 0/{self.early_stopping_patience}")
                else:
                    # No improvement
                    patience_counter += 1
                    self.logger.info(f"⚠ No improvement in fusion accuracy (best: {best_fusion_accuracy * 100:.2f}% at epoch {best_epoch})")
                    self.logger.info(f"  Patience counter: {patience_counter}/{self.early_stopping_patience}")

                    # Check if we should stop
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        self.logger.info(f"Best fusion accuracy: {best_fusion_accuracy * 100:.2f}% at epoch {best_epoch}")
                        self.logger.info(f"Restoring best model weights from epoch {best_epoch}...")

                        # Restore best weights
                        if best_discriminator_weights is not None and best_generator_weights is not None:
                            self.discriminator.set_weights(best_discriminator_weights)
                            self.generator.set_weights(best_generator_weights)
                            self.logger.info("✓ Best model weights restored successfully")

                        # Save the best model
                        model_save_path = "../../../../ModelArchive/discriminator_GLOBAL_AfterFit_ACGAN.h5"
                        self.discriminator.save(model_save_path)
                        print(f"Best model saved at: {model_save_path}")

                        # Log training completion with early stopping
                        self.logger.info("=" * 50)
                        self.logger.info("TRAINING COMPLETED (EARLY STOPPED)")
                        self.logger.info("=" * 50)
                        self.logger.info(f"Best probabilistic fusion accuracy: {best_fusion_accuracy * 100:.2f}%")
                        self.logger.info(f"Best model from epoch: {best_epoch}")
                        self.logger.info(f"Total epochs trained: {epoch + 1}")
                        self.logger.info(f"Training stopped early due to no improvement for {self.early_stopping_patience} epochs")
                        self.logger.info("=" * 50 + "\n")

                        # Return early
                        return self.discriminator.get_weights(), {
                            "best_fusion_accuracy": best_fusion_accuracy,
                            "best_epoch": best_epoch,
                            "total_epochs_trained": epoch + 1,
                            "early_stopped": True
                        }

        # Training completed all epochs
        self.logger.info("=" * 50)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 50)
        self.logger.info(f"Best probabilistic fusion accuracy: {best_fusion_accuracy * 100:.2f}%")
        self.logger.info(f"Best model from epoch: {best_epoch}")
        self.logger.info(f"Total epochs trained: {self.epochs}")
        self.logger.info("Training completed all epochs")
        self.logger.info("=" * 50 + "\n")

        # Restore best weights if we have them
        if best_discriminator_weights is not None and best_generator_weights is not None:
            self.discriminator.set_weights(best_discriminator_weights)
            self.generator.set_weights(best_generator_weights)
            self.logger.info(f"✓ Best model weights from epoch {best_epoch} restored")

        # save model after training completion
        model_save_path = "../../../../ModelArchive/discriminator_GLOBAL_AfterFit_ACGAN.h5"
        self.discriminator.save(model_save_path)
        print(f"Model saved at: {model_save_path}")

        # Send updated weights back to clients
        return self.discriminator.get_weights(), {
            "best_fusion_accuracy": best_fusion_accuracy,
            "best_epoch": best_epoch,
            "total_epochs_trained": self.epochs,
            "early_stopped": False
        }



        # -- Validation Functions (Disc, Gen, NIDS) -- #

    def validation_disc(self):
        """
        Evaluate the discriminator on the validation set using custom evaluation function.
        First, evaluate on real data (with labels = 1) and then on fake data (labels = 0).
        Prints and returns the average total loss along with a metrics dictionary.
        """
        # --- Evaluate on real validation data ---
        val_valid_labels = tf.ones((len(self.x_val), 1))

        # Ensure y_val is one-hot encoded if needed
        if self.y_val.ndim == 1 or self.y_val.shape[1] != self.num_classes:
            y_val_onehot = tf.one_hot(self.y_val, depth=self.num_classes)
        else:
            y_val_onehot = self.y_val

        # Use custom evaluation function for real data
        d_total_loss_real, d_val_loss_real, d_cls_loss_real, d_val_acc_real, d_cls_acc_real = \
            self.evaluate_discriminator(self.x_val, y_val_onehot, val_valid_labels)

        # --- Evaluate on generated (fake) data ---
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        fake_labels = tf.random.uniform(
            (len(self.x_val),), minval=0, maxval=self.num_classes, dtype=tf.int32
        )
        fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)
        fake_valid_labels = tf.zeros((len(self.x_val), 1))
        generated_data = self.generator.predict([noise, fake_labels])

        # Use custom evaluation function for fake data
        d_total_loss_fake, d_val_loss_fake, d_cls_loss_fake, d_val_acc_fake, d_cls_acc_fake = \
            self.evaluate_discriminator(generated_data, fake_labels_onehot, fake_valid_labels)

        # --- Compute average loss ---
        avg_total_loss = 0.5 * (float(d_total_loss_real.numpy()) + float(d_total_loss_fake.numpy()))

        self.logger.info("Validation Discriminator Evaluation:")
        # Log for real data
        self.logger.info(
            f"Real Data -> Total Loss: {float(d_total_loss_real.numpy()):.4f}, "
            f"Validity Loss: {float(d_val_loss_real.numpy()):.4f}, "
            f"Class Loss: {float(d_cls_loss_real.numpy()):.4f}, "
            f"Validity Accuracy: {float(d_val_acc_real.numpy()) * 100:.2f}%, "
            f"Class Accuracy: {float(d_cls_acc_real.numpy()) * 100:.2f}%"
        )
        self.logger.info(
            f"Fake Data -> Total Loss: {float(d_total_loss_fake.numpy()):.4f}, "
            f"Validity Loss: {float(d_val_loss_fake.numpy()):.4f}, "
            f"Class Loss: {float(d_cls_loss_fake.numpy()):.4f}, "
            f"Validity Accuracy: {float(d_val_acc_fake.numpy()) * 100:.2f}%, "
            f"Class Accuracy: {float(d_cls_acc_fake.numpy()) * 100:.2f}%"
        )
        self.logger.info(f"Average Discriminator Loss: {avg_total_loss:.4f}")

        metrics = {
            "Real Total Loss": f"{float(d_total_loss_real.numpy()):.4f}",
            "Real Validity Loss": f"{float(d_val_loss_real.numpy()):.4f}",
            "Real Class Loss": f"{float(d_cls_loss_real.numpy()):.4f}",
            "Real Validity Accuracy": f"{float(d_val_acc_real.numpy()) * 100:.2f}%",
            "Real Class Accuracy": f"{float(d_cls_acc_real.numpy()) * 100:.2f}%",
            "Fake Total Loss": f"{float(d_total_loss_fake.numpy()):.4f}",
            "Fake Validity Loss": f"{float(d_val_loss_fake.numpy()):.4f}",
            "Fake Class Loss": f"{float(d_cls_loss_fake.numpy()):.4f}",
            "Fake Validity Accuracy": f"{float(d_val_acc_fake.numpy()) * 100:.2f}%",
            "Fake Class Accuracy": f"{float(d_cls_acc_fake.numpy()) * 100:.2f}%",
            "Average Total Loss": f"{avg_total_loss:.4f}"
        }
        return avg_total_loss, metrics

    def validation_NIDS(self):
        """
        Evaluate the NIDS model on validation data augmented with generated fake samples.
        Real data is labeled as 1 (benign) and fake/generated data as 0 (attack).
        Prints detailed metrics including a classification report and returns the custom
        NIDS loss along with a metrics dictionary.
        """
        if self.nids is None:
            print("NIDS model is not defined.")
            return None

        # --- Prepare Real Data ---
        X_real = self.x_val
        y_real = np.ones((len(self.x_val),), dtype="int32")  # Real samples labeled 1

        # --- Generate Fake Data ---
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        fake_labels = tf.random.uniform(
            (len(self.x_val),), minval=0, maxval=self.num_classes, dtype=tf.int32
        )
        generated_samples = self.generator.predict([noise, fake_labels])
        # Rescale generated samples from [-1, 1] to [0, 1] so they match the NIDS training data.
        X_fake = (generated_samples + 1) / 2
        y_fake = np.zeros((len(self.x_val),), dtype="int32")  # Fake samples labeled 0

        # --- Compute custom NIDS loss ---
        real_output = self.nids.predict(X_real)
        fake_output = self.nids.predict(X_fake)
        custom_nids_loss = self.nids_loss(real_output, fake_output)

        # --- Evaluate on the Combined Dataset ---
        X_combined = np.vstack([X_real, X_fake])
        y_combined = np.hstack([y_real, y_fake])
        nids_eval = self.nids.evaluate(X_combined, y_combined, verbose=0)
        # Expected order: [loss, accuracy, precision, recall, auc, logcosh]

        # --- Compute Additional Metrics ---
        y_pred_probs = self.nids.predict(X_combined)
        y_pred = (y_pred_probs > 0.5).astype("int32")
        f1 = f1_score(y_combined, y_pred)
        class_report = classification_report(
            y_combined, y_pred, target_names=["Attack (Fake)", "Benign (Real)"]
        )

        self.logger.info("Validation NIDS Evaluation with Augmented Data:")
        self.logger.info(f"Custom NIDS Loss (Real vs Fake): {custom_nids_loss:.4f}")
        self.logger.info(f"Overall NIDS Loss: {nids_eval[0]:.4f}, Accuracy: {nids_eval[1]:.4f}, "
                         f"Precision: {nids_eval[2]:.4f}, Recall: {nids_eval[3]:.4f}, "
                         f"AUC: {nids_eval[4]:.4f}, LogCosh: {nids_eval[5]:.4f}")
        self.logger.info("Classification Report:")
        self.logger.info(class_report)
        self.logger.info(f"F1 Score: {f1:.4f}")

        metrics = {
            "Custom NIDS Loss": f"{custom_nids_loss:.4f}",
            "Loss": f"{nids_eval[0]:.4f}",
            "Accuracy": f"{nids_eval[1]:.4f}",
            "Precision": f"{nids_eval[2]:.4f}",
            "Recall": f"{nids_eval[3]:.4f}",
            "AUC": f"{nids_eval[4]:.4f}",
            "LogCosh": f"{nids_eval[5]:.4f}",
            "F1 Score": f"{f1:.4f}"
        }
        return custom_nids_loss, metrics

    # # -- Evaluate -- #
    # def evaluate(self, parameters, config):
    #
    #     # -- Set the model weights from the Host --#
    #     self.GAN.set_weights(parameters)
    #
    #     # Set the data
    #     X_test = self.x_test
    #     y_test = self.y_test
    #
    #     # --------------------------
    #     # Test Discriminator
    #     # --------------------------
    #     self.logger.info("-- Evaluating Discriminator --")
    #     # run the model
    #     results = self.discriminator.evaluate(X_test, [tf.ones((len(y_test), 1)), y_test], verbose=0)
    #     # Using the updated ordering:
    #     d_loss_total = results[0]
    #     d_loss_validity = results[1]
    #     d_loss_class = results[2]
    #     d_validity_acc = results[3]
    #     d_validity_bin_acc = results[4]
    #     d_validity_auc = results[5]
    #     d_class_acc = results[6]
    #     d_class_cat_acc = results[7]
    #
    #     d_eval_metrics = {
    #         "Loss": f"{d_loss_total:.4f}",
    #         "Validity Loss": f"{d_loss_validity:.4f}",
    #         "Class Loss": f"{d_loss_class:.4f}",
    #         "Validity Accuracy": f"{d_validity_acc * 100:.2f}%",
    #         "Validity Binary Accuracy": f"{d_validity_bin_acc * 100:.2f}%",
    #         "Validity AUC": f"{d_validity_auc * 100:.2f}%",
    #         "Class Accuracy": f"{d_class_acc * 100:.2f}%",
    #         "Class Categorical Accuracy": f"{d_class_cat_acc * 100:.2f}%"
    #     }
    #     self.logger.info(
    #         f"Discriminator Total Loss: {d_loss_total:.4f} | Validity Loss: {d_loss_validity:.4f} | Class Loss: {d_loss_class:.4f}"
    #     )
    #     self.logger.info(
    #         f"Validity Accuracy: {d_validity_acc * 100:.2f}%, Binary Accuracy: {d_validity_bin_acc * 100:.2f}%, AUC: {d_validity_auc * 100:.2f}%"
    #     )
    #     self.logger.info(
    #         f"Class Accuracy: {d_class_acc * 100:.2f}%, Categorical Accuracy: {d_class_cat_acc * 100:.2f}%"
    #     )
    #
    #     # --------------------------
    #     # Test Generator (ACGAN)
    #     # --------------------------
    #     self.logger.info("-- Evaluating Generator --")
    #
    #     # get the noise samples
    #     noise = tf.random.normal((len(y_test), self.latent_dim))
    #     sampled_labels = tf.random.uniform((len(y_test),), minval=0, maxval=self.num_classes, dtype=tf.int32)
    #
    #     # run the model
    #     g_loss = self.ACGAN.evaluate([noise, sampled_labels],
    #                                  [tf.ones((len(y_test), 1)),
    #                                   tf.one_hot(sampled_labels, depth=self.num_classes)],
    #                                  verbose=0)
    #
    #     # Using the updated ordering for ACGAN:
    #     g_loss_total = g_loss[0]
    #     g_loss_validity = g_loss[1]
    #     g_loss_class = g_loss[2]
    #     g_validity_acc = g_loss[3]
    #     g_validity_bin_acc = g_loss[4]
    #     g_validity_auc = g_loss[5]
    #     g_class_acc = g_loss[6]
    #     g_class_cat_acc = g_loss[7]
    #
    #     g_eval_metrics = {
    #         "Loss": f"{g_loss_total:.4f}",
    #         "Validity Loss": f"{g_loss_validity:.4f}",
    #         "Class Loss": f"{g_loss_class:.4f}",
    #         "Validity Accuracy": f"{g_validity_acc * 100:.2f}%",
    #         "Validity Binary Accuracy": f"{g_validity_bin_acc * 100:.2f}%",
    #         "Validity AUC": f"{g_validity_auc * 100:.2f}%",
    #         "Class Accuracy": f"{g_class_acc * 100:.2f}%",
    #         "Class Categorical Accuracy": f"{g_class_cat_acc * 100:.2f}%"
    #     }
    #     self.logger.info(
    #         f"Generator Total Loss: {g_loss_total:.4f} | Validity Loss: {g_loss_validity:.4f} | Class Loss: {g_loss_class:.4f}"
    #     )
    #     self.logger.info(
    #         f"Validity Accuracy: {g_validity_acc * 100:.2f}%, Binary Accuracy: {g_validity_bin_acc * 100:.2f}%, AUC: {g_validity_auc * 100:.2f}%"
    #     )
    #     self.logger.info(
    #         f"Class Accuracy: {g_class_acc * 100:.2f}%, Categorical Accuracy: {g_class_cat_acc * 100:.2f}%"
    #     )
    #
    #     # --------------------------
    #     # Test NIDS
    #     # --------------------------
    #     nids_eval_metrics = None
    #     if self.nids is not None:
    #         self.logger.info("-- Evaluating NIDS --")
    #         # Prepare real test data (labeled as benign, 1)
    #         X_real = X_test
    #         y_real = np.ones((len(X_test),), dtype="int32")
    #
    #         # Generate fake test data (labeled as attack, 0)
    #         noise = tf.random.normal((len(X_test), self.latent_dim))
    #         fake_labels = tf.random.uniform((len(X_test),), minval=0, maxval=self.num_classes, dtype=tf.int32)
    #         generated_samples = self.generator.predict([noise, fake_labels])
    #         # Rescale generated samples from [-1, 1] to [0, 1]
    #         X_fake = (generated_samples + 1) / 2
    #         y_fake = np.zeros((len(X_test),), dtype="int32")
    #
    #         # Compute custom NIDS loss on real and fake outputs
    #         real_output = self.nids.predict(X_real)
    #         fake_output = self.nids.predict(X_fake)
    #         custom_nids_loss = self.nids_loss(real_output, fake_output)
    #
    #         # Combine real and fake data for evaluation
    #         X_combined = np.vstack([X_real, X_fake])
    #         y_combined = np.hstack([y_real, y_fake])
    #         nids_eval_results = self.nids.evaluate(X_combined, y_combined, verbose=0)
    #         # Expected order: [loss, accuracy, precision, recall, auc, logcosh]
    #
    #         # Compute additional metrics
    #         y_pred_probs = self.nids.predict(X_combined)
    #         y_pred = (y_pred_probs > 0.5).astype("int32")
    #         f1 = f1_score(y_combined, y_pred)
    #         class_report = classification_report(
    #             y_combined, y_pred, target_names=["Attack (Fake)", "Benign (Real)"]
    #         )
    #
    #         nids_eval_metrics = {
    #             "Custom NIDS Loss": f"{custom_nids_loss:.4f}",
    #             "Loss": f"{nids_eval_results[0]:.4f}",
    #             "Accuracy": f"{nids_eval_results[1]:.4f}",
    #             "Precision": f"{nids_eval_results[2]:.4f}",
    #             "Recall": f"{nids_eval_results[3]:.4f}",
    #             "AUC": f"{nids_eval_results[4]:.4f}",
    #             "LogCosh": f"{nids_eval_results[5]:.4f}",
    #             "F1 Score": f"{f1:.4f}"
    #         }
    #         self.logger.info(f"NIDS Custom Loss: {custom_nids_loss:.4f}")
    #         self.logger.info(
    #             f"NIDS Evaluation -> Loss: {nids_eval_results[0]:.4f}, Accuracy: {nids_eval_results[1]:.4f}, "
    #             f"Precision: {nids_eval_results[2]:.4f}, Recall: {nids_eval_results[3]:.4f}, "
    #             f"AUC: {nids_eval_results[4]:.4f}, LogCosh: {nids_eval_results[5]:.4f}")
    #         self.logger.info("NIDS Classification Report:")
    #         self.logger.info(class_report)
    #
    #     # Log the overall evaluation metrics using our logging function
    #     self.log_evaluation_metrics(d_eval_metrics, g_eval_metrics, nids_eval_metrics)
    #
    #     return d_loss_total, len(self.x_test), {}

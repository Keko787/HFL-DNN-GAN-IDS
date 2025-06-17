import os
import random
import time
from datetime import datetime
import argparse
import tensorflow as tf
import flwr as fl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


class BinaryWGeneratorClient(fl.client.NumPyClient):
    def __init__(self, gan, nids, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 noise_dim, epochs, steps_per_epoch, learning_rate):
        self.model = gan
        self.nids = nids
        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.x_train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(self.BATCH_SIZE)
        self.x_val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.BATCH_SIZE)

        self.gen_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

        self.generator = self.model.layers[0]
        self.discriminator = self.model.layers[1]

        self.precision = Precision()
        self.recall = Recall()
        self.accuracy = BinaryAccuracy()

        self.gen_accuracy = BinaryAccuracy(name="gen_accuracy")
        self.gen_precision = Precision(name="gen_precision")

        # -- Metric Helper Functions

    def update_generator_metrics(self, fake_output, threshold=0.0):
        # Convert the critic's output on generated samples to binary predictions.
        # For the generator, we want these samples to be classified as real (1).
        fake_preds = tf.cast(fake_output > threshold, tf.int32)
        target_labels = tf.zeros_like(fake_preds)
        self.gen_accuracy.update_state(target_labels, fake_preds)
        self.gen_precision.update_state(target_labels, fake_preds)

    def log_metrics(self, step, gen_loss):
        # Retrieve generator metrics.
        gen_acc = self.gen_accuracy.result().numpy()
        gen_prec = self.gen_precision.result().numpy()

        print(f"Step {step}, G Loss: {gen_loss.numpy():.4f}")
        print(f"Generator Metrics -- Accuracy: {gen_acc:.4f}, Precision: {gen_prec:.4f}")

    def reset_metrics(self):
        self.gen_accuracy.reset_states()
        self.gen_precision.reset_states()

    # Sending training params to host
    def get_parameters(self, config):
        # Combine generator and discriminator weights into a single list
        return self.generator.get_weights()

    # Loss Function

    def discriminator_loss(self, real_output, fake_output, gradient_penalty):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(
            real_output) + 15.0 * gradient_penalty  # Increased from 10.0 to 15.0

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def gradient_penalty(self, real_data, fake_data):
        feature_dim = tf.shape(real_data)[1]

        # Generate alpha and broadcast it
        alpha = tf.random.uniform([tf.shape(real_data)[0], 1], 0., 1., dtype=tf.float32)
        alpha = tf.broadcast_to(alpha, [tf.shape(real_data)[0], feature_dim])

        real_data = tf.cast(real_data, tf.float32)
        fake_data = tf.cast(fake_data, tf.float32)

        interpolated = alpha * real_data + (1 - alpha) * fake_data

        if random.random() < 0.01:  # Log occasionally
            print(
                f"Interpolated Mean: {tf.reduce_mean(interpolated).numpy()}, Std: {tf.math.reduce_std(interpolated).numpy()}")

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = tape.gradient(pred, [interpolated])[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))

        if random.random() < 0.01:
            print(
                f"Gradient Norm Mean: {tf.reduce_mean(grad_norm).numpy()}, Min: {tf.reduce_min(grad_norm).numpy()}, Max: {tf.reduce_max(grad_norm).numpy()}")

        return tf.reduce_mean((grad_norm - 1.0) ** 2)

    # Training
    def fit(self, parameters, config):
        self.generator.set_weights(parameters)
        discriminator = self.model.layers[1]

        for epoch in range(self.epochs):
            for step, (real_data, real_labels) in enumerate(self.x_train_ds.take(self.steps_per_epoch)):
                # generate noise for generator to use.
                real_batch_size = tf.shape(real_data)[0]  # Ensure real batch size
                noise = tf.random.normal([real_batch_size, self.noise_dim])

                # Train Generator
                with tf.GradientTape() as gen_tape:
                    # generate samples
                    generated_samples = self.generator(noise, training=True)

                    # take generated samples
                    fake_output = discriminator(generated_samples, training=True)

                    # generator loss function
                    gen_loss = self.generator_loss(fake_output)

                # update the gradiants and generator weights from gradiants
                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

                # Update metrics using discriminator outputs.
                self.update_generator_metrics(fake_output, 0.5)

                if step % 100 == 0:
                    self.log_metrics(step, gen_loss)

                # Reset metric states after each epoch.
            self.reset_metrics()

            # Evaluate Generator Performance (Optional)
            val_gen_loss = self.evaluate_validation_gen(self.generator, discriminator)
            print(f'Epoch {epoch + 1}, Validation Generator Loss: {val_gen_loss:.4f}')

            # Evaluate NIDS if Available
            if self.nids is not None:
                val_nids_loss = self.evaluate_validation_NIDS(self.generator)
                print(f'Epoch {epoch + 1}, Validation NIDS Loss: {val_nids_loss:.4f}')

        return self.get_parameters(config={}), len(self.x_train_ds), {}

    # validation
    def evaluate_validation_disc(self, generator, discriminator):
        total_disc_loss = 0.0
        num_batches = 0

        for step, (real_data, _) in enumerate(self.x_val_ds):
            # Generate fake samples
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = generator(noise, training=False)

            # Pass real and fake data through the discriminator
            real_output = discriminator(real_data, training=False)
            fake_output = discriminator(generated_samples, training=False)

            # Compute gradient penalty (optional for consistency)
            gp_loss = self.gradient_penalty(real_data, generated_samples)

            # Compute WGAN-GP loss
            disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + 10.0 * gp_loss

            total_disc_loss += disc_loss.numpy()
            num_batches += 1

        return total_disc_loss / num_batches  # Average loss

    def evaluate_validation_gen(self, generator, discriminator):
        # Generate fake samples
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = generator(noise, training=False)

        # Pass fake samples through the discriminator
        fake_output = discriminator(generated_samples, training=False)

        # Compute WGAN-GP generator loss (maximize discriminator's mistake)
        gen_loss = -tf.reduce_mean(fake_output)

        return float(gen_loss.numpy())

    def evaluate_validation_NIDS(self, generator):
        # Generate fake samples
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = generator(noise, training=False)

        # Ensure proper input format for NIDS
        real_data_batches = tf.concat([data for data, _ in self.x_val_ds], axis=0)

        # Get NIDS predictions
        real_output = self.nids(real_data_batches, training=False)
        fake_output = self.nids(generated_samples, training=False)

        # Compute binary cross-entropy loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_loss = bce(tf.zeros_like(real_output), real_output)  # Real samples should be classified as 0
        fake_loss = bce(tf.ones_like(fake_output), fake_output)  # Fake samples should be classified as 1

        nids_loss = real_loss
        gen_loss = fake_loss  # Generator loss to fool NIDS

        print(f'Validation GEN-NIDS Loss: {gen_loss.numpy()}')
        print(f'Validation NIDS Loss: {nids_loss.numpy()}')

        return float(nids_loss.numpy())

    # -- Evaluate -- #
    def evaluate(self, parameters, config):
        total_disc_loss = 0.0
        total_gen_loss = 0.0
        num_batches = 0
        precision_vals = []
        recall_vals = []
        accuracy_vals = []

        self.model.set_weights(parameters)
        generator = self.model.layers[0]
        discriminator = self.model.layers[1]

        for step, (test_data_batch, test_labels_batch) in enumerate(self.x_test_ds):
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = generator(noise, training=False)

            real_output = discriminator(test_data_batch, training=False)
            fake_output = discriminator(generated_samples, training=False)

            gp_loss = self.gradient_penalty(test_data_batch, generated_samples)
            disc_loss = self.discriminator_loss(real_output, fake_output, gp_loss)
            gen_loss = self.generator_loss(fake_output)

            total_disc_loss += disc_loss.numpy()
            total_gen_loss += gen_loss.numpy()
            num_batches += 1

            predictions = tf.concat([real_output, fake_output], axis=0)
            labels = tf.concat([tf.ones_like(real_output), tf.zeros_like(fake_output)], axis=0)

            self.precision.update_state(labels, predictions)
            self.recall.update_state(labels, predictions)
            self.accuracy.update_state(labels, predictions)

            precision_vals.append(self.precision.result().numpy())
            recall_vals.append(self.recall.result().numpy())
            accuracy_vals.append(self.accuracy.result().numpy())

        avg_disc_loss = total_disc_loss / num_batches
        avg_gen_loss = total_gen_loss / num_batches
        avg_precision = sum(precision_vals) / len(precision_vals)
        avg_recall = sum(recall_vals) / len(recall_vals)
        avg_accuracy = sum(accuracy_vals) / len(accuracy_vals)
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-10)

        print(f"Final Evaluation Critic Loss: {avg_disc_loss}")
        print(f"Final Evaluation Generator Loss: {avg_gen_loss}")
        print(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, Accuracy: {avg_accuracy:.4f}, F1 Score: {f1_score:.4f}")

        return avg_disc_loss, len(self.x_test_ds), {
            "precision": avg_precision,
            "recall": avg_recall,
            "accuracy": avg_accuracy,
            "f1_score": f1_score
        }

    def save(self, save_name):
        self.generator.save(f"../../../../../ModelArchive/generator_fed_WGANGP_{save_name}.h5")
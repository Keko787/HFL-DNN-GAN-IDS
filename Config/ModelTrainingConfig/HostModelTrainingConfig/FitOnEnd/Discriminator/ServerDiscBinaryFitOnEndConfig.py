import flwr as fl
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


# Custom FedAvg strategy with server-side model training and saving
class DiscriminatorSyntheticStrategy(fl.server.strategy.FedAvg):
    def __init__(self, gan, generator, discriminator, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE, noise_dim, epochs, steps_per_epoch,
                 dataset_used, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.gan = gan
        self.generator = generator  # Generator is fixed during discriminator training
        # create model
        self.discriminator = discriminator

        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.dataset_used = dataset_used

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val  # Add validation data
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(self.BATCH_SIZE)
        self.x_val_ds = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val)).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices(self.x_test).batch(self.BATCH_SIZE)

        lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.98, staircase=True)

        self.disc_optimizer = Adam(learning_rate=lr_schedule_disc, beta_1=0.5, beta_2=0.999)

        self.disc_accuracy = tf.keras.metrics.BinaryAccuracy(name='disc_accuracy')
        self.disc_precision = tf.keras.metrics.Precision(name='disc_precision')
        self.disc_recall = tf.keras.metrics.Recall(name='disc_recall')

    # -- Metrics--#

    def log_metrics(self, step, disc_loss):
        print(f"Step {step}, D Loss: {disc_loss.numpy():.4f}")
        print(f"Discriminator Metrics -- Accuracy: {self.disc_accuracy.result().numpy():.4f}, "
              f"Precision: {self.disc_precision.result().numpy():.4f}, "
              f"Recall: {self.disc_recall.result().numpy():.4f}")

    def update_metrics(self, real_output=None, fake_output=None):
        if real_output is not None:
            # Update discriminator metrics: real samples are labeled 0, fake samples are labeled 1
            real_labels = tf.zeros_like(real_output)
            fake_labels = tf.ones_like(fake_output)
            all_labels = tf.concat([real_labels, fake_labels], axis=0)
            all_predictions = tf.concat([real_output, fake_output], axis=0)
            self.disc_accuracy.update_state(all_labels, all_predictions)
            self.disc_precision.update_state(all_labels, all_predictions)
            self.disc_recall.update_state(all_labels, all_predictions)

    def reset_metrics(self):
        # Reset discriminator metrics
        self.disc_accuracy.reset_states()
        self.disc_precision.reset_states()
        self.disc_recall.reset_states()

    # -- Loss--#
    def discriminator_loss(self, real_output, fake_output):
        # Create binary labels: 0 for real, 1 for fake
        real_labels = tf.zeros_like(real_output)
        fake_labels = tf.ones_like(fake_output)

        # Compute binary cross-entropy loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_loss = bce(real_labels, real_output)
        fake_loss = bce(fake_labels, fake_output)

        return real_loss + fake_loss


    def aggregate_fit(self, server_round, results, failures):
        # -- Set the model with global weights, Bring in the parameters for the global model -- #
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving global model after round {server_round}...")
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters[0])
            if len(aggregated_weights) == len(self.gan.get_weights()):
                self.discriminator.set_weights(aggregated_weights)
        # EoF Set global weights

        #-- Training Loop --#
        for epoch in range(self.epochs):
            for step, (real_data, real_labels) in enumerate(self.x_train_ds.take(self.steps_per_epoch)):
                # generate noise for generator to use.
                real_batch_size = tf.shape(real_data)[0]  # Ensure real batch size
                noise = tf.random.normal([real_batch_size, self.noise_dim])

                with tf.GradientTape() as disc_tape:
                    # Generate samples
                    generated_samples = self.generator(noise, training=False)

                    # Discriminator predictions
                    real_output = self.discriminator(real_data, training=True)
                    fake_output = self.discriminator(generated_samples, training=True)

                    # Compute losses
                    disc_loss = self.discriminator_loss(real_output, fake_output)

                # Compute gradients and apply updates
                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                # Apply gradient clipping
                gradients_of_discriminator, _ = tf.clip_by_global_norm(gradients_of_discriminator, 5.0)

                self.disc_optimizer.apply_gradients(
                    zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                # Update Metrics
                # After computing real_output and fake_output
                self.update_metrics(real_output, fake_output)

                if step % 100 == 0:
                    self.log_metrics(step, disc_loss)

                # reset Training Metrics
            self.reset_metrics()

            # Validation after each epoch
            val_disc_loss = self.evaluate_validation_disc()
            print(f'Epoch {epoch + 1}, Validation D Loss: {val_disc_loss}')

        # Save the fine-tuned model
        self.discriminator.save("../../../../../ModelArchive/disc_model_fine_tuned.h5")
        print(f"Model fine-tuned and saved after round {server_round}.")

        # Send updated weights back to clients
        return self.discriminator.get_weights(), {}

    # Function to evaluate the discriminator on validation data
        # -- Validate -- #
    def evaluate_validation_disc(self):
        total_disc_loss = 0.0
        num_batches = 0

        for step, (real_data, real_labels) in enumerate(self.x_val_ds):
            # Generate fake samples using the generator
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = self.generator(noise, training=False)

            # Pass real and fake data through the discriminator
            real_output = self.discriminator(real_data, training=False)
            fake_output = self.discriminator(generated_samples, training=False)

            # Compute the discriminator loss using the real and fake outputs
            disc_loss = self.discriminator_loss(real_output, fake_output)
            total_disc_loss += disc_loss.numpy()
            num_batches += 1

        # Average discriminator loss over all validation batches
        avg_disc_loss = total_disc_loss / num_batches
        return avg_disc_loss

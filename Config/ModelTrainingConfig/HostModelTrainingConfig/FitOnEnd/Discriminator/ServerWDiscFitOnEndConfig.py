import flwr as fl
import argparse
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall, BinaryAccuracy
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays



# Custom FedAvg strategy with server-side model training and saving
class WDiscriminatorSyntheticStrategy(fl.server.strategy.FedAvg):
    def __init__(self, gan, nids, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE, noise_dim, epochs, steps_per_epoch,
                 dataset_used, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.model = gan
        self.nids = nids
        self.generator = self.model.layers[0]
        self.discriminator = self.model.layers[1]

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val  # Add validation data
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        self.BATCH_SIZE = BATCH_SIZE
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.dataset_used = dataset_used

        self.x_train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(self.BATCH_SIZE)
        self.x_val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.BATCH_SIZE)

        self.disc_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

        self.precision = Precision()
        self.recall = Recall()
        self.accuracy = BinaryAccuracy()


    def update_critic_metrics(self, real_output, fake_output, threshold=0.0):
        """
        Update classification metrics by thresholding raw scores.
        Real samples are expected to have scores > threshold (label 1),
        while fake samples should be below threshold (label 0).
        """
        # Convert scores to binary predictions based on the threshold.
        real_preds = tf.cast(real_output > threshold, tf.int32)
        fake_preds = tf.cast(fake_output > threshold, tf.int32)

        # Create corresponding labels.
        real_labels = tf.zeros_like(real_preds)
        fake_labels = tf.ones_like(fake_preds)

        # Concatenate predictions and labels.
        all_preds = tf.concat([real_preds, fake_preds], axis=0)
        all_labels = tf.concat([real_labels, fake_labels], axis=0)

        # Update metrics.
        self.accuracy.update_state(all_labels, all_preds)
        self.precision.update_state(all_labels, all_preds)
        self.recall.update_state(all_labels, all_preds)

    def log_metrics(self, step, disc_loss):
        # Retrieve critic (discriminator) metrics.
        acc = self.accuracy.result().numpy()
        prec = self.precision.result().numpy()
        rec = self.recall.result().numpy()

        print(f"Step {step}, D Loss: {disc_loss.numpy():.4f}")
        print(f"Critic Metrics -- Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    def reset_metrics(self):
        self.accuracy.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()


    # Loss Function
    def discriminator_loss(self, real_output, fake_output, gradient_penalty):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(
            real_output) + 15.0 * gradient_penalty  # Increased from 10.0 to 15.0

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def gradient_penalty(self, discriminator, real_data, fake_data):
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
            pred = discriminator(interpolated, training=True)

        grads = tape.gradient(pred, [interpolated])[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))

        if random.random() < 0.01:
            print(
                f"Gradient Norm Mean: {tf.reduce_mean(grad_norm).numpy()}, Min: {tf.reduce_min(grad_norm).numpy()}, Max: {tf.reduce_max(grad_norm).numpy()}")

        return tf.reduce_mean((grad_norm - 1.0) ** 2)

    def aggregate_fit(self, server_round, results, failures):
        # -- Set the model with global weights, Bring in the parameters for the global model --#
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving global model after round {server_round}...")
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters[0])
            if len(aggregated_weights) == len(self.model.get_weights()):
                self.model.set_weights(aggregated_weights)
        # EoF Set global weights

        for epoch in range(self.epochs):
            for step, (real_data, real_labels) in enumerate(self.x_train_ds.take(self.steps_per_epoch)):
                # generate noise for generator to use.
                real_batch_size = tf.shape(real_data)[0]  # Ensure real batch size
                noise = tf.random.normal([real_batch_size, self.noise_dim])

                # Train Discriminator
                for _ in range(5):  # Train discriminator 5 times per generator update
                    with tf.GradientTape() as disc_tape:
                        # generate samples
                        generated_samples = self.generator(noise, training=True)

                        # predict data
                        real_output = self.discriminator(real_data, training=True)  # take real samples
                        fake_output = self.discriminator(generated_samples, training=True)  # take fake smaples

                        # compute loss functions
                        gp_loss = self.gradient_penalty(self.discriminator, real_data, generated_samples)
                        disc_loss = self.discriminator_loss(real_output, fake_output, gp_loss)

                    # update the gradiants and discriminator weights from gradiants
                    gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                    self.disc_optimizer.apply_gradients(
                        zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                    # Update metrics using discriminator outputs.
                    self.update_critic_metrics(real_output, fake_output)

                    if step % 100 == 0:
                        self.log_metrics(step, disc_loss)

                    # Reset metric states after each epoch.
                self.reset_metrics()

            # Evaluate Discriminator (Critic) on Validation Set
            val_disc_loss = self.evaluate_validation_disc(self.generator, self.discriminator)
            print(f'Epoch {epoch + 1}, Validation Critic Loss: {val_disc_loss:.4f}')

            # Evaluate NIDS if Available
            if self.nids is not None:
                val_nids_loss = self.evaluate_validation_NIDS(self.generator)
                print(f'Epoch {epoch + 1}, Validation NIDS Loss: {val_nids_loss:.4f}')

        # Save the fine-tuned model
        self.discriminator.save("../../../../ModelArchive/Wdisc_model_fine_tuned.h5")
        print(f"Model fine-tuned and saved after round {server_round}.")

        # Send updated weights back to clients
        return self.discriminator.get_weights(), {}

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
            gp_loss = self.gradient_penalty(self.discriminator, real_data, generated_samples)

            # Compute WGAN-GP loss
            disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + 10.0 * gp_loss

            total_disc_loss += disc_loss.numpy()
            num_batches += 1

        return total_disc_loss / num_batches  # Average loss

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


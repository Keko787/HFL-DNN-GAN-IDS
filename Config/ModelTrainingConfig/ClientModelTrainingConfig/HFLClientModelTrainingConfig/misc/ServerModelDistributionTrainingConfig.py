import flwr as fl
import tensorflow as tf
import numpy as np

class FLClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train):
        self.model = None
        self.X_train = X_train
        self.y_train = y_train

    def get_parameters(self, config):
        """Return model parameters to be initialized at the client."""
        return [val.numpy() for val in self.model.get_weights()]

    def fit(self, parameters, config):
        """Train the model on local data and return new weights."""
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=5, batch_size=32)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on local validation data."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_train, self.y_train)
        return loss, len(self.X_train), {"accuracy": accuracy}
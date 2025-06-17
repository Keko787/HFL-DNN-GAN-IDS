import flwr as fl
import tensorflow as tf
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays



class LoadModelFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def initialize_parameters(self, client_manager):
        """Send pre-trained model weights to clients on the first round."""
        initial_weights = self.model.get_weights()
        return ndarrays_to_parameters(initial_weights)



# fl.server.start_server(
#     config=fl.server.ServerConfig(num_rounds=5),
#     strategy=LoadModelFedAvg(
#         model = model,
#         min_fit_clients=2,
#         min_evaluate_clients=2,
#         min_available_clients=2
#     )
# )

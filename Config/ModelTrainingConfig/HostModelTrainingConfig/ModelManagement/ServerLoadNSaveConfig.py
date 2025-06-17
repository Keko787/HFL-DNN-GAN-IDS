import flwr as fl
import tensorflow as tf
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


# Load an existing pre-trained model, Save that model after training to the host
class LoadSaveModelFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model=None, model_save_path="global_model.h5", **kwargs):
        super().__init__(**kwargs)
        self.model_save_path = model_save_path
        self.model = model

    def initialize_parameters(self, client_manager):
        """Send pre-trained model weights to clients on the first round."""
        initial_weights =  self.model.get_weights()
        return ndarrays_to_parameters(initial_weights)

    def aggregate_fit(self, server_round, results, failures):
        """Aggregates client results and saves the global model."""
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving global model after round {server_round}...")

            # Convert Parameters object to numpy arrays
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters[0])

            if len(aggregated_weights) == len(self.model.get_weights()):
                self.model.set_weights(aggregated_weights)
                self.model.save(self.model_save_path)
                print(f"Model saved at: {self.model_save_path}")
            else:
                print(
                    f"Warning: Weight mismatch. Expected {len(self.model.get_weights())} but got {len(aggregated_weights)}.")

        return aggregated_parameters

# fl.server.start_server(
#     config=fl.server.ServerConfig(num_rounds=5),
#     strategy=LoadSaveModelFedAvg(
#         model_save_path="global_model.h5",
#         model= model,
#         min_fit_clients=2,
#         min_evaluate_clients=2,
#         min_available_clients=2
#     )
# )

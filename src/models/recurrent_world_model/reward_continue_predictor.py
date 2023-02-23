from tensorflow.keras import layers
from tensorflow.keras import Model

from .utils import symlog


class RewardPredictor(Model):
    def __init__(self, fc_layers: int = 1):
        super().__init__()

        self.fully_connected_layers = [
            layers.Dense("relu", 512) for _ in range(fc_layers)
        ]

        self.output_layer = layers.Dense("linear", 1)

    def call_no_symlog(self, x):
        intermediate_result = x

        for layer in self.fully_connected_layers:
            intermediate_result = layer(intermediate_result)

        output = self.output_layer(intermediate_result)

        return output

    def call(self, x):
        return symlog(self.call_no_symlog(x))


class ContinuePredictor(Model):
    def __init__(self, fc_layers: int = 1):
        super().__init__()

        self.fully_connected_layers = [
            layers.Dense("relu", 512) for _ in range(fc_layers)
        ]

        self.output_layer = layers.Dense("sigmoid", 1)

    def call(self, x):
        intermediate_result = x

        for layer in self.fully_connected_layers:
            intermediate_result = layer(intermediate_result)

        output = self.output_layer(intermediate_result)

        return output

from tensorflow.keras import layers
from tensorflow.keras import Model


class DynamicsPredictor(Model):
    def __init__(self, fc_layers, recurrent_embedding_size=1024):
        super().__init__()

        self.input = layers.Dense("relu", recurrent_embedding_size)

        self.fully_connected_layers = [
            layers.Dense("relu", 512) for _ in range(fc_layers)
        ]

        self.output = [layers.Dense("sigmoid", 32) for _ in range(32)]

    def call(self, x):
        intermediate_result = self.input(x)
        for layer in self.fully_connected_layers:
            intermediate_result = layer

        return [out(intermediate_result) for out in self.output]

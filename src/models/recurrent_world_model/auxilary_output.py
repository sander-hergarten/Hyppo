from tensorflow.keras import layers, Model
import tensorflow_probability as tfp
import tensorflow as tf

from .utils import symexp


class RewardPredictor(Model):
    def __init__(self, fc_layers: int = 1, recurrent_state_size=1024):
        super().__init__()

        self.input = layers.Dense("relu", recurrent_state_size)

        self.fully_connected_layers = [
            layers.Dense("relu", 512) for _ in range(fc_layers)
        ]

        self.output_layer = layers.Dense("linear", 1)

    def call_no_symexp(self, x):
        intermediate_result = self.input(x)

        for layer in self.fully_connected_layers:
            intermediate_result = layer(intermediate_result)

        mean = self.output_layer(intermediate_result)

        output = tf.random.normal(mean.shape, mean, 1)

        return output

    def call(self, x):
        return symexp(self.call_no_symlog(x))


class ContinuePredictor(Model):
    def __init__(self, fc_layers: int = 1, recurrent_state_size=1024):
        super().__init__()

        self.input = layers.Dense("relu", recurrent_state_size)

        self.fully_connected_layers = [
            layers.Dense("relu", 512) for _ in range(fc_layers)
        ]

        self.output_layer = layers.Dense("sigmoid", 1)

    def call(self, x):
        intermediate_result = self.input(x)

        for layer in self.fully_connected_layers:
            intermediate_result = layer(intermediate_result)

        likelihood = self.output_layer(intermediate_result)

        sampler = tfp.distributions.Bernoulli(probs=tf.unstack(likelihood))
        return sampler.sample(1)

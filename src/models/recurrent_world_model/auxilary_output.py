import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, layers
from utils import config, symexp, symlog_loss


class RewardPredictor(Model):
    def __init__(self):
        super().__init__()

        recurrent_state_size = config["model_parameters"]["recurrent_state_size"]
        mlp_layer_size = config["model_sizes"]["mlp_layer_size"]
        activation = config["model_parameters"]["activation"]

        self.input_layer = layers.Dense(recurrent_state_size, activation)

        self.fully_connected_layers = [
            layers.Dense(layer_size, activation) for layer_size in mlp_layer_size
        ]

        self.output_layer = layers.Dense(1, "linear")

    def call_no_symexp(self, x):
        intermediate_result = self.input_layer(x)

        for layer in self.fully_connected_layers:
            intermediate_result = layer(intermediate_result)

        mean = self.output_layer(intermediate_result)

        sampler = tfp.distributions.Normal(loc=mean, scale=1)

        output = sampler.sample(1)

        return output

    def call(self, x):
        return symexp(self.call_no_symlog(x))

    @tf.function
    def loss(self, x, y):
        y_pred = self.call_no_symexp(x)

        return symlog_loss(y, y_pred)


class ContinuePredictor(Model):
    def __init__(self):
        super().__init__()

        mlp_layer_size = config["model_sizes"]["mlp_layer_size"]
        activation = config["model_parameters"]["activation"]

        self.fully_connected_layers = [
            layers.Dense(layer_size, activation) for layer_size in mlp_layer_size
        ]

        self.output_layer = layers.Dense(1, "sigmoid")

        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, x):
        intermediate_result = self.input(x)

        for layer in self.fully_connected_layers:
            intermediate_result = layer(intermediate_result)

        likelihood = self.output_layer(intermediate_result)

        sampler = tfp.distributions.Bernoulli(probs=tf.unstack(likelihood))
        return sampler.sample(1)

    @tf.function
    def loss(self, x, y):
        y_pred = self(x)
        return self.binary_crossentropy(y, y_pred)


class Decoder(Model):
    def __init__(self):
        super().__init__()

        cnn_kernels = config["model_sizes"]["cnn_kernel"]
        cnn_stride = config["model_parameters"]["cnn_stride"]
        cnn_depth = config["model_sizes"]["cnn_depth"]
        mlp_layer_size = config["model_sizes"]["mlp_layer_size"]
        activation = config["model_parameters"]["activation"]

        cnn_kernels.reverse()
        mlp_layer_size.reverse()

        self.mlp_layers = [
            layers.Dense(layer_size, activation) for layer_size in mlp_layer_size
        ]

        self.inter_mlp = layers.Dense(48, activation)

        self.reshape = layers.Reshape((4, 4, 3), input_shape=(48,))

        self.cnn_layers = [
            layers.Conv2DTranspose(
                2**i * cnn_depth,
                kernel,
                activation=activation,
                strides=cnn_stride,
            )
            for i, kernel in enumerate(cnn_kernels)
        ]

        self.resizing_layer = layers.Resizing(64, 64)

    def call_no_symexp(self, x):
        intermediate_result = x

        for layer in self.mlp_layers:
            intermediate_result = layer(intermediate_result)

        intermediate_result = self.inter_mlp(intermediate_result)
        encoded_observation = self.reshape(intermediate_result)

        for layer in self.cnn_layers:
            encoded_observation = layer(encoded_observation)

    def call(self, x):
        return symexp(self.call_no_symexp(x))

    @tf.function
    def loss(self, x, y):
        y_pred = self.call_no_symexp(x)
        processed_y = self.resizing_layer(y)

        return symlog_loss(processed_y, y_pred)

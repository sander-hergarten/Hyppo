import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from utils import config, one_hot, symexp, symlog


class Encoder(Model):
    def __init__(self):
        super().__init__()

        cnn_kernels = config["model_sizes"]["cnn_kernel"]
        cnn_stride = config["model_parameters"]["cnn_stride"]
        cnn_depth = config["model_sizes"]["cnn_depth"]
        mlp_layer_size = config["model_sizes"]["mlp_layer_size"]
        activation = config["model_parameters"]["activation"]

        self.mlp_layers = [
            layers.Dense(layer_size, activation) for layer_size in mlp_layer_size
        ]

        self.cnn_layers = [
            layers.Conv2D(
                2**i * cnn_depth,
                kernel,
                activation=activation,
                strides=cnn_stride,
            )
            for i, kernel in enumerate(cnn_kernels)
        ]

        self.output_layer = layers.Dense(32 * 32, "sigmoid")

        self.distribution_reshape = layers.Reshape((32, 32), input_shape=(32 * 32,))
        self.resizing_layer = layers.Resizing(64, 64)

    def observation_preprocessing(self, observation):
        processed_observation = self.resizing_layer(observation)
        processed_observation = symlog(processed_observation)

        return processed_observation

    def distribution(self, recurrent_state, observation):
        preprocessed_obeservations = self.observation_preprocessing(observation)

        processing_observations = preprocessed_obeservations

        for layer in self.cnn_layers:
            processing_observations = layer(processing_observations)
            processing_observations = layers.LayerNormalization()(
                processing_observations
            )

        processed_observation = layers.Flatten()(processing_observations)
        intermediate_result = layers.Concatenate()(
            [processed_observation, recurrent_state]
        )

        for layer in self.mlp_layers:
            intermediate_result = layer(intermediate_result)

        distribution_flattend = self.output_layer(intermediate_result)

        distribution = self.distribution_reshape(distribution_flattend)

        return distribution

    def sample(self, distribution):
        categorical_one_hot_vector = tf.map_fn(one_hot, distribution)

        return categorical_one_hot_vector

    def call(self, recurrent_state, observation):
        distribution = self.distribution(recurrent_state, observation)

        return self.sample(distribution)


class DynamicsPredictor(Model):
    def __init__(self):
        super().__init__()

        mlp_layer_size = config["model_sizes"]["mlp_layer_size"]
        activation = config["model_parameters"]["activation"]

        self.mlp_layers = [
            layers.Dense(layer_size, activation) for layer_size in mlp_layer_size
        ]

        self.output_layer = layers.Dense(32 * 32, "sigmoid")
        self.distribution_reshape = layers.Reshape((32, 32), input_shape=(32 * 32,))

    def distribution(self, recurrent_state):
        intermediate_result = recurrent_state

        for layer in self.mlp_layers:
            intermediate_result = layer(intermediate_result)

        distribution_flattend = self.output_layer(intermediate_result)

        distribution = self.distribution_reshape(distribution_flattend)

        return distribution

    def sample(self, distribution):
        categorical_one_hot_vector = tf.map_fn(one_hot, distribution)
        return categorical_one_hot_vector

    def call(self, recurrent_state):
        distribution = self.distribution(recurrent_state)

        return self.sample(distribution)


class SequenceModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.recurrent_state_size = config["model_parameters"]["recurrent_state_size"]

        self.mlp_layer_size = config["model_sizes"]["mlp_layer_size"]
        self.gru_recurrent_units = config["model_sizes"]["gru_recurrent_units"]
        self.concat_layer = layers.Concatenate()

        self.gru_layers = [
            layers.GRUCell(units, "relu") for units in self.gru_recurrent_units
        ]

        self.mlp_layers = [
            layers.Dense(layer_size, "relu") for layer_size in self.mlp_layer_size
        ]

        self.output_layer = layers.Dense(self.recurrent_state_size, "sigmoid")

    def call(self, recurrent_state, stochastic_state, action):
        concatinated_inputs = self.concat_layer([stochastic_state, action])

        intermediate_recurrent = concatinated_inputs

        state = recurrent_state

        print("state shape", state.shape)

        for ind, layer in enumerate(self.gru_layers):
            print(ind)
            intermediate_recurrent, state = layer(intermediate_recurrent, state)

        intermediate_dense = intermediate_recurrent

        for layer in self.mlp_layers:
            intermediate_dense = layer(intermediate_dense)

        output = self.output_layer(intermediate_dense)

        return output

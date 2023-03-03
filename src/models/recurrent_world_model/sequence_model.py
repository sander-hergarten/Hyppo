from tensorflow.keras import layers, Model, Input
import tensorflow as tf
from .utils import symexp, symlog, unimix_categoricals, sample_from_distribution


class Encoder(common.Module):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="elu",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=[400, 400, 400, 400],
    ):
        self.shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        ]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        ]
        print("Encoder CNN inputs:", list(self.cnn_keys))
        print("Encoder MLP inputs:", list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

    @tf.function
    def __call__(self, data):
        key, shape = list(self.shapes.items())[0]
        batch_dims = data[key].shape[: -len(shape)]
        data = {
            k: tf.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims) :])
            for k, v in data.items()
        }
        outputs = []
        if self.cnn_keys:
            outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
        if self.mlp_keys:
            outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
        output = tf.concat(outputs, -1)
        return output.reshape(batch_dims + output.shape[1:])

    def _cnn(self, data):
        x = tf.concat(list(data.values()), -1)
        x = x.astype(prec.global_policy().compute_dtype)
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2**i * self._cnn_depth
            x = self.get(f"conv{i}", tfkl.Conv2D, depth, kernel, 2)(x)
            x = self.get(f"convnorm{i}", NormLayer, self._norm)(x)
            x = self._act(x)
        return x.reshape(tuple(x.shape[:-3]) + (-1,))

    def _mlp(self, data):
        x = tf.concat(list(data.values()), -1)
        x = x.astype(prec.global_policy().compute_dtype)
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f"dense{i}", tfkl.Dense, width)(x)
            x = self.get(f"densenorm{i}", NormLayer, self._norm)(x)
            x = self._act(x)
        return x


class Encoder(Model):
    def __init__(self):
        super().__init__()

        self.input_shape = (64, 64)
        self.cnn_kernels = (3, 3, 3, 3)
        self.stride = 2
        self.cnn_depth = 24
        self.mlp_layer_size = [256]

        self.mlp_layers = [
            layers.Dense(layer_size, "relu") for layer_size in self.mlp_layer_size
        ]

        self.cnn_layers = [
            layers.Conv2D(
                2**i * self.cnn_depth, kernel, activation="relu", stride=self.stride
            )
            for i, kernel in enumerate(self.cnn_kernels)
        ]

        self.output = layers.Dense(32 * 32, "sigmoid")
        self.input = Input(self.input_shape)

    def observation_preprocessing(self, observation):
        return symlog(observation)

    def call(self, recurrent_state, observation):
        preprocessed_obeservations = self.observation_preprocessing(observation)

        processing_obsevations = preprocessed_obeservations

        for layer in self.cnn_layers:
            processing_obsevations = layer(processing_obsevations)
            processing_obsevations = layers.LayerNormalization()(processing_obsevations)

        intermediate_result = tf.concat(
            [processing_obsevations, recurrent_state], axis=1
        )

        for layer in self.mlp_layers:
            intermediate_result = layer(intermediate_result)

        distribution_flattend = self.output(intermediate_result)

        distribution = tf.reshape(distribution_flattend, (32, 32))

        balanced_distribution = unimix_categoricals(distribution)

        categorical_one_hot_vector = tf.map_fn(
            sample_from_distribution, balanced_distribution
        )

        return categorical_one_hot_vector


class DynamicsPredictor(Model):
    def __init__(self):
        super().__init__()

        self.mlp_layer_size = [256]

        self.mlp_layers = [
            layers.Dense(layer_size, "relu") for layer_size in self.mlp_layer_size
        ]

        self.output = layers.Dense(32 * 32, "sigmoid")

    def call(self, recurrent_state):
        intermediate_result = recurrent_state

        for layer in self.mlp_layers:
            intermediate_result = layer(intermediate_result)

        distribution_flattend = self.output(intermediate_result)

        distribution = tf.reshape(distribution_flattend, (32, 32))

        balanced_distribution = unimix_categoricals(distribution)

        categorical_one_hot_vector = tf.map_fn(
            sample_from_distribution, balanced_distribution
        )

        return categorical_one_hot_vector


class SequenceModel:
    def __init__(self):

        self.recurrent_state_size = 64

        self.mlp_layer_size = [256]
        self.gru_recurrent_units = [256]

        self.gru_layers = [
            layers.GRU(units, "relu") for units in self.gru_recurrent_units
        ]

        self.mlp_layers = [
            layers.Dense(layer_size, "relu") for layer_size in self.mlp_layer_size
        ]

        self.output = layers.Dense(self.recurrent_state_size, "sigmoid")

    def call(self, previous_recurrent_state, stochastic_latent, action):
        concatinated_inputs = tf.concat(
            [previous_recurrent_state, stochastic_latent, action], axis=1
        )

        intermediate_recurrent = concatinated_inputs

        for layer in self.gru_layers:
            intermediate_recurrent = layer(intermediate_recurrent)

        intermediate_dense = intermediate_recurrent

        for layer in self.mlp_layers:
            intermediate_dense = layer(intermediate_dense)

        output = self.output(intermediate_dense)

        return output

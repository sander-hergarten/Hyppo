from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

# from tensorflow import keras
from tensorflow.keras import regularizers, Model, layers


import tensorflow as tf


# Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, feature_shape, head_size, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.layer_norm_0 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-6)

        self.multihead_attention = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )

        self.dropout_0 = layers.Dropout(dropout)
        self.dropout_1 = layers.Dropout(dropout)

        self.conv_layer_0 = layers.Conv1D(
            filters=ff_dim, kernel_size=1, activation="relu"
        )
        self.conv_layer_1 = layers.Conv1D(filters=feature_shape, kernel_size=1)

    def call(self, inputs):
        x = self.layer_norm_0(inputs)
        x = self.multihead_attention(x, x)
        x = self.dropout_0(x)

        res = x + inputs

        x = self.layer_norm_1(x)
        x = self.conv_layer_0(x)
        x = self.dropout_1(x)
        x = self.conv_layer_1(x)

        return x + res


# Critic
class Critic(Model):
    def __init__(self):
        super().__init__()
        image_embedding_size = 32

        head_size = 256
        num_heads = 3
        ff_dim = 4
        num_transformer_blocks = 4

        cnn_layers = 3

        # Image Encoder
        self.conv_layers = [
            layers.Conv2D(3, 8, activation="relu") for _ in range(cnn_layers)
        ]
        self.max_pooling_layers = [layers.MaxPooling2D() for _ in range(cnn_layers)]

        self.image_embedding = layers.Dense(image_embedding_size)

        # Transformer Architecture
        self.transformer_blocks = [
            TransformerBlock(image_embedding_size, head_size, num_heads, ff_dim)
            for _ in range(num_transformer_blocks)
        ]

        # Dense Architecture
        self.dense_layers = [
            layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            )
            for _ in range(5)
        ]

        self.dropout_layers = [layers.Dropout(0.1) for _ in range(5)]

        # Output
        self.final_dense = layers.Dense(64, activation="relu")
        self.output_layer = layers.Dense(
            50, kernel_initializer="normal", activation="linear"
        )

    @tf.function
    def embedd_image(self, inputs):
        x = inputs
        for cnn_layer, pooling_layer in zip(self.conv_layers, self.max_pooling_layers):
            x = pooling_layer(cnn_layer(x))

        x = layers.Flatten()(x)

        x = self.image_embedding(x)

        return x

    def call(self, inputs):
        # embedd all observations

        feature_embedded_episodes = self.embedd_image(inputs)

        transformered_episodes = tf.expand_dims(feature_embedded_episodes, axis=0)

        del feature_embedded_episodes

        for block in self.transformer_blocks:
            transformered_episodes = block(transformered_episodes)

        dense_output = transformered_episodes

        del transformered_episodes

        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            dense_output = dropout(dense(dense_output))

        dense_output = self.final_dense(dense_output)

        output = self.output_layer(dense_output)

        return output

from dataclasses import dataclass, field

import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from rlu_unplugged_transform import batch_dataset
from utils import config
from world_model import WorldModel

#
# wandb.init(
#     project="Dreamer-V3-World-Model",
# )
#
# wandb.config = config


def main():
    dataset = tfds.load("rlu_atari", split="train[:5%]")
    print("hi")
    dataset_batched = batch_dataset(dataset)

    world_model = tf.keras.Sequential(
        [tf.keras.layers.Dense(10, "relu"), tf.keras.layers.Dense(20, "relu")]
    )

    world_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["model_parameters"]["learning_rate"]
        )
    )

    world_model.fit(dataset_batched, epochs=5)


if __name__ == "__main__":
    main()

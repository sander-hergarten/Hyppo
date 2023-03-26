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
    dataset_batched = batch_dataset(dataset)

    world_model = WorldModel()

    world_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
    world_model.fit(dataset_batched, epochs=5)


if __name__ == "__main__":
    main()

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
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    dataset = tfds.load("rlu_atari", split="train[:5%]")
    dataset_batched = batch_dataset(dataset)

    for ellement in dataset_batched:
        print(ellement)
    # world_model = WorldModel()
    #
    # world_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
    # world_model.fit(dataset_batched, epochs=5, verbose=1)


if __name__ == "__main__":
    main()

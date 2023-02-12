"""test dataset."""
import os
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import json
from collections import defaultdict


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for the coinrun observation_rewards dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
        Run download.py in the data subdirectory to download the necesary 
        json data. Every file contains 50 steps of information
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "observation": tfds.features.Sequence(
                        feature=tfds.features.Tensor(
                            shape=(64, 64, 3), dtype=tf.float32
                        ),
                        length=50,
                    ),
                    "reward": tfds.features.Sequence(feature=tf.float32, length=50),
                }
            ),
            supervised_keys=("observation", "reward"),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        archive_path = dl_manager.manual_dir

        # Dataset is generated in a single split. To retrieve multiple splits use the split api
        return {
            "train": self._generate_examples(archive_path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        beam = tfds.core.lazy_imports.apache_beam

        def base_64_to_np(element: tuple[int, dict[str, list[str]]]):
            """
            function that decodes a base 64 string to the corresponding jpeg image. Designed to fit into apache beam pipeline

            :param element: The ellement of the PCollection
            """

            decoded_observation = []
            key, dictionary = element

            for step in dictionary["observation"]:
                if step == "":
                    decoded_observation.append(
                        tf.convert_to_tensor(np.zeros((64, 64, 3), dtype=np.float32))
                    )
                    continue

                # if the observation is not empty:
                tf_image = tf.io.decode_base64(step)
                tf_tensor = tf.io.decode_jpeg(tf_image)

                decoded_observation.append(tf_tensor.numpy().astype(np.float32))

            export = dictionary
            export["observation"] = decoded_observation
            return (key, export)

        def read_json(path: str) -> dict:
            """
            opens a json file and returns contents as dict

            :param path: the Path to the file
            """

            with open(path) as json_file:
                data = json.load(json_file)

            return data

        path = "/Users/sanderhergarten/Documents/programming/Hyppo/src/datasets/collected_observation_rewards/data/data"

        return (
            "AddPath"
            >> beam.Create(
                list(
                    map(
                        lambda x: path + x,
                        os.listdir(path),
                    )
                )
            )
            | "Read" >> beam.Map(read_json)
            | "DecodeBase64" >> beam.Map(base_64_to_np)
            | "Normalize"
            >> beam.Map(
                lambda x: (
                    x[0],
                    {
                        "observation": np.array(x[1]["observation"]) / 255,
                        "reward": x[1]["reward"],
                    },
                )
            )
        )

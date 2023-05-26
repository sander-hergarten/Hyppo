from collections import namedtuple

import tensorflow as tf


def batch_dataset(dataset: tf.data.Dataset):
    batched_dataset = dataset.map(batcher)
    batched_dataset = batched_dataset.map(test)

    return batched_dataset


@tf.function
def test(element):
    step_list = []
    for dictionary in element:
        step_list.extend([dict(zip(dictionary, t)) for t in zip(*dictionary.values())])

    return step_list


@tf.function
def batcher(element):
    return element["steps"].batch(50)

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
        zip_with_some_help = lambda tensor: tf.reduce_sum(tensor)

        c = tf.stack(list(dictionary.values()), axis=1)
        ziped_dict = tf.map_fn(zip_with_some_help, c, dtype=tf.float32)

        step_list.extend([dict(zip(dictionary, t)) for t in ziped_dict])

    return step_list


@tf.function
def batcher(element):
    return element["steps"].batch(50)

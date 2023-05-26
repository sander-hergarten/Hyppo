from collections import namedtuple

import tensorflow as tf


def batch_dataset(dataset: tf.data.Dataset):
    batch_list = [element["steps"].batch(50) for element in dataset]

    batched_dataset = batch_list.pop(0)

    data = []
    for element in batch_list:
        step_list = []
        for dictionary in element:
            step_list.extend(
                [dict(zip(dictionary, t)) for t in zip(*dictionary.values())]
            )
        data.append({"sequence": step_list})

    dataset = tf.data.Dataset.from_tensor_slices(data)

    return dataset

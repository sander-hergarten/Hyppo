import tensorflow as tf


def batch_dataset(dataset: tf.data.Dataset):
    batch_list = [element["steps"].batch(50) for element in dataset]
    batched_dataset = batch_list

    return batched_dataset

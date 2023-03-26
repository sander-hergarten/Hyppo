import tensorflow as tf


def batch_dataset(dataset: tf.data.Dataset):
    batch_list = [element["steps"].batch(50) for element in dataset]
    batched_dataset = batch_list.pop(0)

    for batch in batch_list:
        batched_dataset = batched_dataset.concatenate(batch)

    return batched_dataset

import tensorflow as tf


def batch_dataset(dataset: tf.data.Dataset):
    batch_list = [element["steps"].batch(50) for element in dataset]

    for batched_dataset in batch_list:
        print(batched_dataset.cardinality().numpy())
        print(list(batched_dataset.take(1)))

    return batch_list

import tensorflow as tf


def batch_dataset(dataset: tf.data.Dataset):
    batch_list = [element["steps"].batch(50) for element in dataset]

    for batched_dataset in batch_list:
        print(batched_dataset.cardinality().numpy())
        for element in batched_dataset:
            print(element.cardinality().numpy())

    return batch_list

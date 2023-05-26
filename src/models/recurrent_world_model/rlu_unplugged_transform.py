import tensorflow as tf


def batch_dataset(dataset: tf.data.Dataset):
    batch_list = [element["steps"].batch(50) for element in dataset]

    batched_dataset = batch_list.pop(0)

    print(batched_dataset.cardinality().numpy())

    for element in batch_list:
        print("dataset size before concat", batched_dataset.cardinality().numpy())
        print("ellements to concat", element.cardinality().numpy())

        batched_dataset = batched_dataset.concatenate(element)
        print("dataset size after concat", batched_dataset.cardinality().numpy())

    return batch_list

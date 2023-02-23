import tensorflow as tf


symlog = lambda x: tf.sign(x) * tf.math.log(tf.abs(x) + 1)
symexp = lambda x: tf.sign(x) * (tf.exp(tf.abs(x)) - 1)

import tensorflow as tf
import tensorflow_probability as tfp


symlog = lambda x: tf.sign(x) * tf.math.log(tf.abs(x) + 1)
symexp = lambda x: tf.sign(x) * (tf.exp(tf.abs(x)) - 1)


@tf.function
def unimix_categoricals(
    distribution_tensor, unimix_ratio: tuple[float, float] = (0.99, 0.01)
):
    """
    function to apply unimix categicals for non exploding KL divergence
    """

    random_offset = tf.random.uniform(shape=tf.shape(distribution_tensor))

    random_offset *= unimix_ratio[1]
    adjusted_distribution_vector = unimix_ratio[0]

    return random_offset + adjusted_distribution_vector


@tf.function
def sample_from_distribution(distribution_tensor, one_hot_encode: bool = True):
    sampler = tfp.distributions.Sample(distribution_tensor)

    if one_hot_encode:
        shape = tf.shape(distribution_tensor)
        sample = sampler.sample()

        return tf.one_hot(sample, shape)

    return sampler.sample()

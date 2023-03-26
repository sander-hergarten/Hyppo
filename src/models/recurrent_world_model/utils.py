import tensorflow as tf
import tensorflow_probability as tfp
import yaml

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
def sample_from_distribution_and_one_hot(distribution_tensor):
    sampler = tfp.distributions.RelacedOneHotCategorical(
        logits=distribution_tensor, temperature=0.01
    )

    return sampler.sample()


@tf.function
def symlog_loss(y_true, y_pred):
    return 0.5 * (y_pred - symlog(y_true)) ** 2


with open(
    "/Users/sanderhergarten/Documents/programming/Hyppo/src/models/recurrent_world_model/config.yaml",
    "r",
) as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

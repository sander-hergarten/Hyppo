import tensorflow as tf
import tensorflow_datasets as tfds
from auxilary_output import ContinuePredictor, Decoder, RewardPredictor
from rlu_unplugged_transform import batch_dataset
from sequence_model import (DynamicsPredictor, Encoder, SequenceModel_Alpha,
                            SequenceModel_Beta)

sequence_model_alpha = SequenceModel_Alpha()
sequence_model_beta = SequenceModel_Beta()

encoder = Encoder()
dynamics_predictor = DynamicsPredictor()

decoder = Decoder()
reward_predictor = RewardPredictor()
continue_predictor = ContinuePredictor()


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
kl_divergence = tf.keras.losses.KLDivergence()


REWARD_THRESHOLD = 0.7

recurrent_state = tf.zeros((1, 256))
stochastic_state = tf.zeros((1, 1024))

action = tf.stack([tf.one_hot(0, 16)])
observation = tf.zeros((1, 64, 64, 3))
reward = tf.zeros((1, 1))
continue_flag = tf.zeros((1, 1), dtype=tf.int32)


with tf.GradientTape(persistent=True) as tape:
    recurrent_state_alpha, gru_output_alpha = sequence_model_alpha(
        recurrent_state=recurrent_state,
        stochastic_state=stochastic_state,
        action=action,
    )

    recurrent_state_alpha = recurrent_state_alpha[0]

    stochastic_dist_alpha = encoder.distribution(recurrent_state_alpha, observation)
    dynamics_dist_alpha = dynamics_predictor.distribution(recurrent_state_alpha)

    loss_dynamics_alpha = kl_divergence(
        tf.stop_gradient(stochastic_dist_alpha), dynamics_dist_alpha
    )
    loss_representation_alpha = kl_divergence(
        tf.stop_gradient(dynamics_dist_alpha), stochastic_dist_alpha
    )

    loss_beta = 0

    if reward > REWARD_THRESHOLD:
        recurrent_state_beta, gru_output_beta = sequence_model_beta(
            recurrent_state=recurrent_state,
            stochastic_state=stochastic_state,
        )

        stochastic_dist_beta = encoder.distribution(recurrent_state_beta, observation)
        dynamics_dist_beta = dynamics_predictor.distribution(recurrent_state_beta)

        model_state_beta = tf.concat(
            [recurrent_state_beta, stochastic_state_beta], axis=1
        )

        loss_decoder_beta = -tf.math.log(decoder.loss_fn(model_state_beta, observation))

        loss_continue_beta = -tf.math.log(
            continue_predictor.loss_fn(model_state_beta, continue_flag)
        )

        loss_prediction_beta = loss_decoder_beta + loss_continue_beta

        loss_dynamics_beta = kl_divergence(dynamics_dist_beta, dynamics_dist_alpha)
        loss_stochastic_beta = kl_divergence(
            stochastic_dist_beta, stochastic_dist_alpha
        )

        loss_sequence_model_beta = kl_divergence(
            tf.stop_gradient(gru_output_alpha), gru_output_beta
        )

        loss_beta = (
            loss_prediction_beta
            + loss_dynamics_beta
            + loss_stochastic_beta
            + loss_sequence_model_beta
        )

    stochastic_state_alpha = encoder.sample(stochastic_dist_alpha)

    model_state_alpha = tf.concat(
        [recurrent_state_alpha, stochastic_state_alpha], axis=1
    )

    loss_decoder_alpha = -tf.math.log(decoder.loss_fn(model_state_alpha, observation))
    loss_reward_alpha = -tf.math.log(
        reward_predictor.loss_fn(model_state_alpha, reward)
    )
    loss_continue_alpha = -tf.math.log(
        continue_predictor.loss_fn(model_state_alpha, continue_flag)
    )

    loss_prediction_alpha = loss_decoder_alpha + loss_reward_alpha + loss_continue_alpha

    loss_dynamics_alpha = tf.math.maximum(1.0, loss_dynamics_alpha)
    loss_representation = tf.math.maximum(1.0, loss_representation_alpha)

    loss_alpha = (
        0.5 * loss_dynamics_alpha
        + 0.1 * loss_representation_alpha
        + 1 * loss_prediction_alpha
    )


grad_var_pairs = []


for model in [
    dynamics_predictor,
    encoder,
    sequence_model_alpha,
    decoder,
    reward_predictor,
    continue_predictor,
]:
    variables = model.trainable_variables
    grads = tape.gradient(loss_alpha, variables)

    grad_var_pairs.extend(list(zip(grads, variables)))

if not isinstance(loss_beta, int):
    for model in [
        sequence_model_beta,
        encoder,
        decoder,
        continue_predictor,
        dynamics_predictor,
    ]:
        variables = model.trainable_variables
        grads = tape.gradient(loss_beta, variables)

        grad_var_pairs.extend(list(zip(grads, variables)))

optimizer.apply_gradients(grad_var_pairs)

import tensorflow as tf
from action_predictor import ActionPredictor
from auxilary_output import ContinuePredictor, Decoder, RewardPredictor
from replay_buffer import ReplayBuffer
from sequence_model import (
    DynamicsPredictor,
    Encoder,
    SequenceModel_Alpha,
    SequenceModel_Beta,
)

from typing import TypedDict

sequence_model_alpha = SequenceModel_Alpha()
sequence_model_beta = SequenceModel_Beta()

encoder = Encoder()
dynamics_predictor = DynamicsPredictor()

decoder = Decoder()
reward_predictor = RewardPredictor()
continue_predictor = ContinuePredictor()

action_predictor = ActionPredictor()

replay_buffer = ReplayBuffer()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
kl_divergence = tf.keras.losses.KLDivergence()
huber_loss = tf.keras.losses.Huber()


REWARD_THRESHOLD = 0.7

recurrent_state = tf.zeros((1, 256))
stochastic_state = tf.zeros((1, 1024))

action = tf.stack([tf.one_hot(0, 16)])
observation = tf.zeros((1, 64, 64, 3))
reward = tf.zeros((1, 1))
continue_flag = tf.zeros((1, 1), dtype=tf.int32)

step_count = 0


class StartSample(TypedDict):
    action: tf.Tensor
    reward: tf.Tensor
    observation: tf.Tensor
    continue_flag: tf.Tensor


EpisodeStartSample = list[StartSample]


@tf.function
def generate_action(
    recurrent_state, stochastic_state, observation: Optional[tf.Tensor] = None
):
    recurrent_state_t1, _ = sequence_model_beta.model(
        recurrent_state=recurrent_state,
        stochastic_state=stochastic_state,
    )

    k = tf.concat([recurrent_state_t1, stochastic_state])

    stochastic_state_t1 = (
        encoder(k, observation) if observation is not None else dynamics_predictor(k)
    )

    model_state_t1 = tf.concat([recurrent_state_t1, stochastic_state_t1], axis=1)

    action = action_predictor.model(model_state_t1)
    return action, recurrent_state_t1, stochastic_state_t1


# ------------------------------
# OFFLINE TRAINING ROUTINES
# ------------------------------
def training_routine_base(sample) -> list:
    recurrent_state = tf.zeros((1, 256))
    stochastic_state = tf.zeros((1, 1024))

    grad_var_pairs = []

    for step in sample:
        action, reward, observation, continue_flag = step

        with tf.GradientTape(persistent=True) as tape:
            recurrent_state_alpha, gru_output_alpha = sequence_model_alpha(
                recurrent_state=recurrent_state,
                stochastic_state=stochastic_state,
                action=action,
            )

            recurrent_state_alpha = recurrent_state_alpha[0]

            stochastic_dist_alpha = encoder.distribution(
                recurrent_state_alpha, observation
            )
            dynamics_dist_alpha = dynamics_predictor.distribution(recurrent_state_alpha)

            loss_dynamics_alpha = kl_divergence(
                tf.stop_gradient(stochastic_dist_alpha), dynamics_dist_alpha
            )
            loss_representation_alpha = kl_divergence(
                tf.stop_gradient(dynamics_dist_alpha), stochastic_dist_alpha
            )

            stochastic_state_alpha = encoder.sample(stochastic_dist_alpha)

            model_state_alpha = tf.concat(
                [recurrent_state_alpha, stochastic_state_alpha], axis=1
            )

            loss_decoder_alpha = -tf.math.log(
                decoder.loss_fn(model_state_alpha, observation)
            )
            loss_reward_alpha = -tf.math.log(
                reward_predictor.loss_fn(model_state_alpha, reward)
            )
            loss_continue_alpha = -tf.math.log(
                continue_predictor.loss_fn(model_state_alpha, continue_flag)
            )

            loss_prediction_alpha = (
                loss_decoder_alpha + loss_reward_alpha + loss_continue_alpha
            )

            loss_dynamics_alpha = tf.math.maximum(1.0, loss_dynamics_alpha)

            loss_alpha = (
                0.5 * loss_dynamics_alpha
                + 0.1 * loss_representation_alpha
                + 1 * loss_prediction_alpha
            )

            recurrent_state = recurrent_state_alpha
            stochastic_state = stochastic_state_alpha

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

    return grad_var_pairs


def training_routine_high_episode_return(sample) -> list:
    recurrent_state = tf.zeros((1, 256))
    stochastic_state = tf.zeros((1, 1024))

    grad_var_pairs = []

    for step in sample:
        action, _, observation, continue_flag = step

        recurrent_state_alpha, gru_output_alpha = sequence_model_alpha(
            recurrent_state=recurrent_state,
            stochastic_state=stochastic_state,
            action=action,
        )

        with tf.GradientTape(persistent=True) as tape:
            stochastic_dist_alpha = encoder.distribution(
                recurrent_state_alpha, observation
            )
            dynamics_dist_alpha = dynamics_predictor.distribution(recurrent_state_alpha)

            recurrent_state_beta, gru_output_beta = sequence_model_beta.model(
                recurrent_state=recurrent_state,
                stochastic_state=stochastic_state,
            )

            stochastic_dist_beta = encoder.distribution(
                recurrent_state_beta, observation
            )
            dynamics_dist_beta = dynamics_predictor.distribution(recurrent_state_beta)

            stochastic_state_beta = encoder.sample(stochastic_dist_beta)
            model_state_beta = tf.concat(
                [recurrent_state_beta, stochastic_state_beta], axis=1
            )

            loss_decoder_beta = -tf.math.log(
                decoder.loss_fn(model_state_beta, observation)
            )

            loss_continue_beta = -tf.math.log(
                continue_predictor.loss_fn(model_state_beta, continue_flag)
            )

            loss_prediction_beta = loss_decoder_beta + loss_continue_beta

            loss_dynamics_beta = kl_divergence(dynamics_dist_beta, dynamics_dist_alpha)
            loss_representation_beta = kl_divergence(
                stochastic_dist_beta, stochastic_dist_alpha
            )

            loss_sequence_model_beta = kl_divergence(
                tf.stop_gradient(gru_output_alpha), gru_output_beta
            )

            loss_beta = (
                loss_prediction_beta
                + loss_dynamics_beta
                + loss_representation_beta
                + loss_sequence_model_beta
            )

        for model in [
            encoder,
            decoder,
            continue_predictor,
            dynamics_predictor,
            sequence_model_beta.model,
        ]:
            variables = model.trainable_variables
            grads = tape.gradient(loss_beta, variables)

            grad_var_pairs.extend(list(zip(grads, variables)))

        variables = sequence_model_beta.model.trainable_variables
        grads = tape.gradient(loss_beta, variables)

        grad_var_pairs.extend(list(zip(grads, variables)))

    return grad_var_pairs


def training_routine_low_episode_return(sample) -> list:
    recurrent_state = tf.zeros((1, 256))
    stochastic_state = tf.zeros((1, 1024))

    grad_var_pairs = []

    for step in sample:
        action, _, observation, continue_flag = step

        recurrent_state_alpha, _ = sequence_model_alpha(
            recurrent_state=recurrent_state,
            stochastic_state=stochastic_state,
            action=action,
        )

        stochastic_state_alpha = encoder(recurrent_state_alpha, observation)

        with tf.GradientTape(persistent=True) as tape:
            pred_action = action_predictor.model(
                recurrent_state_alpha, stochastic_state_alpha
            )

            loss = huber_loss(action, pred_action)

        variables = action_predictor.model.trainable_variables
        grads = tape.gradient(loss, variables)

        grad_var_pairs.extend(list(zip(grads, variables)))

    return grad_var_pairs


# ------------------------------
# AUTOREGRESSIVE TRAINING ROUTINES
# ------------------------------
# TODO: make this paralizable
# TODO: add DQN loss
def training_routine_autoregressive(episode_start: list[EpisodeStartSample]) -> list:
    replay_buffer = ReplayBuffer(1000, batch_size=1, seed=42, gamma=0.99)

    for episode, samples in enumerate(episode_start):
        step_count = 0
        recurrent_state = tf.zeros((1, 256))
        stochastic_state = tf.zeros((1, 1024))

        continue_flag = False
        observation = tf.zeros((1, 84, 84, 3))

        for sample in samples:
            action, reward, observation, continue_flag = sample.values()

            recurrent_state, gru_output = sequence_model_alpha(
                recurrent_state, stochastic_state, action
            )
            stochastic_state = encoder(recurrent_state, observation)

        recurrent_state_beta = recurrent_state
        stochastic_state_beta = stochastic_state

        while not continue_flag:
            step_count += 1

            action, recurrent_state_beta_t1, stochastic_state_beta_t1 = generate_action(
                recurrent_state_beta, stochastic_state_beta, observation
            )

            recurrent_state_alpha_t1 = sequence_model_alpha(
                recurrent_state=recurrent_state,
                stochastic_state=stochastic_state,
                action=action,
            )

            stochastic_state_alpha_t1 = dynamics_predictor(
                recurrent_state_alpha_t1, stochastic_state
            )

            observation_next = decoder(
                recurrent_state_alpha_t1, stochastic_state_alpha_t1
            )
            reward = reward_predictor(
                recurrent_state_alpha_t1, stochastic_state_alpha_t1
            )
            continue_flag = continue_predictor(
                recurrent_state_alpha_t1, stochastic_state_alpha_t1
            )

            step_tuple = (observation, action, reward, observation_next, continue_flag)
            replay_buffer.add(step_tuple)

            observation = observation_next

            recurrent_state = recurrent_state_alpha_t1
            stochastic_state = stochastic_state_alpha_t1

            recurrent_state_beta = recurrent_state_beta_t1
            stochastic_state_beta = stochastic_state_beta_t1


def training_routine_autoregressive_blind(
    episode_start: list[EpisodeStartSample],
) -> list:
    replay_buffer = ReplayBuffer(1000, batch_size=1, seed=42, gamma=0.99)

    for episode, samples in enumerate(episode_start):
        step_count = 0
        recurrent_state = tf.zeros((1, 256))
        stochastic_state = tf.zeros((1, 1024))

        continue_flag = False
        observation = tf.zeros((1, 84, 84, 3))

        for sample in samples:
            action, reward, observation, continue_flag = sample.values()

            recurrent_state, gru_output = sequence_model_alpha(
                recurrent_state, stochastic_state, action
            )
            stochastic_state = encoder(recurrent_state, observation)

        recurrent_state_beta = recurrent_state
        stochastic_state_beta = stochastic_state

        while not continue_flag:
            step_count += 1

            (
                q_values,
                recurrent_state_beta_t1,
                stochastic_state_beta_t1,
            ) = generate_action(recurrent_state_beta, stochastic_state_beta)

            # TODO: action sampling

            recurrent_state_alpha_t1 = sequence_model_alpha(
                recurrent_state=recurrent_state,
                stochastic_state=stochastic_state,
                action=action,
            )

            stochastic_state_alpha_t1 = dynamics_predictor(
                recurrent_state_alpha_t1, stochastic_state
            )

            observation_next = decoder(
                recurrent_state_alpha_t1, stochastic_state_alpha_t1
            )
            reward = reward_predictor(
                recurrent_state_alpha_t1, stochastic_state_alpha_t1
            )
            continue_flag = continue_predictor(
                recurrent_state_alpha_t1, stochastic_state_alpha_t1
            )

            step_tuple = (observation, action, reward, observation_next, continue_flag)
            replay_buffer.add(step_tuple)

            observation = observation_next

            recurrent_state = recurrent_state_alpha_t1
            stochastic_state = stochastic_state_alpha_t1

            recurrent_state_beta = recurrent_state_beta_t1
            stochastic_state_beta = stochastic_state_beta_t1


# ------------------------------
# ONLINE TRAINING ROUTINES
# ------------------------------
def training_routine_base(env, max_episodes) -> list:
    step_count = 0
    episode_count = 0

    while max_episodes > episode_count:
        grad_var_pairs = []
        grad_var_pairs_high_reward = []

        cumulative_reward = 0

        observation = env.reset()
        recurrent_state = tf.zeros((1, 256))
        stochastic_state = tf.zeros((1, 1024))

        continue_flag = False
        while not continue_flag:
            step_count += 1

            action, recurrent_state_beta_t1, stochastic_state_beta_t1 = generate_action(
                recurrent_state, stochastic_state, observation
            )

            recurrent_state_alpha_t1, gru_output_alpha_t1 = sequence_model_alpha(
                recurrent_state=recurrent_state,
                stochastic_state=stochastic_state,
                action=action,
            )

            observation_next, reward, continue_flag, _ = env.step(action)
            cumulative_reward += reward

            step_tuple = (observation, action, reward, observation_next, continue_flag)
            replay_buffer.add(step_tuple)

            observation = observation_next

            recurrent_state = recurrent_state_t1
            stochastic_state = stochastic_state_t1

        episode_count += 1

        if cumulative_reward > REWARD_THRESHOLD:
            grad_var_pairs.extend(grad_var_pairs_high_reward)

        return grad_var_pairs


optimizer.apply_gradients(grad_var_pairs)

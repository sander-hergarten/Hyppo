from typing import Any, Optional

import tensorflow as tf
from auxilary_output import ContinuePredictor, Decoder, RewardPredictor
from sequence_model import DynamicsPredictor, Encoder, SequenceModel
from tensorflow.keras import Model
from utils import config, one_hot, unimix_categoricals

DISCOUNTS = config["model_constants"]["loss_discount"]


class WorldModel(Model):
    encoder = Encoder()
    dynamics_predictor = DynamicsPredictor()
    sequence_model = SequenceModel()

    continue_predictor = ContinuePredictor()
    reward_predictor = RewardPredictor()
    decoder = Decoder()

    recurrent_state = {
        "timestep": 0,
        "value": tf.zeros((config["model_parameters"]["recurrent_state_size"])),
    }
    stochastic_state = {"timestep": 0, "value": tf.zeros((32 * 32))}

    training = False

    kl_divergence = tf.keras.losses.KLDivergence()

    def train_step(self, data):
        self.training = True

        # sequence needs to have dimensions (step, batch_size, step_data)
        tf.print("data", data)

        dataset = tf.data.Dataset.zip(
            tuple(
                [
                    tf.data.Dataset.from_tensor_slices(data[feature])
                    for feature in ["observation", "reward", "is_last", "action"]
                ]
            )
        )

        dataset_iterator = iter(dataset)

        for image, reward, continue_flag, action in dataset_iterator:
            action = self.process_action(action)

            with tf.GradientTape() as tape:
                self.advance_recurrent_state(action)

                (
                    dynamics_loss,
                    representation_loss,
                ) = self.stochastic_timestep_to_recurrent_timestep(image)

                print(dynamics_loss, representation_loss)
                # continue_loss = -tf.math.log(
                #     self.continue_predictor.loss(self.model_state, continue_flag)
                # )
                # reward_loss = -tf.math.log(
                #     self.reward_predictor.loss(self.model_state, reward)
                # )
                # decoder_loss = -tf.math.log(self.decoder.loss(self.model_state, image))

                # prediction_loss = reward_loss + continue_loss + decoder_loss

                # loss = (
                #     DISCOUNTS["PRED"] * prediction_loss
                #     + DISCOUNTS["DYN"] * dynamics_loss
                #     + DISCOUNTS["REP"] * representation_loss
                # )

                loss = dynamics_loss

            for ellement in [
                self.encoder,
                self.dynamics_predictor,
                self.sequence_model,
                self.continue_predictor,
                self.reward_predictor,
            ]:
                variables = ellement.trainable_variables
                gradient = tape.gradient(loss, variables)

                self.optimizer.apply_gradients(zip(gradient, variables))

    @property
    def model_state(self):
        return tf.concat([self.recurrent_state, self.stochastic_state], axis=1)

    def process_action(self, p_action):
        action = tf.one_hot(p_action, 16)

        return action

    def stochastic_timestep_to_recurrent_timestep(
        self, p_observation: Optional[Any] = None
    ):
        observation = tf.stack([p_observation])
        if self.training:
            gradient_distribution_fn = lambda model, *args: unimix_categoricals(
                model.distribution(*args)
            )

            stop_gradient_distribution_fn = lambda model, *args: tf.stop_gradient(
                gradient_distribution_fn(model, *args)
            )

            encoder_distribution = self.encoder.distribution(
                self.recurrent_state["value"], observation
            )

            dynamics_distribution = gradient_distribution_fn(
                self.dynamics_predictor, self.recurrent_state["value"]
            )
            encoder_distribution_stop_grad = stop_gradient_distribution_fn(
                self.encoder, self.recurrent_state["value"], observation
            )

            dynamics_distribution_stop_grad = stop_gradient_distribution_fn(
                self.dynamics_predictor, self.recurrent_state["value"]
            )

            self.stochastic_state["value"] = self.encoder.sample(encoder_distribution)
            self.stochastic_state["timestep"] += 1

            encoder_distribution = unimix_categoricals(encoder_distribution)

            dynamics_loss = self.kl_divergence(
                encoder_distribution_stop_grad, dynamics_distribution
            )
            representation_loss = self.kl_divergence(
                encoder_distribution, dynamics_distribution_stop_grad
            )

            clipped_losses = (
                tf.math.maximum(1, dynamics_loss),
                tf.math.maximum(1, representation_loss),
            )

            return clipped_losses

        self.stochastic_state["value"] = self.dynamics_predictor(self.recurrent_state)
        self.stochastic_state["timestep"] += 1

    def advance_recurrent_state(self, action):
        assert (
            self.stochastic_state["timestep"] == self.recurrent_state["timestep"]
        ), "stochastic and recurrent state are not on the same timestep"

        self.recurrent_state["value"] = self.sequence_model(
            self.recurrent_state["value"], self.stochastic_state["value"], action
        )

        self.recurrent_state["timestep"] += 1

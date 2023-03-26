from typing import Any, Optional

import tensorflow as tf
from auxilary_output import ContinuePredictor, Decoder, RewardPredictor
from sequence_model import DynamicsPredictor, Encoder, SequenceModel
from tensorflow.keras import Model
from utils import config, unimix_categoricals

DISCOUNTS = config["model_constants"]["loss_discount"]


class WorldModel(Model):
    encoder = Encoder()
    dynamics_predictor = DynamicsPredictor()
    sequence_model = SequenceModel()

    continue_predictor = ContinuePredictor()
    reward_predictor = RewardPredictor()
    decoder = Decoder()

    recurrent_state = {"timestep": 0, "value": tf.zeros((64))}
    stochastic_state = {"timestep": 0, "value": tf.zeros((32 * 32))}

    training = False

    kl_divergence = tf.keras.losses.KLDivergence()

    def train_step(self, data):
        print("1")
        self.training = True

        # sequence needs to have dimensions (step, batch_size, step_data)

        for step in data:
            print("2")
            image = step["observation"]
            reward = step["reward"]
            continue_flag = step["is_last"]
            action = step["action"]

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

    def stochastic_timestep_to_recurrent_timestep(
        self, observation: Optional[Any] = None
    ):
        if self.training:
            assert (
                observation
            ), "during training an observation has to be given for stochastic state generation"

            gradient_distribution_fn = lambda model, *args: unimix_categoricals(
                model.distribution(*args)
            )

            stop_gradient_distribution_fn = lambda model, *args: tf.stop_gradient(
                gradient_distribution_fn(model, *args)
            )

            encoder_distribution = self.encoder.distribution(
                self.recurrent_state, observation
            )

            dynamics_distribution = gradient_distribution_fn(
                self.dynamics_predictor, self.recurrent_state
            )
            encoder_distribution_stop_grad = stop_gradient_distribution_fn(
                self.encoder, self.recurrent_state
            )

            dynamics_distribution_stop_grad = stop_gradient_distribution_fn(
                self.dynamics_predictor, self.recurrent_state
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

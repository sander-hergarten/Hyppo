import tensorflow as tf
from tensorflow.keras import Input, Sequential, layers
from utils import config


class ActionPredictor:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()

        # parameters for epsilon-greedy exploration
        self.epsilon = config["model_parameters"]["epsilon"]
        self.epsilon_decay = config["model_parameters"]["epsilon_decay"]
        self.epsilon_min = config["model_parameters"]["epsilon_min"]

    def create_model(self):
        model = Sequential()
        model.add(
            Input(shape=(config["model_parameters"]["recurrent_state_size"] + 1024,))
        )

        for units in config["model_sizes"]["action_predictor_mlp_layer_size"]:
            model.add(
                layers.Dense(units, activation=config["model_parameters"]["activation"])
            )

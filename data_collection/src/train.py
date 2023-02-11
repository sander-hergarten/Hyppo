import wandb
import tracemalloc
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Lambda, Resizing

from tqdm import tqdm


from run_saver import StepSaver

from procgen_env import ProcgenEnvironment
import argparse


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


tracemalloc.start()


from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import tf_py_environment, BatchedPyEnvironment
from tf_agents.networks import categorical_q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

from metrics import NormalizedReturn, VideoSaver

parser = argparse.ArgumentParser()
parser.add_argument("--sweep_id", help="increase output verbosity")
parser.add_argument("--environment", help="increase output verbosity")
args = parser.parse_args()


sweep_threads = 6
parallel_envs = 20


class ObserverStack:
    observers = []

    def __init__(self, batch_size: int):
        """
        This class is and abstraction layer for the observers

        """
        self.reset_observers()
        self.batch_size = batch_size

    def reset_observers(self):
        self.observers = []


def main(env="coinrun"):
    def evaluate_agent(policy, observers, batch_size=10):
        eval_env_py = BatchedPyEnvironment(
            [ProcgenEnvironment(env) for _ in range(batch_size)]
        )

        eval_env = tf_py_environment.TFPyEnvironment(eval_env_py)

        dynamic_episode_driver.DynamicEpisodeDriver(
            eval_env,
            policy,
            observers=observers,
            num_episodes=20,
        ).run()

    wandb.init()

    class StepCounter:
        step_count = 0

        def step_increment(self, batch):
            self.step_count += 20

    step_counter = StepCounter()

    # with tf.profiler.experimental.Profile("logs/"):
    train_py_env = BatchedPyEnvironment(
        [ProcgenEnvironment(env) for _ in range(parallel_envs)]
    )

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)

    # Input encoding
    preprocessing_layers = tf.keras.models.Sequential(
        [Lambda(lambda x: tf.divide(x, 255))]
    )

    conv_layer_params = [
        (32, (3, 3), 1),
        (64, (3, 3), 1),
        (64, (3, 3), 1),
    ]
    fc_layer_params = [512 for _ in range(8)]

    actor_net = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        preprocessing_layers=preprocessing_layers,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    train_step_counter = tf.Variable(0)

    tf_agent = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=actor_net,
        optimizer=optimizer,
        min_q_value=-10,
        max_q_value=10,
        n_step_update=2,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=0.99,
        train_step_counter=train_step_counter,
    )

    tf_agent.initialize()

    returns = []
    last_avg_reward = 0
    buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=1000,
        dataset_drop_remainder=True,
    )

    state_saver = StepSaver(parallel_envs)

    for n in range(100):
        # with tf.profiler.experimental.Trace("logs/", step_num=n, _r=1):

        # Observers
        max_rewards_metric = tf_metrics.MaxReturnMetric(batch_size=parallel_envs)
        normalized_return_metric = NormalizedReturn(env, batch_size=parallel_envs)
        average_episode_length_metric = tf_metrics.AverageEpisodeLengthMetric(
            batch_size=parallel_envs
        )
        average_rewards_metric = tf_metrics.AverageReturnMetric(
            batch_size=parallel_envs
        )

        replay_observer = [
            buffer.add_batch,
            max_rewards_metric,
            average_rewards_metric,
            normalized_return_metric,
            average_episode_length_metric,
            state_saver.add_data_to_queue,
            step_counter.step_increment,
        ]
        dynamic_step_driver.DynamicStepDriver(
            train_env,
            tf_agent.collect_policy,
            observers=replay_observer,
            num_steps=20200,
        ).run()

        state_saver.commit_episodes()

        dataset = buffer.as_dataset(
            sample_batch_size=10,
            num_steps=10,
            num_parallel_calls=16,
            single_deterministic_pass=True,
        )

        iterator = iter(dataset)

        num_train_steps = 10

        for _ in tqdm(range(num_train_steps)):
            trajectories, _ = next(iterator)
            tf_agent.train(experience=trajectories)

        returns.append(max_rewards_metric.result().numpy())
        last_avg_reward = average_rewards_metric.result().numpy()

        max_reward_metric_eval = tf_metrics.MaxReturnMetric(batch_size=10)

        normalized_return_metric_eval = NormalizedReturn(env, batch_size=10)
        video_saver_eval = VideoSaver()

        evaluate_agent(
            tf_agent.policy,
            [max_reward_metric_eval, normalized_return_metric_eval, video_saver_eval],
        )
        video = video_saver_eval.result()

        wandb.log(
            {
                "average_reward_training": last_avg_reward,
                "max_reward_training": returns[-1],
                "average_episode_length": average_episode_length_metric.result().numpy(),
                "normalized_return_training": normalized_return_metric.result(),
                "max_reward_eval": max_reward_metric_eval.result().numpy(),
                "normalized_return_eval": normalized_return_metric_eval.result(),
                "total_steps": step_counter.step_count,
                "video": wandb.Video(np.array(video), fps=15),
            }
        )

    normalized_return_metric_val = NormalizedReturn(env, batch_size=1)

    evaluate_agent(tf_agent.policy, [normalized_return_metric_val], batch_size=1)

    wandb.log({"final_normalized_return": normalized_return_metric_val.result()})


if __name__ == "__main__":
    main()

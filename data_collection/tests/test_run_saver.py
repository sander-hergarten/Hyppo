from tqdm import tqdm
import tensorflow as tf
import numpy as np
from tf_agents.trajectories import Trajectory
from tf_agents.trajectories import time_step as ts
from ..src.run_saver import StepSaver


class RandomTrajectoryFactory:
    def __init__(
        self,
        batch_size: int = 20,
        max_episode_len: int = 1001,
        enable_short_episodes: bool = True,
    ):

        self.batch_size = batch_size
        self.max_episode_len = max_episode_len
        self.enable_short_episodes = enable_short_episodes
        self.step = 0

    def _mode_controller(self):
        chance_of_short_episode = 0.5
        decide_if_short = (
            lambda: self.enable_short_episodes
            and np.random.rand(1)[0] > chance_of_short_episode
        )

        if self.step == 0:
            self.is_short = decide_if_short()

            if self.is_short:
                self.episode_len = np.random.randint(100, self.max_episode_len)
            else:
                self.episode_len = self.max_episode_len

        if self.step == self.episode_len:
            self._reset

    def _reset(self):
        self.step = 0
        self._mode_controller()

    def generate(self) -> Trajectory:
        self._mode_controller()

        step_type_generator = lambda: [
            np.select(
                [
                    self.step == 0,
                    self.step > 0 and self.step < self.episode_len - 1,
                    self.step == self.episode_len - 1,
                ],
                [ts.StepType.FIRST, ts.StepType.MID, ts.StepType.LAST],
            )
            for _ in range(self.batch_size)
        ]

        traj_data = {
            "step_type": step_type_generator(),
            "next_step_type": step_type_generator(),
            "observation": [
                np.random.randint(0, 256, (64, 64, 3)) for _ in range(self.batch_size)
            ],
            "action": np.random.randint(0, 15, (self.batch_size)),
            "policy_info": ["" for _ in range(self.batch_size)],
            "reward": np.random.choice(
                [0 for _ in range(self.episode_len - 1)] + [10], self.batch_size
            ),
            "discount": [0 for _ in range(self.batch_size)],
        }

        for key, values in traj_data.items():
            traj_data[key] = tf.convert_to_tensor(values)

        self.step += 1

        return Trajectory(**traj_data)


def test_step_saver(
    testing_iterations_times_episode_length: int = 10, max_episode_len: int = 10
):
    trajectory_factory = RandomTrajectoryFactory(
        enable_short_episodes=False, max_episode_len=max_episode_len
    )
    run_saver = StepSaver(batch_size=20, extend_length=11)

    for k in tqdm(range(max_episode_len)):
        generated_trajectory = trajectory_factory.generate()
        run_saver.add_data_to_queue(generated_trajectory)

    run_saver.commit_episodes()


if __name__ == "__main__":
    test_step_saver()

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class B4Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.th = 1.0
        # self.viewer = None

        high = np.array([2, 2, 2], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.th,
            high=self.th,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x1, x2, x3 = self.state
        done = False
        # u = np.clip(u, -self.th, self.th)[0]
        # u = 10 * u

        offset = 0
        scala = 1

        # offset = 0.5
        # scala = 2

        u = u[0] - offset
        u = u * scala
        # t = 0.02
        t = 0.05
        x1_new = x1 + (-x1 + x2 - x3) * t
        x2_new = x2 + (-x1 * (x3 + 1) - x2) * t
        x3_new = x3 + (-x1 + u) * t

        self.state = np.array([x1_new, x2_new, x3_new], dtype=np.float32)

        reward = (-abs(x1_new) - abs(x2_new + 0.03)) * 2

        # if 0.05 >= x1_new >= -0.05 and 0 >= x2_new >= -0.05:
        #     reward = 500
        #     done = True
        if 0.04 >= x1_new >= -0.04 and -0.014 >= x2_new >= -0.045:
            reward = 500
            done = True

        # done = bool(
        #     abs(x1_new) > 1.5 or
        #     abs(x2_new) > 1.5
        # ) or done
        #
        # if bool(
        #         abs(x1_new) > 1.5 or
        #         abs(x2_new) > 1.5
        # ):
        #     reward = -600

        return self._get_obs(), reward, done, {}

    def reset(self):
        high = np.array([0.27, 0.1, 0.27])
        low = np.array([0.25, 0.08, 0.25])
        self.state = self.np_random.uniform(low=low, high=high)

        return self._get_obs()

    def _get_obs(self):
        self.state[0] = np.clip(self.state[0], -2, 2)
        self.state[1] = np.clip(self.state[1], -2, 2)
        self.state[2] = np.clip(self.state[2], -2, 2)
        return self.state
        # return np.array(transition([np.cos(theta), np.sin(theta), thetadot]))

    def step_size(self, u, step_size=0.005):

        done = False

        offset = 0
        scala = 1
        u = u[0] - offset
        u = u * scala
        t = 0.1
        time = 0
        state_list = []
        while time <= t:
            x1, x2, x3 = self.state
            x1_new = x1 + (-x1 + x2 - x3) * step_size
            x2_new = x2 + (-x1 * (x3 + 1) - x2) * step_size
            x3_new = x3 + (-x1 + u) * step_size
            state_list.append([x1_new, x2_new])

            self.state = np.array([x1_new, x2_new, x3_new], dtype=np.float32)

            if 0.05 >= x1_new >= -0.05 and 0 >= x2_new >= -0.05:
                done = True
            time = round(time + step_size, 10)
        return self.state, state_list, done

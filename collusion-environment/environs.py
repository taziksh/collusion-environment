import gym
import numpy as np

class Cournot(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        #print(self.action_space)

        self._q = 0

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._q = 0

        return 0, 0 #observation and info

    def _profit_function(self, q):
        return -(q)**2+2*(q)

    def step(self, action):

        self._q = action

        # An episode is done iff the agent has reached the target
        terminated = False
        observation = 0
        reward = self._profit_function(self._q)
        info = 0

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info

print(f"Cournot steps: {Cournot().step(3)}")
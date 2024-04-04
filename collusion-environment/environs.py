import gym
import numpy as np

class Cournot(gym.Env):
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.action_space = gym.spaces.Discrete(3)
        #print(self.action_space)

        # self._q = 0
        self._qs = [0] * self.num_agents

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._qs = [0] * self.num_agents

        return [0] * self.num_agents, {} #observation and info

    def _profit_function(self, q, total_q):
        price = max(0, 1 - total_q * 0.1) # linear
        profit = price * q - (q**2) * 0.05 # quadratic
        return profit

    def step(self, actions):
        self._qs = actions
        total_production = sum(self._qs)

        rewards = []
        for q in self._qs:
            reward = self._profit_function(q, total_production)
            rewards.append(reward)

        observations = [0] * self.num_agents

        # An episode is done iff the agent has reached the target
        terminated = [False] * self.num_agents
        observations = [0] * self.num_agents
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminated, info

# Example Usage
# print(f"Cournot steps: {Cournot().step(3)}")
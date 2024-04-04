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

    def get_price(self):
        total_production = sum(self._qs)
        return max(0, 10 - total_production)
    
    def step(self, actions, agents):
        self._qs = actions

        # for q in self._qs:
        #     reward = self._profit_function(q, total_production)
        #     rewards.append(reward)

        observations = [0] * self.num_agents

        # An episode is done iff the agent has reached the target
        terminated = [False] * self.num_agents
        observations = [0] * self.num_agents

        rewards = []
        for agent in agents:
            reward = agent.get_profit(self.get_price())
            rewards.append(reward)

        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminated, info

# Example Usage
# print(f"Cournot steps: {Cournot().step(3)}")
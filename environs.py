import gym
import numpy as np

class Cournot(gym.Env):
    def __init__(self, num_agents,demand_quantity, demand_factor):
        self.num_agents = num_agents
        self.action_space = gym.spaces.Discrete(3)
        #print(self.action_space)

        self.u = demand_quantity
        self.v = demand_factor

        # self._q = 0
        self._qs = [0] * self.num_agents

        self.sum_qs_traj = []
        self.rewards_traj = []

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._qs = [0] * self.num_agents

        return 0, {} #observation and info

    def get_price(self):
        total_production = sum(self._qs)
        return max(0, self.u - self.v*total_production)
    
    def step(self, actions, agents):
        self._qs = actions

        # for q in self._qs:
        #     reward = self._profit_function(q, total_production)
        #     rewards.append(reward)

        sum_qs = sum(self._qs)

        # An episode is done iff the agent has reached the target
        rewards = []
        price = self.get_price()
        for agent in agents:
            reward = agent.get_profit(price)
            rewards.append(reward)

        self.rewards_traj.append(rewards)
        self.sum_qs_traj.append(sum_qs)

        if self.render_mode == "human":
            self._render_frame()

        return sum_qs, rewards

# Example Usage
# print(f"Cournot steps: {Cournot().step(3)}")
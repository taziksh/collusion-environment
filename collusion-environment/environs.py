import gym
import numpy as np
import random
import math

learning_rate = 0.9
discount_rate = 0.8
#epsilon = 0.5 # for randomly chose
beta = 0.5 # for chosing like in the paper
decay_rate= 0.005

number_actions = 3

class Cournot(gym.Env):
    def __init__(self):

        self.action_space = gym.spaces.Discrete(3)
        #print(self.action_space)

        self._q = 0

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._qs = [0,0,0]

        return 0, 0 #observation and info

    def get_price(self, qs):
        return 50 - sum(qs)

    def step(self, actions, agents):

        self._qs = actions

        # An episode is done iff the agent has reached the target
        terminated = False
        observation = 0

        rewards =[]
        for agent in agents:
            reward = agent.get_profit(self.get_price(self._qs))
            rewards.append(reward)
        info = 0

        if self.render_mode == "human":
            self._render_frame()

        return observation, rewards, terminated, info

#print(Cournot().step(3))

class Agent():
    
    def __init__(self):
        
        self.q = 0
        self.qtable = np.zeros((1, number_actions))

    def run(self):
        
        logit = lambda a: math.exp(self.qtable[0,a]/beta) / sum([math.exp(self.qtable[0,a]/beta) for a in [0,1,2]])
            
        probs = [logit(action) for action in range(0,number_actions)]
            
        collect = 0
        p = []
        for prob in probs:
            collect = prob + collect
            p.append(collect)

        r = random.uniform(0,1)
        if r<p[0]:
            action = 0
        elif r<p[1]:
            action = 1
        else:
            action = 2

        self.q = action

        return action
    
    def update(self, reward, state, action):
        self.qtable[state,action] = (1-learning_rate)*self.qtable[state,action] + learning_rate * reward

    def get_profit(self,price):
        return price*self.q - 2*self.q
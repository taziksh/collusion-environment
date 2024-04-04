import numpy as np
import random
import math

class Agent:
    def __init__(self, num_actions,num_cumul_actions, learning_rate, beta, epsilon, variable_cost, fixed_cost,memory=False):
        self.num_actions = num_actions
        self.num_cumul_actions = num_cumul_actions
        self.q = 0
        self.state = [self.q, 0]
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.w = variable_cost
        self.f = fixed_cost
        self.memory = memory
        #TODO: try np.zeros
        if memory:
            self.qtable = np.zeros((num_actions, num_cumul_actions, num_actions))
        else:
            self.qtable = np.zeros((1, 1, num_actions))

    def reset(self):
        if self.memory:
            self.qtable = np.zeros((self.num_actions, self.num_cumul_actions, self.num_actions))
        else:
            self.qtable = np.zeros((1, 1, self.num_actions))
        self.q = 0
        self.state = [0, 0]

    def softmax(self):
        total_sum = sum([np.exp(self.qtable[self.state[0], self.state[1], a]/self.beta) for a in range(self.num_actions)])

        logit = lambda a: np.exp(self.qtable[self.state[0], self.state[1], a]/self.beta) / total_sum
        probs = [logit(action) for action in range(0,self.num_actions)]
        return probs

    def get_profit(self,price):
        return price*self.q - self.w*self.q - self.f

    def select_action(self, sum_qs):
        if self.memory:
            self.state[1] = sum_qs
        probs = self.softmax()

        collect = 0
        p = []
        for prob in probs:
            collect = prob + collect
            p.append(collect)

        r = random.uniform(0, 1)

        for i in range(0, self.num_actions):
            if r < p[i]:
                action = i
                break

        self.q = action
        if self.memory:
            self.state[0]=action

        return action
               
        # probs = self.softmax()
        # action = np.random.choice(range(self.num_actions), p=probs)
        
        # # Explore
        # if np.random.rand() < self.epsilon:
        #     action = np.random.randint(self.num_actions)
        # # Exploit
        # else:
        #     action = np.argmax(self.qtable[0])
        # return action

    def decrease_epsilon(self, episode, decay_rate=0.99, min_epsilon=0.01):
        self.epsilon = max(min_epsilon, np.exp(-decay_rate*episode))

    def update_qtable(self, action, reward):
        self.qtable[self.state[0], self.state[1], action] = (1 - self.learning_rate) * self.qtable[self.state[0], self.state[1], action] + self.learning_rate * reward

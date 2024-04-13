import numpy as np
import random
import math

class Agent:
    def __init__(self, num_actions,num_cumul_actions, learning_rate,gamma, 
                 beta, epsilon, variable_cost, fixed_cost,memory=False):
        self.num_actions = num_actions
        self.num_cumul_actions = num_cumul_actions
        self.q = 0
        self.state = [self.q, 0]
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
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
        # the following makes no sense if they should learn:
        '''if self.memory:
            self.qtable = np.zeros((self.num_actions, self.num_cumul_actions, self.num_actions))
        else:
            self.qtable = np.zeros((1, 1, self.num_actions))'''
        self.q = 0
        self.state = [0, 0]

    def softmax(self):

        total_sum = sum([np.exp(self.qtable[self.state[0], self.state[1], q]/self.beta) for q in range(self.num_actions)])
        
        # handle to high values which cause math errors:
        if not total_sum < 10000000000:
            q_list = self.qtable[self.state[0], self.state[1]]
            idx = np.argmax(q_list)
            probs = [0]*len(q_list)
            probs[idx] = 1
            #print('---------------- Runtime error solved ----------------------')
            return probs
        
        soft = lambda q: np.exp(self.qtable[self.state[0], self.state[1], q]/self.beta) / total_sum
        probs = [soft(q) for q in range(self.num_actions)]
        return probs

    def get_profit(self,price):
        return price*self.q - self.w*self.q - self.f

    def select_action(self,beta):
        
        self.beta = beta
        probs = self.softmax()

        collect = 0
        p = []
        for prob in probs:
            collect = prob + collect
            p.append(collect)

        r = random.uniform(0, 1)

        action = 0

        for i in range(0, self.num_actions):
            if r < p[i]:
                action = i
                break

        self.q = action

        return action

    def decrease_epsilon(self, episode, decay_rate=0.99, min_epsilon=0.01):
        self.epsilon = max(min_epsilon, np.exp(-decay_rate*episode))

    def update_qtable(self, reward, sum_qs):
        self.qtable[self.state[0], self.state[1], self.q] = (1 - self.learning_rate) * self.qtable[self.state[0], self.state[1], self.q] + self.learning_rate * (reward +self.gamma*max(self.qtable[self.state[0], self.state[1]]))
        if self.memory:
            self.state[0] = self.q
            self.state[1] = sum_qs

import numpy as np
import random
import math

class Agent:
    def __init__(self, num_actions, learning_rate, beta, epsilon):
        self.num_actions = num_actions
        self.q = 0
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        #TODO: try np.zeros
        self.qtable = np.zeros((1, num_actions))

    def softmax(self):
        logit = lambda a: math.exp(self.qtable[0,a]/self.beta) / sum([math.exp(self.qtable[0,a]/self.beta) for a in range(self.num_actions)])
        probs = [logit(action) for action in range(0,self.num_actions)]
        return probs

    def get_profit(self,price):
        return price*self.q - 2*self.q

    def select_action(self):
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
        self.qtable[0, action] = (1 - self.learning_rate) * self.qtable[0, action] + self.learning_rate * reward

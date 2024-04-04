import numpy as np

class Agent:
    def __init__(self, num_actions, learning_rate, beta, epsilon):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        #TODO: try np.zeros
        self.qtable = np.ones((1, num_actions))

    def softmax(self):
        q_exp = np.exp(self.qtable[0] / self.beta)
        probs = q_exp / np.sum(q_exp)
        return probs

    def select_action(self):
        # probs = self.softmax()
        # action = np.random.choice(range(self.num_actions), p=probs)
        
        # Explore
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_actions)
        # Exploit
        else:
            action = np.argmax(self.qtable[0])
        return action

    def decrease_epsilon(self, episode, decay_rate=0.99, min_epsilon=0.01):
        self.epsilon = max(min_epsilon, np.exp(-decay_rate*episode))

    def update_qtable(self, action, reward):
        self.qtable[0, action] = (1 - self.learning_rate) * self.qtable[0, action] + self.learning_rate * reward

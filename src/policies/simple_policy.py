import numpy as np

class SimplePolicy(object):

    def __init__(self, env, ALPHA, GAMMA):
        self.env = env
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.states = np.linspace(-0.20943951, 0.20943951, 10)
        self.V = {}
        for state in range(len(self.states)+1):
            self.V[state] = 0        

    def chooseAction(self, state):
        action = 0 if state < 5 else 1
        return action

    def train(self):
        observation = self.env.reset()
        done = False
        rewards = 0
        while not done:
            s = int(np.digitize(observation[2], self.states))
            a = self.chooseAction(s)
            observation_, reward, done, info = self.env.step(a)
            rewards += reward
            s_ = int(np.digitize(observation_[2], self.states))            
            self.V[s] = self.V[s] + self.ALPHA*(reward + self.GAMMA*self.V[s_] - self.V[s])
            observation = observation_

        return rewards

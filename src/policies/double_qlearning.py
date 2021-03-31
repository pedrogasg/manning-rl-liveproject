import numpy as np
from policies import SarSaPolicy

class DoubleQlearning(SarSaPolicy):
    def initQ(self):
        self.Q1, self.Q2 = {}, {}
        for state in self.states:
            for action in range(2):
                self.Q1[state, action] = 0
                self.Q2[state,action] = 0

    def maxAction(self, Q1, Q2, state):    
        values = np.array([Q1[state,a] + Q2[state,a] for a in range(2)])
        action = np.argmax(values)
        return action

    def train(self):
        done = False
        rewards = 0
        observation = self.env.reset()
        while not done:
            state = self.getState(observation)
            rand = np.random.random()
            action = self.maxAction(self.Q1,self.Q2,state) if rand < (1-self.EPS) else self.env.action_space.sample()
            observation_, reward, done, info = self.env.step(action)
            rewards += reward
            state_ = self.getState(observation_)
            rand = np.random.random()
            if rand <= 0.5:
                action_ = self.maxAction(self.Q1,self.Q1,state_)
                self.Q1[state,action] = self.Q1[state,action] + self.ALPHA*(reward + self.GAMMA*self.Q2[state_,action_] - self.Q1[state,action])
            elif rand > 0.5:
                action_ = self.maxAction(self.Q2,self.Q2,state_)
                self.Q2[state,action] = self.Q2[state,action] + self.ALPHA*(reward + self.GAMMA*self.Q1[state_,action_] - self.Q2[state,action])
            observation = observation_

        self.updateEps()

        return rewards
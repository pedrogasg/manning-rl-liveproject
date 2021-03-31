import numpy as np
from policies import SarSaPolicy

class QLearningPolicy(SarSaPolicy):
    def train(self):
        observation = self.env.reset()        
        state = self.getState(observation)
        rand = np.random.random()
        action = self.chooseAction(state) if rand < (1-self.EPS) else self.env.action_space.sample()
        done = False
        rewards = 0
        while not done:
            observation_, reward, done, info = self.env.step(action)
            rewards += reward
            state_ = self.getState(observation_)
            rand = np.random.random()
            action_ = self.chooseAction(state_) if rand < (1-self.EPS) else self.env.action_space.sample()            
            self.Q[state,action] = self.Q[state,action] + self.ALPHA*(reward + self.GAMMA*self.Q[state_,action_] - self.Q[state,action])
            state, action = state_, action_
        
        self.updateEps()

        return rewards
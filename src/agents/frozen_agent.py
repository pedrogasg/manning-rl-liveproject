import numpy as np

from agents import BaseAgent

class FrozenAgent(BaseAgent):
    def __init__(self, gamma, states, policy):
        self.states = states
        super(FrozenAgent, self).__init__()
            
        self.gamma = gamma # discount parameter	
        self.policy = policy # mapping of states to actions		
        self.statesReturns = {} # states and the discounted returns that followed
		
    def initVariables(self):
        self.v = {}
        for state in self.states:
            self.v[state] = 0 

    def update(self):
        G = 0	
		# assemble discounted future rewards from the agent's memory	
        for state, reward in reversed(self.memory):			
            if state not in self.statesReturns:
                self.statesReturns[state] = [G]
            else:
                self.statesReturns[state].append(G)
            G = reward + self.gamma * G

        # use discounted future rewards to calculate averages for each state
        for state in self.statesReturns:
            self.v[state] = np.mean(self.statesReturns[state])

        self.memory = []

    def chooseAction(self, state):		
        action = self.policy[state]		
        return action
		
    def print(self):
        for state in self.v:
            print(state, '%.5f' % self.v[state])
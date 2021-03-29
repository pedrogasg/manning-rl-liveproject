import numpy as np
from environments import GridWorld

class WindyGridWorld(GridWorld):
    def __init__(self, m, n, wind):
        super(WindyGridWorld, self).__init__(m, n)
        self.wind = wind

    def initGrid(self):
        self.stateSpace = [i for i in range(self.m*self.n)]
        self.agentPosition = 0
        self.stateSpace.remove(28)
    
    def reset(self):
        self.agentPosition = 0
        self.grid = np.zeros((self.m,self.n))
        return self.agentPosition, False

    def step(self, action):
        agentX, agentY = self.getAgentRowAndColumn()
        if agentX > 0:
            resultingState = self.agentPosition + self.actionSpace[action] + \
                            self.wind[agentY] * self.actionSpace['U']
            if resultingState < 0: #if the wind is trying to push agent off grid
                resultingState += self.m
        else:
            if action == 'L' or action == 'R':
                resultingState = self.agentPosition + self.actionSpace[action]
            else:
                resultingState = self.agentPosition + self.actionSpace[action] + \
                            self.wind[agentY] * self.actionSpace['U']
        #reward = -1 if not self.isTerminalState(resultingState) else 0
        reward = -1
        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)
            return resultingState, reward, self.isTerminalState(resultingState), None
        else:
            return self.agentPosition, reward, self.isTerminalState(self.agentPosition), None
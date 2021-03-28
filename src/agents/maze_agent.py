import numpy as np 
from agents import BaseAgent
actionSpace = {'U': (-1,0), 'D': (1,0), 'L': (0,-1), 'R': (0,1)}

class MazeAgent(BaseAgent):
    def __init__(self, maze, alpha=0.15, randomFactor=0.2):
        self.maze = maze
        super(MazeAgent, self).__init__()        
        
        self.memory.append(((0,0), 0))

        self.randomFactor = randomFactor
        self.alpha = alpha   


    def chooseAction(self, state):
        allowedMoves =  self.maze.allowedStates[state]           
        maxG = -10e15    
        nextMove = None 
        randomN = np.random.random()
        if randomN < self.randomFactor:
            nextMove = np.random.choice(allowedMoves)          
        else:            
            for action in allowedMoves:
                newState = tuple([sum(x) for x in zip(state, actionSpace[action])])                                                          
                if self.G[newState] >= maxG:
                    maxG = self.G[newState]
                    nextMove = action            
        return nextMove

    def print(self):
        for i in range(6):            
            for j in range(6):
                if (i,j) in self.G.keys():
                    print('%.6f' % self.G[(i,j)], end='\t')
                else:
                    print('X', end='\t\t')
            print('\n')

    def initVariables(self):
        self.G = {}  # present value of expected future rewards
        for state in self.maze.allowedStates:     
            self.G[state] = np.random.uniform(low=-1.0, high=-0.1)

    def update(self):
        target = 0 # we only learn when we beat the maze

        for prev, reward in reversed(self.memory):                    
            self.G[prev] = self.G[prev] + self.alpha * (target - self.G[prev])            
            target += reward

        self.memory = []
        self.randomFactor -= 10e-5

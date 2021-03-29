import numpy as np

from environments import GridWorld

class PolicyIteration(object):
    def __init__(self, grid:GridWorld, GAMMA, THETA) -> None:
        V = {}
        for state in grid.stateSpacePlus:        
            V[state] = 0
        
        policy = {}
        for state in grid.stateSpace:
            policy[state] = [key for key in grid.actionSpace.keys()]
        self.V = V
        self.policy = policy
        self.grid = grid
        self.stable = False
        self.GAMMA = GAMMA
        self.THETA = THETA

    def evaluatePolicy(self):
        # policy evaluation for the random choice in gridworld
        converged = False
        while not converged:
            DELTA = 0
            for state in self.grid.stateSpace:
                oldV = self.V[state]
                total = 0
                weight = 1 / len(self.policy[state])           
                for action in self.policy[state]:
                    self.grid.setState(state)
                    newState, reward, _, _ = self.grid.step(action)
                    key = (newState, reward, state, action)
                    total += weight*self.grid.p[key]*(reward+self.GAMMA*self.V[newState])
                self.V[state] = total
                DELTA = max(DELTA, np.abs(oldV-self.V[state]))
                converged = True if DELTA < self.THETA else False

    def improvePolicy(self):
        self.stable = True
        newPolicy = {}
        for state in self.grid.stateSpace:       
            oldActions = self.policy[state]                
            value = []
            newAction = []
            for action in self.policy[state]:
                self.grid.setState(state)
                weight = 1 / len(self.policy[state])
                newState, reward, _, _ = self.grid.step(action)
                key = (newState, reward, state, action)
                value.append(np.round(weight*self.grid.p[key]*(reward+self.GAMMA*self.V[newState]), 2))
                newAction.append(action)
            value = np.array(value)        
            best = np.where(value == value.max())[0]        
            bestActions = [newAction[item] for item in best] 
            newPolicy[state] = bestActions

            if oldActions != bestActions:
                self.stable = False
            
        self.policy = newPolicy

    def iterateValues(self):
        converged = False
        while not converged:
            DELTA = 0
            for state in self.grid.stateSpace:
                oldV = self.V[state]
                newV = []            
                for action in self.grid.actionSpace:
                    self.grid.setState(state)
                    newState, reward, _, _ = self.grid.step(action)
                    key = (newState, reward, state, action) 
                    newV.append(self.grid.p[key]*(reward+self.GAMMA*self.V[newState]))                
                newV = np.array(newV)
                bestV = np.where(newV == newV.max())[0]
                bestState = np.random.choice(bestV)
                self.V[state] = newV[bestState]
                DELTA = max(DELTA, np.abs(oldV-self.V[state]))
                converged = True if DELTA < self.THETA else False

        for state in self.grid.stateSpace:
            newValues = []
            actions = []
            for action in self.grid.actionSpace:
                self.grid.setState(state)
                newState, reward, _, _ = self.grid.step(action)
                key = (newState, reward, state, action)
                newValues.append(self.grid.p[key]*(reward+(self.GAMMA*self.V[newState])))
                actions.append(action)
            newValues = np.array(newValues)
            bestActionIDX = np.where(newValues == newValues.max())[0]
            #bestActions = actions[np.random.choice(bestActionIDX)]
            bestActions = actions[bestActionIDX[0]]
            self.policy[state] = bestActions

    def print(self):
        for state in self.policy:
            print(state, self.policy[state])
        print('--------------------')
        for idx, row in enumerate(self.grid.grid):
            for idy, _ in enumerate(row):            
                state = self.grid.m * idx + idy 
                print('%.2f' % self.V[state], end='\t')
            print('\n')
        print('--------------------')


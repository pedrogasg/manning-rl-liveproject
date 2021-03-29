import numpy as np

class MCPolicyFV():
    def __init__(self, grid, GAMMA):
        policy = {}
        for state in grid.stateSpace:
            policy[state] = grid.possibleActions

        V = {}
        for state in grid.stateSpacePlus:
            V[state] = 0

        
        returns = {}
        for state in grid.stateSpace:
            returns[state] = []

        self.policy = policy
        self.V = V
        self.returns = returns
        self.grid = grid
        self.GAMMA = GAMMA

    def episode(self):
        observation, done = self.grid.reset()
        memory = []
        statesReturns = []
        while not done:
            # attempt to follow the policy
            action = np.random.choice(self.policy[observation])            
            observation_, reward, done, info = self.grid.step(action)
            memory.append((observation, action, reward))
            observation = observation_
        # append terminal state
        memory.append((observation, action, reward))

        G = 0        
        last = True
        for state, action, reward in reversed(memory): 
            if last:
                last = False
            else:
                statesReturns.append((state,G))
            G = self.GAMMA*G + reward

        statesReturns.reverse()
        statesVisited = []
        for state, G in statesReturns:
            if state not in statesVisited:
                self.returns[state].append(G)
                self.V[state] = np.mean(self.returns[state])
                statesVisited.append(state)

    def print(self):
        for idx, row in enumerate(self.grid.grid):
            for idy, _ in enumerate(row):            
                state = self.grid.m * idx + idy 
                print('%.2f' % self.V[state], end='\t')
            print('\n')
        print('--------------------')
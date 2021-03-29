import numpy as np

class MCPolicyES():
    def __init__(self, grid, GAMMA):
        policy = {}
        for state in grid.stateSpace:
            policy[state] = np.random.choice(grid.possibleActions)
        Q = {}
        returns = {}
        pairsVisited = {}
        for state in grid.stateSpacePlus:
            for action in grid.possibleActions:
                Q[(state, action)] = 0
                returns[(state,action)] = 0
                pairsVisited[(state,action)] = 0


        self.policy = policy
        self.Q = Q
        self.returns = returns
        self.pairsVisited = pairsVisited
        self.grid = grid
        self.GAMMA = GAMMA

    def episode(self):
        statesActionsReturns = []
        observation = np.random.choice(self.grid.stateSpace)
        action = np.random.choice(self.grid.possibleActions)
        self.grid.setState(observation)
        observation_, reward, done, info = self.grid.step(action)
        memory = [(observation, action, reward)]
        steps = 1
        while not done:
            action = self.policy[observation_]
            steps += 1
            observation, reward, done, info = self.grid.step(action)
            if steps > 15 and not done:
                done = True
                reward = -steps
            memory.append((observation_, action, reward))
            observation_ = observation

        # append the terminal state
        memory.append((observation_, action, reward))
        
        G = 0        
        last = True # start at t = T - 1
        for state, action, reward in reversed(memory):
            if last:
                last = False  
            else:
                statesActionsReturns.append((state,action, G))
            G = self.GAMMA*G + reward

        statesActionsReturns.reverse()
        statesAndActions = []
        for state, action, G in statesActionsReturns:
            if (state, action) not in statesAndActions:
                self.pairsVisited[(state,action)] += 1
                self.returns[(state,action)] += (1 / self.pairsVisited[(state,action)])*(G-self.returns[(state,action)])                   
                self.Q[(state,action)] = self.returns[(state,action)]
                statesAndActions.append((state,action))
                values = np.array([self.Q[(state,a)] for a in self.grid.possibleActions])
                best = np.argmax(values)
                self.policy[state] = self.grid.possibleActions[best]

    def print(self):
        for idx, row in enumerate(self.grid.grid):
            for idy, _ in enumerate(row):            
                state = self.grid.m * idx + idy            
                if state != self.grid.m * self.grid.n - 1:
                    vals = [np.round(self.Q[state,action], 5) for action in self.grid.possibleActions]
                    print(vals, end='\t')
            print('\n')
        print('--------------------')
        for idx, row in enumerate(self.grid.grid):
            for idy, _ in enumerate(row):            
                state = self.grid.m * idx + idy 
                if state in self.grid.stateSpace:
                    string = ''.join(self.policy[state])
                    print(string, end='\t')
                else:
                    print('', end='\t')
            print('\n')
        print('--------------------')  
import numpy as np

class MCAPolicy(object):
    def __init__(self, env, ALPHA, GAMMA, dt = 1.0):
        self.env = env
        self.GAMMA = GAMMA
        self.dt = dt
        self.ALPHA = ALPHA
        self.startBins()
        self.weights = {}
        self.stateSpace = np.mgrid[1:10:1, 1:10:1].reshape(2,-1).T
        for state in self.stateSpace:
            self.weights[tuple(state)] = 0
    
    def startBins(self):
        self.posBins = np.linspace(-1.2, 0.5, 8)
        self.velBins = np.linspace(-0.07, 0.07, 8)
    
    def calculateV(self, state):  
        v = self.weights[state]
        return v

    def updateWeights(self, G, state, t):
        value = self.calculateV(state)
        self.weights[state] += self.ALPHA/t*(G - value)

    def aggregateState(self, obs):
        pos = int(np.digitize(obs[0], self.posBins))
        vel = int(np.digitize(obs[1], self.velBins))
        state = (pos, vel)
        return state

    def policy(self, vel):
        #_, velocity = state
        # 0 - backward, 1 - none, 2 - forward
        if vel < 4: 
            return 0
        elif vel >= 4: 
            return 2

    def train(self):
        observation = self.env.reset()
        done = False
        memory = [] 

        while not done:
            state = self.aggregateState(observation)
            action = self.policy(state[1])
            observation_, reward, done, _ = self.env.step(action)
            #self.env.render()           
            memory.append((state, action, reward))
            observation = observation_ 
        state = self.aggregateState(observation)
        memory.append((state, action, reward))

        G = 0
        last = True
        statesReturns = []        
        for state, action, reward in reversed(memory):
            if last:
                last = False
            else:
                statesReturns.append((state, G))
            G = self.GAMMA*G + reward

        statesReturns.reverse()
        statesVisited = []
        for state, G in statesReturns:                                
            if state not in statesVisited:
                self.updateWeights(G, state, self.dt)
                statesVisited.append(state)



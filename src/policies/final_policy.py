import numpy as np

def getBins(nBins=8, nLayers=8):
    # construct the asymmetric bins
    posTileWidth = (0.5 + 1.2)/nBins*0.5
    velTileWidth = (0.07 + 0.07)/nBins*0.5
    posBins = np.zeros((nLayers,nBins))
    velBins = np.zeros((nLayers,nBins))
    for i in range(nLayers):
        posBins[i] = np.linspace(-1.2+i*posTileWidth, 0.5+i*posTileWidth/2, nBins)
        velBins[i] = np.linspace(-0.07+3*i*velTileWidth, 0.07+3*i*posTileWidth/2, nBins)    
    return posBins, velBins  

class FinalPolicy(object):
    def __init__(self, env, ALPHA, GAMMA, EPSILON, dt = 1.0, nActions=3):
        self.env = env
        self.GAMMA = GAMMA
        self.dt = dt
        self.ALPHA = ALPHA
        self.EPSILON = EPSILON
        self.startBins()
        self.weights = np.zeros(512*nActions)
        self.actions = np.arange(nActions)
    
    def startBins(self):
        self.posBins ,self.velBins = getBins()
    
    def calculateQ(self, state):     
        v = self.weights.dot(state)
        return v

    def updateWeights(self, R, state, state_, t): 
        value = self.calculateQ(state) 
        value_ = self.calculateQ(state_)
        self.weights += self.ALPHA/t*(R + self.GAMMA*value_ - value)*state

    def updateWeightsInGoal(self, state, reward):
        q = self.calculateQ(state)
        self.weights += self.ALPHA/self.dt * (reward - q) * state

    def aggregateState(self, action, obs, nTiles=8, nLayers=8, nActions=3):
        position, velocity = obs
        # 8 tilings of 8x8 grid   
        tiledState = np.zeros(nTiles*nTiles*nTiles*nActions)
        for row in range(nLayers):
            if position > self.posBins[row][0] and position < self.posBins[row][nTiles-1]:
                if velocity > self.velBins[row][0] and velocity < self.velBins[row][nTiles-1]:
                    x = np.digitize(position, self.posBins[row])
                    y = np.digitize(velocity, self.velBins[row])                
                    idx = (x+1)*(y+1)+row*nTiles**2-1+action*nLayers*nTiles**2
                    tiledState[idx] = 1.0
                else:
                    break
            else:
                break            
        return tiledState

    def policy(self, observation):
        rand = np.random.random()
        if rand < 1 - self.EPSILON:
            values = []
            for a_ in self.actions:                        
                sa = self.aggregateState(a_, observation)
                values.append(self.calculateQ(sa))
            values = np.array(values)                    
            best = np.argmax(values)
            return self.actions[best]
        else:
            return int(np.random.choice(self.actions))


    def train(self):
        steps = 0
        done = False
        observation = self.env.reset()

        action = self.policy(observation)

        while not done:
            state = self.aggregateState(action, observation)
            observation_, reward, done, _ = self.env.step(action)
            steps += 1
            if done and steps < self.env._max_episode_steps:
                self.updateWeightsInGoal(state, reward)
                break
            action_ = self.policy(observation_)

            state_ = self.aggregateState(action_, observation_)
            self.updateWeights(reward, state, state_, self.dt)
            action = action_
            observation = observation_

        return steps

        





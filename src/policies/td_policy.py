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

class TDPolicy(object):
    def __init__(self, env, ALPHA, GAMMA, dt = 1.0):
        self.env = env
        self.GAMMA = GAMMA
        self.dt = dt
        self.ALPHA = ALPHA
        self.startBins()
        self.weights = np.zeros(512)
    
    def startBins(self):
        self.posBins ,self.velBins = getBins()
    
    def calculateV(self, state):     
        v = self.weights.dot(state)
        return v

    def updateWeights(self, R, state, state_, t): 
        value = self.calculateV(state) 
        value_ = self.calculateV(state_)
        self.weights += self.ALPHA/t*(R + self.GAMMA*value_ - value)*state

    def aggregateState(self, obs, nTiles=8, nLayers=8):
        position, velocity = obs
        # 8 tilings of 8x8 grid   
        tiledState = np.zeros(nTiles*nTiles*nTiles)
        for row in range(nLayers):
            if position > self.posBins[row][0] and position < self.posBins[row][nTiles-1]:
                if velocity > self.velBins[row][0] and velocity < self.velBins[row][nTiles-1]:
                    x = np.digitize(position, self.posBins[row])
                    y = np.digitize(velocity, self.velBins[row])                
                    idx = (x+1)*(y+1)+row*nTiles**2-1
                    tiledState[idx] = 1.0
                else:
                    break
            else:
                break            
        return tiledState

    def policy(self, velocity):
        # 0 - backward, 1 - none, 2 - forward
        if velocity < 0:
            return 0
        elif velocity >= 0:
            return 2

    def train(self):
        observation = self.env.reset()
        done = False

        while not done:
            state = self.aggregateState(observation)
            action = self.policy(observation[1])
            observation_, reward, done, _ = self.env.step(action)
            #self.env.render()           
            state_ = self.aggregateState(observation_)
            self.updateWeights(reward, state, state_, self.dt)
            observation = observation_ 




